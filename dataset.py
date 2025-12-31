# dataset.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================
# 1) NumbersDataset (.pt) + collate_multi_task
# ============================================================

PaletteMode = Literal["init", "getitem", "batch"]


def _to_nchw_u8(imgs: torch.Tensor) -> torch.Tensor:
    """
    Normaliza imágenes a NCHW uint8.
    Soporta:
      - [N, H, W, C] (NHWC) uint8/float
      - [N, C, H, W] (NCHW) uint8/float
    Si viene float, asume rango 0..255 y convierte a uint8 (redondeando).
    """
    if imgs.ndim != 4:
        raise ValueError(f"Esperaba 4D, vino {tuple(imgs.shape)}")

    if imgs.dtype.is_floating_point:
        imgs = imgs.clamp(0, 255).round().to(torch.uint8)
    else:
        imgs = imgs.to(torch.uint8)

    # NHWC -> NCHW si el último dim parece ser canal
    if imgs.shape[-1] in (1, 3):
        imgs = imgs.permute(0, 3, 1, 2).contiguous()
        return imgs
    # NCHW si el segundo dim parece ser canal
    if imgs.shape[1] in (1, 3):
        return imgs.contiguous()

    raise ValueError(f"No pude inferir layout (NCHW o NHWC), shape={tuple(imgs.shape)}")


def _pick(blob: dict, *names):
    for n in names:
        if n in blob:
            return n
    return None


class NumbersDataset(Dataset):
    """
    Lee un .pt con cualquiera de los siguientes formatos de imagen:
      - RGB: 'images' = [N,3,H,W] u8  o  [N,H,W,3] u8 (o float 0..255)
      - INDEXADO: 'images' = [N,H,W] u8 y 'palette' = [K,3] u8

    Además puede contener (opcionales):
      - 'labels'        : [N] int (número)
      - 'rotation_deg'  o 'rot_deg' : [N] float
      - 'scale_vals'    o 'scale'   : [N] float
      - 'font_index'                : [N] int
      - 'font_vocab'                : list[str]
      - 'color_index'               : [N] int
      - 'color_rgb'                 : [N,3] u8

    Modo de remapeo si es INDEXADO:
      palette_map_mode = "init"    -> remapea en __init__, guarda RGB en RAM (rápido en batch)
      palette_map_mode = "getitem" -> remapea por muestra en __getitem__ (menos RAM)
      palette_map_mode = "batch"   -> remapea por batch en collate (dataset devuelve índices)

    Salida de __getitem__:
      - Siempre entrega 'image':
          * u8 [C,H,W] si keep_uint8=True
          * float32 [C,H,W] (0..1) si keep_uint8=False
        EXCEPTO en "batch", donde __getitem__ entregará 'image_idx' y 'palette'
        y el remapeo a RGB se hace en el collate.
      - 'number', 'rotation_deg', 'scale', 'font_id', 'color_id', 'color_rgb' si disponibles
      - 'meta' con 'font_table' y 'palette' (si return_meta=True)
    """

    def __init__(
        self,
        pt_path: str,
        *,
        transform=None,
        return_meta: bool = False,
        keep_uint8: bool = False,
        palette_map_mode: PaletteMode = "init",  # "init" | "getitem" | "batch"
    ):
        if palette_map_mode not in ("init", "getitem", "batch"):
            raise ValueError("palette_map_mode debe ser 'init', 'getitem' o 'batch'")

        if not os.path.isfile(pt_path):
            raise FileNotFoundError(pt_path)

        blob = torch.load(pt_path, map_location="cpu")

        # ---------- IMÁGENES ----------
        if "images" not in blob:
            raise KeyError(f"El .pt no contiene 'images'. Keys: {list(blob.keys())}")
        imgs = blob["images"]

        self.palette = blob.get("palette", None)
        self._palette_u8: Optional[torch.Tensor] = (
            torch.as_tensor(self.palette, dtype=torch.uint8) if self.palette is not None else None
        )

        self._mode = palette_map_mode
        self.transform = transform
        self.return_meta = bool(return_meta)
        self.keep_uint8 = bool(keep_uint8)

        self.images_u8: Optional[torch.Tensor] = None  # NCHW u8 si ya está en RGB
        self.idx_images: Optional[torch.Tensor] = None  # [N,H,W] u8 si indexado
        self.N: int = 0

        if imgs.ndim == 3 and self._palette_u8 is not None:
            # INDEXADO [N,H,W] + paleta -> según modo
            if self._mode == "init":
                idx = imgs.long()  # [N,H,W]
                rgb = self._palette_u8[idx]  # [N,H,W,3] u8
                self.images_u8 = rgb.permute(0, 3, 1, 2).contiguous()  # NCHW
                self.N = self.images_u8.size(0)
            elif self._mode in ("getitem", "batch"):
                self.idx_images = imgs.to(torch.uint8).contiguous()  # [N,H,W]
                self.N = self.idx_images.size(0)
            else:
                raise AssertionError("Modo inválido (chequeado arriba)")
        else:
            # RGB directo (NHWC/NCHW float/u8)
            self.images_u8 = _to_nchw_u8(imgs)  # NCHW u8
            self.N = self.images_u8.size(0)

        # Si accidentalmente quedó 1 canal, replicamos a 3
        if self.images_u8 is not None and self.images_u8.size(1) == 1:
            self.images_u8 = self.images_u8.repeat(1, 3, 1, 1)

        # ---------- ATRIBUTOS / ETIQUETAS ----------
        if "labels" not in blob:
            raise KeyError("No se encontró 'labels' en el .pt")

        self.labels = torch.as_tensor(blob["labels"], dtype=torch.long)
        if self.labels.numel() != self.N:
            raise ValueError("labels y images tienen distinto N")

        rot_key = _pick(blob, "rotation_deg", "rot_deg")
        self.rot = torch.as_tensor(blob[rot_key], dtype=torch.float32) if rot_key else None

        scale_key = _pick(blob, "scale_vals", "scale")
        self.scale = torch.as_tensor(blob[scale_key], dtype=torch.float32) if scale_key else None

        self.font_id = blob.get("font_index", None)
        if isinstance(self.font_id, torch.Tensor):
            self.font_id = self.font_id.to(torch.long)
        self.font_table: Optional[List[str]] = blob.get("font_vocab", None)

        self.color_id = blob.get("color_index", None)
        if isinstance(self.color_id, torch.Tensor):
            self.color_id = self.color_id.to(torch.long)

        # color_rgb: si ya viene, lo respetamos.
        crgb = blob.get("color_rgb", None)
        if isinstance(crgb, torch.Tensor):
            if crgb.dtype != torch.uint8:
                crgb = crgb.clamp(0, 255).round().to(torch.uint8)
            self.color_rgb = crgb
        else:
            self.color_rgb = None

    def __len__(self) -> int:
        return self.N

    def _remap_index_to_rgb_u8(self, idx_img: torch.Tensor) -> torch.Tensor:
        if self._palette_u8 is None:
            raise RuntimeError("No hay paleta para remapear índices.")
        rgb = self._palette_u8[idx_img.long()]  # [H,W,3] u8
        return rgb.permute(2, 0, 1).contiguous()  # [3,H,W] u8

    def __getitem__(self, i: int) -> Dict[str, Any]:
        # --- imagen ---
        if self._mode == "batch" and self.idx_images is not None:
            # Delega remapeo al collate
            sample_img_dict = {
                "image_idx": self.idx_images[i],  # [H,W] u8
                "palette": self.palette,
            }
        else:
            if self.images_u8 is not None:
                img_u8 = self.images_u8[i]  # [C,H,W] u8
            else:
                img_u8 = self._remap_index_to_rgb_u8(self.idx_images[i])  # [3,H,W] u8

            img = img_u8 if self.keep_uint8 else img_u8.float().div(255.0)

            if self.transform is not None:
                img = self.transform(img)

            sample_img_dict = {"image": img}

        out: Dict[str, Any] = {**sample_img_dict, "number": self.labels[i]}

        if self.rot is not None:
            out["rotation_deg"] = self.rot[i]
        if self.scale is not None:
            out["scale"] = self.scale[i]
        if self.font_id is not None:
            out["font_id"] = self.font_id[i]
        if self.color_id is not None:
            out["color_id"] = self.color_id[i]

        if self.color_rgb is not None:
            out["color_rgb"] = self.color_rgb[i]
        elif (self.color_id is not None) and (self._palette_u8 is not None):
            out["color_rgb"] = self._palette_u8[self.color_id[i].long()]  # [3] u8

        if self.return_meta:
            meta: Dict[str, Any] = {}
            if self.font_table is not None:
                meta["font_table"] = self.font_table
                if "font_id" in out:
                    fid = int(out["font_id"])
                    if 0 <= fid < len(self.font_table):
                        meta["font_name"] = self.font_table[fid]
            if self.palette is not None:
                meta["palette"] = self.palette
            out["meta"] = meta

        return out


def collate_multi_task(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # --- imágenes ---
    if "image" in batch[0]:
        out["image"] = torch.stack([b["image"] for b in batch], dim=0)
    elif "image_idx" in batch[0]:
        idx = torch.stack([b["image_idx"] for b in batch], dim=0).long()  # [B,H,W]
        pal = torch.as_tensor(batch[0]["palette"], dtype=torch.uint8)  # [K,3]
        imgs_u8 = pal[idx]  # [B,H,W,3] u8
        out["image"] = imgs_u8.permute(0, 3, 1, 2).float().div(255.0)  # [B,3,H,W] float
    else:
        raise KeyError("Ni 'image' ni 'image_idx' presentes en el batch.")

    # --- obligatorios/opcionales ---
    out["number"] = torch.stack([b["number"] for b in batch], dim=0)

    if "rotation_deg" in batch[0]:
        out["rotation_deg"] = torch.stack([b["rotation_deg"] for b in batch], dim=0)
        rot_rad = out["rotation_deg"] * (np.pi / 180.0)
        out["rot_rad"] = rot_rad
        out["rot_sin"] = torch.sin(rot_rad)
        out["rot_cos"] = torch.cos(rot_rad)

    if "scale" in batch[0]:
        out["scale"] = torch.stack([b["scale"] for b in batch], dim=0)
    if "font_id" in batch[0]:
        out["font_id"] = torch.stack([b["font_id"] for b in batch], dim=0)
    if "color_id" in batch[0]:
        out["color_id"] = torch.stack([b["color_id"] for b in batch], dim=0)
    if "color_rgb" in batch[0]:
        out["color_rgb"] = torch.stack([b["color_rgb"] for b in batch], dim=0)

    if "meta" in batch[0]:
        out["meta"] = [b["meta"] for b in batch]

    return out


# ============================================================
# 2) IDG (.npz) dataset loader (dsprites / shapes3d / mpi3d)
# ============================================================

IDGDatasetName = Literal["dsprites", "idsprites", "shapes3d", "mpi3d"]
IDGSplitName = Literal["random", "composition", "interpolation", "extrapolation"]
IDGMode = Literal["train", "test"]


def _default_factor_names(dataset: IDGDatasetName, n_factors: int) -> List[str]:
    if dataset in ("dsprites", "idsprites"):
        base6 = ["color", "shape", "scale", "rotation", "pos_x", "pos_y"]
        base5 = ["shape", "scale", "rotation", "pos_x", "pos_y"]
        if n_factors == 6:
            return base6
        if n_factors == 5:
            return base5
        return [f"factor_{i}" for i in range(n_factors)]

    if dataset == "shapes3d":
        base6 = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
        if n_factors == 6:
            return base6
        return [f"factor_{i}" for i in range(n_factors)]

    if dataset == "mpi3d":
        base7 = [
            "object_color",
            "object_shape",
            "object_size",
            "camera_height",
            "background_color",
            "robot_arm_horizontal",
            "robot_arm_vertical",
        ]
        if n_factors == 7:
            return base7
        return [f"factor_{i}" for i in range(n_factors)]

    raise ValueError(f"Dataset no soportado: {dataset}")


def _load_npz_array(npz_path: Union[str, Path], prefer_keys: Sequence[str]) -> np.ndarray:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"No existe: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        for k in prefer_keys:
            if k in data.files:
                return data[k]
        return data[data.files[0]]


def _resolve_file(root: Union[str, Path], dataset: str, filename: str) -> Path:
    root = Path(root)
    p1 = root / filename
    if p1.exists():
        return p1
    p2 = root / dataset / filename
    if p2.exists():
        return p2
    raise FileNotFoundError(
        f"No encontré '{filename}' en '{root}' ni en '{root / dataset}'. "
        f"(Asegúrate de descargar los .npz con ese nombre.)"
    )


def idg_filename(dataset: IDGDatasetName, split: IDGSplitName, mode: IDGMode, kind: Literal["images", "labels"]) -> str:
    if dataset == "idsprites":
        dataset = "dsprites"
    return f"{dataset}_{split}_{mode}_{kind}.npz"


class IDGBenchmarkDataset(Dataset):
    """
    __getitem__ -> (image, latents, factor_names)
      - image: torch.FloatTensor [C,H,W] en [0,1] (por defecto)
      - latents: torch.LongTensor [F] (por defecto)
      - factor_names: List[str]
    """

    def __init__(
        self,
        root: Union[str, Path],
        dataset: IDGDatasetName,
        split: IDGSplitName,
        mode: IDGMode,
        image_as_float: bool = True,
        latents_dtype: torch.dtype = torch.long,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        image_key_candidates: Sequence[str] = ("images", "x", "arr_0"),
        label_key_candidates: Sequence[str] = ("labels", "y", "arr_0"),
    ):
        self.root = Path(root)
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.image_as_float = image_as_float
        self.latents_dtype = latents_dtype
        self.transform = transform

        img_file = idg_filename(dataset, split, mode, "images")
        lab_file = idg_filename(dataset, split, mode, "labels")

        disk_dataset = "dsprites" if dataset == "idsprites" else dataset
        img_path = _resolve_file(self.root, disk_dataset, img_file)
        lab_path = _resolve_file(self.root, disk_dataset, lab_file)

        self._images = _load_npz_array(img_path, image_key_candidates)
        self._labels = _load_npz_array(lab_path, label_key_candidates)

        if len(self._images) != len(self._labels):
            raise ValueError(f"Len mismatch: images={len(self._images)} labels={len(self._labels)}")

        n_factors = 1 if self._labels.ndim == 1 else self._labels.shape[1]
        self.factor_names: List[str] = _default_factor_names(dataset, n_factors)

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def _to_chw_tensor(self, img: np.ndarray) -> torch.Tensor:
        if img.ndim == 2:
            t = torch.from_numpy(img)[None, ...]  # [1,H,W]
        elif img.ndim == 3:
            # [C,H,W] o [H,W,C]
            if img.shape[0] in (1, 3) and img.shape[1] != img.shape[0]:
                t = torch.from_numpy(img)  # ya CHW
            else:
                t = torch.from_numpy(img).permute(2, 0, 1)  # HWC->CHW
        else:
            raise ValueError(f"Forma de imagen no soportada: {img.shape}")

        # ✅ Para ResNet: si viene 1 canal, replicamos a 3
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)

        if self.image_as_float:
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
            if t.max() > 1.5:
                t = t / 255.0
        return t

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        img_np = self._images[idx]
        lab_np = self._labels[idx]

        img = self._to_chw_tensor(img_np)
        if self.transform is not None:
            img = self.transform(img)

        latents = torch.as_tensor(lab_np, dtype=self.latents_dtype)
        return img, latents, self.factor_names


def idg_collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, List[str]]]):
    imgs, lats, names = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    lats = torch.stack(lats, dim=0) if lats[0].ndim > 0 else torch.tensor(lats)
    factor_names = names[0]
    return imgs, lats, factor_names


@dataclass
class IDGDataConfig:
    root: Union[str, Path]
    dataset: IDGDatasetName
    split: IDGSplitName

    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    make_val: bool = True
    val_fraction: float = 0.1
    val_seed: int = 0

    image_as_float: bool = True
    latents_dtype: torch.dtype = torch.long

    shuffle_train: bool = True
    drop_last_train: bool = True


def make_idg_dataloaders(
    cfg: IDGDataConfig,
    build: Iterable[Literal["train", "val", "test"]] = ("train", "val", "test"),
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, DataLoader]:
    build = set(build)
    out: Dict[str, DataLoader] = {}

    train_ds_full = IDGBenchmarkDataset(
        root=cfg.root,
        dataset=cfg.dataset,
        split=cfg.split,
        mode="train",
        image_as_float=cfg.image_as_float,
        latents_dtype=cfg.latents_dtype,
        transform=transform,
    )

    if "train" in build or "val" in build:
        if cfg.make_val and ("val" in build):
            if not (0.0 < cfg.val_fraction < 1.0):
                raise ValueError("val_fraction debe estar en (0,1) si make_val=True")

            n_total = len(train_ds_full)
            n_val = max(1, int(round(n_total * cfg.val_fraction)))
            n_train = n_total - n_val
            gen = torch.Generator().manual_seed(cfg.val_seed)
            train_ds, val_ds = random_split(train_ds_full, [n_train, n_val], generator=gen)

            if "train" in build:
                out["train"] = DataLoader(
                    train_ds,
                    batch_size=cfg.batch_size,
                    shuffle=cfg.shuffle_train,
                    num_workers=cfg.num_workers,
                    pin_memory=cfg.pin_memory,
                    drop_last=cfg.drop_last_train,
                    collate_fn=idg_collate_fn,
                    persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
                )

            if "val" in build:
                out["val"] = DataLoader(
                    val_ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                    pin_memory=cfg.pin_memory,
                    drop_last=False,
                    collate_fn=idg_collate_fn,
                    persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
                )
        else:
            if "train" in build:
                out["train"] = DataLoader(
                    train_ds_full,
                    batch_size=cfg.batch_size,
                    shuffle=cfg.shuffle_train,
                    num_workers=cfg.num_workers,
                    pin_memory=cfg.pin_memory,
                    drop_last=cfg.drop_last_train,
                    collate_fn=idg_collate_fn,
                    persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
                )

    if "test" in build:
        test_ds = IDGBenchmarkDataset(
            root=cfg.root,
            dataset=cfg.dataset,
            split=cfg.split,
            mode="test",
            image_as_float=cfg.image_as_float,
            latents_dtype=cfg.latents_dtype,
            transform=transform,
        )
        out["test"] = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
            collate_fn=idg_collate_fn,
            persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        )

    return out
