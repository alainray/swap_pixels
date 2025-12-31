# train.py
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from dataset import (
    NumbersDataset,
    collate_multi_task,
    IDGDataConfig,
    make_idg_dataloaders,
    IDGBenchmarkDataset,
)
from model import DNModel, Manipulator


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    # dataset selector
    dataset: str = "numbers"  # numbers | dsprites | idsprites | shapes3d | mpi3d
    # Numbers (.pt)
    pt_path: str = "../datasets/exports/train_64x64.pt"
    # IDG (.npz)
    idg_root: str = "../datasets"
    idg_split: str = "composition"  # random|composition|interpolation|extrapolation

    # tasks toggle (numbers)
    enable_number: bool = True
    enable_scale: bool = True
    enable_rot: bool = True
    enable_font: bool = True
    enable_color: bool = True

    # training
    batch_size: int = 256
    epochs: int = 100
    lr: float = 2e-3
    weight_decay: float = 1e-4
    device: str = "cuda"

    train_mode: str = "manip_swap"  # baseline | manipulation | swap | manip_swap

    # architecture
    encoder_name: str = "resnet18"
    z_dim: int = 512
    use_pretrained_encoder: bool = False

    # heads behavior
    split_heads_inv_eq: bool = False  # for swap/manip_swap we force True
    inv_factors: List[str] = field(default_factory=list)  # for IDG: which factors are "inv"

    # classification/regression (numbers)
    use_classification: bool = False
    num_classes_number: int = 5000
    num_classes_scale: int = 6
    num_classes_rot: int = 8
    NUM_LOSS_WEIGHT: float = 50.0

    # scale/rot discretization for metrics
    scale_start: float = 1.0
    scale_step: float = 0.2
    rot_step: float = 45.0

    # logging / output
    out_dir: str = "./runs_dn"
    run_name: str = "baseline"
    log_csv: str = "metrics.csv"
    num_workers: int = 0
    pin_memory: bool = True

    val_split_pct: float = 0.1
    debug_max_samples: Optional[int] = None
    print_freq: int = 500
    checkpoint_freq: int = 10

    seed: int = 111
    pair_seed: int = 12345

    resume_from: Optional[str] = None
    auto_resume: bool = False

    # internal (set at runtime)
    task_type: str = "numbers"  # numbers | factors
    num_classes_for_classification: Dict[str, int] = field(default_factory=dict)


def parse_args_to_cfg() -> TrainConfig:
    p = argparse.ArgumentParser("train.py (Numbers + IDG npz)")

    # dataset choice
    p.add_argument("--dataset", type=str, choices=["numbers", "dsprites", "idsprites", "shapes3d", "mpi3d"], default=TrainConfig.dataset)

    # numbers
    p.add_argument("--pt_path", type=str, default=TrainConfig.pt_path)

    # idg
    p.add_argument("--idg_root", type=str, default=TrainConfig.idg_root)
    p.add_argument("--idg_split", type=str, choices=["random", "composition", "interpolation", "extrapolation"], default=TrainConfig.idg_split)
    p.add_argument("--inv_factors", type=str, default=None, help="CSV: e.g. shape,object_hue (IDG only)")

    # training / io
    p.add_argument("--out_dir", type=str, default=TrainConfig.out_dir)
    p.add_argument("--run_name", type=str, default=TrainConfig.run_name)
    p.add_argument("--log_csv", type=str, default=TrainConfig.log_csv)

    p.add_argument("--device", type=str, default=TrainConfig.device)
    p.add_argument("--encoder", dest="encoder_name", type=str, default=TrainConfig.encoder_name)
    p.add_argument("--z_dim", type=int, default=TrainConfig.z_dim)
    p.add_argument("--use_pretrained_encoder", action="store_true")

    p.add_argument("--train_mode", type=str, choices=["baseline", "manipulation", "swap", "manip_swap"], default=TrainConfig.train_mode)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--val_percent", dest="val_split_pct", type=float, default=TrainConfig.val_split_pct)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--pair_seed", type=int, default=TrainConfig.pair_seed)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)

    p.add_argument("--split_heads_inv_eq", action="store_true")

    # numbers toggles
    p.add_argument("--use_classification", action="store_true")
    p.add_argument("--no_number", dest="enable_number", action="store_false")
    p.add_argument("--no_scale", dest="enable_scale", action="store_false")
    p.add_argument("--no_rot", dest="enable_rot", action="store_false")
    p.add_argument("--no_font", dest="enable_font", action="store_false")
    p.add_argument("--no_color", dest="enable_color", action="store_false")

    # logging/debug
    p.add_argument("--log_every", dest="print_freq", type=int, default=TrainConfig.print_freq)
    p.add_argument("--save_every", dest="checkpoint_freq", type=int, default=TrainConfig.checkpoint_freq)
    p.add_argument("--debug_max_samples", type=int, default=None)
    p.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")

    # resume
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--auto_resume", action="store_true")

    args = p.parse_args()
    cfg = TrainConfig()
    for k, v in vars(args).items():
        setattr(cfg, k, v)

    # parse inv_factors
    if args.inv_factors is not None:
        cfg.inv_factors = [x.strip() for x in args.inv_factors.split(",") if x.strip()]
    else:
        cfg.inv_factors = []

    return cfg


# ============================================================
# Utils
# ============================================================

def set_global_seeds(seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_meta_from_pt(pt_path: str) -> Dict[str, int]:
    blob = torch.load(pt_path, map_location="cpu")
    num_fonts = len(blob.get("font_vocab", []))
    if "palette" in blob:
        num_colors = int(blob["palette"].shape[0])
    elif "color_rgb" in blob:
        num_colors = int(blob["color_rgb"].unique(dim=0).size(0))
    else:
        num_colors = 0
    labels = blob["labels"]
    y_max = int(labels.max().item()) if torch.is_tensor(labels) else int(max(labels))
    return {"num_fonts": num_fonts, "num_colors": num_colors, "y_max": y_max}


def infer_num_classes_from_idg(ds: IDGBenchmarkDataset) -> Dict[str, int]:
    labels = ds._labels
    if labels.ndim == 1:
        labels = labels[:, None]
    out = {}
    for j, name in enumerate(ds.factor_names):
        col = labels[:, j]
        mn, mx = int(col.min()), int(col.max())
        out[name] = (mx + 1) if mn >= 0 else int(np.unique(col).size)
    return out


def normalize_batch_to_dict(batch):
    """
    Numbers: batch ya es dict
    IDG: batch = (imgs, lats, factor_names) -> dict con keys factor_names
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        imgs, lats, names = batch
        if lats.ndim == 1:
            lats = lats[:, None]
        out = {"image": imgs}
        for j, n in enumerate(names):
            out[n] = lats[:, j]
        return out, list(names)
    return batch, None


def to_device_batch(batch: Dict, device: str) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ============================================================
# Loss Computers
# ============================================================

class LossComputerNumbers:
    def __init__(self, cfg: TrainConfig, y_max: int, have_scale: bool, have_rot: bool):
        self.cfg = cfg
        self.y_max = max(1, int(y_max))
        self.have_scale = have_scale
        self.have_rot = have_rot
        self.use_cls = bool(cfg.use_classification)

    def targets_from_batch(self, batch: Dict) -> Dict:
        y = {}
        y["number_norm"] = batch["number"].float() / float(self.y_max)
        y["number_raw"] = batch["number"].long()

        if self.have_scale and "scale" in batch:
            y["scale"] = batch["scale"].float()

        if self.have_rot and "rotation_deg" in batch:
            y["rotation_deg"] = batch["rotation_deg"].float()
            if "rot_sin" in batch and "rot_cos" in batch and "rot_rad" in batch:
                y["rot_sin"] = batch["rot_sin"].float()
                y["rot_cos"] = batch["rot_cos"].float()
                y["rot_rad"] = batch["rot_rad"].float()

        if "font_id" in batch:
            y["font_id"] = batch["font_id"].long()
        if "color_id" in batch:
            y["color_id"] = batch["color_id"].long()
        return y

    def compute(self, preds: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = 0.0
        metrics: Dict[str, float] = {}

        # ---- number ----
        if self.use_cls:
            if "number_logits" in preds:
                tgt = targets["number_raw"]
                ln = F.cross_entropy(preds["number_logits"], tgt) * float(self.cfg.NUM_LOSS_WEIGHT)
                loss = loss + ln
                pred = preds["number_logits"].argmax(1)
                metrics["acc_num"] = (pred == tgt).float().mean().item()
                metrics["mae_num"] = F.l1_loss(pred.float(), tgt.float()).item()
        else:
            ln = F.mse_loss(preds["number"], targets["number_norm"]) * float(self.cfg.NUM_LOSS_WEIGHT)
            loss = loss + ln
            pred_val = preds["number"].detach() * self.y_max
            pred_int = torch.round(pred_val).clamp(0, self.y_max).long()
            metrics["acc_num"] = (pred_int == targets["number_raw"]).float().mean().item()
            metrics["mae_num"] = (F.l1_loss(preds["number"], targets["number_norm"]).item() * self.y_max)

        # ---- scale ----
        if self.have_scale and "scale" in targets:
            tgt_val = targets["scale"]
            if self.use_cls and "scale_logits" in preds:
                tgt_idx = torch.round((tgt_val - self.cfg.scale_start) / self.cfg.scale_step).long()
                ls = F.cross_entropy(preds["scale_logits"], tgt_idx)
                loss = loss + ls
                pred_idx = preds["scale_logits"].argmax(1)
                metrics["acc_scale"] = (pred_idx == tgt_idx).float().mean().item()
                pred_val = pred_idx.float() * self.cfg.scale_step + self.cfg.scale_start
                metrics["mae_scale"] = F.l1_loss(pred_val, tgt_val).item()
            elif (not self.use_cls) and ("scale" in preds):
                ls = F.mse_loss(preds["scale"], tgt_val)
                loss = loss + ls
                metrics["mae_scale"] = F.l1_loss(preds["scale"], tgt_val).item()
                pred_steps = torch.round((preds["scale"].detach() - self.cfg.scale_start) / self.cfg.scale_step)
                pred_disc = pred_steps * self.cfg.scale_step + self.cfg.scale_start
                metrics["acc_scale"] = (torch.abs(pred_disc - tgt_val) < 1e-4).float().mean().item()

        # ---- rotation ----
        if self.have_rot and "rotation_deg" in targets:
            if self.use_cls and "rot_logits" in preds:
                tgt_deg = torch.remainder(targets["rotation_deg"], 360.0)
                tgt_idx = torch.round(tgt_deg / self.cfg.rot_step).long() % int(self.cfg.num_classes_rot)
                lr = F.cross_entropy(preds["rot_logits"], tgt_idx)
                loss = loss + lr
                pred_idx = preds["rot_logits"].argmax(1)
                metrics["acc_rot"] = (pred_idx == tgt_idx).float().mean().item()
            elif (not self.use_cls) and ("rot_sin" in preds) and ("rot_cos" in preds) and ("rot_rad" in targets):
                lr_sin = F.mse_loss(preds["rot_sin"], targets["rot_sin"])
                lr_cos = F.mse_loss(preds["rot_cos"], targets["rot_cos"])
                lr = lr_sin + lr_cos
                loss = loss + lr
                # accuracy discretizada
                pred_rad = torch.atan2(preds["rot_sin"].detach(), preds["rot_cos"].detach())
                pred_deg = torch.remainder(pred_rad * (180.0 / math.pi), 360.0)
                tgt_deg = torch.remainder(targets["rotation_deg"], 360.0)
                pred_class = torch.round(pred_deg / self.cfg.rot_step) * self.cfg.rot_step
                metrics["acc_rot"] = (torch.abs(pred_class - tgt_deg) < 1e-3).float().mean().item()

        # ---- font / color ----
        if "font_id" in targets and "font_logits" in preds:
            lf = F.cross_entropy(preds["font_logits"], targets["font_id"])
            loss = loss + lf
            metrics["acc_font"] = (preds["font_logits"].argmax(1) == targets["font_id"]).float().mean().item()

        if "color_id" in targets and "color_logits" in preds:
            lc = F.cross_entropy(preds["color_logits"], targets["color_id"])
            loss = loss + lc
            metrics["acc_color"] = (preds["color_logits"].argmax(1) == targets["color_id"]).float().mean().item()

        metrics["loss"] = float(loss.item())
        return loss, metrics


class LossComputerFactors:
    def __init__(self, factor_names: List[str]):
        self.factor_names = list(factor_names)

    def targets_from_batch(self, batch: Dict) -> Dict:
        return {n: batch[n].long() for n in self.factor_names}

    def compute(self, preds: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = 0.0
        metrics: Dict[str, float] = {}
        accs = []
        for n in self.factor_names:
            logits = preds[f"{n}_logits"]
            tgt = targets[n]
            l = F.cross_entropy(logits, tgt)
            loss = loss + l
            acc = (logits.argmax(1) == tgt).float().mean().item()
            metrics[f"acc_{n}"] = acc
            accs.append(acc)
        metrics["acc_mean"] = float(np.mean(accs)) if accs else 0.0
        metrics["loss"] = float(loss.item())
        return loss, metrics


# ============================================================
# Swap/Manip helpers
# ============================================================

def build_targets_swap_mixed_numbers(yi: dict, yj: dict, inv_from: str) -> dict:
    assert inv_from in ("i", "j")
    src_inv = yi if inv_from == "i" else yj
    src_eq = yj if inv_from == "i" else yi
    t = {}
    for k in ("number_norm", "number_raw"):
        t[k] = src_inv[k]
    for k in ("scale", "rot_sin", "rot_cos", "rotation_deg", "rot_rad", "font_id", "color_id"):
        if k in src_eq:
            t[k] = src_eq[k]
    return t


def build_targets_swap_mixed_factors(yi: dict, yj: dict, inv_from: str, factor_names: List[str], inv_factors: List[str]) -> dict:
    assert inv_from in ("i", "j")
    inv_set = set(inv_factors)
    src_inv = yi if inv_from == "i" else yj
    src_eq = yj if inv_from == "i" else yi
    out = {}
    for n in factor_names:
        out[n] = src_inv[n] if n in inv_set else src_eq[n]
    return out


def control_vector_numbers(yi: dict, yj: dict, device: torch.device) -> torch.Tensor:
    """
    ctrl_dim = 5 fixed: [num, scale, rot_sin, font, color]
    """
    d_num = (yj["number_norm"] - yi["number_norm"]).unsqueeze(-1)

    d_scale = (yj["scale"] - yi["scale"]).unsqueeze(-1) if ("scale" in yi and "scale" in yj) else torch.zeros_like(d_num)
    d_rot = (yj["rot_sin"] - yi["rot_sin"]).unsqueeze(-1) if ("rot_sin" in yi and "rot_sin" in yj) else torch.zeros_like(d_num)
    d_font = (yj["font_id"].float() - yi["font_id"].float()).unsqueeze(-1) if ("font_id" in yi and "font_id" in yj) else torch.zeros_like(d_num)
    d_color = (yj["color_id"].float() - yi["color_id"].float()).unsqueeze(-1) if ("color_id" in yi and "color_id" in yj) else torch.zeros_like(d_num)

    return torch.cat([d_num, d_scale, d_rot, d_font, d_color], dim=1).to(device)


def control_vector_factors(yi: dict, yj: dict, factor_names: List[str], num_classes: Dict[str, int], device: torch.device) -> torch.Tensor:
    vecs = []
    for n in factor_names:
        denom = max(1, int(num_classes[n]) - 1)
        vecs.append(((yj[n].float() - yi[n].float()) / float(denom)).unsqueeze(-1))
    return torch.cat(vecs, dim=1).to(device)


# ============================================================
# Epoch runners
# ============================================================

def log_batch_metrics(epoch, it, metrics, header="", max_keys=12):
    keys = ["loss"]
    extras = sorted([k for k in metrics.keys() if k.startswith("acc_") or k.startswith("mae_") or k.startswith("acc")])
    keys += extras[:max_keys]
    parts = []
    for k in keys:
        if k in metrics:
            v = metrics[k]
            if "acc" in k:
                parts.append(f"{k}:{v*100:.1f}%")
            else:
                parts.append(f"{k}:{v:.4f}")
    print(f"[{header}] Ep {epoch} | Batch {it} | " + " | ".join(parts))


def run_epoch_baseline(model, losscomp, loader, cfg: TrainConfig, epoch, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    agg = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Ep {epoch} {'Train' if is_train else 'Val'}")
    for it, batch in pbar:
        batch, _ = normalize_batch_to_dict(batch)
        batch = to_device_batch(batch, cfg.device)
        imgs = batch["image"]

        outs = model(imgs)
        targets = losscomp.targets_from_batch(batch)
        loss, metr = losscomp.compute(outs, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        agg.append(metr)
        if is_train and it % cfg.print_freq == 0 and it > 0:
            log_batch_metrics(epoch, it, metr, "Base")

    keys = agg[0].keys()
    return {k: float(sum(d[k] for d in agg) / len(agg)) for k in keys}


def run_epoch_swap(model, losscomp, loader, cfg: TrainConfig, epoch, optimizer=None, factor_names: Optional[List[str]] = None):
    is_train = optimizer is not None
    model.train(is_train)
    agg = []

    assert cfg.z_dim % 2 == 0, "swap requiere z_dim par"
    d_half = cfg.z_dim // 2

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Ep {epoch} Swap {'Train' if is_train else 'Val'}")
    for it, batch in pbar:
        batch, _ = normalize_batch_to_dict(batch)
        batch = to_device_batch(batch, cfg.device)

        imgs = batch["image"]
        B = imgs.size(0)

        # perm determinista por batch
        g = torch.Generator(device=imgs.device)
        g.manual_seed(cfg.pair_seed + it)
        perm = torch.randperm(B, generator=g, device=imgs.device)

        imgs_j = imgs[perm] # if is_train else imgs

        yi = losscomp.targets_from_batch(batch)

        batch_p = {k: (v[perm] if torch.is_tensor(v) else v) for k, v in batch.items() if torch.is_tensor(v)}
        yj = losscomp.targets_from_batch(batch_p)

        z_i = model.encode(imgs)
        z_j = model.encode(imgs_j)

        # aux loss on original
        preds_aux = model.heads_full(z_i)
        loss_aux, metr_aux = losscomp.compute(preds_aux, yi)

        z_inv_i, z_eq_i = z_i[:, :d_half], z_i[:, d_half:]
        z_inv_j, z_eq_j = z_j[:, :d_half], z_j[:, d_half:]

        # ij: [inv_j, eq_i]
        z_swapped_ij = torch.cat([z_inv_j, z_eq_i], dim=1)
        preds_ij = model.heads_full(z_swapped_ij)

        # ji: [inv_i, eq_j]
        z_swapped_ji = torch.cat([z_inv_i, z_eq_j], dim=1)
        preds_ji = model.heads_full(z_swapped_ji)

        if cfg.task_type == "factors":
            assert factor_names is not None
            tgt_ij = build_targets_swap_mixed_factors(yi, yj, inv_from="j", factor_names=factor_names, inv_factors=cfg.inv_factors)
            tgt_ji = build_targets_swap_mixed_factors(yi, yj, inv_from="i", factor_names=factor_names, inv_factors=cfg.inv_factors)
        else:
            tgt_ij = build_targets_swap_mixed_numbers(yi, yj, inv_from="j")
            tgt_ji = build_targets_swap_mixed_numbers(yi, yj, inv_from="i")

        loss_i, metr_i = losscomp.compute(preds_ij, tgt_ij)
        loss_j, metr_j = losscomp.compute(preds_ji, tgt_ji)

        loss = 0.5 * (loss_i + loss_j) + loss_aux

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        metr = {**{f"i_{k}": v for k, v in metr_i.items()}, **{f"j_{k}": v for k, v in metr_j.items()},
                "loss": loss.item(), "loss_aux": float(loss_aux.item())}
        agg.append(metr)

        if is_train and it % cfg.print_freq == 0 and it > 0:
            # para swap, imprime las i_* (si existen)
            compact = {"loss": metr["loss"]}
            for k, v in metr.items():
                if k.startswith("i_acc_") or k.startswith("i_mae_") or k.startswith("i_acc"):
                    compact[k.replace("i_", "")] = v
            log_batch_metrics(epoch, it, compact, "Swap")

    keys = agg[0].keys()
    return {k: float(sum(d[k] for d in agg) / len(agg)) for k in keys}

def run_epoch_eval_plain(model, losscomp, loader, cfg: TrainConfig, epoch, header="Eval"):
    model.eval()
    agg = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Ep {epoch} {header}")
    with torch.no_grad():
        for it, batch in pbar:
            batch, _ = normalize_batch_to_dict(batch)
            batch = to_device_batch(batch, cfg.device)

            imgs = batch["image"]
            outs = model(imgs)  # forward normal: heads_full(encode(x))

            targets = losscomp.targets_from_batch(batch)
            loss, metr = losscomp.compute(outs, targets)

            agg.append(metr)

    keys = agg[0].keys()
    return {k: float(sum(d[k] for d in agg) / len(agg)) for k in keys}

def run_epoch_manip(model, manip, losscomp, loader, cfg: TrainConfig, epoch, optimizer=None,
                   factor_names: Optional[List[str]] = None, num_classes: Optional[Dict[str, int]] = None):
    is_train = optimizer is not None
    model.train(is_train)
    manip.train(is_train)
    agg = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Ep {epoch} Manip {'Train' if is_train else 'Val'}")
    for it, batch in pbar:
        batch, _ = normalize_batch_to_dict(batch)
        batch = to_device_batch(batch, cfg.device)
        imgs = batch["image"]
        B = imgs.size(0)

        g = torch.Generator(device=imgs.device)
        g.manual_seed(cfg.pair_seed + it)
        perm = torch.randperm(B, generator=g, device=imgs.device)

        batch_p = {k: (v[perm] if torch.is_tensor(v) else v) for k, v in batch.items() if torch.is_tensor(v)}

        yi = losscomp.targets_from_batch(batch)
        yj = losscomp.targets_from_batch(batch_p)

        z_i = model.encode(imgs)

        # aux on original
        preds_aux = model.heads_full(z_i)
        loss_aux, metr_aux = losscomp.compute(preds_aux, yi)

        # ctrl
        if cfg.task_type == "factors":
            assert factor_names is not None and num_classes is not None
            ctrl = control_vector_factors(yi, yj, factor_names, num_classes, imgs.device)
        else:
            ctrl = control_vector_numbers(yi, yj, imgs.device)

        z_i_hat = manip(z_i, ctrl)
        preds_hat = model.heads_full(z_i_hat)
        loss_main, metr_main = losscomp.compute(preds_hat, yj)

        loss = loss_main + loss_aux

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        metr = {**{f"main_{k}": v for k, v in metr_main.items()},
                **{f"aux_{k}": v for k, v in metr_aux.items()},
                "loss": float(loss.item())}
        agg.append(metr)

        if is_train and it % cfg.print_freq == 0 and it > 0:
            compact = {"loss": metr["loss"]}
            for k, v in metr.items():
                if k.startswith("main_acc_") or k.startswith("main_acc"):
                    compact[k.replace("main_", "")] = v
            log_batch_metrics(epoch, it, compact, "Manip")

    keys = agg[0].keys()
    return {k: float(sum(d[k] for d in agg) / len(agg)) for k in keys}


def run_epoch_manip_swap(model, manip, losscomp, loader, cfg: TrainConfig, epoch, optimizer=None,
                         factor_names: Optional[List[str]] = None, num_classes: Optional[Dict[str, int]] = None):
    is_train = optimizer is not None
    model.train(is_train)
    manip.train(is_train)
    agg = []

    assert cfg.z_dim % 2 == 0, "manip_swap requiere z_dim par"
    d_half = cfg.z_dim // 2

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Ep {epoch} ManipSwap {'Train' if is_train else 'Val'}")
    for it, batch in pbar:
        batch, _ = normalize_batch_to_dict(batch)
        batch = to_device_batch(batch, cfg.device)
        imgs = batch["image"]
        B = imgs.size(0)

        g = torch.Generator(device=imgs.device)
        g.manual_seed(cfg.pair_seed + it)
        perm = torch.randperm(B, generator=g, device=imgs.device)

        batch_p = {k: (v[perm] if torch.is_tensor(v) else v) for k, v in batch.items() if torch.is_tensor(v)}

        yi = losscomp.targets_from_batch(batch)
        yj = losscomp.targets_from_batch(batch_p)

        z_i = model.encode(imgs)
        z_j = model.encode(imgs[perm])

        # aux on originals
        preds_aux_i = model.heads_full(z_i)
        loss_aux_i, metr_aux_i = losscomp.compute(preds_aux_i, yi)
        preds_aux_j = model.heads_full(z_j)
        loss_aux_j, metr_aux_j = losscomp.compute(preds_aux_j, yj)
        loss_aux = 0.5 * (loss_aux_i + loss_aux_j)

        # ctrl
        if cfg.task_type == "factors":
            assert factor_names is not None and num_classes is not None
            ctrl_i = control_vector_factors(yi, yj, factor_names, num_classes, imgs.device)
            ctrl_j = control_vector_factors(yj, yi, factor_names, num_classes, imgs.device)
        else:
            ctrl_i = control_vector_numbers(yi, yj, imgs.device)
            ctrl_j = control_vector_numbers(yj, yi, imgs.device)

        z_i_hat = manip(z_i, ctrl_i)
        z_j_hat = manip(z_j, ctrl_j)

        z_inv_i_hat, z_eq_i_hat = z_i_hat[:, :d_half], z_i_hat[:, d_half:]
        z_inv_j_hat, z_eq_j_hat = z_j_hat[:, :d_half], z_j_hat[:, d_half:]

        z_final_i = torch.cat([z_inv_j_hat, z_eq_i_hat], dim=1)  # inv from j, eq from i
        z_final_j = torch.cat([z_inv_i_hat, z_eq_j_hat], dim=1)  # inv from i, eq from j

        preds_i = model.heads_full(z_final_i)
        preds_j = model.heads_full(z_final_j)

        if cfg.task_type == "factors":
            assert factor_names is not None
            tgt_i = build_targets_swap_mixed_factors(yi, yj, inv_from="j", factor_names=factor_names, inv_factors=cfg.inv_factors)
            tgt_j = build_targets_swap_mixed_factors(yi, yj, inv_from="i", factor_names=factor_names, inv_factors=cfg.inv_factors)
        else:
            tgt_i = build_targets_swap_mixed_numbers(yi, yj, inv_from="j")
            tgt_j = build_targets_swap_mixed_numbers(yi, yj, inv_from="i")

        loss_i, metr_i = losscomp.compute(preds_i, tgt_i)
        loss_j, metr_j = losscomp.compute(preds_j, tgt_j)
        loss_main = 0.5 * (loss_i + loss_j)

        loss = loss_main + loss_aux

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        metr = {**{f"i_{k}": v for k, v in metr_i.items()},
                **{f"j_{k}": v for k, v in metr_j.items()},
                "loss": float(loss.item()),
                "loss_aux": float(loss_aux.item())}
        agg.append(metr)

        if is_train and it % cfg.print_freq == 0 and it > 0:
            compact = {"loss": metr["loss"]}
            for k, v in metr.items():
                if k.startswith("i_acc_") or k.startswith("i_acc"):
                    compact[k.replace("i_", "")] = v
            log_batch_metrics(epoch, it, compact, "M-Swap")

    keys = agg[0].keys()
    return {k: float(sum(d[k] for d in agg) / len(agg)) for k in keys}


def run_one_split(model, manip, losscomp, loader, cfg: TrainConfig, epoch, optimizer,
                  factor_names=None, num_classes=None, split_name="train"):

    # ✅ EVALUACIÓN: nunca uses swap/manip; mide desempeño natural
    if optimizer is None:
        return run_epoch_eval_plain(model, losscomp, loader, cfg, epoch, header=split_name)

    # 🚀 ENTRENAMIENTO: aquí sí aplica el modo elegido
    if cfg.train_mode == "baseline":
        return run_epoch_baseline(model, losscomp, loader, cfg, epoch, optimizer)
    if cfg.train_mode == "swap":
        return run_epoch_swap(model, losscomp, loader, cfg, epoch, optimizer, factor_names=factor_names)
    if cfg.train_mode == "manipulation":
        return run_epoch_manip(model, manip, losscomp, loader, cfg, epoch, optimizer,
                               factor_names=factor_names, num_classes=num_classes)
    if cfg.train_mode == "manip_swap":
        return run_epoch_manip_swap(model, manip, losscomp, loader, cfg, epoch, optimizer,
                                    factor_names=factor_names, num_classes=num_classes)
    raise ValueError(f"train_mode desconocido: {cfg.train_mode}")


# ============================================================
# Checkpoint / resume helpers
# ============================================================

def get_epoch_num_from_ckpt(p: Path) -> int:
    m = re.search(r"ckpt_ep(\d+)\.pt", p.name)
    return int(m.group(1)) if m else -1


def load_checkpoint_with_fallback(checkpoint_path: str, device: str, ckpt_dir: Optional[Path] = None) -> Tuple[Optional[dict], Optional[str]]:
    def _try(path: Path):
        print(f"🔄 Intentando cargar: {path}")
        ckpt = torch.load(path, map_location=device)
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise ValueError("Checkpoint inválido (falta model_state).")
        return ckpt

    tried = set()
    cur = Path(checkpoint_path)

    while True:
        tried.add(cur)
        try:
            ckpt = _try(cur)
            print(f"✅ OK: {cur.name}")
            return ckpt, str(cur)
        except Exception as e:
            print(f"⚠️ Falló {cur.name}: {e}")
            # si no hay fallback, salir
            if ckpt_dir is None or cur.parent.resolve() != ckpt_dir.resolve():
                break
            # buscar otro ckpt
            cands = [p for p in ckpt_dir.glob("ckpt_ep*.pt") if p.exists() and p not in tried]
            if not cands:
                break
            cands = sorted(cands, key=get_epoch_num_from_ckpt)
            cur = cands[-1]

    return None, None


# ============================================================
# Main
# ============================================================

def main():
    cfg = parse_args_to_cfg()
    set_global_seeds(cfg.seed, cfg.device)

    # force split_heads_inv_eq for swap modes (recommended)
    if cfg.train_mode in ("swap", "manip_swap") and not cfg.split_heads_inv_eq:
        print("⚠️ Forzando split_heads_inv_eq=True porque train_mode usa swap.")
        cfg.split_heads_inv_eq = True

    # dirs
    os.makedirs(cfg.out_dir, exist_ok=True)
    run_dir = Path(cfg.out_dir) / cfg.run_name / f"seed_{cfg.seed}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    csv_path = run_dir / cfg.log_csv

    # build dataloaders
    have_scale = False
    have_rot = False
    factor_names: Optional[List[str]] = None
    num_classes: Optional[Dict[str, int]] = None
    test_dl: Optional[DataLoader] = None

    if cfg.dataset == "numbers":
        cfg.task_type = "numbers"
        meta = infer_meta_from_pt(cfg.pt_path)

        cfg.num_classes_for_classification = {}
        if cfg.enable_number:
            cfg.num_classes_for_classification["number"] = int(cfg.num_classes_number)
        if cfg.enable_scale:
            cfg.num_classes_for_classification["scale"] = int(cfg.num_classes_scale)
        if cfg.enable_rot:
            cfg.num_classes_for_classification["rotation"] = int(cfg.num_classes_rot)
        if cfg.enable_font and meta["num_fonts"] > 0:
            cfg.num_classes_for_classification["font"] = int(meta["num_fonts"])
        if cfg.enable_color and meta["num_colors"] > 0:
            cfg.num_classes_for_classification["color"] = int(meta["num_colors"])

        full_ds = NumbersDataset(cfg.pt_path, palette_map_mode="batch", keep_uint8=False, return_meta=True)

        if cfg.debug_max_samples is not None and cfg.debug_max_samples < len(full_ds):
            idx = torch.randperm(len(full_ds))[: cfg.debug_max_samples]
            full_ds = Subset(full_ds, idx)
            print(f"DEBUG subset numbers: {len(full_ds)}")

        n_total = len(full_ds)
        n_val = int(n_total * cfg.val_split_pct)
        n_train = n_total - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                              collate_fn=collate_multi_task)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                            collate_fn=collate_multi_task)

        sample = next(iter(train_dl))
        have_scale = ("scale" in sample) and cfg.enable_scale
        have_rot = ("rotation_deg" in sample) and cfg.enable_rot

        losscomp = LossComputerNumbers(cfg, y_max=meta["y_max"], have_scale=have_scale, have_rot=have_rot)

    else:
        # IDG datasets
        cfg.task_type = "factors"
        cfg.use_classification = True

        idg_cfg = IDGDataConfig(
            root=cfg.idg_root,
            dataset=cfg.dataset,
            split=cfg.idg_split,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=(cfg.num_workers > 0),
            make_val=True,
            val_fraction=cfg.val_split_pct,
            val_seed=42,
            image_as_float=True,
            latents_dtype=torch.long,
            shuffle_train=True,
            drop_last_train=True,
        )
        dls = make_idg_dataloaders(idg_cfg, build=("train", "val", "test"))
        train_dl, val_dl, test_dl = dls["train"], dls["val"], dls["test"]

        # dataset base (antes del split) para nombres/cardinalidades
        base_train: IDGBenchmarkDataset = train_dl.dataset.dataset
        factor_names = list(base_train.factor_names)
        num_classes = infer_num_classes_from_idg(base_train)

        cfg.num_classes_for_classification = dict(num_classes)

        if cfg.split_heads_inv_eq and len(cfg.inv_factors) == 0:
            # default razonable
            default_inv = "shape" if cfg.dataset == "shapes3d" else ("object_shape" if cfg.dataset == "mpi3d" else factor_names[0])
            cfg.inv_factors = [default_inv]
            print(f"⚠️ split_heads_inv_eq=True y inv_factors vacío -> usando {cfg.inv_factors}")

        losscomp = LossComputerFactors(factor_names)

    print("======== CONFIG ========")
    print(cfg)
    print("========================")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoint dir: {ckpt_dir}")

    # build model + manip
    if cfg.train_mode in ("swap", "manip_swap"):
        assert cfg.z_dim % 2 == 0, "swap/manip_swap requieren z_dim par"

    model = DNModel(cfg, have_scale=have_scale, have_rot=have_rot).to(cfg.device)

    ctrl_dim = (len(factor_names) if (cfg.task_type == "factors" and factor_names is not None) else 5)
    manip = Manipulator(cfg.z_dim, ctrl_dim=ctrl_dim).to(cfg.device) if cfg.train_mode in ("manipulation", "manip_swap") else None

    params = list(model.parameters()) + (list(manip.parameters()) if manip is not None else [])
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # resume
    start_epoch = 1
    ckpt_to_load = None
    if cfg.resume_from is not None:
        ckpt_to_load = cfg.resume_from
    elif cfg.auto_resume:
        cands = sorted(list(ckpt_dir.glob("ckpt_ep*.pt")), key=get_epoch_num_from_ckpt)
        if cands:
            ckpt_to_load = str(cands[-1])

    if ckpt_to_load is not None and os.path.isfile(ckpt_to_load):
        ckpt, used = load_checkpoint_with_fallback(ckpt_to_load, cfg.device, ckpt_dir=ckpt_dir)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if manip is not None and "manip_state" in ckpt:
                manip.load_state_dict(ckpt["manip_state"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"✅ Resume desde {used} -> epoch {start_epoch}")
        else:
            print("⚠️ Resume falló, starting from scratch.")

    # csv init / truncate
    if (not csv_path.exists()) or start_epoch == 1:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "split", "metric", "value"])
    else:
        # truncate >= start_epoch
        rows = []
        with open(csv_path, "r", newline="") as f:
            r = csv.reader(f)
            header = next(r)
            rows.append(header)
            for row in r:
                try:
                    ep = int(row[0])
                    if ep < start_epoch:
                        rows.append(row)
                except Exception:
                    rows.append(row)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerows(rows)

    def log_metrics(epoch: int, split: str, metrics: Dict[str, float]):
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for k, v in metrics.items():
                w.writerow([epoch, split, k, v])

    # training loop
    print(f"🚀 Training: {cfg.dataset} | mode={cfg.train_mode} | epochs {start_epoch}->{cfg.epochs}")
    for ep in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()

        tr = run_one_split(model, manip, losscomp, train_dl, cfg, ep, optimizer,
                           factor_names=factor_names, num_classes=num_classes)
        with torch.no_grad():
            va = run_one_split(model, manip, losscomp, val_dl, cfg, ep, None,
                               factor_names=factor_names, num_classes=num_classes, split_name="val")

        dt = time.time() - t0
        log_metrics(ep, "train", tr)
        log_metrics(ep, "val", va)

        print(f"[{ep:03d}] time={dt:.1f}s | train.loss={tr.get('loss'):.4f} | val.loss={va.get('loss'):.4f}")

        if ep % cfg.checkpoint_freq == 0:
            ckpt_path = ckpt_dir / f"ckpt_ep{ep}.pt"
            save = {
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": vars(cfg),
            }
            if manip is not None:
                save["manip_state"] = manip.state_dict()
            torch.save(save, ckpt_path)
            print(f"💾 Saved: {ckpt_path.name}")

        # optional: evaluate IDG test at end
        if test_dl is not None:
            with torch.no_grad():
                te = run_one_split(model, manip, losscomp, test_dl, cfg, ep, None,
                                   factor_names=factor_names, num_classes=num_classes, 
                                   split_name="test")
            log_metrics(ep, "test", te)
            print(f"🧪 TEST | loss={te.get('loss'):.4f} | acc_mean={te.get('acc_mean', float('nan')):.4f}")


if __name__ == "__main__":
    main()
