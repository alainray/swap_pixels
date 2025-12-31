# model.py
import torch
import torch.nn as nn
import torchvision.models as tvm
from typing import Dict, Sequence


def make_encoder(name: str, z_dim: int, pretrained: bool = False):
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        feat_dim = m.fc.in_features
        layers = list(m.children())[:-1]
        encoder = nn.Sequential(*layers)
    elif name == "resnet34":
        m = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT if pretrained else None)
        feat_dim = m.fc.in_features
        layers = list(m.children())[:-1]
        encoder = nn.Sequential(*layers)
    else:
        raise ValueError(f"Encoder no soportado: {name}")
    projector = nn.Linear(feat_dim, z_dim)
    return encoder, projector, feat_dim


class HeadsNumbers(nn.Module):
    """
    Heads para Numbers:
      - Regresión: number (norm), scale, rot(sin/cos)
      - Clasificación: number_logits, scale_logits, rot_logits, font_logits, color_logits
    split_inv_eq:
      - number usa z_inv
      - resto usa z_eq
    """
    def __init__(
        self,
        z_dim: int,
        num_classes: dict,
        have_scale: bool,
        have_rot: bool,
        use_classification: bool = False,
        split_inv_eq: bool = False,
    ):
        super().__init__()
        self.use_classification = use_classification
        self.split_inv_eq = split_inv_eq
        self.have_scale = have_scale
        self.have_rot = have_rot

        if self.split_inv_eq:
            assert (z_dim % 2) == 0, "z_dim debe ser par para split inv/eq"
            self.d_half = z_dim // 2

        if not use_classification:
            self.num = nn.Linear(z_dim, 1)
            self.scale = nn.Linear(z_dim, 1) if have_scale else None
            self.rot = nn.Linear(z_dim, 2) if have_rot else None
        else:
            n_num = int(num_classes.get("number", 5000))
            n_scl = int(num_classes.get("scale", 6))
            n_rot = int(num_classes.get("rotation", 8))
            self.num = nn.Linear(z_dim, n_num)
            self.scale = nn.Linear(z_dim, n_scl) if have_scale else None
            self.rot = nn.Linear(z_dim, n_rot) if have_rot else None

        self.font = nn.Linear(z_dim, int(num_classes["font"])) if "font" in num_classes else None
        self.color = nn.Linear(z_dim, int(num_classes["color"])) if "color" in num_classes else None

    def _make_task_views(self, z: torch.Tensor):
        if not self.split_inv_eq:
            return z, z
        d = self.d_half
        z_inv = z[:, :d]
        z_eq = z[:, d:]
        z_num = torch.cat([z_inv, torch.zeros_like(z_eq)], dim=1)
        z_oth = torch.cat([torch.zeros_like(z_inv), z_eq], dim=1)
        return z_num, z_oth

    def forward(self, z: torch.Tensor):
        out = {}
        z_num, z_oth = self._make_task_views(z)

        if not self.use_classification:
            out["number"] = self.num(z_num).squeeze(-1)
            if self.scale is not None:
                out["scale"] = self.scale(z_oth).squeeze(-1)
            if self.rot is not None:
                r = self.rot(z_oth)
                out["rot_sin"], out["rot_cos"] = r[:, 0], r[:, 1]
        else:
            out["number_logits"] = self.num(z_num)
            if self.scale is not None:
                out["scale_logits"] = self.scale(z_oth)
            if self.rot is not None:
                out["rot_logits"] = self.rot(z_oth)

        if self.font is not None:
            out["font_logits"] = self.font(z_oth)
        if self.color is not None:
            out["color_logits"] = self.color(z_oth)

        return out


class GenericHeads(nn.Module):
    """
    Un Linear por factor:
      out[f"{name}_logits"] = [B, n_classes]
    Opcional split inv/eq:
      - factores en inv_factors usan z_inv
      - el resto usa z_eq
    """
    def __init__(
        self,
        z_dim: int,
        num_classes: Dict[str, int],
        split_inv_eq: bool = False,
        inv_factors: Sequence[str] = (),
    ):
        super().__init__()
        self.num_classes = dict(num_classes)
        self.split_inv_eq = bool(split_inv_eq)
        self.inv_factors = set(inv_factors)

        if self.split_inv_eq:
            assert z_dim % 2 == 0, "z_dim debe ser par para split inv/eq"
            self.d_half = z_dim // 2

        self.heads = nn.ModuleDict({k: nn.Linear(z_dim, int(v)) for k, v in self.num_classes.items()})

    def _views(self, z: torch.Tensor):
        if not self.split_inv_eq:
            return z, z
        d = self.d_half
        z_inv, z_eq = z[:, :d], z[:, d:]
        z_inv_view = torch.cat([z_inv, torch.zeros_like(z_eq)], dim=1)
        z_eq_view = torch.cat([torch.zeros_like(z_inv), z_eq], dim=1)
        return z_inv_view, z_eq_view

    def forward(self, z: torch.Tensor):
        out = {}
        z_inv_view, z_eq_view = self._views(z)
        for name, head in self.heads.items():
            if self.split_inv_eq:
                z_use = z_inv_view if name in self.inv_factors else z_eq_view
            else:
                z_use = z
            out[f"{name}_logits"] = head(z_use)
        return out


class DNModel(nn.Module):
    def __init__(self, cfg, have_scale: bool, have_rot: bool):
        super().__init__()
        enc, proj, _ = make_encoder(cfg.encoder_name, cfg.z_dim, pretrained=cfg.use_pretrained_encoder)
        self.encoder = enc
        self.projector = proj
        self.z_dim = cfg.z_dim

        task_type = getattr(cfg, "task_type", "numbers")

        if task_type == "factors":
            self.heads_full = GenericHeads(
                cfg.z_dim,
                cfg.num_classes_for_classification,
                split_inv_eq=getattr(cfg, "split_heads_inv_eq", False),
                inv_factors=getattr(cfg, "inv_factors", []),
            )
        else:
            self.heads_full = HeadsNumbers(
                cfg.z_dim,
                cfg.num_classes_for_classification,
                have_scale,
                have_rot,
                use_classification=cfg.use_classification,
                split_inv_eq=getattr(cfg, "split_heads_inv_eq", False),
            )

    def encode(self, x):
        f = self.encoder(x)
        f = torch.flatten(f, 1)
        z = self.projector(f)
        return z

    def forward(self, x):
        z = self.encode(x)
        return {"z_full": z, **self.heads_full(z)}


class Manipulator(nn.Module):
    def __init__(self, z_dim: int, ctrl_dim: int = 5, hidden: int = 2048):
        super().__init__()
        self.ctrl_embed = nn.Sequential(
            nn.Linear(ctrl_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.Linear(z_dim + 128, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z, ctrl):
        e = self.ctrl_embed(ctrl)
        h = torch.cat([z, e], dim=1)
        return z + self.net(h)
