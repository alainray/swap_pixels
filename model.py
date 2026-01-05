# model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def make_encoder(name: str, pretrained: bool = False):
    """
    Devuelve (backbone_sin_fc, feat_dim).
    """
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        feat_dim = m.fc.in_features
        backbone = nn.Sequential(*list(m.children())[:-1])  # -> [B, feat_dim, 1, 1]
        return backbone, feat_dim
    if name == "resnet34":
        m = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT if pretrained else None)
        feat_dim = m.fc.in_features
        backbone = nn.Sequential(*list(m.children())[:-1])
        return backbone, feat_dim
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
        feat_dim = m.fc.in_features
        backbone = nn.Sequential(*list(m.children())[:-1])
        return backbone, feat_dim
    raise ValueError(f"Encoder no soportado: {name}")


class HeadsFull(nn.Module):
    """
    Produce dict outputs:
      - factors:  {f"{name}_logits": [B,C]}
      - numbers:  keys like number/number_logits, scale/scale_logits, rot_logits or rot_sin/rot_cos, font_logits, color_logits
    Supports:
      - cfg.factorwise_heads (split representation into equal chunks per factor/task)
      - cfg.split_heads_inv_eq + cfg.inv_factors (old behavior)
    """

    def __init__(self, cfg, have_scale: bool, have_rot: bool):
        super().__init__()
        self.cfg = cfg
        self.have_scale = have_scale
        self.have_rot = have_rot

        self.task_type = cfg.task_type
        self.z_dim = int(cfg.z_dim)
        self.factorwise = bool(getattr(cfg, "factorwise_heads", False))
        self.split_inv_eq = bool(getattr(cfg, "split_heads_inv_eq", False))
        self.inv_set = set(getattr(cfg, "inv_factors", []))

        if self.task_type == "factors":
            self.factor_names: List[str] = list(getattr(cfg, "factor_names_order", []))
            if len(self.factor_names) == 0:
                # fallback (order not guaranteed)
                self.factor_names = list(cfg.num_classes_for_classification.keys())

            self.F = len(self.factor_names)
            assert self.F > 0, "factor_names_order vacío; setéalo en train.py (cfg.factor_names_order = factor_names)."

            if self.factorwise:
                assert self.z_dim % self.F == 0, f"factorwise_heads requiere z_dim divisible por F={self.F}"
                self.z_per = self.z_dim // self.F
            else:
                self.z_per = None

            if (not self.factorwise) and self.split_inv_eq:
                assert self.z_dim % 2 == 0, "split_heads_inv_eq requiere z_dim par"
                self.d_half = self.z_dim // 2
            else:
                self.d_half = None

            self.heads = nn.ModuleDict()
            for i, name in enumerate(self.factor_names):
                ncls = int(cfg.num_classes_for_classification[name])
                in_dim = self._in_dim_for_factor(i, name)
                self.heads[name] = nn.Linear(in_dim, ncls)

        else:
            # numbers
            # Define task chunks for factorwise split:
            # rotation counts as one "task chunk" even if it outputs sin+cos
            self.use_cls = bool(cfg.use_classification)
            self.num_tasks_order: List[str] = []
            if cfg.enable_number:
                self.num_tasks_order.append("number")
            if have_scale and cfg.enable_scale:
                self.num_tasks_order.append("scale")
            if have_rot and cfg.enable_rot:
                self.num_tasks_order.append("rotation")
            if cfg.enable_font and ("font" in cfg.num_classes_for_classification):
                self.num_tasks_order.append("font")
            if cfg.enable_color and ("color" in cfg.num_classes_for_classification):
                self.num_tasks_order.append("color")

            self.T = max(1, len(self.num_tasks_order))
            if self.factorwise:
                assert self.z_dim % self.T == 0, f"factorwise_heads en numbers requiere z_dim divisible por T={self.T}"
                self.z_per_num = self.z_dim // self.T
            else:
                self.z_per_num = None

            if (not self.factorwise) and self.split_inv_eq:
                assert self.z_dim % 2 == 0, "split_heads_inv_eq requiere z_dim par"
                self.d_half = self.z_dim // 2
            else:
                self.d_half = None

            # build heads
            self.number_head = None
            self.number_logits = None
            self.scale_head = None
            self.scale_logits = None
            self.rot_logits = None
            self.rot_sin = None
            self.rot_cos = None
            self.font_logits = None
            self.color_logits = None

            # helper to get chunk for a named task
            def in_dim_for_task(task: str) -> int:
                if self.factorwise:
                    return self.z_per_num
                if self.split_inv_eq:
                    # number uses inv half; others use eq half
                    return self.d_half
                return self.z_dim

            # number
            if cfg.enable_number:
                if self.use_cls:
                    self.number_logits = nn.Linear(in_dim_for_task("number"), int(cfg.num_classes_number))
                else:
                    self.number_head = nn.Linear(in_dim_for_task("number"), 1)

            # scale
            if have_scale and cfg.enable_scale:
                if self.use_cls:
                    self.scale_logits = nn.Linear(in_dim_for_task("scale"), int(cfg.num_classes_scale))
                else:
                    self.scale_head = nn.Linear(in_dim_for_task("scale"), 1)

            # rotation
            if have_rot and cfg.enable_rot:
                if self.use_cls:
                    self.rot_logits = nn.Linear(in_dim_for_task("rotation"), int(cfg.num_classes_rot))
                else:
                    self.rot_sin = nn.Linear(in_dim_for_task("rotation"), 1)
                    self.rot_cos = nn.Linear(in_dim_for_task("rotation"), 1)

            # font/color
            if cfg.enable_font and ("font" in cfg.num_classes_for_classification):
                self.font_logits = nn.Linear(in_dim_for_task("font"), int(cfg.num_classes_for_classification["font"]))
            if cfg.enable_color and ("color" in cfg.num_classes_for_classification):
                self.color_logits = nn.Linear(in_dim_for_task("color"), int(cfg.num_classes_for_classification["color"]))

    def _slice_factorwise(self, z: torch.Tensor, idx: int) -> torch.Tensor:
        a = idx * self.z_per
        b = (idx + 1) * self.z_per
        return z[:, a:b]

    def _slice_num_task(self, z: torch.Tensor, task: str) -> torch.Tensor:
        if self.factorwise:
            idx = self.num_tasks_order.index(task)
            a = idx * self.z_per_num
            b = (idx + 1) * self.z_per_num
            return z[:, a:b]
        if self.split_inv_eq:
            if task == "number":
                return z[:, :self.d_half]
            return z[:, self.d_half:]
        return z

    def _in_dim_for_factor(self, idx: int, name: str) -> int:
        if self.factorwise:
            return self.z_per
        if self.split_inv_eq:
            return self.z_dim // 2
        return self.z_dim

    def _slice_for_factor(self, z: torch.Tensor, idx: int, name: str) -> torch.Tensor:
        if self.factorwise:
            return self._slice_factorwise(z, idx)
        if self.split_inv_eq:
            d_half = self.z_dim // 2
            if name in self.inv_set:
                return z[:, :d_half]
            return z[:, d_half:]
        return z

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        if self.task_type == "factors":
            for i, name in enumerate(self.factor_names):
                x = self._slice_for_factor(z, i, name)
                out[f"{name}_logits"] = self.heads[name](x)
            return out

        # numbers
        if self.number_logits is not None:
            x = self._slice_num_task(z, "number")
            out["number_logits"] = self.number_logits(x)
        if self.number_head is not None:
            x = self._slice_num_task(z, "number")
            out["number"] = self.number_head(x).squeeze(1)

        if self.scale_logits is not None:
            x = self._slice_num_task(z, "scale")
            out["scale_logits"] = self.scale_logits(x)
        if self.scale_head is not None:
            x = self._slice_num_task(z, "scale")
            out["scale"] = self.scale_head(x).squeeze(1)

        if self.rot_logits is not None:
            x = self._slice_num_task(z, "rotation")
            out["rot_logits"] = self.rot_logits(x)
        if self.rot_sin is not None and self.rot_cos is not None:
            x = self._slice_num_task(z, "rotation")
            out["rot_sin"] = self.rot_sin(x).squeeze(1)
            out["rot_cos"] = self.rot_cos(x).squeeze(1)

        if self.font_logits is not None:
            x = self._slice_num_task(z, "font")
            out["font_logits"] = self.font_logits(x)
        if self.color_logits is not None:
            x = self._slice_num_task(z, "color")
            out["color_logits"] = self.color_logits(x)

        return out


class DNModel(nn.Module):
    def __init__(self, cfg, have_scale: bool = False, have_rot: bool = False):
        super().__init__()
        self.cfg = cfg
        self.have_scale = have_scale
        self.have_rot = have_rot

        backbone, feat_dim = make_encoder(cfg.encoder_name, pretrained=bool(cfg.use_pretrained_encoder))
        self.backbone = backbone
        self.projector = nn.Linear(feat_dim, int(cfg.z_dim))

        self.heads_full = HeadsFull(cfg, have_scale=have_scale, have_rot=have_rot)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # ensure 3-channel for resnet
        if x.ndim == 4 and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        h = self.backbone(x)                 # [B, feat_dim, 1, 1]
        h = h.flatten(1)                     # [B, feat_dim]
        z = self.projector(h)                # [B, z_dim]
        return z

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        return self.heads_full(z)


class Manipulator(nn.Module):
    """
    Simple manipulator: z_hat = z + MLP([z, ctrl])
    """
    def __init__(self, z_dim: int, ctrl_dim: int, hidden: int = 512):
        super().__init__()
        self.z_dim = int(z_dim)
        self.ctrl_dim = int(ctrl_dim)

        self.net = nn.Sequential(
            nn.Linear(self.z_dim + self.ctrl_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.z_dim),
        )

    def forward(self, z: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, ctrl], dim=1)
        dz = self.net(x)
        return z + dz
