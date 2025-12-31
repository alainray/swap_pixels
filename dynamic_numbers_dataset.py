
from __future__ import annotations
import os, glob, platform, random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Sequence, Callable
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset

# Resample compat
try:
    RESAMP_BICUBIC = Image.Resampling.BICUBIC
except Exception:
    RESAMP_BICUBIC = Image.BICUBIC

def _default_font_dirs() -> List[str]:
    sysname = platform.system().lower()
    dirs = []
    if "linux" in sysname:
        dirs += ["/usr/share/fonts", "/usr/local/share/fonts", str(Path.home()/".fonts")]
    elif "darwin" in sysname:
        dirs += ["/System/Library/Fonts", "/Library/Fonts", str(Path.home()/ "Library/Fonts")]
    elif "windows" in sysname:
        dirs += [r"C:\\Windows\\Fonts"]
    return [d for d in dirs if os.path.isdir(d)]

def _collect_font_paths(fonts_dir: Optional[str]=None, extra_paths: Optional[Sequence[str]]=None, limit: Optional[int]=None) -> List[str]:
    paths = []
    search_dirs = []
    if fonts_dir:
        search_dirs.append(fonts_dir)
    search_dirs += _default_font_dirs()
    exts = ("*.ttf","*.otf","*.ttc","*.otc")
    for d in search_dirs:
        for ext in exts:
            paths.extend(glob.glob(str(Path(d)/"**"/ext), recursive=True))
    if extra_paths:
        for p in extra_paths:
            if os.path.isfile(p):
                paths.append(p)
    paths = sorted(set(paths))
    if limit:
        paths = paths[:limit]
    return paths

def _normalize_font_basename(name: str) -> str:
    base = os.path.basename(name)
    base = os.path.splitext(base)[0]
    base = base.lower().replace(" ", "").replace("_", "").replace("-", "")
    return base

def _seeded_rng(seed: int, *xs: int) -> random.Random:
    s = seed
    for x in xs:
        s = (s * 6364136223846793005 + (x+1442695040888963407)) & ((1<<64)-1)
    return random.Random(s)

def _np_color(rng: random.Random):
    return (rng.randint(0,255), rng.randint(0,255), rng.randint(0,255))

def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)

def _sample_from_range_or_list(
    rng: random.Random,
    lo_hi: Tuple[float, float],
    values: Optional[Sequence],
    as_int: bool = False
):
    if values is not None and len(values) > 0:
        return rng.choice(list(values))
    lo, hi = lo_hi
    if as_int:
        return rng.randint(int(lo), int(hi))
    else:
        return rng.uniform(float(lo), float(hi))

def _resolve_single_font_basename(basename: str, fonts_dir: Optional[str]=None) -> Optional[str]:
    want = _normalize_font_basename(basename)
    dirs = []
    if fonts_dir:
        dirs.append(fonts_dir)
    dirs += _default_font_dirs()
    exts = ("*.ttf","*.otf","*.ttc","*.otc")
    for d in dirs:
        for ext in exts:
            for p in glob.glob(str(Path(d)/"**"/ext), recursive=True):
                if _normalize_font_basename(os.path.basename(p)) == want:
                    return p
    return None

class DynamicNumbersDataset(Dataset):
    def __init__(
        self,
        length: int,
        image_size: int = 128,
        background: Tuple[int,int,int] = (255,255,255),
        min_digits: int = 1,
        max_digits: int = 5,
        fixed_width_5: bool = False,
        # rotation: range or values
        rotation_deg: Tuple[float,float] = (-30.0, 30.0),
        rotation_values: Optional[Sequence[float]] = None,
        # per-digit scale: range or list
        per_digit_scale: Tuple[float,float] = (0.8, 1.2),
        per_digit_scale_values: Optional[Sequence[float]] = None,
        # per-digit spacing: range or list (integers)
        per_digit_spacing_px: Tuple[int,int] = (4, 16),
        per_digit_spacing_values: Optional[Sequence[int]] = None,
        # per-digit color list (RGB) or full-random if None
        per_digit_color_values: Optional[Sequence[Tuple[int,int,int]]] = None,
        # fonts
        fonts_dir: Optional[str] = None,
        font_paths: Optional[Sequence[str]] = None,
        fonts_limit: Optional[int] = 40,
        restrict_font_basenames: Optional[Sequence[str]] = None,
        require_restricted_fonts: bool = False,
        # labels / sampling
        seed: int = 123,
        enumerate_numbers: bool = False,
        return_tensor: bool = True,
        color_mode: str = "RGB",
        allow_repeat_fonts: bool = True,
        cache_glyphs: bool = True,
        min_value: int = 0,
        max_value: int = 99999,
        # flexible filters
        parity: Optional[str] = None,
        hundreds_between: Optional[Tuple[int,int]] = None,
        allowed_labels: Optional[Sequence[int]] = None,
        allowed_label_fn: Optional[Callable[[int], bool]] = None,
        precompute_allowed: bool = True,
        # NEW: tie (share across digits of the same number)
        tie_digits_font: bool = False,
        tie_digits_scale: bool = False,
        tie_digits_color: bool = False,
        # NEW: fixed factors (apply to all digits of all numbers if given)
        fixed_font_path: Optional[str] = None,
        fixed_font_basename: Optional[str] = None,
        fixed_scale: Optional[float] = None,
        fixed_color: Optional[Tuple[int,int,int]] = None,
    ):
        assert 1 <= min_digits <= max_digits <= 5
        if min_value > max_value:
            raise ValueError(f"min_value ({min_value}) must be <= max_value ({max_value}).")
        if fixed_width_5 and max_value > 99999:
            raise ValueError("fixed_width_5=True requiere max_value <= 99999 para no truncar dígitos.")
        if min_value < 0:
            raise ValueError("Los labels deben ser no negativos.")
        if parity is not None and parity not in ("even", "odd"):
            raise ValueError("parity debe ser 'even', 'odd' o None.")

        self.length = int(length)
        self.image_size = int(image_size)
        self.background = tuple(background)
        self.min_digits = int(min_digits)
        self.max_digits = int(max_digits)
        self.fixed_width_5 = bool(fixed_width_5)
        # latent specs
        self.rotation_deg = rotation_deg
        self.rotation_values = list(rotation_values) if rotation_values is not None else None
        self.per_digit_scale = per_digit_scale
        self.per_digit_scale_values = list(per_digit_scale_values) if per_digit_scale_values is not None else None
        self.per_digit_spacing_px = per_digit_spacing_px
        self.per_digit_spacing_values = list(per_digit_spacing_values) if per_digit_spacing_values is not None else None
        self.per_digit_color_values = list(per_digit_color_values) if per_digit_color_values is not None else None

        self.base_size_px = 96
        self.seed = int(seed)
        self.enumerate_numbers = bool(enumerate_numbers)
        self.return_tensor = bool(return_tensor)
        self.color_mode = color_mode.upper()
        self.allow_repeat_fonts = bool(allow_repeat_fonts)
        self.cache_glyphs = bool(cache_glyphs)
        self.min_value = int(min_value)
        self.max_value = int(max_value)

        # label filters
        self.parity = parity
        self.hundreds_between = hundreds_between
        self.allowed_labels = set(int(x) for x in allowed_labels) if allowed_labels is not None else None
        self.allowed_label_fn = allowed_label_fn
        self.precompute_allowed = bool(precompute_allowed)

        # collect fonts
        self._font_paths = _collect_font_paths(fonts_dir, font_paths, fonts_limit)
        if not self._font_paths:
            self._font_paths = []

        # apply whitelist by basename
        self.restrict_font_basenames = [str(x) for x in restrict_font_basenames] if restrict_font_basenames is not None else None
        self.require_restricted_fonts = bool(require_restricted_fonts)
        if self.restrict_font_basenames:
            allow = { _normalize_font_basename(x) for x in self.restrict_font_basenames }
            filtered = []
            for p in list(self._font_paths):
                if _normalize_font_basename(os.path.basename(p)) in allow:
                    filtered.append(p)
            self._font_paths = filtered
            if self.require_restricted_fonts and not self._font_paths:
                raise ValueError("Ninguna de las fuentes solicitadas fue hallada en fonts_dir/directorios del sistema.")

        # NEW: ties and fixed factors
        self.tie_digits_font = bool(tie_digits_font)
        self.tie_digits_scale = bool(tie_digits_scale)
        self.tie_digits_color = bool(tie_digits_color)

        self.fixed_font_path = fixed_font_path if fixed_font_path else None
        self.fixed_font_basename = fixed_font_basename if fixed_font_basename else None
        self._fixed_font_resolved: Optional[str] = None
        if self.fixed_font_path and os.path.isfile(self.fixed_font_path):
            self._fixed_font_resolved = self.fixed_font_path
        elif self.fixed_font_basename:
            self._fixed_font_resolved = _resolve_single_font_basename(self.fixed_font_basename, fonts_dir=fonts_dir)

        self.fixed_scale = float(fixed_scale) if fixed_scale is not None else None
        self.fixed_color = tuple(fixed_color) if fixed_color is not None else None

        self._worker_fonts: Optional[List[str]] = None
        self._worker_font_objs: Dict[Tuple[str,int], ImageFont.FreeTypeFont] = {}
        self._glyph_cache: Dict[Tuple[str,int,str], Image.Image] = {}
        self._epoch = 0
        self._worker_id = 0

        # allowed label cache
        self._allowed_labels: Optional[List[int]] = None
        if self.precompute_allowed:
            self._allowed_labels = self._compute_allowed_labels()
            if len(self._allowed_labels) == 0:
                raise ValueError("No hay labels válidos tras aplicar filtros en el rango especificado.")

    # ---------- runtime control ---------- #
    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def set_worker_id(self, wid: int):
        self._worker_id = int(wid)
        self._worker_fonts = list(self._font_paths)
        self._worker_font_objs.clear()
        self._glyph_cache.clear()
        if not self.precompute_allowed:
            self._allowed_labels = self._compute_allowed_labels()

    # ---------- runtime font whitelist control ---------- #
    def set_font_whitelist(self, basenames_or_paths, fonts_dir: Optional[str] = None, strict: bool = False):
        paths = []
        search_dirs = []
        if fonts_dir:
            search_dirs.append(fonts_dir)
        search_dirs += _default_font_dirs()
        exts = ("*.ttf","*.otf","*.ttc","*.otc")
        for d in search_dirs:
            for ext in exts:
                paths.extend(glob.glob(str(Path(d)/"**"/ext), recursive=True))
        paths = sorted(set(paths))

        allow = set()
        for x in basenames_or_paths:
            if os.path.isfile(str(x)):
                allow.add(_normalize_font_basename(os.path.basename(str(x))))
            else:
                allow.add(_normalize_font_basename(str(x)))

        filtered = [p for p in paths if _normalize_font_basename(os.path.basename(p)) in allow]
        if strict and not filtered:
            raise ValueError("No se encontraron fuentes que coincidan con la whitelist")
        self._font_paths = filtered
        self.restrict_font_basenames = list(basenames_or_paths)
        self.require_restricted_fonts = bool(strict)
        self._worker_fonts = list(self._font_paths)
        self._worker_font_objs.clear()
        self._glyph_cache.clear()
        return list(self._font_paths)

    # ---------- runtime label whitelist control ---------- #
    def set_allowed_labels(self, labels, clip_to_range: bool = True) -> int:
        labs = set(int(x) for x in labels)
        if clip_to_range:
            lo, hi = self.min_value, self.max_value
            labs = {x for x in labs if lo <= x <= hi}
        if not labs:
            raise ValueError("La lista de labels permitidos quedó vacía tras aplicar clip_to_range.")
        self.allowed_labels = set(labs)
        self._allowed_labels = sorted(labs)
        self.precompute_allowed = True
        return len(self._allowed_labels)

    def clear_allowed_labels(self) -> bool:
        self.allowed_labels = None
        self._allowed_labels = None
        return True

    # ---------- runtime ties & fixed factors ---------- #
    def set_shared_factors(self, *, font: Optional[bool]=None, scale: Optional[bool]=None, color: Optional[bool]=None):
        if font is not None: self.tie_digits_font = bool(font)
        if scale is not None: self.tie_digits_scale = bool(scale)
        if color is not None: self.tie_digits_color = bool(color)

    def set_fixed_factors(self, *, font_path: Optional[str]=None, font_basename: Optional[str]=None,
                          scale: Optional[float]=None, color: Optional[Tuple[int,int,int]]=None,
                          fonts_dir: Optional[str]=None, strict: bool=False) -> Optional[str]:
        if font_path == "":
            font_path = None
        if font_basename == "":
            font_basename = None

        self.fixed_scale = float(scale) if scale is not None else self.fixed_scale
        self.fixed_color = tuple(color) if color is not None else self.fixed_color

        resolved = None
        if font_path is not None:
            if os.path.isfile(font_path):
                resolved = font_path
            elif strict:
                raise ValueError(f"font_path no existe: {font_path}")
        elif font_basename is not None:
            resolved = _resolve_single_font_basename(font_basename, fonts_dir=fonts_dir)
            if strict and resolved is None:
                raise ValueError(f"No se pudo resolver la fuente por basename: {font_basename}")

        if resolved is not None:
            self._fixed_font_resolved = resolved
            self.fixed_font_path = resolved
            self.fixed_font_basename = None
        elif font_path is None and font_basename is None:
            # clear
            self._fixed_font_resolved = None
            self.fixed_font_path = None
            self.fixed_font_basename = None

        # reset caches because font selection might change glyphs
        self._worker_font_objs.clear()
        self._glyph_cache.clear()
        return self._fixed_font_resolved

    # ---------- label filtering ---------- #
    def _label_ok(self, n: int) -> bool:
        if self.allowed_labels is not None and n not in self.allowed_labels:
            return False
        if self.parity == "even" and (n % 2 != 0):
            return False
        if self.parity == "odd" and (n % 2 != 1):
            return False
        if self.hundreds_between is not None:
            lo_h, hi_h = self.hundreds_between
            h = n // 100
            if not (lo_h <= h <= hi_h):
                return False
        if self.allowed_label_fn is not None:
            if not self.allowed_label_fn(n):
                return False
        return True

    def _compute_allowed_labels(self) -> List[int]:
        lo, hi = self.min_value, self.max_value
        out = [n for n in range(lo, hi+1) if self._label_ok(n)]
        return out

    # ---------- glyph & compose ---------- #
    def _get_font(self, font_path: str, size_px: int) -> ImageFont.ImageFont:
        key = (font_path, size_px)
        ft = self._worker_font_objs.get(key, None)
        if ft is None:
            try:
                if font_path:
                    ft = ImageFont.truetype(font_path, size_px)
                else:
                    ft = ImageFont.load_default()
            except Exception:
                ft = ImageFont.load_default()
            self._worker_font_objs[key] = ft
        return ft

    def _glyph_rgba(self, font_path: str, char: str) -> Image.Image:
        assert char in "0123456789"
        key = (font_path, self.base_size_px, char)
        if self.cache_glyphs and key in self._glyph_cache:
            return self._glyph_cache[key]

        font = self._get_font(font_path, self.base_size_px)
        bbox = font.getbbox(char, anchor=None)
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])

        img = Image.new("L", (w, h), 0)
        drw = ImageDraw.Draw(img)
        drw.text((-bbox[0], -bbox[1]), char, font=font, fill=255)

        rgba = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        rgba.putalpha(img)
        if self.cache_glyphs:
            self._glyph_cache[key] = rgba
        return rgba

    def _compose_number(
        self,
        digits: List[str],
        rng: random.Random,
        font_paths: List[str],
    ) -> Tuple[Image.Image, Dict[str,Any]]:
        # --------- FONT selection (shared or per-digit) --------- #
        sel_fonts: List[str] = []
        shared_font: Optional[str] = None
        if self._fixed_font_resolved:
            shared_font = self._fixed_font_resolved
        elif self.tie_digits_font:
            if font_paths:
                shared_font = rng.choice(font_paths)
            else:
                shared_font = ""  # default
        # If shared_font decided, apply to all digits, else sample per-digit as before
        if shared_font is not None:
            sel_fonts = [shared_font for _ in digits]
        else:
            if self.allow_repeat_fonts:
                sel_fonts = [rng.choice(font_paths) if font_paths else "" for _ in digits]
            else:
                if font_paths and len(font_paths) >= len(digits):
                    sel_fonts = random.sample(font_paths, len(digits))
                else:
                    sel_fonts = [rng.choice(font_paths) if font_paths else "" for _ in digits]

        per_digit = []
        glyph_imgs = []
        total_w = 0
        max_h  = 0

        # --------- SCALE/COLOR selection (shared or per-digit) --------- #
        shared_scale = None
        if self.fixed_scale is not None:
            shared_scale = float(self.fixed_scale)
        elif self.tie_digits_scale:
            shared_scale = float(_sample_from_range_or_list(rng, self.per_digit_scale, self.per_digit_scale_values, as_int=False))

        shared_color = None
        if self.fixed_color is not None:
            shared_color = tuple(self.fixed_color)
        elif self.tie_digits_color:
            if self.per_digit_color_values is not None and len(self.per_digit_color_values) > 0:
                shared_color = rng.choice(self.per_digit_color_values)
            else:
                shared_color = _np_color(rng)

        for i, (ch, fpath) in enumerate(zip(digits, sel_fonts)):
            # scale
            if shared_scale is not None:
                scale = float(shared_scale)
            else:
                scale = float(_sample_from_range_or_list(rng, self.per_digit_scale, self.per_digit_scale_values, as_int=False))
            # spacing (kept per-digit)
            spacing = int(_sample_from_range_or_list(rng, self.per_digit_spacing_px, self.per_digit_spacing_values, as_int=True))
            # color
            if shared_color is not None:
                color = tuple(shared_color)
            else:
                if self.per_digit_color_values is not None and len(self.per_digit_color_values) > 0:
                    color = rng.choice(self.per_digit_color_values)
                else:
                    color = _np_color(rng)

            g = self._glyph_rgba(fpath, ch)
            w, h = g.size
            w2 = max(1, int(round(w * scale)))
            h2 = max(1, int(round(h * scale)))
            g2 = g.resize((w2, h2), resample=RESAMP_BICUBIC)
            if g2.mode != "RGBA":
                g2 = g2.convert("RGBA")
            arr = np.array(g2, dtype=np.uint8)
            alpha = arr[..., 3:4].astype(np.float32) / 255.0
            rgb = arr[..., :3].astype(np.float32)
            col = np.array(color, dtype=np.float32)[None,None,:]
            rgb = (col * alpha + rgb * 0.0 * (1 - alpha))
            arr[..., :3] = _ensure_uint8(rgb)
            g_col = Image.fromarray(arr, mode="RGBA")

            glyph_imgs.append((g_col, spacing))
            total_w += g_col.size[0] + spacing
            max_h = max(max_h, g_col.size[1])

            per_digit.append({
                "char": ch,
                "font_path": fpath,
                "scale": scale,
                "color": color,
                "spacing": spacing,
                "glyph_size": g_col.size,
            })

        total_w = max(1, total_w)
        max_h = max(1, max_h)
        canvas = Image.new("RGBA", (total_w, max_h), (0,0,0,0))
        x = 0
        for g_col, spacing in glyph_imgs:
            y = (max_h - g_col.size[1]) // 2
            canvas.alpha_composite(g_col, (x, y))
            x += g_col.size[0] + spacing

        meta = {"per_digit": per_digit, "canvas_size": canvas.size}
        return canvas, meta

    def _finalize(self, rgba: Image.Image, angle: float) -> Image.Image:
        rotated = rgba.rotate(angle, resample=RESAMP_BICUBIC, expand=True, fillcolor=(0,0,0,0))
        bg = Image.new("RGBA", rotated.size, (0,0,0,0))
        bg.alpha_composite(rotated, (0,0))
        W = H = self.image_size
        out = Image.new(self.color_mode, (W, H), self.background)
        rw, rh = bg.size
        scale = min(W / rw, H / rh) if rw>0 and rh>0 else 1.0
        if scale < 1.0:
            bg = bg.resize((max(1,int(rw*scale)), max(1,int(rh*scale))), resample=RESAMP_BICUBIC)
            rw, rh = bg.size
        x0 = (W - rw)//2
        y0 = (H - rh)//2
        if bg.mode != "RGBA":
            bg = bg.convert("RGBA")
        out = out.convert("RGBA")
        out.alpha_composite(bg, (x0, y0))
        out = out.convert(self.color_mode)
        return out

    # ---------- direct rendering APIs ---------- #
    def _broadcast_len(self, values, L, cast=None):
        if values is None:
            return None
        if isinstance(values, (list, tuple)):
            if len(values) != L:
                raise ValueError(f"Se esperaban {L} valores, recibidos {len(values)}.")
            out = list(values)
        else:
            out = [values] * L
        if cast is not None:
            out = [cast(x) for x in out]
        return out

    def _compose_number_with_attrs(self, digits, attrs):
        glyph_imgs = []
        total_w = 0
        max_h  = 0
        per_digit_meta = []
        for ch, a in zip(digits, attrs):
            fpath = a.get("font_path", "") or ""
            scale = float(a.get("scale", 1.0))
            spacing = int(a.get("spacing", 0))
            color = tuple(a.get("color", (0,0,0)))

            g = self._glyph_rgba(fpath, ch)
            w, h = g.size
            w2 = max(1, int(round(w * scale)))
            h2 = max(1, int(round(h * scale)))
            g2 = g.resize((w2, h2), resample=RESAMP_BICUBIC)
            if g2.mode != "RGBA":
                g2 = g2.convert("RGBA")
            arr = np.array(g2, dtype=np.uint8)
            alpha = arr[..., 3:4].astype(np.float32) / 255.0
            rgb = arr[..., :3].astype(np.float32)
            col = np.array(color, dtype=np.float32)[None,None,:]
            rgb = (col * alpha + rgb * 0.0 * (1 - alpha))
            arr[..., :3] = (np.clip(rgb, 0, 255)).astype(np.uint8)
            g_col = Image.fromarray(arr, mode="RGBA")

            glyph_imgs.append((g_col, spacing))
            total_w += g_col.size[0] + spacing
            max_h = max(max_h, g_col.size[1])

            per_digit_meta.append({
                "char": ch, "font_path": fpath, "scale": scale, "color": color,
                "spacing": spacing, "glyph_size": g_col.size,
            })

        total_w = max(1, total_w)
        max_h = max(1, max_h)
        canvas = Image.new("RGBA", (total_w, max_h), (0,0,0,0))
        x = 0
        for g_col, spacing in glyph_imgs:
            y = (max_h - g_col.size[1]) // 2
            canvas.alpha_composite(g_col, (x, y))
            x += g_col.size[0] + spacing

        meta = {"per_digit": per_digit_meta, "canvas_size": canvas.size}
        return canvas, meta

    def render_label(self, label: int, *, rotation: float = 0.0,
                     per_digit_scale=None, per_digit_spacing=None, per_digit_color=None, per_digit_font=None,
                     image_size: int = None, background=None, color_mode: str = None, return_tensor: bool = None,
                     fixed_width_5: bool = None):
        if image_size is None: image_size = self.image_size
        if background is None: background = self.background
        if color_mode is None: color_mode = self.color_mode
        if return_tensor is None: return_tensor = self.return_tensor
        if fixed_width_5 is None: fixed_width_5 = self.fixed_width_5

        old_size, old_bg, old_mode = self.image_size, self.background, self.color_mode
        old_fw5 = self.fixed_width_5
        self.image_size, self.background, self.color_mode = int(image_size), tuple(background), color_mode.upper()
        self.fixed_width_5 = bool(fixed_width_5)

        try:
            if self.fixed_width_5:
                digits = list(f"{int(label):05d}")
            else:
                digits = list(str(int(label)))
            L = len(digits)

            scales  = self._broadcast_len(per_digit_scale,  L, float) if per_digit_scale is not None else [1.0]*L
            spacings= self._broadcast_len(per_digit_spacing, L, int)   if per_digit_spacing is not None else [0]*L
            colors  = self._broadcast_len(per_digit_color,  L, tuple) if per_digit_color is not None else [(0,0,0)]*L
            fonts   = self._broadcast_len(per_digit_font,   L, str)   if per_digit_font is not None else [""]*L

            attrs = []
            for i in range(L):
                attrs.append({"scale": scales[i], "spacing": spacings[i], "color": colors[i], "font_path": fonts[i]})
            canvas, meta_digits = self._compose_number_with_attrs(digits, attrs)
            pil_img = self._finalize(canvas, float(rotation))

            meta = {"rotation_deg": float(rotation), "digits": digits, **meta_digits}
            if return_tensor:
                arr = np.asarray(pil_img, dtype=np.uint8)
                if self.color_mode == "L":
                    arr = arr[..., 1:2]  # ensure C
                tensor = torch.from_numpy(arr).permute(2,0,1).float().div(255.0)
                return {"image": tensor, "label": int(label), "meta": meta}
            else:
                return {"image": pil_img, "label": int(label), "meta": meta}
        finally:
            self.image_size, self.background, self.color_mode = old_size, old_bg, old_mode
            self.fixed_width_5 = old_fw5

    def render_from_meta(self, meta: Dict[str, Any], label: int = None, **override):
        if label is None:
            digits_meta = meta.get("digits", None)
            if digits_meta is None:
                raise ValueError("Debes pasar 'label' o incluir 'digits' en meta.")
            label = int("".join(digits_meta))

        per_digit = meta.get("per_digit", [])
        fonts = [d.get("font_path","") for d in per_digit]
        scales = [float(d.get("scale", 1.0)) for d in per_digit]
        spacings = [int(d.get("spacing", 0)) for d in per_digit]
        colors = [tuple(d.get("color", (0,0,0))) for d in per_digit]
        rotation = float(meta.get("rotation_deg", 0.0))

        rotation = override.get("rotation", rotation)
        scales   = override.get("per_digit_scale", scales)
        spacings = override.get("per_digit_spacing", spacings)
        colors   = override.get("per_digit_color", colors)
        fonts    = override.get("per_digit_font", fonts)

        return self.render_label(label, rotation=rotation, per_digit_scale=scales, per_digit_spacing=spacings,
                                 per_digit_color=colors, per_digit_font=fonts,
                                 image_size=override.get("image_size", None) or self.image_size,
                                 background=override.get("background", None) or self.background,
                                 color_mode=override.get("color_mode", None) or self.color_mode,
                                 return_tensor=override.get("return_tensor", None) if override.get("return_tensor", None) is not None else self.return_tensor,
                                 fixed_width_5=override.get("fixed_width_5", None) if override.get("fixed_width_5", None) is not None else self.fixed_width_5)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str,Any]:
        rng = _seeded_rng(self.seed, self._epoch, self._worker_id, idx)

        if self._allowed_labels is not None:
            if self.enumerate_numbers:
                label = self._allowed_labels[idx % len(self._allowed_labels)]
            else:
                label = rng.choice(self._allowed_labels)
        else:
            lo, hi = self.min_value, self.max_value
            guard = 0
            while True:
                candidate = rng.randrange(lo, hi + 1)
                if self._label_ok(candidate):
                    label = candidate
                    break
                guard += 1
                if guard > 10000:
                    raise RuntimeError("No se pudo samplear un label válido; verifica los filtros.")

        if self.fixed_width_5:
            digits = list(f"{label:05d}")
        else:
            digits = list(str(label))

        rot = float(_sample_from_range_or_list(rng, self.rotation_deg, self.rotation_values, as_int=False))
        canvas, meta_digits = self._compose_number(digits, rng, self._worker_fonts or [])
        pil_img = self._finalize(canvas, rot)
        meta = {"epoch": self._epoch, "worker_id": self._worker_id, "rotation_deg": rot, "digits": digits, **meta_digits}

        if self.return_tensor:
            arr = np.asarray(pil_img, dtype=np.uint8)
            if self.color_mode == "L":
                arr = arr[..., None]
            tensor = torch.from_numpy(arr).permute(2,0,1).float().div(255.0)
            return {"image": tensor, "label": label, "meta": meta}
        else:
            return {"image": pil_img, "label": label, "meta": meta}
