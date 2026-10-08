"""Relative height from a single image: Depth Anything V2 (ONNX) on tiles.

The model returns relative inverse depth (higher = closer to the camera).
Seen from a satellite, closer means taller, so we use it directly as a
relative height signal. Large images get a global pass plus overlapping
518 px tiles; every tile is aligned to the global pass with a least-squares
scale/shift and feather-blended, which removes seams. Test-time augmentation
(flips/rotation) gives a per-pixel uncertainty.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "depth_anything_v2_vits.onnx"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

Progress = Callable[[float, str], None]


class OnnxDepthModel:
    is_ai = True

    def __init__(self, path: str | Path):
        import onnxruntime as ort

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        size = inp.shape[-1]
        self.size = size if isinstance(size, int) else 518
        self.name = f"Depth Anything V2 Small (ONNX, {self.session.get_providers()[0]})"

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: x})[0]
        out = np.asarray(out, np.float32).reshape(out.shape[-2], out.shape[-1])
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


class HeuristicDepthModel:
    """Fallback when no AI model is installed. NOT a depth model: it only
    turns bright, compact, locally raised blobs into a relative signal so the
    rest of the app can be demonstrated. Results are labelled accordingly."""

    is_ai = False
    size = 518
    name = "Heuristic fallback (no AI model installed)"

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        g = cv2.GaussianBlur(g, (0, 0), 1.5)
        k = max(9, (min(g.shape) // 40) | 1)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
        broad = cv2.GaussianBlur(g, (0, 0), min(g.shape) / 15)
        return 0.7 * tophat / (tophat.max() + 1e-6) + 0.3 * broad


def load_default_model():
    path = Path(os.environ.get("DEPTHWIZARD_MODEL", DEFAULT_MODEL))
    if path.exists():
        return OnnxDepthModel(path)
    return HeuristicDepthModel()


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xm, ym = x.mean(), y.mean()
    var = np.mean((x - xm) ** 2)
    if var <= np.finfo(np.float32).eps:            # featureless: carry the level only
        return 0.0, float(ym)
    a = np.mean((x - xm) * (y - ym)) / var
    return float(a), float(ym - a * xm)


def align_params(t: np.ndarray, g: np.ndarray, trim_frac: float = 0.05) -> tuple[float, float]:
    """alpha, beta minimising |alpha*t + beta - g|^2 over all but the largest residuals.

    Robust start (median / MAD), then two trimmed refits: a plain least-squares start lets a few
    extreme pixels (water noise) drag the slope to ~0, after which trimming can't find them.
    (Ported from the original DepthWizard depth/stitch.py.)"""
    x = t.ravel().astype(np.float64)
    y = g.ravel().astype(np.float64)
    if x.size > 40000:
        idx = np.random.default_rng(0).choice(x.size, 40000, replace=False)
        x, y = x[idx], y[idx]
    mx, my = np.median(x), np.median(y)
    sx, sy = np.median(np.abs(x - mx)), np.median(np.abs(y - my))
    if sx <= np.finfo(np.float32).eps:
        return 0.0, float(my)
    a, b = sy / sx, my - sy / sx * mx
    for _ in range(2):
        resid = np.abs(a * x + b - y)
        keep = resid <= np.quantile(resid, 1.0 - trim_frac)
        a, b = _fit_affine(x[keep], y[keep])
    if a <= 0:                                     # never flip a tile's sign
        a = (y.std() + 1e-6) / (x.std() + 1e-6)
        b = float(y.mean() - a * x.mean())
    return a, b


def _align(pred: np.ndarray, ref: np.ndarray, trim_frac: float = 0.05) -> np.ndarray:
    a, b = align_params(pred, ref, trim_frac)
    return a * pred + b


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets of tiles covering [0, length); the last tile sits flush with the end."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def taper(n: int, flat_start: bool, flat_end: bool) -> np.ndarray:
    """1D sin^2 (Hann) window; at 50% overlap neighbouring windows sum to 1. A side on the image
    boundary is held at full weight, otherwise edge pixels would get ~zero total weight."""
    w = np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2
    half = n // 2
    if flat_start:
        w[:half] = 1.0
    if flat_end:
        w[half:] = 1.0
    return w.astype(np.float32)


def window2d(y: int, x: int, th: int, tw: int, height: int, width: int) -> np.ndarray:
    return np.outer(taper(th, y == 0, y + th >= height), taper(tw, x == 0, x + tw >= width)) + 1e-6


def seam_ratio(d: np.ndarray, tile: int, stride: int) -> float:
    """Mean |gradient| across tile borders / mean elsewhere. ~1.0 means invisible seams."""
    h, w = d.shape
    gx, gy = np.abs(np.diff(d, axis=1)), np.abs(np.diff(d, axis=0))

    def borders(n):
        edges = {e for s0 in tile_origins(n, tile, stride) for e in (s0, s0 + tile) if 0 < e < n}
        m = np.zeros(n - 1, bool)
        m[[e - 1 for e in edges]] = True
        return m
    on_x, on_y = borders(w), borders(h)
    border = np.concatenate([gx[:, on_x].ravel(), gy[on_y, :].ravel()])
    inner = np.concatenate([gx[:, ~on_x].ravel(), gy[~on_y, :].ravel()])
    return float(border.mean() / max(inner.mean(), 1e-12))


def _tta(model, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = model.infer(rgb)
    variants = [base,
                model.infer(rgb[:, ::-1].copy())[:, ::-1],
                model.infer(rgb[::-1].copy())[::-1],
                np.rot90(model.infer(np.rot90(rgb).copy()), -1)]
    aligned = np.stack([base] + [_align(v, base) for v in variants[1:]])
    return aligned.mean(0), aligned.std(0)


def estimate_relative_height(rgb: np.ndarray, model, tile: int = 518, overlap: float = 0.5,
                             tta: bool = True, progress: Progress | None = None,
                             trim_frac: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return (relative height in [0, 1], uncertainty in the same units).

    Global pass G, then native-resolution tiles at 50% overlap, each aligned to G with a robust
    trimmed affine fit and blended with Hann windows (original DepthWizard Stage 2)."""
    report = progress or (lambda f, m: None)
    h, w = rgb.shape[:2]
    report(0.05, "Global depth pass")
    if tta:
        glob, unc = _tta(model, rgb)
    else:
        glob, unc = model.infer(rgb), np.zeros((h, w), np.float32)

    rel = glob
    if max(h, w) > tile * 1.25:
        stride = max(1, int(tile * (1 - overlap)))
        acc = np.zeros((h, w), np.float64)
        wsum = np.zeros((h, w), np.float64)
        boxes = [(y, x) for y in tile_origins(h, tile, stride) for x in tile_origins(w, tile, stride)]
        for i, (y, x) in enumerate(boxes):
            report(0.1 + 0.8 * i / len(boxes), f"Depth tile {i + 1}/{len(boxes)}")
            crop = rgb[y:y + tile, x:x + tile]
            pred = _align(model.infer(crop), glob[y:y + tile, x:x + tile], trim_frac)
            wt = window2d(y, x, pred.shape[0], pred.shape[1], h, w)
            acc[y:y + tile, x:x + tile] += wt * pred
            wsum[y:y + tile, x:x + tile] += wt
        rel = (acc / wsum).astype(np.float32)

    lo, hi = np.percentile(rel, [0.5, 99.5])
    span = max(hi - lo, 1e-6)
    report(0.95, "Normalising depth")
    return np.clip((rel - lo) / span, 0, 1).astype(np.float32), (unc / span).astype(np.float32)
