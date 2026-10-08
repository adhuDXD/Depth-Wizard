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


def _align(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Least-squares scale/shift so that s*pred + t ~= ref."""
    p, r = pred.ravel(), ref.ravel()
    if mask is not None:
        m = mask.ravel()
        p, r = p[m], r[m]
    if p.size > 20000:
        idx = np.random.default_rng(0).choice(p.size, 20000, replace=False)
        p, r = p[idx], r[idx]
    A = np.stack([p, np.ones_like(p)], 1)
    (s, t), *_ = np.linalg.lstsq(A, r, rcond=None)
    if not np.isfinite(s) or s <= 0:
        s = (r.std() + 1e-6) / (p.std() + 1e-6)
        t = r.mean() - s * p.mean()
    return s * pred + t


def _feather(h: int, w: int, ramp: int) -> np.ndarray:
    def ramp1d(n):
        x = np.ones(n, np.float32)
        r = min(ramp, n // 2)
        if r > 0:
            edge = (np.arange(r, dtype=np.float32) + 1) / (r + 1)
            x[:r], x[-r:] = edge, edge[::-1]
        return x
    return np.outer(ramp1d(h), ramp1d(w)) + 1e-3


def _tile_starts(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile, step)) + [n - tile]
    return sorted(set(starts))


def _tta(model, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = model.infer(rgb)
    variants = [base,
                model.infer(rgb[:, ::-1].copy())[:, ::-1],
                model.infer(rgb[::-1].copy())[::-1],
                np.rot90(model.infer(np.rot90(rgb).copy()), -1)]
    aligned = np.stack([base] + [_align(v, base) for v in variants[1:]])
    return aligned.mean(0), aligned.std(0)


def estimate_relative_height(rgb: np.ndarray, model, tile: int = 518, overlap: float = 0.25,
                             tta: bool = True, progress: Progress | None = None
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Return (relative height in [0, 1], uncertainty in the same units)."""
    report = progress or (lambda f, m: None)
    h, w = rgb.shape[:2]
    report(0.05, "Global depth pass")
    if tta:
        glob, unc = _tta(model, rgb)
    else:
        glob, unc = model.infer(rgb), np.zeros((h, w), np.float32)

    rel = glob
    if max(h, w) > tile * 1.25:
        step = int(tile * (1 - overlap))
        acc = np.zeros((h, w), np.float64)
        wsum = np.zeros((h, w), np.float64)
        boxes = [(y, x) for y in _tile_starts(h, tile, step) for x in _tile_starts(w, tile, step)]
        for i, (y, x) in enumerate(boxes):
            report(0.1 + 0.8 * i / len(boxes), f"Depth tile {i + 1}/{len(boxes)}")
            crop = rgb[y:y + tile, x:x + tile]
            pred = _align(model.infer(crop), glob[y:y + tile, x:x + tile])
            wt = _feather(*pred.shape, ramp=int(tile * overlap))
            acc[y:y + tile, x:x + tile] += wt * pred
            wsum[y:y + tile, x:x + tile] += wt
        rel = (acc / wsum).astype(np.float32)

    lo, hi = np.percentile(rel, [0.5, 99.5])
    span = max(hi - lo, 1e-6)
    report(0.95, "Normalising depth")
    return np.clip((rel - lo) / span, 0, 1).astype(np.float32), (unc / span).astype(np.float32)
