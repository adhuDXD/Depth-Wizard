"""Colour maps, hillshade and PNG encoding for map layers."""
from __future__ import annotations

import cv2
import numpy as np

TERRAIN = [(0.0, (40, 70, 140)), (0.15, (60, 140, 160)), (0.35, (90, 170, 100)), (0.55, (200, 200, 110)),
           (0.75, (180, 120, 70)), (1.0, (245, 245, 245))]
HEIGHTS = [(0.0, (20, 24, 40)), (0.25, (60, 40, 120)), (0.5, (190, 60, 110)), (0.75, (250, 150, 60)),
           (1.0, (250, 245, 160))]
ERROR = [(0.0, (40, 120, 220)), (0.5, (245, 245, 245)), (1.0, (215, 40, 40))]
CONFIDENCE = np.array([[215, 40, 40], [240, 180, 40], [40, 170, 90]], np.uint8)


def colormap(values: np.ndarray, stops, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    v = values.astype(np.float32)
    finite = np.isfinite(v)
    if vmin is None or vmax is None:
        lo, hi = np.percentile(v[finite], [1, 99]) if finite.any() else (0, 1)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    t = np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    xs = [s[0] for s in stops]
    out = np.stack([np.interp(t, xs, [s[1][c] for s in stops]) for c in range(3)], -1)
    out[~finite] = 0
    return out.astype(np.uint8)


def hillshade(z: np.ndarray, cell: float = 1.0, az: float = 315, el: float = 45, exaggerate: float = 1.0) -> np.ndarray:
    gy, gx = np.gradient(z.astype(np.float32) * exaggerate, cell)
    a, e = np.radians(az), np.radians(el)
    sun = np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    n = np.dstack([-gx, gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return np.clip(n @ sun, 0, 1)


def shaded(z: np.ndarray, stops, cell: float, exaggerate: float = 1.0, **kw) -> np.ndarray:
    col = colormap(z, stops, **kw).astype(np.float32)
    hs = hillshade(z, cell, exaggerate=exaggerate)
    return np.clip(col * (0.45 + 0.65 * hs[..., None]), 0, 255).astype(np.uint8)


def png_bytes(img: np.ndarray) -> bytes:
    if img.ndim == 2:
        enc = img
    elif img.shape[2] == 4:
        enc = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    else:
        enc = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", enc)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def composite_map(rgb: np.ndarray, overlay: np.ndarray, payload: dict, route: dict | None = None) -> np.ndarray:
    """Base image + hazard overlay + arrows + zone labels, for print/export."""
    h, w = rgb.shape[:2]
    ov = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    a = ov[..., 3:4] / 255 * 0.85
    out = (rgb.astype(np.float32) * (1 - a) + ov[..., :3] * a).astype(np.uint8)
    out = np.ascontiguousarray(out)
    lw = max(1, w // 500)
    for u0, v0, u1, v1, _ in payload.get("arrows", []):
        cv2.arrowedLine(out, (int(u0 * w), int(v0 * h)), (int(u1 * w), int(v1 * h)), (255, 255, 255), lw + 2,
                        tipLength=0.4, line_type=cv2.LINE_AA)
        cv2.arrowedLine(out, (int(u0 * w), int(v0 * h)), (int(u1 * w), int(v1 * h)), (20, 20, 20), lw,
                        tipLength=0.4, line_type=cv2.LINE_AA)
    if route and route.get("path"):
        pts = np.array([[p[0] * w, p[1] * h] for p in route["path"]], np.int32)
        cv2.polylines(out, [pts], False, (255, 255, 255), lw + 4, cv2.LINE_AA)
        cv2.polylines(out, [pts], False, (230, 30, 120), lw + 2, cv2.LINE_AA)
    fs = max(0.5, w / 1400)
    for z in payload.get("zones", []):
        p = (int(z["u"] * w), int(z["v"] * h))
        cv2.circle(out, p, int(14 * fs), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, p, int(14 * fs), (20, 120, 60), 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(z["name"], cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.putText(out, z["name"], (p[0] - tw // 2, p[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 120, 60), 2,
                    cv2.LINE_AA)
    for b in payload.get("bottlenecks", []):
        p = (int(b["u"] * w), int(b["v"] * h))
        cv2.circle(out, p, int(13 * fs), (230, 40, 40), -1, cv2.LINE_AA)
        cv2.putText(out, "!", (p[0] - int(4 * fs), p[1] + int(8 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out
