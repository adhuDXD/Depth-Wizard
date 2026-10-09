"""Is this a top-down image of land? Rejects selfies, pets, street photos, documents and drawings
before any processing, with a reason the user can act on.

Tests on a 512 px copy of the image. Strong evidence rejects the image (the user can still press
"Process anyway"); weak evidence only adds a warning and processing continues.

1 vivid colour  Satellite colour is muted by the atmosphere and the sensor. More than 12% strongly
                saturated pixels means a phone photo, a painting or a graphic.
2 flat colour   Drawings, logos, screenshots and charts have large areas of one exact colour.
3 document      Grey, bright paper background with sharp dark strokes.
4 dark          Mostly black (night sky, lens cap, a dark screenshot).
5 depth jumps   A camera on the ground sees near objects in front of a far background, so the AI
                depth map has big steps. From orbit everything is about equally far away, but dense
                towns at 0.3 m still give steps at every roof edge (up to 0.11 in tests), so only
                steps above 0.15, or above 0.08 with sky along the top of the picture, reject.
                Smaller steps (0.08-0.15) only warn.

Tested on 72 satellite scenes from 18 Maxar disaster events (incl. Namchi) plus the demo town (towns, farms,
forest, desert, snow, flood water, cloud): none rejected. Of 93 other pictures, 73 were rejected
and 8 more warned. Details in tests/test_inputcheck.py.

Georeferenced GeoTIFFs skip the check: a raster with map coordinates is a map by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DEPTH_EDGE_REJECT = 0.15
DEPTH_EDGE_WARN = 0.08
VIVID_MAX = 0.12
FLAT_MAX = 0.25
DARK_MAX = 0.6

REASONS = {
    "depth": "it looks like a photo taken from the ground: the depth model sees objects close to the "
             "camera in front of a distant background",
    "sky": "it looks like a photo taken from the ground: there is sky along the top and objects close "
           "to the camera",
    "vivid": "its colours are far brighter than in satellite or aerial images (a phone photo, painting "
             "or graphic)",
    "flat": "large areas are one flat colour, as in a drawing, logo, chart or screenshot",
    "document": "it looks like a page of text or a scanned document",
    "dark": "it is almost completely black",
}


@dataclass
class InputCheck:
    ok: bool
    reasons: list[str] = field(default_factory=list)      # keys of REASONS
    scores: dict = field(default_factory=dict)
    warning: str = ""                                     # weak evidence: processed, but say so

    @property
    def message(self) -> str:
        if self.ok:
            return ""
        why = "; ".join(REASONS[r] for r in self.reasons)
        return ("This does not look like a satellite, aerial or drone image of land: " + why + ". "
                "DepthWizard needs a top-down view of the ground (a satellite image, a Google Earth or "
                "Bhuvan screenshot in satellite view, or a drone map).")


def _small(rgb: np.ndarray, side: int = 512) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = side / max(h, w)
    if s >= 1:
        return rgb
    return cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def check_image(rgb: np.ndarray, model=None) -> InputCheck:
    """`model` is the depth model; the depth test runs only when it is the real AI model."""
    rgb = _small(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat, val = hsv[..., 1] / 255, hsv[..., 2] / 255
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    vivid = float(((sat > 0.6) & (val > 0.35)).mean())
    flat = float(((np.diff(gray, axis=1)[:-1] == 0) & (np.diff(gray, axis=0)[:, :-1] == 0)).mean())
    dark = float((gray < 25).mean())

    hist = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float32)
    mode = int(np.argmax(np.convolve(hist, np.ones(9) / 9, mode="same")))
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    sharp = float((grad > 200).mean())
    document = float(sat.mean()) < 0.02 and mode > 180 and sharp > 0.1

    # sky: a smooth, bright, blue or white band along the top that is not also along the bottom
    # (clouds seen from above are usually everywhere, not only at the top)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1), cv2.CV_32F))
    hue = hsv[..., 0] * 2
    skyish = (lap < 2.0) & (val > 0.45) & (((hue > 180) & (hue < 260) & (sat > 0.08)) | (sat < 0.15))
    band = max(1, rgb.shape[0] // 6)
    sky = float(skyish[:band].mean()) > 0.6 and float(skyish[-band:].mean()) < 0.25

    scores = {"vivid": round(vivid, 3), "flat": round(flat, 3), "dark": round(dark, 3),
              "document": document, "sky_on_top": sky}
    reasons = []
    warning = ""
    if model is not None and getattr(model, "is_ai", False):
        d = model.infer(rgb).astype(np.float32)
        p5, p95 = np.percentile(d, [5, 95])
        gy, gx = np.gradient(d)
        depth_edge = float(np.percentile(np.hypot(gx, gy), 99) / (p95 - p5 + 1e-6))
        scores["depth_edge"] = round(depth_edge, 3)
        if depth_edge > DEPTH_EDGE_REJECT:
            reasons.append("depth")
        elif depth_edge > DEPTH_EDGE_WARN:
            if sky:
                reasons.append("sky")
            else:
                warning = ("This may not be a top-down image (the depth model sees large height jumps). "
                           "Heights are only meaningful for satellite, aerial or drone views.")
    if vivid > VIVID_MAX:
        reasons.append("vivid")
    if flat > FLAT_MAX:
        reasons.append("flat")
    if document:
        reasons.append("document")
    if dark > DARK_MAX:
        reasons.append("dark")
    return InputCheck(ok=not reasons, reasons=reasons, scores=scores, warning="" if reasons else warning)
