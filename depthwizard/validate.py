"""Compare a DSM against a reference (LiDAR, stereo or synthetic truth)."""
from __future__ import annotations

import numpy as np

OBJECT_MIN_HEIGHT_M = 2.5
NMAD_K = 1.4826          # makes NMAD equal the standard deviation for normal errors


def metrics(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> dict:
    m = np.isfinite(pred) & np.isfinite(ref)
    if mask is not None:
        m &= mask
    if m.sum() < 10:
        return {"rmse": None, "mae": None, "bias": None, "corr": None, "nmad": None,
                "offset_free_rmse": None, "n": int(m.sum())}
    d = pred[m] - ref[m]
    corr = np.corrcoef(pred[m], ref[m])[0, 1] if np.std(pred[m]) > 0 and np.std(ref[m]) > 0 else None
    med = float(np.median(d))
    # NMAD and offset-free RMSE as in the original DepthWizard evals/metrics.py: robust to a few
    # extreme outliers, and to a constant datum offset between the DSM and the reference
    return {"rmse": round(float(np.sqrt(np.mean(d ** 2))), 2), "mae": round(float(np.mean(np.abs(d))), 2),
            "bias": round(float(np.mean(d)), 2), "corr": None if corr is None else round(float(corr), 3),
            "nmad": round(float(NMAD_K * np.median(np.abs(d - med))), 2),
            "offset_free_rmse": round(float(np.sqrt(np.mean((d - med) ** 2))), 2),
            "n": int(m.sum())}


def compare(dsm: np.ndarray, dtm: np.ndarray | None, ref: np.ndarray, building: np.ndarray) -> dict:
    """DepthWizard vs DEM-only.

    "objects" are pixels where the *reference* stands > 2.5 m above the terrain
    (buildings and trees), so the score does not depend on our own detections.
    Without a DEM we fall back to the detected-building mask.
    """
    objects = (ref - dtm) > OBJECT_MIN_HEIGHT_M if dtm is not None else building
    out = {"depthwizard": {"all": metrics(dsm, ref), "buildings": metrics(dsm, ref, objects)},
           "object_definition": "reference > 2.5 m above DEM" if dtm is not None else "detected buildings"}
    if dtm is not None:
        out["dem_only"] = {"all": metrics(dtm, ref), "buildings": metrics(dtm, ref, objects)}
    return out
