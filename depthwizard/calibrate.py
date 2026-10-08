"""Turn relative model output into heights in metres, honestly.

Steps
1. Split the relative surface into ground (morphological opening) and
   height-above-ground (nDSM).
2. Collect metric anchors:
   * shadow anchors  - building height = shadow length x tan(sun elevation)
   * GCP anchors     - known elevations at clicked points
3. Robust fit (median ratio + MAD rejection, optional smooth spatial field)
   with leave-one-out error, so every output carries its own accuracy figure.
4. Terrain comes from the DEM (+ datum offset, + GCP bias) when available.

When nothing gives a scale, the output stays a labelled *relative* DSM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

DEM_SIGMA_M = 5.0          # typical vertical error of 30 m national DEMs
MIN_SUN_ELEVATION = 10.0


@dataclass
class CalibParams:
    gsd: float | None = None
    sun_elevation: float | None = None
    sun_azimuth: float | None = None
    gcps: list[dict] = field(default_factory=list)   # {"u","v" in [0,1], "elevation" m}
    geoid_offset_m: float = 0.0
    building_height_prior_m: float | None = None     # "tallest buildings are about X m"
    ground_window_m: float = 60.0


@dataclass
class HeightModel:
    rel: np.ndarray
    unc_rel: np.ndarray
    terrain_rel: np.ndarray
    ndsm_rel: np.ndarray
    ndsm: np.ndarray
    dtm: np.ndarray | None
    dsm: np.ndarray
    sigma: np.ndarray
    confidence: np.ndarray       # 0 red, 1 amber, 2 green
    building: np.ndarray
    tree: np.ndarray
    water: np.ndarray
    shadow: np.ndarray
    mode: str                    # absolute | above_ground | terrain_only | relative
    height_unit: str             # m | relative
    info: dict


# ---------------------------------------------------------------- masks

def _remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab]


def detect_shadows(rgb: np.ndarray, water: np.ndarray | None = None) -> np.ndarray:
    v = rgb.max(axis=2).astype(np.float32)
    v = cv2.GaussianBlur(v, (3, 3), 0)
    otsu, _ = cv2.threshold(v.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = min(otsu, 0.6 * np.median(v))
    m = v < thr
    if water is not None:
        m &= ~water
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    return _remove_small(m, 4)


def detect_vegetation(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / (f.sum(axis=2) + 1)
    return exg > 0.06


def detect_water(rgb: np.ndarray, gsd: float | None) -> np.ndarray:
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    gray = f.mean(axis=2)
    local_std = np.sqrt(np.maximum(
        ndimage.uniform_filter(gray ** 2, 7) - ndimage.uniform_filter(gray, 7) ** 2, 0))
    m = (b > g * 1.03) & (b > r * 1.12) & (local_std < 10)
    min_px = int(2000 / gsd ** 2) if gsd else 400
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return _remove_small(m, max(min_px, 50))


def ground_surface(rel: np.ndarray, gsd: float | None, window_m: float) -> np.ndarray:
    h, w = rel.shape
    k = int(round(window_m / gsd)) if gsd else max(h, w) // 25
    k = int(np.clip(k, 5, max(h, w) // 3))
    smooth = ndimage.gaussian_filter(rel, 1.0)
    opened = ndimage.grey_opening(smooth, size=(k, k))
    return ndimage.gaussian_filter(opened, k / 3).astype(np.float32)


def _detrend(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    yy, xx = np.mgrid[0:h:8, 0:w:8]
    A = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], 1)
    coef, *_ = np.linalg.lstsq(A, a[::8, ::8].ravel(), rcond=None)
    Y, X = np.mgrid[0:h, 0:w]
    out = a - (coef[0] * X + coef[1] * Y + coef[2])
    lo, hi = np.percentile(out, [1, 99])
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


# ---------------------------------------------------------------- anchors

def shadow_anchors(building_lab: np.ndarray, n_lab: int, shadow: np.ndarray, ndsm_rel: np.ndarray,
                   gsd: float, sun_el: float, sun_az: float, max_len_m: float = 150.0,
                   dtm: np.ndarray | None = None) -> list[dict]:
    """Building height from the length of the shadow it casts.

    On sloping ground (hill towns) a shadow falling uphill is shorter and one
    falling downhill longer. With a terrain model the height is
    h = L * (tan(sun elevation) + s), where s is the terrain slope along the
    shadow direction (rise per metre, uphill positive).

    Shadows point away from the sun: in image coordinates (row down = south,
    col right = east) that direction is (cos A, -sin A) for sun azimuth A.
    """
    if sun_el < MIN_SUN_ELEVATION or gsd is None:
        return []
    a = np.radians(sun_az)
    d = np.array([np.cos(a), -np.sin(a)])
    h, w = shadow.shape
    bmask = building_lab > 0

    # building pixels whose next pixel along the shadow direction is outside the building
    rows, cols = np.nonzero(bmask)
    nr = np.rint(rows + d[0]).astype(int)
    nc = np.rint(cols + d[1]).astype(int)
    inside = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
    rows, cols, nr, nc = rows[inside], cols[inside], nr[inside], nc[inside]
    edge = (building_lab[nr, nc] != building_lab[rows, cols]) & shadow[nr, nc]
    rows, cols = rows[edge], cols[edge]
    if rows.size == 0:
        return []
    labels = building_lab[rows, cols]

    max_steps = int(max_len_m / gsd)
    length = np.zeros(rows.size, np.int32)
    alive = np.ones(rows.size, bool)
    valid = np.zeros(rows.size, bool)
    for t in range(1, max_steps + 1):
        r = np.rint(rows + t * d[0]).astype(int)
        c = np.rint(cols + t * d[1]).astype(int)
        oob = (r < 0) | (r >= h) | (c < 0) | (c >= w)
        r, c = np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)
        in_shadow = shadow[r, c] & ~oob
        ends_clean = alive & ~in_shadow & ~oob & ~bmask[r, c]
        valid |= ends_clean
        alive &= in_shadow
        length += alive
        if not alive.any():
            break

    anchors = []
    tan_el = np.tan(np.radians(sun_el))
    along = None
    if dtm is not None:
        gr, gc = np.gradient(ndimage.gaussian_filter(dtm.astype(np.float32), 3), gsd)
        along = gr * d[0] + gc * d[1]          # terrain rise per metre along the shadow
    for lab in np.unique(labels[valid]):
        sel = valid & (labels == lab)
        if sel.sum() < 3:
            continue
        q25, q75 = np.percentile(length[sel], [25, 75])
        L_px = q75 + 0.5
        k = tan_el
        if along is not None:
            k = max(tan_el + float(np.median(along[rows[sel], cols[sel]])), 0.2 * tan_el)
        height_m = float(L_px * gsd * k)
        spread_m = float(max(q75 - q25, 1.0) * gsd * k)
        comp = building_lab == lab
        r_val = float(np.percentile(ndsm_rel[comp], 75))
        if height_m < 2.0 or r_val <= 0.005:
            continue
        cy, cx = ndimage.center_of_mass(comp)
        anchors.append({"source": "shadow", "row": float(cy), "col": float(cx),
                        "r": r_val, "h": height_m, "spread_m": spread_m, "label": int(lab),
                        "weight": float(min(sel.sum(), 20))})
    return anchors


def fit_scale(anchors: list[dict]) -> dict | None:
    """Robust metres-per-relative-unit with leave-one-out error."""
    usable = [a for a in anchors if a["r"] > 0.005 and a["h"] > 0]
    if not usable:
        return None
    k = np.array([a["h"] / a["r"] for a in usable])
    wts = np.array([a["weight"] for a in usable])

    def wmedian(vals, ws):
        o = np.argsort(vals)
        cw = np.cumsum(ws[o])
        return float(vals[o][np.searchsorted(cw, cw[-1] / 2)])

    med = wmedian(k, wts)
    mad = 1.4826 * np.median(np.abs(k - med))
    tol = max(3 * mad, 0.25 * med)
    inl = np.abs(k - med) <= tol
    scale = wmedian(k[inl], wts[inl])
    for a, keep in zip(usable, inl):
        a["inlier"] = bool(keep)

    idx = np.nonzero(inl)[0]
    errors = []
    if idx.size >= 3:
        for i in idx:
            others = idx[idx != i]
            s_i = wmedian(k[others], wts[others])
            errors.append(s_i * usable[i]["r"] - usable[i]["h"])
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    hs = np.array([usable[i]["h"] for i in idx])
    return {"scale": scale, "n_anchors": len(usable), "n_inliers": int(idx.size),
            "loo_rmse_m": rmse, "loo_mae_m": float(np.mean(np.abs(errors))) if errors else None,
            "median_anchor_h": float(np.median(hs)), "anchors": usable}


def spatial_scale_field(anchors: list[dict], scale: float, shape: tuple[int, int]) -> np.ndarray | None:
    inl = [a for a in anchors if a.get("inlier")]
    if len(inl) < 12:
        return None
    h, w = shape
    gh, gw = max(2, h // 32), max(2, w // 32)
    gy, gx = np.mgrid[0:gh, 0:gw]
    gy = (gy + 0.5) * h / gh
    gx = (gx + 0.5) * w / gw
    sigma = np.hypot(h, w) / 4
    num = np.zeros((gh, gw))
    den = np.full((gh, gw), 3.0)   # prior weight pulls towards the global scale
    for a in inl:
        wgt = a["weight"] * np.exp(-((gy - a["row"]) ** 2 + (gx - a["col"]) ** 2) / (2 * sigma ** 2))
        num += wgt * np.log(a["h"] / a["r"] / scale)
        den += wgt
    field_ = np.exp(num / den).astype(np.float32)
    return scale * cv2.resize(field_, (w, h), interpolation=cv2.INTER_CUBIC)


def _fuse_heights(ndsm_rel, unc_rel, scale_map, building, tree, building_lab, fit, info):
    """Shadow-first fusion.

    * Buildings with a measured shadow get that physical height (flat roof).
    * Other buildings get the model estimate, shrunk towards the typical
      measured height by how well the model agrees with the measurements
      (rho = correlation between model value and measured height). A model
      that does not track heights is not trusted to invent them.
    """
    model_est = ndsm_rel * scale_map
    # below the object threshold the surface is ground: nDSM = 0
    ndsm = np.where(building | tree, model_est, 0.0)
    sigma_n = np.where(tree, np.maximum(2 * unc_rel * scale_map, 0.35 * model_est), 0.7)
    if not fit:
        sigma_n = np.where(building, np.maximum(2 * unc_rel * scale_map, 0.35 * ndsm), sigma_n)
        return ndsm, sigma_n

    anchors = fit["anchors"]
    r = np.array([a["r"] for a in anchors])
    hm_ = np.array([a["h"] for a in anchors])
    med = float(np.median(hm_))
    rho = float(np.corrcoef(r, hm_)[0, 1]) if len(anchors) >= 8 and r.std() > 0 and hm_.std() > 0 else 1.0
    rho = max(0.0, rho) if np.isfinite(rho) else 0.0
    pred = med + rho * (fit["scale"] * r - med)
    model_rmse = float(np.sqrt(np.mean((pred - hm_) ** 2)))

    ndsm = np.where(building, med + rho * (model_est - med), ndsm)
    ndsm = np.maximum(ndsm, 0)
    sigma_n = np.where(building, model_rmse, sigma_n)

    measured = 0
    spreads = []
    for a in anchors:
        lab = a.get("label")
        if a["source"] != "shadow" or lab is None:
            continue
        comp = building_lab == lab
        ndsm[comp] = a["h"]
        sigma_n[comp] = np.hypot(a.get("spread_m", 1.0), 1.0)
        spreads.append(a.get("spread_m", 1.0))
        measured += 1
    total = int(building_lab.max())
    info.update({
        "buildings_measured_by_shadow": measured,
        "buildings_estimated_by_model": max(total - measured, 0),
        "model_height_agreement_rho": round(rho, 2),
        "model_estimate_rmse_m": round(model_rmse, 2),
        "shadow_measurement_spread_m": round(float(np.median(spreads)), 2) if spreads else None,
    })
    if measured:
        info["notes"].append(
            f"{measured} building heights measured from their shadows; "
            f"{max(total - measured, 0)} estimated from the AI model (agreement with measurements rho={rho:.2f}).")
    return ndsm, sigma_n


# ---------------------------------------------------------------- main

def calibrate(rgb: np.ndarray, rel: np.ndarray, unc_rel: np.ndarray, dem: np.ndarray | None,
              p: CalibParams) -> HeightModel:
    h, w = rel.shape
    gsd = p.gsd
    info: dict = {"notes": []}

    ground_rel = ground_surface(rel, gsd, p.ground_window_m)
    ndsm_rel = np.clip(rel - ground_rel, 0, None)
    terrain_rel = _detrend(ground_rel)

    water = detect_water(rgb, gsd)
    shadow = detect_shadows(rgb, water)
    veg = detect_vegetation(rgb)

    thr = max(0.02, 0.2 * float(np.percentile(ndsm_rel, 99)))
    elevated = (ndsm_rel > thr) & ~water
    elevated = cv2.morphologyEx(elevated.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    min_bldg_px = int(20 / gsd ** 2) if gsd else 15
    tree = elevated & veg
    building = _remove_small(elevated & ~veg, max(min_bldg_px, 6))
    n_lab, building_lab = cv2.connectedComponents(building.astype(np.uint8), connectivity=8)

    # ---- terrain
    dtm = None
    gcp_ground, gcp_elev = [], []
    for g in p.gcps:
        r = int(np.clip(round(g["v"] * (h - 1)), 0, h - 1))
        c = int(np.clip(round(g["u"] * (w - 1)), 0, w - 1))
        (gcp_ground if ndsm_rel[r, c] <= thr else gcp_elev).append((r, c, float(g["elevation"])))

    if dem is not None:
        sig = max(1.0, 15.0 / gsd) if gsd else 2.0
        dtm = ndimage.gaussian_filter(dem.astype(np.float32), sig) + p.geoid_offset_m
        info["terrain_source"] = "DEM"
        if p.geoid_offset_m:
            info["notes"].append(f"Vertical datum offset of {p.geoid_offset_m:+.1f} m applied to the DEM.")
        if gcp_ground:
            bias = float(np.median([e - dtm[r, c] for r, c, e in gcp_ground]))
            dtm += bias
            info["gcp_dem_bias_m"] = bias
            info["notes"].append(f"DEM shifted by {bias:+.1f} m to match {len(gcp_ground)} ground GCPs.")
    elif gcp_ground and gsd:
        pts = np.array(gcp_ground, float)
        if len(pts) >= 3:
            A = np.c_[pts[:, 1], pts[:, 0], np.ones(len(pts))]
            coef, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
            Y, X = np.mgrid[0:h, 0:w]
            dtm = (coef[0] * X + coef[1] * Y + coef[2]).astype(np.float32)
        else:
            dtm = np.full((h, w), pts[:, 2].mean(), np.float32)
        info["terrain_source"] = f"plane through {len(pts)} ground GCPs"

    # ---- metric anchors
    anchors: list[dict] = []
    if gsd and p.sun_elevation is not None and p.sun_azimuth is not None:
        anchors += shadow_anchors(building_lab, n_lab, shadow, ndsm_rel, gsd, p.sun_elevation, p.sun_azimuth,
                                  dtm=dtm)
        info["shadow_anchors"] = len(anchors)
        if dtm is not None:
            info["notes"].append("Shadow heights corrected for terrain slope (from the DEM).")
    elif not gsd:
        info["notes"].append("Pixel size unknown: shadow calibration disabled.")
    else:
        info["notes"].append("Sun angles unknown: shadow calibration disabled.")

    if dtm is not None:
        for r, c, e in gcp_elev:
            anchors.append({"source": "gcp", "row": r, "col": c, "r": float(ndsm_rel[r, c]),
                            "h": e - float(dtm[r, c]), "weight": 20.0})

    # ---- height scale
    fit = fit_scale(anchors) if anchors else None
    scale_map = None
    if fit:
        sources = sorted({a["source"] for a in fit["anchors"] if a.get("inlier")})
        info["scale_source"] = "+".join(sources) or "none"
        scale_map = spatial_scale_field(fit["anchors"], fit["scale"], (h, w))
        if scale_map is None:
            scale_map = np.full((h, w), fit["scale"], np.float32)
        else:
            info["notes"].append("Scale varies smoothly across the scene (>= 12 anchors).")
        info.update({k: v for k, v in fit.items() if k != "anchors"})
        info["anchors"] = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.items()}
                           for a in fit["anchors"]]
    elif p.building_height_prior_m:
        bvals = ndsm_rel[building] if building.any() else ndsm_rel
        ref = float(np.percentile(bvals, 99))
        if ref > 0:
            scale_map = np.full((h, w), p.building_height_prior_m / ref, np.float32)
            info["scale_source"] = "user prior"
            info["notes"].append(
                f"Building heights scaled so the tallest are ~{p.building_height_prior_m:.0f} m (user estimate, unverified).")

    # ---- assemble
    if scale_map is not None:
        height_unit = "m"
        ndsm, sigma_n = _fuse_heights(ndsm_rel, unc_rel, scale_map, building, tree, building_lab, fit, info)
    else:
        ndsm = ndsm_rel
        height_unit = "relative"
        sigma_n = 2 * unc_rel + 0.02

    if dtm is not None and height_unit == "m":
        mode, dsm = "absolute", dtm + ndsm
        sigma = np.sqrt(sigma_n ** 2 + DEM_SIGMA_M ** 2)
    elif dtm is not None:
        mode, dsm, sigma = "terrain_only", dtm.copy(), np.full((h, w), DEM_SIGMA_M, np.float32)
        info["notes"].append("Terrain is in metres but building heights could not be scaled. "
                             "Add sun angles + pixel size, GCPs, or a typical building height.")
    elif height_unit == "m":
        mode, dsm, sigma = "above_ground", ndsm.copy(), sigma_n
        info["notes"].append("Heights are metres above local ground; no DEM, so terrain shape is not in metres.")
    else:
        mode, dsm, sigma = "relative", rel.copy(), sigma_n
        info["notes"].append("Relative DSM (0-1): no scale information, so no metres are claimed.")

    # ---- confidence
    if height_unit == "m":
        green = sigma_n < np.maximum(2.0, 0.2 * ndsm)
        amber = sigma_n < np.maximum(5.0, 0.5 * ndsm)
    else:
        q60, q90 = np.percentile(unc_rel, [60, 90])
        green, amber = unc_rel <= q60, unc_rel <= q90
    conf = np.where(green, 2, np.where(amber, 1, 0)).astype(np.uint8)
    conf[shadow] = np.minimum(conf[shadow], 1)
    conf[water] = 0

    info.update({
        "mode": mode, "height_unit": height_unit,
        "buildings_detected": int(n_lab - 1),
        "confidence_share": {k: round(float((conf == v).mean()), 3) for k, v in
                             (("green", 2), ("amber", 1), ("red", 0))},
    })
    return HeightModel(rel=rel, unc_rel=unc_rel, terrain_rel=terrain_rel, ndsm_rel=ndsm_rel,
                       ndsm=ndsm.astype(np.float32), dtm=dtm, dsm=dsm.astype(np.float32),
                       sigma=sigma.astype(np.float32), confidence=conf, building=building,
                       tree=tree, water=water, shadow=shadow, mode=mode,
                       height_unit=height_unit, info=info)
