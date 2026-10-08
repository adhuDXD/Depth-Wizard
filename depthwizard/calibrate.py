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

from . import config
from .dem import Dem

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
    off_nadir: float | None = None                   # degrees from vertical
    view_azimuth: float | None = None                # degrees, from the ground towards the satellite
    footprints: tuple | None = None                  # (label image, tagged heights, source) from osm.py
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


def _local_std(gray: np.ndarray, size: int = 5) -> np.ndarray:
    return np.sqrt(np.maximum(ndimage.uniform_filter(gray ** 2, size) - ndimage.uniform_filter(gray, size) ** 2, 0))


def detect_vegetation(rgb: np.ndarray) -> np.ndarray:
    """Green AND textured: tree canopy is leafy, a green-painted roof is smooth."""
    f = rgb.astype(np.float32)
    exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / (f.sum(axis=2) + 1)
    return (exg > 0.06) & (_local_std(f.mean(axis=2)) > 5)


def roof_segments(rgb: np.ndarray, ndsm_rel: np.ndarray, thr: float, shadow: np.ndarray,
                  gsd: float | None) -> np.ndarray:
    """Whole roofs: uniform-colour patches bounded by edges whose median relative
    height is raised. Snapping to these patches recovers roof parts the depth
    model under-estimates, and gives buildings clean, straight outlines."""
    lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    grad = sum(np.hypot(cv2.Sobel(lab_img[..., c], cv2.CV_32F, 1, 0), cv2.Sobel(lab_img[..., c], cv2.CV_32F, 0, 1))
               for c in range(3))
    uniform = (grad < 60) & (_local_std(rgb.astype(np.float32).mean(axis=2)) < 6) & ~shadow
    uniform = cv2.morphologyEx(uniform.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(uniform, connectivity=4)
    if n <= 1:
        return np.zeros(rgb.shape[:2], bool)
    area = stats[:, cv2.CC_STAT_AREA]
    px_m2 = gsd ** 2 if gsd else 0.25
    med = ndimage.median(ndsm_rel, lab, np.arange(n))
    ok = (area * px_m2 >= 20) & (area * px_m2 <= 2000) & (med > 0.5 * thr)
    ok[0] = False
    return cv2.dilate(ok[lab].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def detect_water(rgb: np.ndarray, gsd: float | None, exclude: np.ndarray | None = None) -> np.ndarray:
    """Rivers, lakes and ponds: blue-hued, saturated, flat (not raised like a blue roof),
    and large. `exclude` masks raised objects from the height model."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    hue = hsv[..., 0] * 360 / 256
    sat = hsv[..., 1] / 255
    val = hsv[..., 2] / 255
    m = (hue > 185) & (hue < 250) & (sat > 0.25) & (val > 0.08) & (val < 0.8)
    k = max(3, int(round(2.0 / gsd)) | 1) if gsd else 3          # ~2 m
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    min_px = max(int(2000 / gsd ** 2) if gsd else 400, 50)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    raised_frac = (ndimage.mean(exclude, lab, np.arange(n)) if exclude is not None and n > 1
                   else np.zeros(n))
    for i in range(1, n):
        # judged per region: a blue roof is almost entirely raised, a river mostly flat
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_px and raised_frac[i] < 0.6
    return keep[lab]


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
                   dtm: np.ndarray | None = None, off_nadir: float | None = None,
                   view_az: float | None = None) -> list[dict]:
    """Building height from the length of the shadow it casts.

    On sloping ground (hill towns) a shadow falling uphill is shorter and one
    falling downhill longer. With a terrain model the height is
    h = L * (tan(sun elevation) + s), where s is the terrain slope along the
    shadow direction (rise per metre, uphill positive).

    Shadows point away from the sun: in image coordinates (row down = south,
    col right = east) that direction is (cos A, -sin A) for sun azimuth A.

    Off-nadir views: the roof is drawn displaced from its footprint by h*tan(off_nadir), away from
    the satellite (view_az = direction from the ground to the satellite). The part of that shift
    along the shadow hides (or, if opposite, lengthens) the visible shadow:
        visible L = h * (1 / (tan(el) + s) - tan(off_nadir) * cos(view_az - sun_az))
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
    lean_along = 0.0
    if off_nadir and view_az is not None:
        lean_along = float(np.tan(np.radians(off_nadir)) * np.cos(np.radians(view_az - sun_az)))
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
        per_m = 1.0 / k                                  # ground shadow length per metre of height
        if off_nadir:
            per_m -= lean_along
            if per_m < 0.15 / k:                         # roof hides almost all of its shadow
                continue
        height_m = float(L_px * gsd / per_m)
        spread_m = float(max(q75 - q25, 1.0) * gsd / per_m)
        comp = building_lab == lab
        r_val = float(np.percentile(ndsm_rel[comp], 75))
        if height_m < 2.0 or r_val <= 0.005:
            continue
        cy, cx = ndimage.center_of_mass(comp)
        anchors.append({"source": "shadow", "row": float(cy), "col": float(cx),
                        "r": r_val, "h": height_m, "spread_m": spread_m, "label": int(lab),
                        "weight": float(min(sel.sum(), 20))})
    return _clean_shadow_anchors(anchors)


def _clean_shadow_anchors(anchors: list[dict]) -> list[dict]:
    """Drop measurements that disagree with themselves (shadow length varies wildly along the
    edge: usually a shaded road or slope read as a shadow) or with the rest of the town."""
    if not anchors:
        return anchors
    hmax = config.get("calibration.max_building_height_m")
    ok = [a for a in anchors if a["h"] <= hmax and a["spread_m"] <= max(6.0, 0.5 * a["h"])]
    if len(ok) >= 8:
        hs = np.array([a["h"] for a in ok])
        med = float(np.median(hs))
        mad = 1.4826 * float(np.median(np.abs(hs - med)))
        ok = [a for a in ok if a["h"] <= med + 5 * max(mad, 1.0)]
    return ok


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


DIRECT_BLUR_MAX_SIGMA_PX = 4.0


def lowpass(d: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur; for wide sigma blur at reduced resolution (same result, much faster).
    Ported from the original DepthWizard calibrate/fit.py."""
    if sigma_px <= DIRECT_BLUR_MAX_SIGMA_PX:
        return cv2.GaussianBlur(d.astype(np.float32), (0, 0), sigma_px, borderType=cv2.BORDER_REFLECT)
    h, w = d.shape
    f = int(sigma_px // DIRECT_BLUR_MAX_SIGMA_PX)
    small = cv2.resize(d.astype(np.float32), (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    sigma_small = np.sqrt(max(sigma_px ** 2 - f ** 2 / 12.0, 0.0)) / f
    small = cv2.GaussianBlur(small, (0, 0), max(sigma_small, 0.1), borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def block_mean(x: np.ndarray, block: int) -> np.ndarray:
    h, w = x.shape
    hh, ww = h // block * block, w // block * block
    return x[:hh, :ww].reshape(hh // block, block, ww // block, block).mean(axis=(1, 3))


def band_scale(d: np.ndarray, dem: np.ndarray, dem_res_m: float, pixel_m: float) -> tuple[float, float, int]:
    """Metres per model unit, learnt from the band the DEM itself resolves (its cell size up to a
    few cells). Both grids are area-averaged to DEM cells and high-passed alike; the DEM's detail is
    regressed on the model's. Returns (scale, correlation, cells). Ported from the original
    DepthWizard; a low correlation means the model disagrees with the terrain and is not trusted."""
    block = max(1, round(dem_res_m / pixel_m))
    mc, dc = block_mean(d, block), block_mean(dem, block)
    ok = np.isfinite(mc) & np.isfinite(dc)
    cells = int(ok.sum())
    if cells < config.get("calibration.band_min_cells"):
        return 0.0, 0.0, cells
    coarse = config.get("calibration.band_coarse_cells")

    def detail(grid):
        filled = np.where(ok, grid, grid[ok].mean()).astype(np.float32)
        return (filled - cv2.GaussianBlur(filled, (0, 0), coarse, borderType=cv2.BORDER_REFLECT))[ok].astype(np.float64)
    x, y = detail(mc), detail(dc)
    if x.std() <= 0 or y.std() <= 0:
        return 0.0, 0.0, cells
    return float(np.dot(x, y) / np.dot(x, x)), float(np.corrcoef(x, y)[0, 1]), cells


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
    # mapped heights (OSM) win over shadow measurements of the same building
    for a in sorted(anchors, key=lambda a: a["source"] == "osm"):
        lab = a.get("label")
        if a["source"] not in ("shadow", "osm") or lab is None:
            continue
        comp = building_lab == lab
        ndsm[comp] = a["h"]
        sigma_n[comp] = np.hypot(a.get("spread_m", 1.0), 1.0)
        spreads.append(a.get("spread_m", 1.0))
        measured += 1
    total = int(building_lab.max())
    info.update({
        "buildings_measured_by_shadow": sum(a["source"] == "shadow" for a in anchors),
        "buildings_measured": measured,
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

def calibrate(rgb: np.ndarray, rel: np.ndarray, unc_rel: np.ndarray, dem: "Dem | np.ndarray | None",
              p: CalibParams) -> HeightModel:
    h, w = rel.shape
    gsd = p.gsd
    info: dict = {"notes": []}

    ground_rel = ground_surface(rel, gsd, p.ground_window_m)
    ndsm_rel = np.clip(rel - ground_rel, 0, None)
    terrain_rel = _detrend(ground_rel)

    thr = max(0.02, 0.2 * float(np.percentile(ndsm_rel, 99)))
    raised = cv2.dilate(((ndsm_rel > thr) * 255).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    water = detect_water(rgb, gsd, exclude=raised)
    shadow = detect_shadows(rgb, water)
    veg = detect_vegetation(rgb)

    elevated = (ndsm_rel > thr) & ~water
    elevated = cv2.morphologyEx(elevated.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    min_bldg_px = int(20 / gsd ** 2) if gsd else 15
    roofs = roof_segments(rgb, ndsm_rel, thr, shadow, gsd) & ~water
    tree = elevated & veg & ~roofs
    building = _remove_small((elevated & ~veg) | roofs, max(min_bldg_px, 6))
    osm_anchors: list[dict] = []
    if p.footprints is not None:
        # mapped footprints: real outlines, one building each; detections outside them are kept
        f_lab, f_heights, f_src = p.footprints
        inside = f_lab > 0
        _, extra = cv2.connectedComponents((building & ~inside).astype(np.uint8), connectivity=8)
        building_lab = np.where(inside, f_lab, np.where(extra > 0, extra + f_lab.max(), 0)).astype(np.int32)
        building = building_lab > 0
        tree &= ~inside
        n_lab = int(building_lab.max()) + 1
        for i, hgt in enumerate(f_heights, start=1):
            comp = f_lab == i
            if hgt and comp.any():
                cy, cx = ndimage.center_of_mass(comp)
                osm_anchors.append({"source": "osm", "row": float(cy), "col": float(cx), "label": i,
                                    "r": float(np.percentile(ndsm_rel[comp], 75)), "h": float(hgt),
                                    "spread_m": 1.5, "weight": 10.0})
        info["osm_buildings"] = int(f_lab.max())
        info["osm_tagged_heights"] = len(osm_anchors)
        info["notes"].append(f"{int(f_lab.max())} building footprints from {f_src}"
                             + (f", {len(osm_anchors)} with tagged heights." if osm_anchors else "."))
    else:
        n_lab, building_lab = cv2.connectedComponents(building.astype(np.uint8), connectivity=8)

    # ---- terrain (DEM already on the image grid in sea-level heights, see dem.get_dem)
    if isinstance(dem, np.ndarray):
        dem = Dem(heights=dem + (p.geoid_offset_m or 0.0), source="uploaded", res_m=30.0,
                  datum="unknown", kind="surface")
    D = None
    gcp_ground, gcp_elev = [], []
    for g in p.gcps:
        r = int(np.clip(round(g["v"] * (h - 1)), 0, h - 1))
        c = int(np.clip(round(g["u"] * (w - 1)), 0, w - 1))
        (gcp_ground if ndsm_rel[r, c] <= thr else gcp_elev).append((r, c, float(g["elevation"])))

    sigma_dem_px = None
    if dem is not None:
        D = dem.heights.astype(np.float32).copy()
        info["notes"] += dem.notes
        info.update(terrain_source="DEM", dem_source=dem.source, dem_datum=dem.datum,
                    dem_res_m=round(dem.res_m, 1), dem_kind=dem.kind)
        if gcp_ground:
            bias = float(np.median([e - D[r, c] for r, c, e in gcp_ground]))
            D += bias
            info["gcp_dem_bias_m"] = bias
            info["notes"].append(f"DEM shifted by {bias:+.1f} m to match {len(gcp_ground)} ground GCPs.")
        if gsd:
            sigma_dem_px = config.get("calibration.band_sigma_k") * dem.res_m / gsd
    elif gcp_ground and gsd:
        pts = np.array(gcp_ground, float)
        if len(pts) >= 3:
            A = np.c_[pts[:, 1], pts[:, 0], np.ones(len(pts))]
            coef, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
            Y, X = np.mgrid[0:h, 0:w]
            D = (coef[0] * X + coef[1] * Y + coef[2]).astype(np.float32)
        else:
            D = np.full((h, w), pts[:, 2].mean(), np.float32)
        info["terrain_source"] = f"plane through {len(pts)} ground GCPs"
    surface_dem = dem is not None and dem.kind == "surface" and sigma_dem_px is not None

    # ---- metric anchors
    anchors: list[dict] = []
    if gsd and p.sun_elevation is not None and p.sun_azimuth is not None:
        anchors += shadow_anchors(building_lab, n_lab, shadow, ndsm_rel, gsd, p.sun_elevation, p.sun_azimuth,
                                  dtm=D, off_nadir=p.off_nadir, view_az=p.view_azimuth)
        if p.off_nadir and p.view_azimuth is not None:
            info["notes"].append(f"Shadow heights corrected for the {p.off_nadir:.0f} degree off-nadir view "
                                 "(leaning buildings hide or lengthen their shadows).")
        info["shadow_anchors"] = len(anchors)
        if D is not None:
            info["notes"].append("Shadow heights corrected for terrain slope (from the DEM).")
    elif not gsd:
        info["notes"].append("Pixel size unknown: shadow calibration disabled.")
    else:
        info["notes"].append("Sun angles unknown: shadow calibration disabled.")

    anchors += osm_anchors
    if D is not None and gcp_elev:
        # on a surface DEM the GCP stands above the DEM's blurred surface: compare with the
        # model's detail in the same band (original Method B, GCP scale source)
        feat = ndsm_rel - lowpass(ndsm_rel, sigma_dem_px) if surface_dem else ndsm_rel
        for r, c, e in gcp_elev:
            anchors.append({"source": "gcp", "row": r, "col": c, "r": float(feat[r, c]),
                            "h": e - float(D[r, c]), "weight": 20.0})

    # ---- height scale: GCPs + shadows, then the DEM band, then the user's estimate
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
    if scale_map is None and dem is not None and gsd:
        s_band, r_band, cells = band_scale(rel, D, dem.res_m, gsd)
        info.update(band_r=round(r_band, 3), band_cells=cells)
        if r_band >= config.get("calibration.band_min_r") and s_band > 0:
            scale_map = np.full((h, w), s_band, np.float32)
            info["scale_source"] = "dem_band"
            info["notes"].append(f"No shadows or GCPs: building scale learnt from the detail the DEM itself "
                                 f"resolves (agreement r = {r_band:.2f}; medium confidence).")
    if scale_map is None and p.building_height_prior_m:
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

    dtm = None
    if D is not None and height_unit == "m":
        if surface_dem:
            # the DEM already holds the buildings blurred over ~one cell: remove that blur from the
            # ground and add the sharp model detail, so nothing is counted twice
            dtm = D - lowpass(ndsm.astype(np.float32), sigma_dem_px)
            info["notes"].append("Surface DEM: blurred buildings removed from the ground before adding sharp "
                                 "building heights (no double counting).")
        else:
            dtm = D
        mode, dsm = "absolute", dtm + ndsm
        sigma = np.sqrt(sigma_n ** 2 + DEM_SIGMA_M ** 2)
    elif D is not None:
        dtm = D
        mode, dsm, sigma = "terrain_only", D.copy(), np.full((h, w), DEM_SIGMA_M, np.float32)
        info["notes"].append("Terrain is in metres but building heights could not be scaled, so the DSM is the "
                             "DEM. Add sun angles + pixel size, GCPs, or a typical building height.")
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
