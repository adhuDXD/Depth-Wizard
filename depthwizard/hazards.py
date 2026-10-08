"""Hazard scenarios and escape-route planning on a coarse analysis grid.

flood       water rises `level` above stream level (HAND model). Tall buildings
            whose roofs stay well above water become vertical-evacuation refuges.
landslide   susceptibility from slope, flow convergence and bare ground, plus a
            downslope run-out zone. `level` = caution (0..1).
earthquake  debris from a collapsing building can reach ~`level` x its height
            into the street. Open ground beyond that reach is an assembly area.

All heights used here come from the calibrated height model, so the routing
is only as metric as the DSM; units are carried through and stated.
"""
from __future__ import annotations

import string
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from .calibrate import HeightModel
from .hydrology import Hydrology, compute_hydrology, height_above_sources
from .routing import RouteField, arrows, evacuation_field, trace

FLOOR_HEIGHT_M = 3.0
M2_FLOOR_PER_PERSON = 15.0       # residential floor area per person (assumption, editable)
M2_PER_EVACUEE = 3.5             # Sphere minimum covered area per person
MIN_SAFE_AREA_M2 = 300.0
REFUGE_MIN_HEIGHT_M = 9.0        # ~3 storeys
MAX_GRID = 320

HAZARDS = {
    "flood": {"label": "Flood", "param": "Water level rise", "default": 0.0, "min": 0.0, "max": 15.0},
    "landslide": {"label": "Landslide", "param": "Caution level", "default": 0.5, "min": 0.0, "max": 1.0},
    "earthquake": {"label": "Earthquake", "param": "Debris reach (x building height)", "default": 0.5, "min": 0.2, "max": 1.2},
}


@dataclass
class Terrain:
    factor: int
    shape: tuple[int, int]
    cell_m: float
    metric_xy: bool
    z: np.ndarray
    z_unit: str
    ndsm: np.ndarray
    height_unit: str
    building: np.ndarray
    tree: np.ndarray
    water: np.ndarray
    veg: np.ndarray
    slope_deg: np.ndarray
    slope_is_metric: bool
    population: np.ndarray
    hydro: Hydrology
    flood_ref: np.ndarray        # height above the flood source (water body, else stream network)
    flood_seed: np.ndarray       # where the water starts
    water_bodies: list[dict]


def _down(a: np.ndarray, shape: tuple[int, int], interp=cv2.INTER_AREA) -> np.ndarray:
    return cv2.resize(a.astype(np.float32), (shape[1], shape[0]), interpolation=interp)


def build_terrain(hm: HeightModel, gsd: float | None, rgb: np.ndarray) -> Terrain:
    H, W = hm.dsm.shape
    factor = max(1, int(np.ceil(max(H, W) / MAX_GRID)))
    shape = (int(np.ceil(H / factor)), int(np.ceil(W / factor)))
    cell_m = (gsd or 1.0) * factor
    if hm.dtm is not None:
        z, z_unit = _down(hm.dtm, shape), "m"
    else:
        z, z_unit = _down(hm.terrain_rel, shape), "relative"
    building = _down(hm.building, shape) > 0.4
    ndsm = cv2.resize(ndimage.maximum_filter(hm.ndsm * hm.building, size=factor),
                      (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    ndsm = np.where(building, ndsm, _down(hm.ndsm, shape))
    tree = _down(hm.tree, shape) > 0.4
    water = _down(hm.water, shape) > 0.4

    f = rgb.astype(np.float32)
    exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / (f.sum(axis=2) + 1)
    veg = _down((exg > 0.06).astype(np.float32), shape)

    zs = ndimage.gaussian_filter(z, 1.0)
    gy, gx = np.gradient(zs, cell_m)
    grad = np.hypot(gx, gy)
    slope_is_metric = z_unit == "m" and gsd is not None
    if slope_is_metric:
        slope = np.degrees(np.arctan(grad))
    else:   # relative terrain: map the 99th percentile gradient to 45 degrees
        slope = 45.0 * np.clip(grad / (np.percentile(grad, 99) + 1e-9), 0, 2)

    cell_area = cell_m ** 2
    if hm.height_unit == "m" and gsd:
        floors = np.maximum(1, np.round(ndsm / FLOOR_HEIGHT_M))
    else:
        floors = np.ones(shape)
    population = np.where(building, cell_area * floors / M2_FLOOR_PER_PERSON, 0.0)

    hydro = compute_hydrology(z, cell_m)

    # Water bodies seen in the image are where a flood starts. Without any,
    # fall back to the drainage network derived from the terrain.
    min_water_cells = max(6, int(500 / cell_area)) if gsd else 12
    water = ndimage.binary_fill_holes(water)
    wlab, wids = _components(water, min_water_cells, max_n=50)
    water = np.isin(wlab, wids)
    water_bodies = []
    for i in wids:
        comp = wlab == i
        cy, cx = np.unravel_index(np.argmax(ndimage.distance_transform_edt(comp)), comp.shape)
        rr, cc = np.nonzero(comp)
        elong = max(np.ptp(rr), np.ptp(cc)) + 1
        water_bodies.append({
            "kind": "river" if elong ** 2 > 6 * comp.sum() else "lake / pond",
            "area_m2": round(float(comp.sum() * cell_area)) if gsd else None,
            "u": (cx + 0.5) / shape[1], "v": (cy + 0.5) / shape[0]})
    if water.any():
        flood_seed = water
        flood_ref = height_above_sources(hydro, water)
    else:
        flood_seed = hydro.drainage
        flood_ref = hydro.hand
    return Terrain(factor=factor, shape=shape, cell_m=cell_m, metric_xy=gsd is not None, z=z,
                   z_unit=z_unit, ndsm=ndsm, height_unit=hm.height_unit, building=building, tree=tree,
                   water=water, veg=veg, slope_deg=slope, slope_is_metric=slope_is_metric,
                   population=population, hydro=hydro, flood_ref=flood_ref, flood_seed=flood_seed,
                   water_bodies=water_bodies)


def _zone_name(i: int) -> str:
    letters = string.ascii_uppercase
    return letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]


def _components(mask: np.ndarray, min_cells: int, max_n: int = 40) -> tuple[np.ndarray, list[int]]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_cells]
    ids.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    return lab, ids[:max_n]


@dataclass
class ScenarioResult:
    field: RouteField
    overlay: np.ndarray          # RGBA uint8 on the analysis grid
    time_overlay: np.ndarray
    danger: np.ndarray
    payload: dict


def run_scenario(T: Terrain, kind: str, level: float | None = None) -> ScenarioResult:
    if kind not in HAZARDS:
        raise ValueError(f"Unknown hazard '{kind}'")
    spec = HAZARDS[kind]
    level = float(spec["default"] if level is None else level)
    h, w = T.shape
    cell_area = T.cell_m ** 2
    min_cells = max(4, int(MIN_SAFE_AREA_M2 / cell_area)) if T.metric_xy else 12
    warnings: list[str] = []
    mult = np.ones((h, w), np.float32)
    mult[T.tree] = 1.3
    mult[T.water] = 15.0
    passable = np.ones((h, w), bool)
    overlay = np.zeros((h, w, 4), np.uint8)
    target_offset = np.zeros((h, w), np.float32)
    zone_kind: dict[int, str] = {}
    zone_capacity: dict[int, float] = {}
    targets = np.zeros((h, w), np.int32)

    metric_heights = T.height_unit == "m" and T.metric_xy
    if kind == "flood":
        ref = T.flood_ref
        has_water = bool(T.water.any())
        if T.z_unit == "relative":
            span = float(np.ptp(ref)) or 1.0
            level_z = level / 15.0 * span          # slider maps to a share of local relief
            margin = 0.03 * span
            warnings.append("No DEM: water level is relative to local relief, not metres.")
        else:
            level_z, margin = level, 1.0
        if not has_water:
            warnings.append("No river or lake detected in the image: the flood starts from the lowest "
                            "drainage lines of the terrain.")
        # the flood grows outward from the water body, only through cells it can actually reach
        reach = (ref < level_z) | T.flood_seed if level_z > 0 else T.flood_seed & T.water
        flooded = ndimage.binary_propagation(T.flood_seed & reach, structure=np.ones((3, 3), bool), mask=reach)
        depth = np.where(flooded, np.clip(level_z - ref, 0, None), 0.0)
        depth[T.water] = level_z + (1.0 if T.z_unit == "m" else margin)   # existing water is deep
        danger = flooded | T.water
        depth_n = np.clip(depth / (1.0 if T.z_unit == "m" else max(margin, 1e-6)), 0, 3)
        mult += (1.0 + depth_n) * danger
        mult[T.building & ~danger] = 4.0
        safe = (ref >= level_z + margin) & ~T.building & ~T.water & (T.slope_deg < 25)
        lab, ids = _components(safe, min_cells)
        for k, i in enumerate(ids):
            targets[lab == i] = k + 1
            zone_kind[k + 1] = "High ground"
            zone_capacity[k + 1] = (lab == i).sum() * cell_area / M2_PER_EVACUEE
        next_id = len(ids) + 1
        if metric_heights:
            blab, bids = _components(T.building, 1, max_n=10000)
            for i in bids:
                comp = blab == i
                roof = float(T.ndsm[comp].max())
                wdepth = float(depth[comp].max())
                if wdepth > 0 and roof >= REFUGE_MIN_HEIGHT_M and roof - wdepth >= FLOOR_HEIGHT_M:
                    floors_dry = int((roof - wdepth) // FLOOR_HEIGHT_M)
                    targets[comp] = next_id
                    target_offset[comp] = 300.0     # prefer high ground if within ~5 extra minutes
                    mult[comp] = 1.5
                    zone_kind[next_id] = "Refuge building"
                    zone_capacity[next_id] = comp.sum() * cell_area * floors_dry / M2_PER_EVACUEE
                    next_id += 1
                    if next_id > 80:
                        break
        else:
            warnings.append("Building heights are not in metres, so vertical-evacuation refuges are off.")
        new_water = danger & ~T.water
        alpha = np.clip(90 + 160 * depth_n / 3, 0, 230).astype(np.uint8)
        overlay[new_water] = np.c_[np.tile([30, 110, 230], (int(new_water.sum()), 1)), alpha[new_water]]
        overlay[T.water] = [15, 55, 160, 210]
        hazard_name = f"{level:g} {'m' if T.z_unit == 'm' else '(relative)'} flood"

    elif kind == "landslide":
        if not T.slope_is_metric:
            warnings.append("Slopes are approximate: terrain is not in metres (no DEM).")
        acc = np.log1p(T.hydro.accumulation)
        acc_n = acc / (acc.max() or 1)
        S = (0.6 * np.clip((T.slope_deg - 15) / 25, 0, 1) + 0.25 * acc_n * (T.slope_deg > 10)
             + 0.15 * (1 - T.veg))
        thr = 0.7 - 0.4 * np.clip(level, 0, 1)
        source = S >= thr
        runout = np.zeros(h * w, bool)
        front = np.flatnonzero(source.ravel())
        steps = int(100.0 / T.cell_m) if T.metric_xy else 20
        rec = T.hydro.receivers
        for _ in range(steps):
            front = rec[front]
            front = front[front >= 0]
            if front.size == 0:
                break
            runout[front] = True
        runout = runout.reshape(h, w) & ~source
        danger = source | runout
        mult[danger] = 10.0
        mult[T.building & ~danger] = 4.0
        buffer_cells = max(1, int(30 / T.cell_m)) if T.metric_xy else 3
        far = ndimage.distance_transform_edt(~danger) > buffer_cells
        safe = far & (T.slope_deg < 12) & ~T.building & ~T.water
        lab, ids = _components(safe, min_cells)
        for k, i in enumerate(ids):
            targets[lab == i] = k + 1
            zone_kind[k + 1] = "Stable open ground"
            zone_capacity[k + 1] = (lab == i).sum() * cell_area / M2_PER_EVACUEE
        overlay[source] = [220, 40, 40, 170]
        overlay[runout] = [245, 140, 30, 150]
        hazard_name = f"landslide (caution {level:.1f})"

    else:  # earthquake
        heights = T.ndsm.copy()
        if not metric_heights:
            heights = np.where(T.building, 9.0, 0.0)
            warnings.append("Building heights not in metres: assuming 9 m (3 storeys) for every building.")
        dist, (ir, ic) = ndimage.distance_transform_edt(~T.building, return_indices=True)
        dist_m = dist * T.cell_m
        h_near = heights[ir, ic]
        reach = level * h_near
        danger = ~T.building & (dist_m <= reach)
        mult[danger] = 6.0
        mult[T.building] = 25.0
        safe = ~T.building & ~T.water & (dist_m >= reach + 5.0) & (dist_m >= 10.0)
        lab, ids = _components(safe, min_cells)
        for k, i in enumerate(ids):
            targets[lab == i] = k + 1
            zone_kind[k + 1] = "Open assembly area"
            zone_capacity[k + 1] = (lab == i).sum() * cell_area / M2_PER_EVACUEE
        overlay[danger] = [245, 140, 30, 160]
        overlay[T.building] = [120, 120, 120, 90]
        hazard_name = f"earthquake (debris reach {level:.1f} x height)"

    # safe zones / refuges on the overlay
    is_refuge = np.isin(targets, [k for k, v in zone_kind.items() if v == "Refuge building"])
    overlay[(targets > 0) & ~is_refuge] = [40, 190, 90, 150]
    overlay[is_refuge] = [20, 200, 210, 200]

    pop = T.population if T.metric_xy else T.building.astype(np.float64)
    field = evacuation_field(T.z if T.slope_is_metric else None, T.cell_m, mult, passable,
                             targets, target_offset, pop)

    # ---- per-zone statistics
    zones = []
    for zid, zkind in zone_kind.items():
        m = targets == zid
        if not m.any():
            continue
        cy, cx = np.unravel_index(np.argmax(ndimage.distance_transform_edt(m)), m.shape)  # most interior cell
        assigned = float(pop[(field.target_id == zid)].sum())
        cap = zone_capacity[zid]
        zones.append({"id": int(zid), "name": _zone_name(len(zones)), "kind": zkind,
                      "u": (cx + 0.5) / w, "v": (cy + 0.5) / h,
                      "area_m2": round(float(m.sum() * cell_area)) if T.metric_xy else None,
                      "capacity": round(cap) if T.metric_xy else None,
                      "assigned_people": round(assigned) if T.metric_xy else None,
                      "overloaded": bool(T.metric_xy and assigned > cap)})
    zid_to_name = {z["id"]: z["name"] for z in zones}

    # ---- bottlenecks (cells most evacuees pass through)
    flow = np.where(targets > 0, 0, field.flow)
    total_pop = float(pop.sum())
    bottlenecks = []
    if total_pop > 0:
        fl = flow.copy()
        radius = max(3, max(h, w) // 25)
        for _ in range(5):
            i = int(np.argmax(fl))
            r, c = divmod(i, w)
            if fl[r, c] < 0.08 * total_pop:
                break
            bottlenecks.append({"u": (c + 0.5) / w, "v": (r + 0.5) / h,
                                "people": round(float(fl[r, c])) if T.metric_xy else None,
                                "share": round(float(fl[r, c] / total_pop), 2),
                                "zone": zid_to_name.get(int(field.target_id[r, c]))})
            fl[max(0, r - radius):r + radius + 1, max(0, c - radius):c + radius + 1] = 0

    # ---- statistics + plain-language summary
    t = field.time_s
    occupied = pop > 0
    risk_area = (danger | T.building) if kind == "earthquake" else danger
    in_danger = occupied & risk_area
    reachable = np.isfinite(t)
    people_danger = float(pop[in_danger].sum())
    blab, bids = _components(T.building, 1, max_n=100000)
    bldg_hit = len(set(np.unique(blab[T.building & ndimage.binary_dilation(danger)])) - {0})
    t_danger = t[in_danger & reachable] / 60
    stats = {
        "buildings_total": len(bids),
        "buildings_at_risk": int(bldg_hit),
        "people_at_risk": round(people_danger) if T.metric_xy else None,
        "people_trapped": round(float(pop[occupied & ~reachable].sum())) if T.metric_xy else None,
        "median_minutes": round(float(np.median(t_danger)), 1) if t_danger.size else None,
        "p90_minutes": round(float(np.percentile(t_danger, 90)), 1) if t_danger.size else None,
        "danger_area_share": round(float(danger.mean()), 3),
        "safe_zones": len(zones),
    }

    summary = _summary(hazard_name, stats, zones, bottlenecks, T.metric_xy)
    if kind == "flood":
        if T.water_bodies:
            kinds = sorted({wb["kind"] for wb in T.water_bodies})
            src = f"{len(T.water_bodies)} water bod{'y' if len(T.water_bodies) == 1 else 'ies'} detected ({', '.join(kinds)})"
        else:
            src = "No water body detected; using the terrain's drainage lines"
        if level == 0:
            summary = [f"{src}. Water level 0: normal conditions, nothing flooded yet.",
                       "Raise the water level to see the flood spread out from the water."]
        else:
            summary.insert(0, f"{src}. The flood spreads out from there.")
    spacing = max(4, max(h, w) // 30)
    near_danger = ndimage.binary_dilation(risk_area, iterations=max(1, spacing // 2))
    arrow_list = arrows(field, spacing, skip=(targets > 0) | ~near_danger)

    tmin = np.where(reachable, t / 60, np.nan)
    time_overlay = _time_overlay(tmin)
    payload = {"hazard": kind, "level": level, "hazard_name": hazard_name, "stats": stats,
               "water_bodies": T.water_bodies if kind == "flood" else [],
               "summary": summary, "warnings": warnings, "zones": zones, "bottlenecks": bottlenecks,
               "arrows": arrow_list, "grid": [h, w], "units": {
                   "time": "minutes" if T.metric_xy else "approx. (pixel size unknown)",
                   "level": "m" if T.z_unit == "m" else "relative"}}
    return ScenarioResult(field=field, overlay=overlay, time_overlay=time_overlay, danger=danger, payload=payload)


def _time_overlay(tmin: np.ndarray) -> np.ndarray:
    h, w = tmin.shape
    out = np.zeros((h, w, 4), np.uint8)
    stops = [(0, (40, 190, 90)), (5, (40, 190, 90)), (10, (240, 200, 40)), (15, (245, 140, 30)), (30, (215, 40, 40))]
    xs = [s[0] for s in stops]
    ok = np.isfinite(tmin)
    for ch in range(3):
        out[..., ch][ok] = np.interp(tmin[ok], xs, [s[1][ch] for s in stops]).astype(np.uint8)
    out[..., 3][ok] = 150
    out[~ok] = [90, 0, 120, 200]
    return out


def _summary(hazard_name, stats, zones, bottlenecks, metric) -> list[str]:
    lines = []
    if metric:
        lines.append(f"Scenario: {hazard_name}. {stats['buildings_at_risk']} of {stats['buildings_total']} buildings "
                     f"are in the danger zone (about {stats['people_at_risk']:,} people, estimated).")
    else:
        lines.append(f"Scenario: {hazard_name}. {stats['buildings_at_risk']} of {stats['buildings_total']} buildings "
                     f"are in the danger zone.")
    if stats["median_minutes"] is not None and metric:
        lines.append(f"Half of them can walk to safety within {stats['median_minutes']} min; "
                     f"90% within {stats['p90_minutes']} min.")
    if zones:
        ranked = sorted(zones, key=lambda z: -(z["assigned_people"] or 0))[:3]
        parts = []
        for z in ranked:
            s = f"Zone {z['name']} ({z['kind'].lower()}"
            if metric and z["capacity"] is not None:
                s += f", ~{z['assigned_people']:,} people / room for {z['capacity']:,}"
            parts.append(s + ")")
        lines.append("Main safe places: " + "; ".join(parts) + ".")
        over = [z["name"] for z in zones if z["overloaded"]]
        if over:
            lines.append("Over capacity: zone " + ", ".join(over) + ". Spread people to other zones.")
    else:
        lines.append("No safe zone was found inside this image. Evacuate beyond the mapped area.")
    if bottlenecks:
        b = bottlenecks[0]
        lines.append(f"Main choke point carries {int(b['share'] * 100)}% of evacuees "
                     f"(marked ! on the map). Station marshals there.")
    if metric and stats["people_trapped"]:
        lines.append(f"About {stats['people_trapped']:,} people have no safe path. Plan boat/air rescue.")
    return lines


def route_from(T: Terrain, res: ScenarioResult, u: float, v: float) -> dict:
    h, w = T.shape
    r = int(np.clip(v * h, 0, h - 1))
    c = int(np.clip(u * w, 0, w - 1))
    t = res.field.time_s[r, c]
    if not np.isfinite(t):
        return {"reachable": False, "path": [], "message": "No safe route from here. Stay high and call for rescue."}
    path = trace(res.field, r, c)
    dist = 0.0
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dist += np.hypot(r1 - r0, c1 - c0) * T.cell_m
    end = path[-1]
    zid = int(res.field.target_id[end])
    zone = next((z for z in res.payload["zones"] if z["id"] == zid), None)
    return {"reachable": True,
            "path": [[(cc + 0.5) / w, (rr + 0.5) / h] for rr, cc in path],
            "minutes": round(float(t) / 60, 1),
            "distance_m": round(dist) if T.metric_xy else None,
            "zone": zone["name"] if zone else None,
            "zone_kind": zone["kind"] if zone else None}
