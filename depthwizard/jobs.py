"""Processing jobs: image -> height model -> terrain, plus layers and exports."""
from __future__ import annotations

import base64
import html
import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio import features

from . import render
from .calibrate import CalibParams, HeightModel, calibrate
from .depth import estimate_relative_height, load_default_model
from .hazards import ScenarioResult, Terrain, build_terrain, route_from, run_scenario
from .scene import Scene, load_dem, load_image
from .sun import sun_position
from .validate import compare

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Job:
    id: str
    name: str
    params: dict
    status: str = "queued"
    progress: float = 0.0
    message: str = "Waiting"
    error: str | None = None
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    scene: Scene | None = None
    hm: HeightModel | None = None
    terrain: Terrain | None = None
    scenario: ScenarioResult | None = None
    validation: dict | None = None
    truth: np.ndarray | None = None
    sun_source: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class JobManager:
    def __init__(self, model=None):
        self.model = model or load_default_model()
        self.jobs: dict[str, Job] = {}
        self.pool = ThreadPoolExecutor(max_workers=1)

    def submit(self, name: str, image: str, dem: str | None, truth: str | None, params: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], name=name, params=params)
        self.jobs[job.id] = job
        self.pool.submit(self._run, job, image, dem, truth)
        return job

    def run_sync(self, name: str, image: str, dem: str | None = None, truth: str | None = None,
                 params: dict | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], name=name, params=params or {})
        self.jobs[job.id] = job
        self._run(job, image, dem, truth)
        return job

    # ------------------------------------------------------------------ pipeline
    def _run(self, job: Job, image: str, dem_path: str | None, truth_path: str | None) -> None:
        def report(f: float, msg: str):
            job.progress, job.message = round(f, 3), msg

        try:
            job.status = "running"
            p = job.params
            report(0.02, "Reading image")
            scene = load_image(image, gsd_hint=p.get("gsd"))
            notes = []

            el, az = p.get("sun_elevation"), p.get("sun_azimuth")
            if el is not None and az is not None:
                job.sun_source = "entered by user"
            elif scene.sun_elevation is not None and scene.sun_azimuth is not None:
                el, az = scene.sun_elevation, scene.sun_azimuth
                job.sun_source = "image metadata"
            elif p.get("acquired") and scene.georeferenced:
                when = datetime.fromisoformat(p["acquired"].replace("Z", "+00:00"))
                lon, lat = scene.center_lonlat()
                el, az = sun_position(when, lat, lon)
                job.sun_source = "computed from acquisition time"
            scene.sun_elevation, scene.sun_azimuth = el, az

            if dem_path:
                report(0.05, "Aligning DEM")
                try:
                    scene.dem = load_dem(dem_path, scene)
                except ValueError as e:
                    notes.append(f"DEM ignored: {e}")
            job.scene = scene

            def depth_progress(f, m):
                report(0.08 + 0.72 * f, m)
            rel, unc = estimate_relative_height(scene.rgb, self.model, tta=p.get("tta", True),
                                                progress=depth_progress)

            report(0.82, "Calibrating heights")
            cp = CalibParams(gsd=scene.gsd, sun_elevation=el, sun_azimuth=az, gcps=p.get("gcps") or [],
                             geoid_offset_m=float(p.get("geoid_offset_m") or 0.0),
                             building_height_prior_m=p.get("building_height_prior_m"))
            hm = calibrate(scene.rgb, rel, unc, scene.dem, cp)
            hm.info["notes"] = notes + hm.info["notes"]
            job.hm = hm

            report(0.9, "Analysing terrain and drainage")
            job.terrain = build_terrain(hm, scene.gsd, scene.rgb)

            if truth_path:
                report(0.96, "Validating against reference")
                self.attach_reference(job, truth_path)
            report(1.0, "Done")
            job.status = "done"
        except Exception as e:  # noqa: BLE001 - surfaced to the user
            traceback.print_exc()
            job.status, job.error, job.message = "error", str(e), "Failed"

    def attach_reference(self, job: Job, ref_path: str) -> dict:
        ref = load_dem(ref_path, job.scene)
        job.truth = ref
        job.validation = compare(job.hm.dsm, job.hm.dtm, ref, job.hm.building)
        return job.validation

    # ------------------------------------------------------------------ queries
    def scenario(self, job: Job, hazard: str, level: float | None) -> dict:
        with job.lock:
            job.scenario = run_scenario(job.terrain, hazard, level)
            return job.scenario.payload

    def route(self, job: Job, u: float, v: float) -> dict:
        if job.scenario is None:
            raise ValueError("Run a hazard scenario first.")
        return route_from(job.terrain, job.scenario, u, v)


# ---------------------------------------------------------------------- helpers

def meta(job: Job, model) -> dict:
    out = {"id": job.id, "name": job.name, "status": job.status, "progress": job.progress,
           "message": job.message, "error": job.error, "created": job.created,
           "model": {"name": model.name, "is_ai": model.is_ai}}
    if job.status != "done":
        return out
    s, hm = job.scene, job.hm
    h, w = s.shape
    finite = hm.dsm[np.isfinite(hm.dsm)]
    out.update({
        "width": w, "height": h, "gsd": s.gsd, "georeferenced": s.georeferenced, "crs": s.crs,
        "center_lonlat": s.center_lonlat(),
        "extent_m": [w * s.gsd, h * s.gsd] if s.gsd else None,
        "sun": {"elevation": s.sun_elevation, "azimuth": s.sun_azimuth, "source": job.sun_source},
        "mode": hm.mode, "height_unit": hm.height_unit, "has_dem": s.dem is not None,
        "dsm_range": [float(finite.min()), float(finite.max())],
        "building_height_p95": float(np.percentile(hm.ndsm[hm.building], 95)) if hm.building.any() else None,
        "calibration": hm.info, "validation": job.validation,
        "grid": list(job.terrain.shape) if job.terrain else None,
    })
    return out


def heightmap(job: Job, max_dim: int = 512) -> tuple[bytes, int, int]:
    dsm = job.hm.dsm
    h, w = dsm.shape
    s = max(h, w) / max_dim
    if s > 1:
        dsm = cv2.resize(dsm, (round(w / s), round(h / s)), interpolation=cv2.INTER_AREA)
    return dsm.astype("<f4").tobytes(), dsm.shape[1], dsm.shape[0]


def layer_png(job: Job, name: str) -> bytes:
    s, hm = job.scene, job.hm
    cell = s.gsd or 1.0
    if name == "image":
        img = s.rgb
    elif name == "height":
        ex = 1.0 if hm.height_unit == "m" else 200.0 / cell
        img = render.shaded(hm.dsm, render.TERRAIN, cell, exaggerate=ex)
    elif name == "buildings":
        vmax = float(np.percentile(hm.ndsm[hm.building], 99)) if hm.building.any() else None
        col = render.colormap(hm.ndsm, render.HEIGHTS, vmin=0, vmax=vmax)
        gray = cv2.cvtColor(cv2.cvtColor(s.rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB) // 2
        m = (hm.building | hm.tree)[..., None]
        img = np.where(m, col, gray)
    elif name == "confidence":
        img = render.CONFIDENCE[hm.confidence]
    elif name == "drainage":
        t = job.terrain
        img = render.colormap(t.hydro.hand, render.TERRAIN)
        img[t.hydro.drainage] = [20, 60, 200]
    elif name == "error":
        if job.truth is None:
            raise KeyError(name)
        img = render.colormap(hm.dsm - job.truth, render.ERROR, -6, 6)
    elif name in ("hazard", "time"):
        if job.scenario is None:
            raise KeyError(name)
        img = job.scenario.overlay if name == "hazard" else job.scenario.time_overlay
    else:
        raise KeyError(name)
    return render.png_bytes(img)


def point_info(job: Job, u: float, v: float) -> dict:
    hm, T = job.hm, job.terrain
    h, w = hm.dsm.shape
    r, c = int(np.clip(v * h, 0, h - 1)), int(np.clip(u * w, 0, w - 1))
    tr, tc = int(np.clip(v * T.shape[0], 0, T.shape[0] - 1)), int(np.clip(u * T.shape[1], 0, T.shape[1] - 1))
    out = {"surface": float(hm.dsm[r, c]), "height_above_ground": float(hm.ndsm[r, c]),
           "ground": float(hm.dtm[r, c]) if hm.dtm is not None else None,
           "uncertainty": float(hm.sigma[r, c]), "confidence": ["low", "medium", "high"][int(hm.confidence[r, c])],
           "is_building": bool(hm.building[r, c]), "is_water": bool(hm.water[r, c]),
           "slope_deg": round(float(T.slope_deg[tr, tc]), 1),
           "height_above_drainage": float(T.hydro.hand[tr, tc]), "unit": hm.height_unit}
    if job.scene.georeferenced:
        x, y = job.scene.transform @ (c + 0.5, r + 0.5)
        lon, lat = Transformer.from_crs(job.scene.crs, "EPSG:4326", always_xy=True).transform(x, y)
        out["lonlat"] = [round(lon, 6), round(lat, 6)]
    if job.scenario is not None:
        t = job.scenario.field.time_s[tr, tc]
        out["minutes_to_safety"] = round(float(t) / 60, 1) if np.isfinite(t) else None
        out["in_danger"] = bool(job.scenario.danger[tr, tc])
    return out


def profile(job: Job, u0: float, v0: float, u1: float, v1: float, n: int = 200) -> dict:
    hm = job.hm
    h, w = hm.dsm.shape
    us, vs = np.linspace(u0, u1, n), np.linspace(v0, v1, n)
    rows = np.clip((vs * h).astype(int), 0, h - 1)
    cols = np.clip((us * w).astype(int), 0, w - 1)
    cell = job.scene.gsd or 1.0
    length = float(np.hypot((u1 - u0) * w, (v1 - v0) * h) * cell)
    return {"distance": np.linspace(0, length, n).round(2).tolist(),
            "surface": hm.dsm[rows, cols].round(2).tolist(),
            "ground": hm.dtm[rows, cols].round(2).tolist() if hm.dtm is not None else None,
            "distance_unit": "m" if job.scene.gsd else "px", "unit": hm.height_unit}


# ---------------------------------------------------------------------- exports

def dsm_geotiff(job: Job, which: str = "dsm") -> bytes:
    from rasterio.io import MemoryFile
    s, hm = job.scene, job.hm
    arr = {"dsm": hm.dsm, "ndsm": hm.ndsm, "uncertainty": hm.sigma}[which]
    h, w = arr.shape
    profile_ = {"driver": "GTiff", "width": w, "height": h, "count": 1, "dtype": "float32", "compress": "deflate"}
    if s.georeferenced:
        profile_.update(crs=s.crs, transform=s.transform)
    with MemoryFile() as mem:
        with mem.open(**profile_) as dst:
            dst.write(arr.astype(np.float32), 1)
            dst.update_tags(DEPTHWIZARD_MODE=hm.mode, UNIT=hm.height_unit, LAYER=which)
        return mem.read()


def _to_lonlat(job: Job):
    s = job.scene
    h, w = s.shape
    if not s.georeferenced:
        return lambda u, v: [round(u, 5), round(v, 5)], False
    tf = Transformer.from_crs(s.crs, "EPSG:4326", always_xy=True)

    def f(u, v):
        x, y = s.transform @ (u * w, v * h)
        lon, lat = tf.transform(x, y)
        return [round(lon, 7), round(lat, 7)]
    return f, True


def geojson(job: Job) -> dict:
    if job.scenario is None:
        raise ValueError("Run a hazard scenario first.")
    conv, geo = _to_lonlat(job)
    p = job.scenario.payload
    feats = []
    T = job.terrain
    gh, gw = T.shape
    for geom, val in features.shapes(job.scenario.danger.astype(np.uint8), mask=job.scenario.danger):
        rings = [[conv(x / gw, y / gh) for x, y in ring] for ring in geom["coordinates"]]
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": rings},
                      "properties": {"kind": "danger_zone", "hazard": p["hazard_name"]}})
    for z in p["zones"]:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": conv(z["u"], z["v"])},
                      "properties": {k: z[k] for k in z if k not in ("u", "v", "kind")}
                      | {"kind": "safe_zone", "zone_type": z["kind"]}})
    for b in p["bottlenecks"]:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": conv(b["u"], b["v"])},
                      "properties": {"kind": "choke_point", "people": b["people"], "share": b["share"]}})
    for a in p["arrows"]:
        feats.append({"type": "Feature",
                      "geometry": {"type": "LineString", "coordinates": [conv(a[0], a[1]), conv(a[2], a[3])]},
                      "properties": {"kind": "escape_direction", "minutes_to_safety": a[4]}})
    out = {"type": "FeatureCollection", "features": feats,
           "properties": {"hazard": p["hazard_name"], "summary": p["summary"],
                          "generated_by": "DepthWizard", "mode": job.hm.mode}}
    if not geo:
        out["properties"]["coordinates"] = "normalised image coordinates (image not georeferenced)"
    return out


def kml(job: Job) -> str:
    conv, geo = _to_lonlat(job)
    if not geo:
        raise ValueError("KML needs a georeferenced (GeoTIFF) image.")
    p = job.scenario.payload
    def c(u, v):
        lon, lat = conv(u, v)
        return f"{lon},{lat},0"
    parts = []
    for z in p["zones"]:
        desc = html.escape(f"{z['kind']}; capacity {z['capacity']}; assigned {z['assigned_people']}")
        parts.append(f"<Placemark><name>Safe zone {z['name']}</name><description>{desc}</description>"
                     f"<styleUrl>#safe</styleUrl><Point><coordinates>{c(z['u'], z['v'])}</coordinates></Point></Placemark>")
    for i, b in enumerate(p["bottlenecks"]):
        parts.append(f"<Placemark><name>Choke point {i + 1}</name><styleUrl>#choke</styleUrl>"
                     f"<Point><coordinates>{c(b['u'], b['v'])}</coordinates></Point></Placemark>")
    for a in p["arrows"]:
        parts.append(f"<Placemark><styleUrl>#arrow</styleUrl><LineString><coordinates>{c(a[0], a[1])} {c(a[2], a[3])}"
                     f"</coordinates></LineString></Placemark>")
    return ("<?xml version='1.0' encoding='UTF-8'?><kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
            f"<name>DepthWizard: {html.escape(p['hazard_name'])}</name>"
            "<Style id='safe'><IconStyle><color>ff50c828</color></IconStyle></Style>"
            "<Style id='choke'><IconStyle><color>ff2828e6</color></IconStyle></Style>"
            "<Style id='arrow'><LineStyle><color>ff202020</color><width>2</width></LineStyle></Style>"
            + "".join(parts) + "</Document></kml>")


def report_html(job: Job, route: dict | None = None) -> str:
    if job.scenario is None:
        raise ValueError("Run a hazard scenario first.")
    p = job.scenario.payload
    img = render.composite_map(job.scene.rgb, job.scenario.overlay, p, route)
    b64 = base64.b64encode(render.png_bytes(img)).decode()
    info = job.hm.info
    esc = html.escape
    rows = "".join(
        f"<tr><td>{esc(z['name'])}</td><td>{esc(z['kind'])}</td><td>{z['area_m2'] or '-'}</td>"
        f"<td>{z['capacity'] or '-'}</td><td>{z['assigned_people'] or '-'}</td>"
        f"<td>{'OVER CAPACITY' if z['overloaded'] else 'ok'}</td></tr>" for z in p["zones"])
    acc = info.get("loo_rmse_m")
    acc_txt = (f"Building heights checked against {info.get('n_inliers')} independent anchors: "
               f"typical error +/- {acc:.1f} m." if acc else "No independent accuracy check was possible for this image.")
    when = datetime.now().strftime("%d %b %Y %H:%M")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Evacuation plan: {esc(job.name)}</title>
<style>body{{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#111}}h1{{margin:0 0 4px;font-size:22px}}
.muted{{color:#555}}img{{width:100%;border:1px solid #ccc}}table{{border-collapse:collapse;width:100%;margin-top:8px}}
td,th{{border:1px solid #ccc;padding:4px 6px;text-align:left}}li{{margin:4px 0}}.warn{{background:#fff4d6;padding:8px;border-radius:6px}}
@media print{{body{{margin:8mm}}}}</style></head><body>
<h1>Evacuation plan: {esc(p['hazard_name'])}</h1>
<div class="muted">{esc(job.name)} · generated {when} by DepthWizard · surface model: {esc(job.hm.mode)}</div>
<h2>What to do</h2><ul>{''.join(f'<li>{esc(s)}</li>' for s in p['summary'])}</ul>
<img src="data:image/png;base64,{b64}" alt="Evacuation map">
<p class="muted">Arrows point the fastest walking direction to safety. Green = safe zone (letters), cyan = refuge building,
blue/red/orange = danger zone, red ! = choke point{', pink line = selected route' if route else ''}.</p>
<h2>Safe zones</h2><table><tr><th>Zone</th><th>Type</th><th>Area m²</th><th>Capacity</th><th>Assigned</th><th>Status</th></tr>{rows}</table>
<h2>How reliable is this?</h2><p>{esc(acc_txt)}</p>
<ul>{''.join(f'<li>{esc(n)}</li>' for n in info.get('notes', []) + p.get('warnings', []))}</ul>
<p class="warn">Pre-disaster planning aid made from a single satellite image. Not a live navigation system.
Check routes on the ground before relying on them. Population figures are estimates from building volume.</p>
<script>window.onload=()=>setTimeout(()=>window.print(),400)</script></body></html>"""
