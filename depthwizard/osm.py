"""OpenStreetMap buildings: footprints and tagged heights.

Implements the original DepthWizard plan's "OSM buildings" scale source (ARCHITECTURE.md,
Stage 3b, source 2), which was cut there. Footprints replace the model's blobs with real
outlines; buildings tagged `height` or `building:levels` (x 3 m) become metric anchors.

Sources: an uploaded GeoJSON (export from overpass-turbo.eu or a Geofabrik extract), or the
Overpass API, fetched once per scene and cached (the app still works offline without it).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize

from . import config

log = logging.getLogger(__name__)
LEVEL_HEIGHT_M = 3.0
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _num(text) -> float | None:
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", str(text or ""))
    return float(m.group(1)) if m else None


def tagged_height(props: dict) -> float | None:
    h = _num(props.get("height"))
    if h:
        return h
    levels = _num(props.get("building:levels"))
    return levels * LEVEL_HEIGHT_M + 1.0 if levels else None


def fetch_overpass(bounds_lonlat, timeout: int = 60) -> dict:
    """Building ways in the bounds as GeoJSON-like features (cached under data/cache/osm)."""
    w, s, e, n = bounds_lonlat
    key = hashlib.sha1(f"{w:.5f},{s:.5f},{e:.5f},{n:.5f}".encode()).hexdigest()[:16]
    cache = config.resolve("data/cache/osm") / f"{key}.geojson"
    if cache.exists():
        return json.loads(cache.read_text())
    query = f'[out:json][timeout:{timeout}];way["building"]({s},{w},{n},{e});out tags geom;'
    data = urllib.parse.urlencode({"data": query}).encode()
    with urllib.request.urlopen(OVERPASS_URL, data=data, timeout=timeout) as r:
        raw = json.load(r)
    feats = []
    for el in raw.get("elements", []):
        geom = el.get("geometry")
        if geom and len(geom) >= 4:
            ring = [[p["lon"], p["lat"]] for p in geom]
            feats.append({"type": "Feature", "properties": el.get("tags", {}),
                          "geometry": {"type": "Polygon", "coordinates": [ring]}})
    fc = {"type": "FeatureCollection", "features": feats}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(fc))
    return fc


def load_geojson(path: str | Path) -> dict:
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    if fc.get("type") == "Feature":
        fc = {"type": "FeatureCollection", "features": [fc]}
    return fc


def rasterize_buildings(fc: dict, scene) -> tuple[np.ndarray, list[float | None]]:
    """Label image (0 = none, i = footprint i) on the scene grid, and each footprint's tagged
    height. GeoJSON coordinates are lon/lat (EPSG:4326) unless they already look projected."""
    h, w = scene.shape
    to_scene = Transformer.from_crs("EPSG:4326", scene.crs, always_xy=True)
    inv = ~scene.transform
    shapes, heights = [], []
    for f in fc.get("features", []):
        g = f.get("geometry") or {}
        polys = [g["coordinates"]] if g.get("type") == "Polygon" else \
            (g["coordinates"] if g.get("type") == "MultiPolygon" else [])
        for poly in polys:
            rings = []
            for ring in poly:
                pts = np.asarray(ring, float)
                if np.abs(pts).max() <= 360:                   # lon/lat
                    xs, ys = to_scene.transform(pts[:, 0], pts[:, 1])
                else:
                    xs, ys = pts[:, 0], pts[:, 1]
                cols, rows = inv @ (np.asarray(xs), np.asarray(ys))
                rings.append(list(zip(cols, rows)))
            heights.append(tagged_height(f.get("properties") or {}))
            shapes.append(({"type": "Polygon", "coordinates": rings}, len(heights)))
    if not shapes:
        return np.zeros((h, w), np.int32), []
    lab = rasterize(shapes, out_shape=(h, w), fill=0, dtype="int32")   # pixel grid, identity transform
    return lab, heights


def get_buildings(scene, path: str | Path | None = None, allow_fetch: bool = False):
    """(label image, heights, source) or None."""
    if not scene.georeferenced:
        return None
    try:
        if path:
            fc, src = load_geojson(path), "uploaded GeoJSON"
        elif allow_fetch:
            from .dem import scene_bounds_lonlat
            fc, src = fetch_overpass(scene_bounds_lonlat(scene)), "OpenStreetMap (Overpass)"
        else:
            return None
    except (OSError, ValueError) as e:
        log.warning("OSM buildings unavailable: %s", e)
        return None
    lab, heights = rasterize_buildings(fc, scene)
    if lab.max() == 0:
        return None
    return lab, heights, src
