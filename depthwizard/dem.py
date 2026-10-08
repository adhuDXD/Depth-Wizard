"""Calibration DEM: an uploaded file (CartoDEM, SRTM, ...) or Copernicus GLO-30 fetched once and
cached, resampled onto the image grid, with its vertical datum made consistent (EGM2008).

Ported from the original DepthWizard repo (depthwizard/calibrate/dem.py, evals/datum_check.py).
The Copernicus fetch is the only network call; with an uploaded DEM the app runs offline.
"""
from __future__ import annotations

import logging
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy import ndimage

from . import config

log = logging.getLogger(__name__)

# Vertical datum per known source.
DATUMS = {
    "copernicus_glo30": "EGM2008",
    # Measured on V3R1 tile 88E27N: 46 m below Copernicus, matching the EGM2008 geoid (-43.8 m).
    "cartodem": "WGS84 ellipsoid (CartoDEM V3R1)",
    "srtm": "EGM96",
    "nasadem": "EGM96",
    "synthetic": "WGS84 ellipsoid (synthetic, CartoDEM-like)",
}
ELLIPSOIDAL_SOURCES = {"cartodem", "synthetic"}
# 30 m surface models contain blurred buildings and trees; a bare-earth DTM does not.
SURFACE_SOURCES = {"copernicus_glo30", "cartodem", "srtm", "nasadem", "synthetic"}


@dataclass
class Dem:
    heights: np.ndarray          # (H, W) float32 metres on the image grid
    source: str
    res_m: float                 # native ground resolution
    datum: str
    kind: str = "surface"        # surface | terrain
    notes: list[str] = field(default_factory=list)


def infer_source(path: Path, tags: dict | None = None) -> str:
    """Name the DEM source from tags or path, e.g. data/dem/cartodem/cdng45e.tif -> cartodem."""
    if tags and tags.get("DW_DEM_SOURCE"):
        return tags["DW_DEM_SOURCE"]
    text = str(path).lower()
    for key, name in (("carto", "cartodem"), ("copernicus", "copernicus_glo30"),
                      ("nasadem", "nasadem"), ("srtm", "srtm")):
        if key in text:
            return name
    stem = path.stem.lower()
    if len(stem) == 7 and stem.startswith("cdn") and stem[4:6].isdigit():
        return "cartodem"          # Bhuvan tile names, e.g. cdng45e
    return path.stem


def copernicus_tile_names(west: float, south: float, east: float, north: float) -> list[str]:
    """1x1 degree GLO-30 COG names; each is named after its south-west corner."""
    names = []
    for lat in range(math.floor(south), math.ceil(north)):
        for lon in range(math.floor(west), math.ceil(east)):
            ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
            names.append(f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM")
    return names


def fetch_copernicus(bounds_lonlat, cache_dir: Path | None = None) -> list[Path]:
    """Download the GLO-30 tiles covering the bounds once. Ocean tiles (404) are skipped."""
    cache_dir = cache_dir or config.resolve(config.get("dem.cache_dir"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    url_t, timeout = config.get("dem.copernicus_url"), config.get("dem.fetch_timeout_s")
    paths = []
    for name in copernicus_tile_names(*bounds_lonlat):
        path = cache_dir / f"{name}.tif"
        if not path.exists():
            url = url_t.format(name=name)
            log.info("Fetching %s", url)
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp, \
                        open(path.with_suffix(".part"), "wb") as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                raise
            path.with_suffix(".part").replace(path)
        paths.append(path)
    if not paths:
        raise FileNotFoundError("No Copernicus DEM tiles cover this scene.")
    return paths


def scene_bounds_lonlat(scene) -> tuple[float, float, float, float]:
    h, w = scene.shape
    left, top = scene.transform @ (0, 0)
    right, bottom = scene.transform @ (w, h)
    return transform_bounds(scene.crs, "EPSG:4326", left, bottom, right, top, densify_pts=21)


def _res_m(src) -> float:
    if src.crs and src.crs.is_geographic:
        cx, cy = src.transform @ (src.width / 2, src.height / 2)
        lat = math.radians(cy)
        dx = abs(src.transform.a) * 111320 * math.cos(lat)
        dy = abs(src.transform.e) * 110540
        return math.sqrt(dx * dy)
    return math.sqrt(abs(src.transform.a * src.transform.e))


def resample_to_scene(files: list[Path], scene, resampling=Resampling.bilinear) -> tuple[np.ndarray, float, dict]:
    """Mosaic DEM file(s) onto the image grid; first file wins where they overlap."""
    h, w = scene.shape
    heights = np.full((h, w), np.nan, np.float32)
    res_m, tags = math.nan, {}
    for path in files:
        tmp = np.full((h, w), np.nan, np.float32)
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"{path.name} has no coordinate reference system.")
            reproject(source=rasterio.band(src, 1), destination=tmp, src_nodata=src.nodata,
                      dst_transform=scene.transform, dst_crs=scene.crs, dst_nodata=np.nan,
                      resampling=resampling)
            if math.isnan(res_m):
                res_m = _res_m(src)
                tags = src.tags()
        fill = np.isnan(heights) & np.isfinite(tmp)
        heights[fill] = tmp[fill]
    if not np.isfinite(heights).any():
        raise ValueError("The DEM does not overlap the image.")
    return heights, res_m, tags


def _fill(a: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def geoid_undulation(scene) -> np.ndarray | None:
    """EGM2008 geoid height N on the image grid, or None if the grid file is missing."""
    grid = config.resolve(config.get("dem.geoid_grid"))
    if not grid.exists():
        return None
    h, w = scene.shape
    n = np.full((h, w), np.nan, np.float32)
    with rasterio.open(grid) as src:
        reproject(source=rasterio.band(src, 1), destination=n, src_nodata=src.nodata,
                  dst_transform=scene.transform, dst_crs=scene.crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
        n = n * src.scales[0] + src.offsets[0]
    return n if np.isfinite(n).all() else None


def get_dem(scene, path: str | Path | None = None, manual_offset_m: float | None = None,
            kind: str | None = None, allow_fetch: bool | None = None) -> Dem | None:
    """The calibration DEM on the scene grid, in EGM2008 (sea-level) heights where possible.

    1. uploaded file, else Copernicus GLO-30 (auto-fetched, cached) for georeferenced scenes
    2. ellipsoidal sources (CartoDEM) -> EGM2008 with the geoid grid, else measured against
       Copernicus (datum check), else the manual offset
    """
    if not scene.georeferenced:
        return None
    notes: list[str] = []
    allow_fetch = config.get("dem.auto_fetch") if allow_fetch is None else allow_fetch
    if path:
        files = [Path(path)]
        heights, res_m, tags = resample_to_scene(files, scene)
        source = infer_source(files[0], tags)
    elif allow_fetch:
        try:
            files = fetch_copernicus(scene_bounds_lonlat(scene))
        except (OSError, ValueError) as e:
            log.warning("Copernicus fetch failed: %s", e)
            return None
        heights, res_m, tags = resample_to_scene(files, scene)
        source = "copernicus_glo30"
        notes.append("DEM: Copernicus GLO-30 (fetched automatically, cached for offline use).")
    else:
        return None

    datum = DATUMS.get(source, "unknown")
    if manual_offset_m:
        heights = heights + manual_offset_m
        notes.append(f"Vertical datum offset of {manual_offset_m:+.1f} m applied to the DEM (entered).")
        datum = f"{datum}, shifted {manual_offset_m:+.1f} m by the user"
    elif source in ELLIPSOIDAL_SOURCES:
        n = geoid_undulation(scene)
        if n is not None:
            heights = heights - n
            notes.append(f"{source}: converted from the WGS84 ellipsoid to EGM2008 sea-level heights "
                         f"(geoid N = {np.nanmedian(n):.1f} m).")
            datum = "EGM2008 (converted from the WGS84 ellipsoid)"
        else:
            offset, how = None, ""
            if tags_n := (tags.get("DW_GEOID_N") if path else None):
                offset, how = float(tags_n), "geoid height recorded in the file"
            elif allow_fetch and source != "synthetic":
                offset, how = datum_offset_vs_copernicus(scene, heights), "measured against Copernicus"
            if offset is not None:
                heights = heights - offset
                notes.append(f"{source}: ellipsoidal heights converted to sea level ({how}; {offset:+.1f} m).")
                datum = f"EGM2008 ({how})"
            else:
                notes.append(f"{source} heights are ellipsoidal (~40-50 m above sea level in India). Run "
                             "scripts/fetch_geoid.py or enter the datum offset.")
    dem_kind = kind or ("surface" if source in SURFACE_SOURCES else "surface")
    heights = _fill(heights).astype(np.float32)
    if scene.gsd:
        # bilinear resampling of a 30 m grid leaves flat facets; a light blur (0.4 cell) hides them
        heights = ndimage.gaussian_filter(heights, 0.4 * res_m / scene.gsd)
    return Dem(heights=heights, source=source, res_m=float(res_m),
               datum=datum, kind=dem_kind, notes=notes)


def datum_offset_vs_copernicus(scene, heights: np.ndarray) -> float | None:
    """Median (DEM - Copernicus): a large, uniform difference is a vertical datum offset."""
    try:
        cop, _, _ = resample_to_scene(fetch_copernicus(scene_bounds_lonlat(scene)), scene)
    except (OSError, ValueError):
        return None
    d = (heights - cop)[np.isfinite(heights) & np.isfinite(cop)]
    if d.size < 100:
        return None
    med = float(np.median(d))
    nmad = 1.4826 * float(np.median(np.abs(d - med)))
    if abs(med) >= config.get("dem.datum_check_min_offset_m") and nmad < abs(med) / 2:
        return med
    return None


def to_lonlat(scene):
    return Transformer.from_crs(scene.crs, "EPSG:4326", always_xy=True)
