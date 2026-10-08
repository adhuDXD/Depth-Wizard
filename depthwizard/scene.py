"""Loading satellite images (PNG/JPG/GeoTIFF) and DEMs onto one common grid.

Georeferenced inputs are reprojected to the local UTM zone so that one pixel
has a known size in metres. Plain images keep pixel coordinates; their pixel
size (GSD) can be supplied by the user.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy import ndimage

MAX_DIM = 1024
GEOTIFF_SUFFIXES = {".tif", ".tiff", ".gtif", ".geotiff"}


@dataclass
class Scene:
    rgb: np.ndarray                      # (H, W, 3) uint8
    gsd: float | None                    # metres per pixel, None if unknown
    transform: Affine | None = None      # pixel -> projected (UTM) coordinates
    crs: str | None = None               # e.g. "EPSG:32645"
    dem: np.ndarray | None = None        # (H, W) metres, on the image grid
    sun_elevation: float | None = None   # degrees, from metadata if present
    sun_azimuth: float | None = None     # degrees clockwise from north

    @property
    def georeferenced(self) -> bool:
        return self.transform is not None and self.crs is not None

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[:2]

    def center_lonlat(self) -> tuple[float, float] | None:
        if not self.georeferenced:
            return None
        h, w = self.shape
        x, y = self.transform @ (w / 2, h / 2)
        to_wgs = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        return to_wgs.transform(x, y)


def utm_crs_for(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def _stretch_to_uint8(band: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if band.dtype == np.uint8:
        return band
    vals = band[valid]
    if vals.size == 0:
        return np.zeros(band.shape, np.uint8)
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1
    out = np.clip((band.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def _fill_nan(a: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    if bad.all():
        return np.zeros_like(a)
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _read_sun_tags(src) -> tuple[float | None, float | None]:
    tags = {k.upper(): v for k, v in src.tags().items()}
    def pick(*names):
        for n in names:
            if n in tags:
                try:
                    return float(tags[n])
                except ValueError:
                    pass
        return None
    el = pick("SUN_ELEVATION", "SUNELEVATION", "SUN_ELEV", "SOLAR_ELEVATION")
    az = pick("SUN_AZIMUTH", "SUNAZIMUTH", "SUN_AZ", "SOLAR_AZIMUTH")
    return el, az


def load_image(path: str | Path, gsd_hint: float | None = None, max_dim: int = MAX_DIM) -> Scene:
    path = Path(path)
    if path.suffix.lower() in GEOTIFF_SUFFIXES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(path) as src:
                if src.crs is not None:
                    return _load_georeferenced(src, max_dim)
                arr = src.read(list(range(1, min(src.count, 3) + 1)))
        arr = np.moveaxis(arr, 0, -1)
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        valid = np.ones(arr.shape[:2], bool)
        rgb = np.stack([_stretch_to_uint8(arr[..., i], valid) for i in range(3)], -1)
        img = Image.fromarray(rgb)
    else:
        img = Image.open(path).convert("RGB")
    return _plain_scene(img, gsd_hint, max_dim)


def _plain_scene(img: Image.Image, gsd_hint: float | None, max_dim: int) -> Scene:
    w, h = img.size
    scale = max(w, h) / max_dim
    if scale > 1:
        img = img.resize((round(w / scale), round(h / scale)), Image.LANCZOS)
    else:
        scale = 1.0
    gsd = gsd_hint * scale if gsd_hint else None
    return Scene(rgb=np.asarray(img, np.uint8).copy(), gsd=gsd)


def _load_georeferenced(src, max_dim: int) -> Scene:
    src_crs = CRS.from_user_input(src.crs)
    left, bottom, right, top = src.bounds
    to_wgs = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs.transform((left + right) / 2, (bottom + top) / 2)
    if src_crs.is_projected and src_crs.axis_info[0].unit_name in ("metre", "meter"):
        dst_crs = src_crs
    else:
        dst_crs = utm_crs_for(lon, lat)

    transform, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
    scale = max(w, h) / max_dim
    if scale > 1:
        transform = transform * Affine.scale(scale)
        w, h = int(np.ceil(w / scale)), int(np.ceil(h / scale))

    nb = min(src.count, 3)
    bands = []
    for b in range(1, nb + 1):
        dst = np.full((h, w), np.nan, np.float32)
        reproject(
            source=rasterio.band(src, b), destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=transform, dst_crs=dst_crs, dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        bands.append(dst)
    valid = np.all([np.isfinite(b) for b in bands], axis=0)
    src_is_byte = src.dtypes[0] == "uint8"
    rgb_bands = []
    for b in bands:
        b = _fill_nan(b)
        rgb_bands.append(np.clip(b, 0, 255).astype(np.uint8) if src_is_byte else _stretch_to_uint8(b, valid))
    if nb == 1:
        rgb_bands = rgb_bands * 3
    el, az = _read_sun_tags(src)
    return Scene(
        rgb=np.stack(rgb_bands, -1), gsd=abs(transform.a), transform=transform,
        crs=dst_crs.to_string(), sun_elevation=el, sun_azimuth=az,
    )


def load_dem(path: str | Path, scene: Scene) -> np.ndarray:
    """Resample a DEM GeoTIFF onto the scene grid (metres)."""
    if not scene.georeferenced:
        raise ValueError("A DEM can only be aligned with a georeferenced (GeoTIFF) image.")
    h, w = scene.shape
    dst = np.full((h, w), np.nan, np.float32)
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("The DEM file has no coordinate reference system.")
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=scene.transform, dst_crs=scene.crs, dst_nodata=np.nan,
            resampling=Resampling.cubic,
        )
    if not np.isfinite(dst).any():
        raise ValueError("The DEM does not overlap the image.")
    return _fill_nan(dst)
