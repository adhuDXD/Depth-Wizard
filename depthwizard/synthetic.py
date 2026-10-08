"""Synthetic hill-town scene with known ground truth.

Lets the app be demonstrated and *validated* without any downloads: we know
the true DSM, so DEM-only vs DepthWizard errors can be measured honestly.
The scene is written as GeoTIFFs (image, coarse DEM with a datum offset, truth)
and then goes through exactly the same pipeline as user uploads.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

SIZE = 1024
GSD = 0.5
CRS = "EPSG:32645"                 # UTM 45N (Sikkim)
ORIGIN = (634000.0, 3006500.0)     # near Namchi
SUN_ELEVATION = 52.0
SUN_AZIMUTH = 140.0
GEOID_N = -43.8                    # EGM2008 geoid height at Namchi: CartoDEM (ellipsoidal) = H + N


def _terrain(n: int, rng) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n] / n
    hill = 1450 + 90 * y + 18 * np.sin(2.2 * np.pi * x + 1.0) * (0.5 + y) + 30 * np.tanh((y - 0.42) / 0.07)
    valley_x = 0.18 + 0.06 * np.sin(3 * np.pi * y)
    valley = -22 * np.exp(-((x - valley_x) / 0.07) ** 2)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), 25) * 60
    return (hill + valley + noise).astype(np.float32)


def _shadows(dsm: np.ndarray, gsd: float, el: float, az: float) -> np.ndarray:
    a = np.radians(az)
    toward_sun = np.array([-np.cos(a), np.sin(a)])      # (drow, dcol)
    tan_el = np.tan(np.radians(el))
    n = dsm.shape[0]
    shade = np.zeros_like(dsm, bool)
    for t in range(1, int(40 / (gsd * tan_el)) + 1):
        dr, dc = np.rint(toward_sun * t).astype(int)
        shifted = np.full_like(dsm, -np.inf)
        rs, cs = slice(max(0, -dr), n - max(0, dr)), slice(max(0, -dc), n - max(0, dc))
        rd, cd = slice(max(0, dr), n - max(0, -dr)), slice(max(0, dc), n - max(0, -dc))
        shifted[rs, cs] = dsm[rd, cd]
        shade |= shifted > dsm + t * gsd * tan_el + 0.3
    return shade


def make_scene(seed: int = 7):
    rng = np.random.default_rng(seed)
    n = SIZE
    terrain = _terrain(n, rng)
    yy, xx = np.mgrid[0:n, 0:n]

    land = np.zeros((n, n), np.uint8)          # 0 grass, 1 road, 2 river, 3 trees
    river_x = (0.18 + 0.06 * np.sin(3 * np.pi * yy / n)) * n
    land[np.abs(xx - river_x) < 9] = 2
    for k, y0 in enumerate([260, 520, 790]):
        cy = y0 + 25 * np.sin(xx / 90 + k)
        land[(np.abs(yy - cy) < 5) & (land != 2)] = 1
    for x0 in (470, 760):
        land[(np.abs(xx - x0 - 20 * np.sin(yy / 120)) < 4) & (land != 2)] = 1

    bldg_h = np.zeros((n, n), np.float32)
    roofs = np.zeros((n, n, 3), np.float32)
    palette = np.array([[178, 60, 50], [70, 110, 160], [190, 190, 185], [150, 150, 150], [210, 205, 190], [90, 140, 120]])
    placed = 0
    for _ in range(4000):
        if placed >= 170:
            break
        cx, cy = rng.integers(320, n - 40), rng.integers(40, n - 40)
        w2, h2 = rng.integers(10, 26), rng.integers(10, 24)
        y0, y1, x0, x1 = cy - h2, cy + h2, cx - w2, cx + w2
        if y0 < 0 or x0 < 0 or y1 >= n or x1 >= n:
            continue
        pad = 6
        if (land[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] != 0).any() or \
                bldg_h[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad].any():
            continue
        floors = int(rng.choice([1, 2, 2, 3, 3, 4, 4, 5, 6, 8]))
        bldg_h[y0:y1, x0:x1] = floors * 3.1 + 1.2
        roofs[y0:y1, x0:x1] = palette[rng.integers(len(palette))] * rng.uniform(0.85, 1.1)
        placed += 1

    tree_h = np.zeros((n, n), np.float32)
    for _ in range(700):
        cx, cy = rng.integers(0, n), rng.integers(0, n)
        if cx > 360 and rng.random() < 0.75:
            continue
        r = rng.integers(5, 12)
        m = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
        m &= (land == 0) & (bldg_h == 0)
        tree_h[m] = np.maximum(tree_h[m], rng.uniform(8, 16) * np.sqrt(1 - ((xx[m] - cx) ** 2 + (yy[m] - cy) ** 2) / (r * r)))
    land[tree_h > 0] = 3

    dsm = terrain + bldg_h + tree_h
    dsm[land == 2] -= 1.5

    # ---- render
    tex = rng.normal(0, 1, (n, n)).astype(np.float32)
    base = np.zeros((n, n, 3), np.float32)
    base[:] = [118, 128, 82]
    base += cv2.GaussianBlur(tex, (0, 0), 6)[..., None] * 25
    base[land == 1] = [165, 160, 150]
    base[land == 2] = [70, 100, 125]
    base[land == 3] = [45, 85, 45]
    base[land == 3] += (tex[land == 3] * 12)[:, None]
    base[bldg_h > 0] = roofs[bldg_h > 0]
    gy, gx = np.gradient(dsm, GSD)
    a, e = np.radians(SUN_AZIMUTH), np.radians(SUN_ELEVATION)
    sun = np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    nrm = np.dstack([-gx, gy, np.ones_like(gx)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    shade = np.clip(nrm @ sun, 0, 1)
    img = base * (0.35 + 0.75 * shade[..., None])
    sh = _shadows(dsm, GSD, SUN_ELEVATION, SUN_AZIMUTH)
    img[sh] = img[sh] * 0.4 + np.array([8, 12, 25])
    img += rng.normal(0, 3, img.shape)
    rgb = np.clip(img, 0, 255).astype(np.uint8)

    # ---- coarse ~30 m surface DEM, like CartoDEM: buildings and trees blurred in, ellipsoidal
    k = int(30 / GSD)
    coarse = cv2.resize(dsm, (n // k + 1, n // k + 1), interpolation=cv2.INTER_AREA)
    return {"rgb": rgb, "truth": dsm.astype(np.float32), "dem": coarse + GEOID_N,
            "dem_gsd": GSD * n / coarse.shape[0], "terrain": terrain, "building_h": bldg_h, "tree_h": tree_h}


def write_scene(folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    s = make_scene()
    tr = from_origin(ORIGIN[0], ORIGIN[1], GSD, GSD)
    paths = {k: folder / f"{k}.tif" for k in ("image", "dem", "truth")}
    with rasterio.open(paths["image"], "w", driver="GTiff", width=SIZE, height=SIZE, count=3, dtype="uint8",
                       crs=CRS, transform=tr) as dst:
        dst.write(np.moveaxis(s["rgb"], -1, 0))
        dst.update_tags(SUN_ELEVATION=SUN_ELEVATION, SUN_AZIMUTH=SUN_AZIMUTH)
    dem = s["dem"]
    with rasterio.open(paths["dem"], "w", driver="GTiff", width=dem.shape[1], height=dem.shape[0], count=1,
                       dtype="float32", crs=CRS, transform=from_origin(ORIGIN[0], ORIGIN[1], s["dem_gsd"], s["dem_gsd"])) as dst:
        dst.write(dem, 1)
        dst.update_tags(DW_DEM_SOURCE="synthetic", DW_GEOID_N=GEOID_N)
    with rasterio.open(paths["truth"], "w", driver="GTiff", width=SIZE, height=SIZE, count=1, dtype="float32",
                       crs=CRS, transform=tr) as dst:
        dst.write(s["truth"], 1)
    return {k: str(v) for k, v in paths.items()}
