"""Real demo scene: Namchi, South Sikkim. WorldView-3, 14 March 2022, ~0.37 m, true colour.

Imagery: Maxar Open Data Program (event "India-Floods-Oct-2023"), licence CC BY-NC 4.0.
Credit "Maxar Open Data Program" wherever the imagery is shown. Non-commercial use only.
(Ported from the team's original DepthWizard repo.)

The town centre sits where four "visual" COG tiles meet. This downloads them once
(~290 MB) into data/demo/namchi/raw/, mosaics them and crops a square around the town:

    python scripts/fetch_demo_namchi.py          -> data/demo/namchi/namchi.tif
    python scripts/fetch_demo_namchi.py --size 2048
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.merge import merge

REPO = Path(__file__).resolve().parent.parent
BASE_URL = ("https://maxar-opendata.s3.amazonaws.com/events/India-Floods-Oct-2023/ard/45/"
            "{quadkey}/2022-03-14/1040010073381800-visual.tif")
QUADKEYS = ["120220211230", "120220211231", "120220211232", "120220211233"]
CENTRE_LONLAT = (88.363, 27.165)   # Namchi town centre
OUT_DIR = REPO / "data" / "demo" / "namchi"
ATTRIBUTION = "Maxar Open Data Program, WorldView-3 2022-03-14, CC BY-NC 4.0"


META_URL = BASE_URL.replace("-visual.tif", ".json")


def download(url: str, dest: Path) -> Path | None:
    if dest.exists():
        return dest
    print(f"downloading {url}")
    part = dest.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, part)
    except urllib.error.HTTPError as e:
        if e.code == 404:                      # not every quadkey exists for this capture
            print("  (not available, skipped)")
            return None
        raise
    part.replace(dest)
    return dest


def sun_view_angles() -> dict:
    """Sun and view angles from the capture's STAC metadata (needed for shadow heights)."""
    for q in QUADKEYS:
        try:
            with urllib.request.urlopen(META_URL.format(quadkey=q), timeout=60) as r:
                p = json.load(r)["properties"]
            return {"SUN_ELEVATION": p["view:sun_elevation"], "SUN_AZIMUTH": p["view:sun_azimuth"],
                    "OFF_NADIR": p["view:off_nadir"], "VIEW_AZIMUTH": p["view:azimuth"],
                    "ACQUIRED": p["datetime"].replace(" ", "T")}
        except (urllib.error.URLError, KeyError):
            continue
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=4096, help="crop size in pixels (4096 = ~1.5 km)")
    args = ap.parse_args()
    raw = OUT_DIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = [p for q in QUADKEYS if (p := download(BASE_URL.format(quadkey=q), raw / f"{q}.tif"))]
    if not paths:
        print("No imagery could be downloaded.")
        return 1

    sources = [rasterio.open(p) for p in paths]
    try:
        crs, res = sources[0].crs, sources[0].res[0]
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(*CENTRE_LONLAT)
        half = args.size / 2 * res
        rgb, transform = merge(sources, bounds=(x - half, y - half, x + half, y + half),
                               res=res, indexes=[1, 2, 3], nodata=0)
    finally:
        for src in sources:
            src.close()

    out = OUT_DIR / "namchi.tif"
    profile = {"driver": "GTiff", "width": rgb.shape[2], "height": rgb.shape[1], "count": 3,
               "dtype": "uint8", "crs": crs, "transform": transform, "compress": "deflate",
               "tiled": True, "photometric": "RGB"}
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(rgb.astype(np.uint8))
        dst.update_tags(SOURCE=ATTRIBUTION, CENTRE_LONLAT=str(CENTRE_LONLAT), **sun_view_angles())
    print(f"{out}: {rgb.shape[2]}x{rgb.shape[1]} px at {res:.3f} m, {crs}, "
          f"coverage {float(rgb.any(axis=0).mean()):.1%}")
    print(f"Credit: {ATTRIBUTION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
