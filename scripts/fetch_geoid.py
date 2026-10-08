"""Download the EGM2008 geoid grid once (2.5 arc-minute, public domain, ~81 MB).

With it, ellipsoidal DEMs such as CartoDEM V3R1 are converted to EGM2008 (sea-level) heights
offline, the datum Copernicus GLO-30 already uses. Ported from the original DepthWizard repo.

    python scripts/fetch_geoid.py          -> data/geoid/us_nga_egm08_25.tif
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from depthwizard import config  # noqa: E402

URLS = ["https://cdn.proj.org/us_nga_egm08_25.tif",
        "https://raw.githubusercontent.com/OSGeo/PROJ-data/master/us_nga/us_nga_egm08_25.tif"]


def main() -> int:
    dest = config.resolve(config.get("dem.geoid_grid"))
    if dest.exists():
        print(f"Already present: {dest}")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        try:
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest.with_suffix(".part"))
            dest.with_suffix(".part").replace(dest)
            print(f"Saved to {dest}")
            return 0
        except OSError as e:
            print(f"  failed: {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
