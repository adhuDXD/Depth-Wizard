"""Download Depth Anything V2 Small (ONNX, ~99 MB, Apache-2.0) into models/."""
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx"
DEST = Path(__file__).resolve().parent.parent / "models" / "depth_anything_v2_vits.onnx"


def main() -> int:
    if DEST.exists():
        print(f"Already present: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    print(f"Downloading {URL}")

    def hook(blocks, size, total):
        if total > 0:
            print(f"\r  {min(blocks * size, total) / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(URL, tmp, hook)
    tmp.rename(DEST)
    print(f"\nSaved to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
