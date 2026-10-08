#!/usr/bin/env sh
# DepthWizard web app (Linux/macOS). First run installs everything; later runs start in seconds.
set -e
cd "$(dirname "$0")"
cmp -s requirements.txt .venv/requirements.installed 2>/dev/null || rm -f .venv/installed.ok
if [ ! -f .venv/installed.ok ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/installed.ok
  cp requirements.txt .venv/requirements.installed
fi
[ -f models/depth_anything_v2_vits.onnx ] || .venv/bin/python scripts/download_model.py || true
[ -f data/geoid/us_nga_egm08_25.tif ] || .venv/bin/python scripts/fetch_geoid.py || true
echo "DepthWizard: open http://localhost:8000"
exec .venv/bin/python -m depthwizard.server "$@"
