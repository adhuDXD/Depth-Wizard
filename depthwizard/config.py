"""Tunables from configs/default.yaml (or DEPTHWIZARD_CONFIG), with safe built-in defaults."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = REPO_ROOT / "configs" / "default.yaml"

DEFAULTS: dict = {
    "input": {"max_long_side": 2048, "stretch_percentiles": [2, 98]},
    "depth": {"tile_size": 518, "tile_overlap": 0.5, "align_trim_frac": 0.05, "tta": True},
    "dem": {"auto_fetch": True,
            "copernicus_url": "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif",
            "fetch_timeout_s": 60, "cache_dir": "data/cache/copernicus",
            "geoid_grid": "data/geoid/us_nga_egm08_25.tif", "datum_check_min_offset_m": 10.0},
    "calibration": {"band_sigma_k": 1.0, "band_coarse_cells": 4.0, "band_min_r": 0.1,
                    "band_min_cells": 64, "ground_window_m": 60.0,
                    "max_building_height_m": 120.0},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if k not in base:
            raise ValueError(f"Unknown config key '{k}'")
        if isinstance(base[k], dict):
            out[k] = _merge(base[k], v)
        else:
            if base[k] is not None and not isinstance(v, type(base[k])) and \
                    not (isinstance(base[k], float) and isinstance(v, int)):
                raise ValueError(f"Config '{k}': expected {type(base[k]).__name__}, got {v!r}")
            out[k] = v
    return out


@lru_cache(maxsize=1)
def load() -> dict:
    path = Path(os.environ.get("DEPTHWIZARD_CONFIG", DEFAULT_FILE))
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    return _merge(DEFAULTS, data or {})


def get(dotted: str):
    node = load()
    for part in dotted.split("."):
        node = node[part]
    return node


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
