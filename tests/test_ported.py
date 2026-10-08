"""Tests for the ideas ported from the original DepthWizard repo."""
import json
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from depthwizard import config
from depthwizard.calibrate import CalibParams, band_scale, calibrate, lowpass
from depthwizard.dem import copernicus_tile_names, infer_source
from depthwizard.depth import align_params, estimate_relative_height, seam_ratio, taper
from depthwizard.jobs import read_gcps_csv
from depthwizard.osm import rasterize_buildings, tagged_height
from depthwizard.scene import Scene


def _scene(n=200, gsd=0.5):
    return Scene(rgb=np.full((n, n, 3), 150, np.uint8), gsd=gsd,
                 transform=from_origin(634000.0, 3006500.0, gsd, gsd), crs="EPSG:32645")


def test_copernicus_tile_names_and_sources():
    assert copernicus_tile_names(88.3, 27.1, 88.4, 27.2) == ["Copernicus_DSM_COG_10_N27_00_E088_00_DEM"]
    assert len(copernicus_tile_names(87.9, 26.9, 88.1, 27.1)) == 4
    assert infer_source(Path("dem/cdng45e.tif")) == "cartodem"
    assert infer_source(Path("data/dem/cartodem/x.tif")) == "cartodem"
    assert infer_source(Path("srtm_n27.tif")) == "srtm"


def test_align_params_ignores_outliers():
    rng = np.random.default_rng(0)
    t = rng.random((100, 100))
    g = 3.0 * t + 2.0
    t2 = t.copy()
    t2[:3, :3] = 500.0                       # water noise
    a, b = align_params(t2, g)
    assert abs(a - 3.0) < 0.05 and abs(b - 2.0) < 0.05


def test_hann_windows_sum_to_one_at_half_overlap():
    w = taper(518, False, False)
    total = w[259:] + w[:259]
    assert np.allclose(total, 1.0, atol=1e-5)


class _RampModel:
    is_ai, name = True, "ramp"

    def infer(self, rgb):
        h, w = rgb.shape[:2]
        return (np.linspace(0, 1, w)[None, :] * np.ones((h, 1)) * 3.0 + 7.0).astype(np.float32)


def test_no_visible_seams():
    rel, _ = estimate_relative_height(np.zeros((900, 1300, 3), np.uint8), _RampModel(), tta=False)
    assert seam_ratio(rel, 518, 259) < 1.2


def test_band_scale_recovers_metres_per_unit():
    rng = np.random.default_rng(1)
    d = lowpass(rng.random((600, 600)).astype(np.float32), 25)       # smooth model output
    dem = 8.0 * d + 100 + rng.normal(0, 0.001, d.shape)
    s, r, cells = band_scale(d, dem, dem_res_m=30.0, pixel_m=1.0)
    assert cells >= 64 and r > 0.9 and abs(s - 8.0) < 0.5


def test_off_nadir_shadow_correction():
    """A 12 m box, sun from the south (az 180, el 45), seen 20 deg off-nadir from the south:
    the roof is drawn shifted north, hiding part of its own shadow."""
    gsd, el, h_true, on = 0.5, 45.0, 12.0, 20.0
    n = 240
    rgb = np.full((n, n, 3), 150, np.uint8)
    rel = np.zeros((n, n), np.float32)
    lean_px = int(round(h_true * np.tan(np.radians(on)) / gsd))           # 9 px north
    r0 = 110 - lean_px
    rel[r0:r0 + 40, 100:140] = 1.0
    rgb[r0:r0 + 40, 100:140] = [200, 90, 80]
    shadow_px = int(round(h_true / np.tan(np.radians(el)) / gsd))         # 24 px north of the footprint
    rgb[110 - shadow_px:r0, 100:140] = [25, 28, 40]                       # only the visible part
    params = dict(gsd=gsd, sun_elevation=el, sun_azimuth=180.0)
    naive = calibrate(rgb, rel, np.zeros_like(rel), None, CalibParams(**params))
    fixed = calibrate(rgb, rel, np.zeros_like(rel), None,
                      CalibParams(**params, off_nadir=on, view_azimuth=180.0))
    assert naive.info["anchors"][0]["h"] < 0.75 * h_true                  # uncorrected: too short
    assert abs(fixed.info["anchors"][0]["h"] - h_true) < 1.5


def test_gcp_csv_formats(tmp_path):
    sc = _scene()
    f = tmp_path / "g.csv"
    f.write_text("x,y,height_m\n634050,3006450,1500\n")
    g = read_gcps_csv(str(f), sc)
    assert abs(g[0]["u"] - 0.5) < 1e-6 and abs(g[0]["v"] - 0.5) < 1e-6 and g[0]["elevation"] == 1500
    f.write_text("u,v,elevation\n0.25,0.75,10\n")
    assert read_gcps_csv(str(f), sc)[0]["u"] == 0.25


def test_osm_footprints_rasterise_with_heights():
    sc = _scene()
    from pyproj import Transformer
    to_ll = Transformer.from_crs(sc.crs, "EPSG:4326", always_xy=True)
    ring = [to_ll.transform(634000 + x, 3006500 - y) for x, y in ((20, 20), (40, 20), (40, 40), (20, 40), (20, 20))]
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"building": "yes", "building:levels": "4"},
         "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
    lab, heights = rasterize_buildings(fc, sc)
    assert heights == [13.0]
    assert abs((lab == 1).sum() - 40 * 40) < 120                   # 20 m x 20 m at 0.5 m
    assert lab[60, 60] == 1 and lab[10, 10] == 0
    assert tagged_height({"height": "21 m"}) == 21.0


def test_config_rejects_bad_types():
    with pytest.raises(ValueError):
        config._merge(config.DEFAULTS, {"calibration": {"band_min_r": "high"}})
    with pytest.raises(ValueError):
        config._merge(config.DEFAULTS, {"nope": 1})
    assert config.get("depth.tile_size") == 518
