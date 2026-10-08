from datetime import datetime, timezone

import numpy as np

from depthwizard.calibrate import CalibParams, calibrate, fit_scale
from depthwizard.depth import HeuristicDepthModel, estimate_relative_height
from depthwizard.hydrology import compute_hydrology
from depthwizard.routing import evacuation_field, trace, tobler_speed
from depthwizard.sun import sun_position


def test_sun_position_equinox_noon():
    # Delhi, March equinox, ~local solar noon (77.2E -> 12:00 solar ~ 06:51 UTC)
    el, az = sun_position(datetime(2024, 3, 20, 6, 51, tzinfo=timezone.utc), 28.61, 77.21)
    assert abs(el - (90 - 28.61)) < 1.5
    assert abs(az - 180) < 5


def test_tobler_is_fastest_slightly_downhill():
    s = tobler_speed(np.array([-0.05, 0.0, 0.2, -0.3]))
    assert s[0] == s.max()
    assert s[2] < s[1] and s[3] < s[1]


def test_hand_valley():
    h, w = 60, 80
    x = np.abs(np.arange(w) - w // 2)[None, :].repeat(h, 0).astype(float)
    z = 2.0 * x + np.linspace(10, 0, h)[:, None]       # V valley draining south
    hy = compute_hydrology(z, cell=1.0)
    center = hy.hand[:, w // 2]
    assert center.max() < 1e-2                           # stream cells
    assert hy.hand[30, w // 2 + 20] > 30                 # 20 cells up the slope ~ 40 m


def test_evacuation_field_points_to_target():
    h, w = 20, 30
    targets = np.zeros((h, w), np.int32)
    targets[:, -2:] = 1
    f = evacuation_field(None, 2.0, np.ones((h, w), np.float32), np.ones((h, w), bool), targets,
                         population=np.ones((h, w)))
    assert np.isfinite(f.time_s).all()
    assert np.all(np.diff(f.time_s[10, :-2]) < 0)        # closer to the target -> less time
    path = trace(f, 10, 0)
    assert path[-1][1] >= w - 2
    assert (f.target_id == 1).all()


def test_flood_blocks_route_through_penalised_cells():
    h, w = 21, 21
    mult = np.ones((h, w), np.float32)
    mult[:, 10] = 1000.0                                  # deep water column
    targets = np.zeros((h, w), np.int32)
    targets[10, 20] = 1
    targets[0, 0] = 2
    f = evacuation_field(None, 1.0, mult, np.ones((h, w), bool), targets)
    assert f.target_id[10, 2] == 2                        # avoids crossing the water


def test_shadow_anchor_measures_height():
    gsd, el = 0.5, 45.0
    n = 200
    rgb = np.full((n, n, 3), 150, np.uint8)
    rel = np.zeros((n, n), np.float32)
    rel[80:120, 80:120] = 1.0
    rgb[80:120, 80:120] = [200, 90, 80]
    L = int(round(12.0 / np.tan(np.radians(el)) / gsd))   # 24 px
    rgb[80 - L:80, 80:120] = [25, 28, 40]                 # sun from the south (az 180): shadow falls north
    hm = calibrate(rgb, rel, np.zeros_like(rel), None, CalibParams(gsd=gsd, sun_elevation=el, sun_azimuth=180))
    assert hm.info["shadow_anchors"] == 1
    assert abs(hm.info["anchors"][0]["h"] - 12.0) < 1.0
    assert hm.mode == "above_ground"
    assert abs(float(np.median(hm.ndsm[90:110, 90:110])) - 12.0) < 1.0


def test_relative_when_no_scale():
    rgb = np.full((64, 64, 3), 120, np.uint8)
    rel = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    hm = calibrate(rgb, rel, np.zeros_like(rel), None, CalibParams())
    assert hm.mode == "relative" and hm.height_unit == "relative"


def test_fit_scale_rejects_outlier():
    anchors = [{"r": 0.1 * i, "h": 10.0 * i, "weight": 1.0} for i in range(1, 8)]
    anchors.append({"r": 0.5, "h": 500.0, "weight": 1.0})
    fit = fit_scale(anchors)
    assert abs(fit["scale"] - 100.0) < 1e-6
    assert fit["n_inliers"] == 7


class _RampModel:
    """Fake depth model: a smooth function of position, so tiles must align seamlessly."""
    is_ai, size, name = True, 518, "ramp"

    def infer(self, rgb):
        h, w = rgb.shape[:2]
        return (np.linspace(0, 1, w)[None, :] * np.ones((h, 1)) * 3.0 + 7.0).astype(np.float32)


def test_tiles_have_no_seams():
    rgb = np.zeros((900, 1300, 3), np.uint8)
    rel, unc = estimate_relative_height(rgb, _RampModel(), tta=False)
    jumps = np.abs(np.diff(rel, axis=1)).max()
    assert jumps < 0.02


def test_heuristic_model_runs():
    rgb = (np.random.default_rng(1).random((300, 400, 3)) * 255).astype(np.uint8)
    rel, unc = estimate_relative_height(rgb, HeuristicDepthModel())
    assert rel.shape == (300, 400) and 0 <= rel.min() and rel.max() <= 1


def _town(dtm, n=300, ndsm_rel=None, unit="m"):
    """A grid of 10 m tall houses on the given ground (None: no DEM)."""
    from depthwizard.calibrate import HeightModel
    b = np.zeros((n, n), bool)
    for r in range(20, n - 20, 40):
        for c in range(20, n - 20, 40):
            b[r:r + 20, c:c + 20] = True
    ndsm = np.where(b, 10.0, 0.0).astype(np.float32)
    z = np.zeros((n, n), np.float32)
    rel = (ndsm / 10.0 if ndsm_rel is None else ndsm_rel).astype(np.float32)
    return HeightModel(rel=rel, unc_rel=z, terrain_rel=z, ndsm_rel=rel, ndsm=ndsm, dtm=dtm,
                       dsm=(ndsm if dtm is None else dtm + ndsm), sigma=z, confidence=z.astype(np.uint8),
                       building=b, tree=np.zeros_like(b), water=np.zeros_like(b), shadow=np.zeros_like(b),
                       mode="absolute" if dtm is not None else "above_ground", height_unit=unit, info={})


def test_landslides_never_start_on_buildings():
    from depthwizard.hazards import build_terrain, run_scenario
    n, gsd = 300, 1.0
    yy = np.mgrid[0:n, 0:n][0].astype(np.float32)
    dtm = yy * gsd * np.tan(np.radians(30))                # a 30 degree hillside
    hm = _town(dtm)
    rgb = np.full((n, n, 3), 140, np.uint8)
    T = build_terrain(hm, gsd, rgb)
    assert T.hillside.mean() > 0.4                         # the mountain is recognised
    assert not (T.hillside & T.building).any()
    res = run_scenario(T, "landslide", 1.0)
    source = (res.overlay[..., 0] == 220) & (res.overlay[..., 1] == 40)
    assert source.any() and not (source & T.building).any()


def test_flat_town_without_dem_has_no_landslides():
    from depthwizard.hazards import build_terrain, run_scenario
    n, gsd = 300, 1.0
    rng = np.random.default_rng(0)
    hm = _town(None)
    hm.rel = hm.rel + rng.normal(0, 0.002, hm.rel.shape).astype(np.float32)   # model noise on flat ground
    T = build_terrain(hm, gsd, np.full((n, n, 3), 140, np.uint8))
    assert T.hillside.mean() < 0.02
    res = run_scenario(T, "landslide", 1.0)
    assert res.payload["stats"]["danger_area_share"] < 0.02
    assert "almost flat" in res.payload["summary"][0]
