"""End-to-end: synthetic town through the full pipeline and the HTTP API.

Uses the heuristic fallback model so the tests run without the ONNX download.
"""
import json
import time

import pytest
from fastapi.testclient import TestClient

from depthwizard import jobs as J
from depthwizard import server
from depthwizard.depth import DEFAULT_MODEL, HeuristicDepthModel
from depthwizard.synthetic import write_scene


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    server.manager = J.JobManager(model=HeuristicDepthModel())
    return TestClient(server.app)


@pytest.fixture(scope="module")
def demo_job(client):
    r = client.post("/api/demo")
    assert r.status_code == 200
    jid = r.json()["id"]
    for _ in range(600):
        m = client.get(f"/api/jobs/{jid}").json()
        if m["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert m["status"] == "done", m.get("error")
    return m


def test_demo_is_absolute_and_validated(demo_job):
    assert demo_job["mode"] == "absolute"
    assert demo_job["georeferenced"]
    assert demo_job["sun"]["source"] == "image metadata"
    assert demo_job["calibration"]["buildings_measured_by_shadow"] > 20
    v = demo_job["validation"]
    # the fallback heuristic is not a depth model: it may not gain, but must not lose much
    assert v["depthwizard"]["buildings"]["rmse"] < 1.05 * v["dem_only"]["buildings"]["rmse"]


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="ONNX model not downloaded")
def test_ai_model_beats_dem_on_demo_town():
    jm = J.JobManager()
    d = J.DATA_DIR / "demo"
    if not (d / ".v3").exists():
        write_scene(d)
        (d / ".v3").touch()
    job = jm.run_sync("demo", str(d / "image.tif"), str(d / "dem.tif"), str(d / "truth.tif"), {"osm_fetch": False})
    v = job.validation
    assert v["depthwizard"]["all"]["rmse"] < v["dem_only"]["all"]["rmse"]
    assert v["depthwizard"]["buildings"]["rmse"] < 0.8 * v["dem_only"]["buildings"]["rmse"]


@pytest.mark.parametrize("hazard", ["flood", "landslide", "earthquake"])
def test_scenarios(client, demo_job, hazard):
    jid = demo_job["id"]
    p = client.post(f"/api/jobs/{jid}/scenario", json={"hazard": hazard}).json()
    assert p["summary"] and p["zones"]
    assert p["stats"]["buildings_total"] > 0
    r = client.post(f"/api/jobs/{jid}/route", json={"u": 0.7, "v": 0.45}).json()
    assert "reachable" in r
    assert client.get(f"/api/jobs/{jid}/layer/hazard.png").status_code == 200


def test_exports(client, demo_job):
    jid = demo_job["id"]
    client.post(f"/api/jobs/{jid}/scenario", json={"hazard": "flood", "level": 4})
    gj = client.get(f"/api/jobs/{jid}/export/evacuation.geojson").json()
    kinds = {f["properties"]["kind"] for f in gj["features"]}
    assert {"safe_zone", "danger_zone"} <= kinds
    lon, lat = next(f for f in gj["features"] if f["geometry"]["type"] == "Point")["geometry"]["coordinates"]
    assert 88 < lon < 89 and 27 < lat < 27.3                # near Namchi
    assert client.get(f"/api/jobs/{jid}/export/evacuation.kml").text.startswith("<?xml")
    assert b"Evacuation plan" in client.get(f"/api/jobs/{jid}/export/report.html").content
    tif = client.get(f"/api/jobs/{jid}/export/dsm.tif")
    assert tif.status_code == 200 and tif.content[:2] in (b"II", b"MM")


def test_point_and_profile(client, demo_job):
    jid = demo_job["id"]
    p = client.get(f"/api/jobs/{jid}/point?u=0.5&v=0.5").json()
    assert p["unit"] == "m" and "lonlat" in p
    prof = client.get(f"/api/jobs/{jid}/profile?u0=0.1&v0=0.1&u1=0.9&v1=0.9").json()
    assert len(prof["surface"]) == 200 and prof["distance_unit"] == "m"


def test_rejects_bad_upload(client):
    r = client.post("/api/jobs", files={"image": ("x.exe", b"abc", "application/octet-stream")})
    assert r.status_code == 400
    r = client.post("/api/jobs", files={"image": ("a.png", b"abc", "image/png")}, data={"params": json.dumps({"gsd": -1})})
    assert r.status_code == 422


def test_flood_starts_from_detected_water(client, demo_job):
    jid = demo_job["id"]
    shares = []
    for level in (0, 3, 8):
        p = client.post(f"/api/jobs/{jid}/scenario", json={"hazard": "flood", "level": level}).json()
        assert p["water_bodies"], "the demo river should be detected"
        shares.append(p["stats"]["danger_area_share"])
    assert shares[0] < 0.05                  # level 0: only the river itself
    assert shares[0] < shares[1] < shares[2]  # the flood grows outward with the level
