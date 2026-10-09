"""The "is this an image of land?" check.

Measured during development with the AI model (scripts not shipped, images not redistributable):
  satellite scenes, 18 Maxar Open Data events (incl. Namchi) + demo town: 0 of 72 rejected, 3 warned
    (dense towns at 0.3 m, still processed)
  other pictures (people, pets, food, rooms, streets, ground-level landscapes, paintings,
    documents, logos, charts): 73 of 93 rejected, 8 more warned
  not caught: close-up textures (grass, gravel, brick) and oblique photos from a plane, which
    look like ground seen from above
The checks below need no model download: colour, flat-area, document and darkness tests.
"""
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image

from depthwizard import jobs as J
from depthwizard import server
from depthwizard.depth import DEFAULT_MODEL, HeuristicDepthModel, OnnxDepthModel
from depthwizard.inputcheck import check_image
from depthwizard.synthetic import write_scene


@pytest.fixture(scope="module")
def satellite(tmp_path_factory):
    folder = tmp_path_factory.mktemp("town")
    write_scene(folder)
    with rasterio.open(folder / "image.tif") as src:
        return np.moveaxis(src.read([1, 2, 3]), 0, -1)


def _drawing():
    img = np.full((400, 600, 3), 255, np.uint8)
    cv2.rectangle(img, (50, 50), (250, 200), (30, 120, 220), -1)
    cv2.circle(img, (420, 250), 100, (240, 60, 40), -1)
    return img


def _document():
    img = np.full((800, 600, 3), 245, np.uint8)
    for i in range(30):
        cv2.putText(img, "Lorem ipsum dolor sit amet", (30, 40 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (20, 20, 20), 2)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _vivid_photo():
    rng = np.random.default_rng(0)
    hsv = np.zeros((300, 300, 3), np.uint8)
    hsv[..., 0] = rng.integers(0, 180, (30, 30)).repeat(10, 0).repeat(10, 1)
    hsv[..., 1] = 220
    hsv[..., 2] = rng.integers(120, 255, (300, 300))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def test_satellite_image_is_accepted(satellite):
    r = check_image(satellite, HeuristicDepthModel())
    assert r.ok, r.message
    tile = satellite[100:612, 300:812]                        # a crop, as a user might upload
    assert check_image(tile).ok


@pytest.mark.parametrize("make,reason", [(_drawing, "flat"), (_document, "document"),
                                         (_vivid_photo, "vivid"),
                                         (lambda: np.zeros((200, 200, 3), np.uint8), "dark")])
def test_non_map_images_are_rejected(make, reason):
    r = check_image(make())
    assert not r.ok and reason in r.reasons
    assert "not look like a satellite" in r.message


@pytest.mark.skipif(not Path(DEFAULT_MODEL).exists(), reason="AI model not downloaded")
def test_real_satellite_scene_passes_the_depth_test():
    namchi = J.DATA_DIR / "demo" / "namchi" / "namchi.tif"
    if not namchi.exists():
        pytest.skip("Namchi scene not downloaded")
    with rasterio.open(namchi) as src:
        rgb = np.moveaxis(src.read([1, 2, 3], out_shape=(3, 1024, 1024)), 0, -1)
    r = check_image(rgb, OnnxDepthModel(DEFAULT_MODEL))
    assert r.ok and r.scores["depth_edge"] < 0.08


def test_upload_rejected_then_forced():
    server.manager = J.JobManager(model=HeuristicDepthModel())
    client = TestClient(server.app)
    buf = io.BytesIO()
    Image.fromarray(_document()).save(buf, "PNG")

    def run(force):
        r = client.post("/api/jobs", files={"image": ("notes.png", buf.getvalue(), "image/png")},
                        data={"params": json.dumps({"force": force})})
        assert r.status_code == 200
        jid = r.json()["id"]
        for _ in range(300):
            m = client.get(f"/api/jobs/{jid}").json()
            if m["status"] in ("done", "error"):
                return m
            time.sleep(0.1)
        raise AssertionError("job did not finish")

    m = run(False)
    assert m["status"] == "error" and m["rejected"] and "document" in m["error"]
    m = run(True)
    assert m["status"] == "done" and not m["rejected"]
    assert any("Process anyway" in n for n in m["calibration"]["notes"])
