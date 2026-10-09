"""DepthWizard web server: REST API + the browser front end.

Run:  uvicorn depthwizard.server:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000 (or this machine's LAN address from other PCs).
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from . import __version__, jobs as J
from .hazards import HAZARDS
from .synthetic import write_scene

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
UPLOADS = J.DATA_DIR / "uploads"
MAX_UPLOAD = 1024 ** 3   # 1 GB
ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gtif", ".geotiff"}


class GCP(BaseModel):
    u: float = Field(ge=0, le=1)
    v: float = Field(ge=0, le=1)
    elevation: float


class JobParams(BaseModel):
    gsd: float | None = Field(default=None, gt=0, le=1000, description="metres per pixel (plain images)")
    sun_elevation: float | None = Field(default=None, ge=0, le=90)
    sun_azimuth: float | None = Field(default=None, ge=0, le=360)
    acquired: str | None = Field(default=None, description="ISO date-time (UTC) of acquisition")
    geoid_offset_m: float | None = 0.0
    building_height_prior_m: float | None = Field(default=None, gt=0, le=500)
    gcps: list[GCP] = []
    tta: bool = True
    off_nadir: float | None = Field(default=None, ge=0, le=60)
    view_azimuth: float | None = Field(default=None, ge=0, le=360, description="ground -> satellite")
    dem_kind: str | None = Field(default=None, pattern="^(surface|terrain)$")
    osm_fetch: bool = True
    force: bool = Field(default=False, description="skip the 'is this an image of land?' check")


class ScenarioReq(BaseModel):
    hazard: str
    level: float | None = None


class PointReq(BaseModel):
    u: float = Field(ge=0, le=1)
    v: float = Field(ge=0, le=1)


manager = J.JobManager(store=J.DATA_DIR / "jobs")
app = FastAPI(title="DepthWizard", version=__version__)


@app.middleware("http")
async def no_stale_frontend(request, call_next):
    """Always serve the current front end, so an updated copy is never hidden by the browser cache."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _job(job_id: str, ready: bool = True) -> J.Job:
    job = manager.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    if ready and job.status != "done":
        raise HTTPException(409, f"Job is {job.status}")
    if ready:
        manager.ensure_loaded(job)
    return job


async def _save(upload: UploadFile, folder: Path, stem: str, allowed: set | None = None) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in (allowed or ALLOWED):
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Use PNG, JPG or GeoTIFF.")
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{stem}{suffix}"
    size = 0
    with dest.open("wb") as f:
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, "File too large")
            f.write(chunk)
    return str(dest)


@app.get("/api/health")
def health():
    return {"version": __version__, "model": manager.model.name, "model_is_ai": manager.model.is_ai,
            "hazards": HAZARDS}


@app.get("/api/jobs")
def list_jobs():
    return [{"id": j.id, "name": j.name, "status": j.status, "created": j.created}
            for j in sorted(manager.jobs.values(), key=lambda j: j.created, reverse=True)]


@app.post("/api/jobs")
async def create_job(image: UploadFile = File(...), dem: UploadFile | None = File(None),
                     reference: UploadFile | None = File(None), gcps: UploadFile | None = File(None),
                     osm: UploadFile | None = File(None), params: str = Form("{}")):
    try:
        p = JobParams(**json.loads(params or "{}"))
    except (ValidationError, json.JSONDecodeError) as e:
        raise HTTPException(422, f"Bad parameters: {e}") from e
    folder = UPLOADS / uuid.uuid4().hex
    img = await _save(image, folder, "image")
    dem_path = await _save(dem, folder, "dem") if dem is not None and dem.filename else None
    ref_path = await _save(reference, folder, "reference") if reference is not None and reference.filename else None
    extra = {}
    if gcps is not None and gcps.filename:
        extra["gcps_csv"] = await _save(gcps, folder, "gcps", {".csv", ".txt"})
    if osm is not None and osm.filename:
        extra["osm_path"] = await _save(osm, folder, "osm", {".geojson", ".json"})
    job = manager.submit(Path(image.filename or "image").name, img, dem_path, ref_path, p.model_dump() | extra)
    return J.meta(job, manager.model)


NAMCHI = J.DATA_DIR / "demo" / "namchi" / "namchi.tif"


@app.get("/api/demos")
def demos():
    return {"synthetic": True, "namchi": NAMCHI.exists()}


@app.post("/api/demo")
def create_demo(scene: str = "synthetic"):
    if scene == "namchi":
        if not NAMCHI.exists():
            raise HTTPException(404, "Run `python scripts/fetch_demo_namchi.py` once to download the Namchi scene.")
        job = manager.submit("Namchi, Sikkim (Maxar WorldView-3, 2022)", str(NAMCHI), None, None,
                             JobParams().model_dump())
        return J.meta(job, manager.model)
    folder = J.DATA_DIR / "demo"
    if not (folder / "truth.tif").exists() or not (folder / ".v3").exists():
        write_scene(folder)
        (folder / ".v3").touch()
    job = manager.submit("Demo: synthetic hill town (Namchi-like)", str(folder / "image.tif"),
                         str(folder / "dem.tif"), str(folder / "truth.tif"),
                         JobParams(osm_fetch=False).model_dump())
    return J.meta(job, manager.model)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _job(job_id, ready=False)
    if job.status == "done":
        manager.ensure_loaded(job)
    return J.meta(job, manager.model)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = _job(job_id, ready=False)
    manager.jobs.pop(job_id, None)
    if job.stored is not None or (J.DATA_DIR / "jobs" / job_id).exists():
        shutil.rmtree(J.DATA_DIR / "jobs" / job_id, ignore_errors=True)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/layer/{name}.png")
def layer(job_id: str, name: str):
    job = _job(job_id)
    try:
        data = J.layer_png(job, name)
    except KeyError:
        raise HTTPException(404, f"Layer '{name}' is not available")
    return Response(data, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/jobs/{job_id}/heightmap")
def heightmap(job_id: str, which: str = "dsm"):
    try:
        data, w, h = J.heightmap(_job(job_id), which=which)
    except KeyError:
        raise HTTPException(404, "No DEM for this result")
    return Response(data, media_type="application/octet-stream",
                    headers={"X-Width": str(w), "X-Height": str(h), "Access-Control-Expose-Headers": "X-Width, X-Height"})


@app.post("/api/jobs/{job_id}/scenario")
def scenario(job_id: str, req: ScenarioReq):
    job = _job(job_id)
    if req.hazard not in HAZARDS:
        raise HTTPException(400, f"Unknown hazard. Choose one of {list(HAZARDS)}")
    return manager.scenario(job, req.hazard, req.level)


@app.post("/api/jobs/{job_id}/route")
def route(job_id: str, req: PointReq):
    try:
        return manager.route(_job(job_id), req.u, req.v)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@app.get("/api/jobs/{job_id}/point")
def point(job_id: str, u: float, v: float):
    return J.point_info(_job(job_id), u, v)


@app.get("/api/jobs/{job_id}/profile")
def profile(job_id: str, u0: float, v0: float, u1: float, v1: float):
    return J.profile(_job(job_id), u0, v0, u1, v1)


@app.post("/api/jobs/{job_id}/validate")
async def validate(job_id: str, reference: UploadFile = File(...)):
    job = _job(job_id)
    if not job.scene.georeferenced:
        raise HTTPException(400, "Validation needs a georeferenced (GeoTIFF) image.")
    path = await _save(reference, UPLOADS / uuid.uuid4().hex, "reference")
    try:
        return manager.attach_reference(job, path)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/jobs/{job_id}/export/{kind}")
def export(job_id: str, kind: str, u: float | None = None, v: float | None = None):
    job = _job(job_id)
    stem = Path(job.name).stem.replace(" ", "_")[:40] or "depthwizard"
    try:
        if kind in ("dsm.tif", "ndsm.tif", "uncertainty.tif"):
            return Response(J.dsm_geotiff(job, kind[:-4]), media_type="image/tiff",
                            headers={"Content-Disposition": f'attachment; filename="{stem}_{kind}"'})
        if kind == "evacuation.geojson":
            return JSONResponse(J.geojson(job), headers={
                "Content-Disposition": f'attachment; filename="{stem}_evacuation.geojson"'})
        if kind == "evacuation.kml":
            return Response(J.kml(job), media_type="application/vnd.google-earth.kml+xml",
                            headers={"Content-Disposition": f'attachment; filename="{stem}_evacuation.kml"'})
        if kind == "report.html":
            route_ = manager.route(job, u, v) if u is not None and v is not None and job.scenario else None
            return HTMLResponse(J.report_html(job, route_))
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    raise HTTPException(404, "Unknown export")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND), name="frontend")


def main():
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="DepthWizard web app")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to serve the whole LAN")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
