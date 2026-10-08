# What was taken from the original DepthWizard repo

Source: [Aaronpaul2006/DepthWizard-Single-View-Height-Estimation-and-3D-Flythrough](https://github.com/Aaronpaul2006/DepthWizard-Single-View-Height-Estimation-and-3D-Flythrough)
(SIH26175, the desktop/Tauri version). This web app keeps that project's accuracy work and adds
shadow physics and disaster routing on top.

## Ported

| Original | Here | Notes |
|---|---|---|
| `depth/stitch.py`, `depth/tiling.py`: global pass, 518 px tiles at **50% overlap**, **robust median/MAD + trimmed alignment**, **Hann windows** that sum to 1 | `depthwizard/depth.py` (`align_params`, `tile_origins`, `taper`, `window2d`, `seam_ratio`) | Same algorithm; ONNX Runtime instead of PyTorch |
| `calibrate/fit.py` **Method B**: terrain from the DEM + model detail high-passed at the DEM's blur (σ = k·dem_res/pixel), so a surface DEM's blurred buildings are not counted twice | `calibrate.py` (`lowpass`, surface-DEM ground) | Demo: bias +2.07 m → 0.00 m, RMSE 4.11 → 3.48 m |
| `calibrate/fit.py` **DEM-band scale** (the DEM's own resolvable detail regressed on the model's, gated by correlation) | `calibrate.py` (`band_scale`) | Used when there are no shadows or GCPs |
| `calibrate/fit.py` **GCP scale** + `calibrate/gcp.py` CSV (`x,y,height_m` or `lon,lat,height_m`) | `jobs.read_gcps_csv`, `calibrate.py` | Upload in "Image details" |
| `calibrate/dem.py`: **Copernicus GLO-30 auto-fetch + cache**, DEM source detection (CartoDEM tile names), datum table | `depthwizard/dem.py` | GeoTIFFs get metres without uploading a DEM |
| `calibrate/dem.py` + `scripts/fetch_geoid.py`: **CartoDEM (ellipsoidal) → EGM2008** with the NGA geoid grid | `dem.py`, `scripts/fetch_geoid.py` | Namchi: N = −43.8 m, as measured in the original |
| `evals/datum_check.py`: CartoDEM vs Copernicus offset | `dem.datum_offset_vs_copernicus` | Automatic fallback when the geoid grid is missing |
| `evals/metrics.py`: **NMAD**, **offset-free RMSE** | `validate.py` | Shown in Expert → Validation |
| `configs/default.yaml` + type-checked loader ("all tunables in one file") | `configs/default.yaml`, `depthwizard/config.py` | |
| `io/read.py`: nodata/alpha masks, downsample with a warning | `scene.py` | Large inputs averaged down to `input.max_long_side` (2048) |
| `scripts/fetch_demo_namchi.py`: **real Namchi WorldView-3 scene** (Maxar ODP, CC BY-NC 4.0) | `scripts/fetch_demo_namchi.py` | Also saves sun/view angles from the STAC metadata. "Real: Namchi" button |
| Viewer: **fly mode** (pointer lock, WASD) | `frontend/view3d.js` | F to start, Esc to stop |
| Viewer: **DEM only** toggle (key B) | `view3d.setHeights`, `/heightmap?which=dem` | Shows what the model adds to the 30 m DEM |
| Viewer: **slope** surface | layer `slope` | Same colour ramp |
| API: finished jobs restored from disk; viewer terrain switcher; `?job=` link | `jobs._save_job/_load_job`, "Previous results" | Results live in `data/jobs/` |
| ARCHITECTURE.md plan, cut-list #2: **OSM buildings** (footprints + `height` / `building:levels`) | `depthwizard/osm.py` | Was never built in the original. Overpass fetch (cached) or uploaded GeoJSON |

## New here (not in the original)

* Shadow-measured building heights, with **terrain-slope correction** and **off-nadir lean correction**
  (the original listed off-nadir lean as an unsolved failure case; Namchi is 26° off-nadir).
* Shadow-first fusion; roof segmentation; water-body detection.
* Flood / landslide / earthquake escape routing, refuges, choke points, printable plan.
* Web app on CPU (≈0.8 GB) instead of the 4.7 GB CUDA desktop build.

## Not ported, and why

* **GAMUS fine-tune** (`scripts/finetune_gamus.py`): the original's own log says it was better on GAMUS
  but not on the LiDAR split, so the app kept the stock model. Same decision here.
* **NAIP + 3DEP LiDAR evaluation harness** (`evals/`, 72 tiles): needs USGS downloads and a GPU-hours
  budget; its published table (REPORT.md) stays the reference for US data. Indian validation is the
  next step (ICESat-2 / GEDI / field survey, see IMPROVEMENT_PLAN.md).
* **Tauri desktop shell**: replaced by the browser app (`run_local.bat`).
