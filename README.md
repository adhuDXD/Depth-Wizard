# DepthWizard

**One satellite image → a metric 3D surface model → a disaster evacuation plan.**

DepthWizard is a web app. Load a satellite image (PNG, JPG or GeoTIFF) and it:

1. estimates the height of every building, tree and terrain feature (Depth Anything V2 + shadow geometry + DEM),
2. says honestly how reliable each height is (metres vs relative, with an error figure),
3. simulates a **flood, landslide or earthquake** and draws **escape arrows**, safe zones, refuge buildings and choke points,
4. exports a printable evacuation plan, GeoJSON/KML for phones, and GeoTIFF surfaces.

It runs on one ordinary office PC (CPU is enough). Everyone else on the network only needs a browser.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_model.py          # Depth Anything V2 Small, ONNX, ~99 MB (optional but recommended)
python -m depthwizard.server              # http://localhost:8000
python -m depthwizard.server --host 0.0.0.0   # serve the whole office LAN
```

Click **Try demo town** for a synthetic Namchi-like hill town with known ground truth. Or with Docker:

```bash
docker build -t depthwizard . && docker run -p 8000:8000 depthwizard
```

Without the model file the app still works, but it uses a clearly labelled non-AI fallback.

**Footprint:** about 0.8 GB installed (Python libraries ~735 MB + model 99 MB). No PyTorch, CUDA or GPU needed, and processing a 1024 px scene takes ~8 s on a laptop CPU. The previous desktop build was 4.7 GB.

---

## How it works

```
image ──► reproject to UTM ──► Depth Anything V2 (global pass + 518 px tiles,
  │                             least-squares aligned, feather-blended, flip/rotate TTA)
  │                                   │ relative height + uncertainty
  │                                   ▼
  ├──► shadows ─► building height = shadow length × tan(sun elevation)   ◄── sun angles from
  │                                   │                                       metadata / date-time
DEM ─► + datum offset ─► terrain (DTM)│◄── GCPs (fix DEM bias, add scale)
                                      ▼
            shadow-first fusion ─► DSM + per-pixel uncertainty + confidence (green/amber/red)
                                      ▼
   HAND hydrology · slope · debris reach ─► hazard zones ─► Dijkstra (Tobler walking speed)
                                      ▼
            escape arrows · safe zones · refuges · choke points · shelter capacity
```

### Height estimation (`depthwizard/depth.py`, `calibrate.py`)

| Step | What it does |
|---|---|
| Tiled AI depth | Global pass plus overlapping tiles. Each tile is fitted to the global pass, so there are no seams. Test-time augmentation gives an uncertainty map. |
| Ground / object split | A morphological opening separates terrain from objects. Objects split into buildings and trees using an excess-green vegetation index. |
| **Shadow measurement** | Each building's shadow is walked along the sun azimuth. Height = length × GSD × (tan(elevation) + **terrain slope along the shadow**). The slope comes from the DEM: on a hillside, shadows falling uphill are shorter and ones falling downhill longer. Truncated shadows and ones that hit image edges are discarded. |
| **Shadow-first fusion** | Buildings with a measured shadow get that physical height. The others get the AI estimate, **shrunk toward typical measured heights by how well the AI agrees with the measurements** (ρ). The app reports ρ, so an AI that doesn't track heights isn't trusted to invent them. |
| Robust scale | Median ratio with MAD outlier rejection, a smooth spatial field once there are ≥ 12 anchors, and leave-one-out error. |
| Terrain | DEM resampled to the image grid, plus the vertical datum offset (e.g. −46 m for CartoDEM at Namchi), plus a bias fix from ground GCPs. |

**Honesty modes.** Every output is labelled with one of these modes:

| Mode | When |
|---|---|
| `absolute` | DEM + a height scale. Metres above sea level. |
| `above_ground` | Pixel size + sun angles (or GCPs or a typical-height estimate), but no DEM. Metres above local ground. |
| `terrain_only` | DEM, but no way to scale buildings. |
| `relative` | Nothing gives a scale. Output is 0–1 and **no metres are claimed**. |

### Escape routing (`depthwizard/hydrology.py`, `hazards.py`, `routing.py`)

* **Flood.** Priority-flood depression filling, D8 flow, accumulation and **HAND** (height above nearest drainage). A water rise of *h* floods every cell with HAND < *h*. **Buildings whose roof stays ≥ 3 m above the water and that are ≥ 9 m tall become vertical-evacuation refuges.** This uses the estimated heights.
* **Landslide.** Susceptibility from slope, flow convergence and bare ground, plus a downslope run-out along D8 paths.
* **Earthquake.** Debris from a building can reach ~0.5 × its height. Streets inside that reach are penalised, and open ground beyond it becomes an assembly area. This also uses the estimated heights.
* **Routing.** 8-connected grid. Cost = distance ÷ **Tobler hiking speed** (uphill/downhill aware) × hazard penalty. One Dijkstra run from a virtual "safety" node on the reversed graph gives every cell its time-to-safety and next step.
* **People.** Population is estimated from building volume (floors × footprint ÷ 15 m²/person). Shelter capacity uses the Sphere 3.5 m²/person standard. Flow through the route tree gives the choke points.

### Demo-town validation (synthetic, known truth; `Expert → Validation`)

| Surface RMSE | Whole scene | Buildings & trees |
|---|---|---|
| 30 m DEM only (datum-corrected) | 6.09 m | 11.54 m |
| DepthWizard, Depth Anything V2 + shadows | **4.14 m** | **7.01 m** |
| DepthWizard, non-AI fallback | 6.32 m | 11.45 m |

Shadow-measured building heights reach 3.1 m RMSE (r = 0.91) against the truth with the slope correction, and 5.5 m (r = 0.78) without it. On this synthetic render, the AI's within-building values do **not** track height (ρ ≈ 0), so the app leans on shadows and says so. Real Indian validation still needs ICESat-2 / GEDI / field data; see `docs/IMPROVEMENT_PLAN.md`.

---

## Using it

* **Guided mode** (default), in three steps: load image → read the height-model card → pick a hazard and move the slider. Tap the map for the walking route from any spot. **Print evacuation plan** opens a one-page report.
* **Expert mode** adds:
  * layers: surface, building heights, confidence, drainage, error, time-to-safety
  * an elevation profile and a point inspector
  * a validation upload (reference DSM GeoTIFF)
  * GeoTIFF downloads
  * the table of measured anchors
* **Image details**: pixel size, sun elevation/azimuth or acquisition time, DEM datum offset, typical building height. GeoTIFF metadata fills these in automatically when present (`SUN_ELEVATION` / `SUN_AZIMUTH` tags).

## HTTP API

| Method | Path | |
|---|---|---|
| GET | `/api/health` | model + hazard definitions |
| POST | `/api/jobs` | multipart: `image`, optional `dem`, `reference`, `params` (JSON) |
| POST | `/api/demo` | synthetic demo town |
| GET | `/api/jobs/{id}` | status / progress / calibration / validation |
| GET | `/api/jobs/{id}/layer/{image,height,buildings,confidence,drainage,error,hazard,time}.png` | map layers |
| GET | `/api/jobs/{id}/heightmap` | float32 grid for 3D (`X-Width`, `X-Height` headers) |
| POST | `/api/jobs/{id}/scenario` | `{"hazard": "flood", "level": 3}` → summary, zones, arrows, choke points |
| POST | `/api/jobs/{id}/route` | `{"u": 0.5, "v": 0.5}` → path to safety |
| GET | `/api/jobs/{id}/point`, `/profile` | inspector and elevation profile |
| POST | `/api/jobs/{id}/validate` | reference DSM upload |
| GET | `/api/jobs/{id}/export/{dsm.tif,ndsm.tif,uncertainty.tif,evacuation.geojson,evacuation.kml,report.html}` | exports |

## Tests

```bash
python -m pytest
```

The test run covers unit tests (sun position, Tobler, HAND, routing, shadow height, robust scale, seam-free tiling) and the end-to-end API on the demo town. The AI-vs-DEM test runs when the model file is present.

## Limitations (stated in the app too)

* The shadow method assumes near-nadir views, flat ground at the building foot and flat roofs. Off-nadir lean is not corrected.
* Roads are not used. Routing is over open ground and streets as seen in the image. An OSM road layer is a planned addition.
* There is no fine-tuning yet. The DEM-conditioned retraining in the plan is future work.
* It is a pre-disaster planning aid, not live navigation.

## Layout

```
depthwizard/   scene (I/O, UTM), depth (ONNX tiles), calibrate (shadows, fusion), hydrology (HAND),
               routing (Dijkstra), hazards (scenarios), jobs (pipeline, exports), server (FastAPI),
               synthetic (demo town), validate, render, sun
frontend/      index.html, app.js, map2d.js (canvas map), view3d.js (Three.js), vendor/three
tests/         pytest suite
docs/          improvement plan answering the review
```
