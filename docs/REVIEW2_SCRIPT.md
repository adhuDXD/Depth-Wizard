# DepthWizard: Review 2 script

Smart India Hackathon 2026 · SIH26175 · ISRO · App version 2.2

**Total time:** about 12 minutes (8 min talk, 4 min live demo), then questions.

> *Italic notes are stage directions.* Everything else is spoken.

---

## Part A: Opening (45 s)

> Good morning. We are team DepthWizard, problem statement SIH26175 from ISRO.
>
> At Review 1 we showed a desktop prototype that put an AI depth model on top of a 30-metre DEM. The feedback was clear:
> - it was a wrapper around a pretrained model;
> - it had a 4.7 GB install that needed a gaming GPU;
> - it had no Indian test data;
> - it gave no practical decisions for disaster managers.
>
> Today we will show you what we changed. Every change answers one of those points.

---

## Part B: What changed since Review 1 (2 min)

> **One: heights now come from physics, not just AI.**
> A building's shadow tells you its height. Height equals shadow length times the tangent of the sun's elevation. We correct this for two real-world effects:
> - sloping ground, because Indian hill towns are on slopes;
> - leaning buildings in off-nadir images.
>
> The AI now gives the *shape*; physics gives the *metres*.
>
> **Two: we tested on a real Indian scene.** It is Namchi, in South Sikkim: a 0.3-metre WorldView-3 image from the Maxar Open Data Program.
>
> **Three: it now gives disaster decisions.**
> - flood, landslide and earthquake scenarios;
> - escape arrows for every street, and safe zones with capacity;
> - refuge buildings, choke points, and a printable evacuation plan.
>
> **Four: it now runs anywhere.**
> - a browser app that installs in about 0.8 GB, down from 4.7 GB;
> - no GPU needed: about 40 seconds for a 2048-pixel scene on an ordinary laptop;
> - one office PC can serve the whole team over the local network.
>
> **Five: we kept the best of our Review-1 engineering.** That includes:
> - the tile stitching and the Method B calibration;
> - Copernicus DEM download;
> - the CartoDEM datum correction;
> - the fly-through viewer.

---

## Part C: How it works (3 min)

*Show the pipeline slide.*

> The pipeline has seven stages.
>
> **Stage 1: Read the image.**
> - PNG, JPG or GeoTIFF; GeoTIFFs are reprojected to UTM so every pixel is in metres;
> - we read the sun angle, off-nadir angle and capture time from the metadata.
>
> **Stage 2: AI relative height.**
> - Depth Anything V2 Small runs through ONNX Runtime on the CPU;
> - a global pass, then 518-pixel tiles with 50% overlap;
> - each tile is aligned to the global pass with a robust median/MAD fit, then blended with Hann windows so there are no seams;
> - flipping and rotating the image gives us an uncertainty map.
>
> **Stage 3: Find objects.**
> - ground is separated from objects by morphological opening;
> - vegetation is green *and* textured, so green roofs aren't counted as trees;
> - roofs are found as uniform-colour patches;
> - water is blue, flat and large;
> - shadows are dark regions.
>
> **Stage 4: Measure metres.** We use four sources, in order of trust:
> 1. Ground control points;
> 2. Shadows, with corrections for slope and building lean;
> 3. OpenStreetMap tagged heights;
> 4. The DEM band: the detail the 30 m DEM itself can see, regressed against the AI, and used only if they agree.
>
> If none is available, we label the output *relative* and claim no metres.
>
> **Stage 5: Terrain.**
> - the Copernicus 30 m DEM is downloaded automatically;
> - CartoDEM's ellipsoidal heights are converted to sea level with the EGM2008 geoid: that's minus 43.8 metres at Namchi;
> - because the DEM is a surface model, its blurred buildings are removed before we add our sharp ones, so nothing is counted twice.
>
> **Stage 6: Hazards.**
> - **Flood:** height above the nearest water body, spreading out from detected rivers.
> - **Landslide:** slope, water convergence and bare soil.
> - **Earthquake:** debris reaching half a building's height.
>
> **Stage 7: Escape routes.**
> - walking speed comes from Tobler's hiking function, so uphill is slower;
> - one Dijkstra search from all safe zones gives every spot its direction and walking time;
> - choke points come from how many people pass each cell.

---

## Part D: Results (1 min)

*Show the results slide.*

> On our test town with known ground truth:
> - the 30-metre DEM alone is off by **10.7 metres** on buildings and trees;
> - DepthWizard is off by **5.3 metres**: **half the error**;
> - across the whole scene, error drops from **5.7 to 3.5 metres**.
>
> Our correction for surface DEMs removed a 2-metre double-counting bias.
>
> On the real Namchi image, the app measured about 200 buildings from their shadows. The median is 12 metres, which is about 4 storeys and plausible for Namchi. Namchi has no ground truth, so we do not claim an accuracy number there.

---

## Part E: Live demo (4 min)

*Before starting: `run_local.bat` is running; Namchi is already processed and in "Previous results".*

1. **Open "Real: Namchi" from Previous results.**
   > This is the real 0.3 m image. The card says: Copernicus DEM, EGM2008 sea level, 26-degree off-nadir corrected, about 200 heights from shadows.
2. **Switch to Expert → Heights layer.**
   > Taller buildings in yellow, lower in purple.
3. **Click a building with Inspect point.**
   > Its height in metres, its uncertainty, and its slope.
4. **Switch to 3D. Press B.**
   > This is what the 30-metre DEM alone knows: no buildings. Press B again: this is what DepthWizard adds.
5. **Press F to fly.** Fly over the town with WASD, then press Esc.
6. **Open the demo town, choose Flood, and drag the slider from 0 to 6 m.**
   > At zero only the river shows. Watch the water spread out from it. The arrows show the way out, and the cyan buildings are tall refuges.
7. **Tap a house in the flood zone.**
   > The pink line is that household's walking route and its time.
8. **Click Earthquake, then Print evacuation plan.**
   > This one page goes to the ward officer.

*If the demo fails, play the backup video.*

---

## Part F: Honest limits and next steps (1 min)

> Our limits:
> - Our accuracy figure is from a synthetic town. Indian ground truth is next.
> - In dense real towns, adjacent buildings merge, and some small houses are missed.
> - Routes use open ground, not yet the road network.
>
> Next steps:
> 1. **Validate on Indian terrain** with free NASA spaceborne LiDAR (ICESat-2 and GEDI), a field survey, and a Cartosat stereo pair.
> 2. **Add the road network**, and OSM footprints where Indian towns are mapped.
> 3. **Test with a district disaster office**, and record a usability score.

## Part G: Close (15 s)

> DepthWizard tells a disaster officer where the water goes, which streets get blocked, and which way to run. It works from one satellite image, on one office PC, with heights you can check. Thank you.

---

# Appendix 1: Software used

| Layer | Software | Version / note | Used for |
|---|---|---|---|
| Language | **Python** | 3.11–3.13 | Whole backend |
| AI model | **Depth Anything V2 Small** | ONNX export, ~99 MB, Apache-2.0 | Relative depth from one image |
| AI runtime | **ONNX Runtime** | CPU (CUDA optional) | Runs the model without PyTorch |
| Numerics | **NumPy**, **SciPy** | `scipy.ndimage`, `scipy.sparse.csgraph` | Arrays, filters, distance transforms, Dijkstra |
| Image processing | **OpenCV** (headless) | | Resizing, morphology, Sobel, connected components, colour spaces |
| Images | **Pillow** | | PNG/JPG reading |
| Geospatial | **GDAL** via **rasterio** | | GeoTIFF read/write, reprojection, rasterising footprints |
| Coordinates | **PROJ** via **pyproj** | | UTM zones, lat/lon conversion |
| Config | **PyYAML** | `configs/default.yaml` | All tunable settings in one file |
| Web server | **FastAPI** + **Uvicorn** | | REST API, serves the web app |
| Front end | **HTML / CSS / JavaScript** (ES modules) | no build step | Guided and Expert interface |
| 2D map | **HTML Canvas** | | Map, overlays, arrows, routes |
| 3D | **Three.js** r170 | OrbitControls, PointerLockControls | 3D terrain, fly mode |
| Testing | **pytest**, **Playwright** (Chromium) | 28 automated tests | Unit, API and browser tests |
| Packaging | **Docker**, `run_local.bat` / `.sh` | | One-click install and start |
| Version control | **Git / GitHub** | | `adhuDXD/Depth-Wizard` |

## Data used

| Data | Source | Licence | Used for |
|---|---|---|---|
| Satellite image (demo) | Maxar Open Data Program, WorldView-3, Namchi, 14 March 2022, 0.3 m | CC BY-NC 4.0 | Real Indian test scene |
| Elevation | **Copernicus GLO-30** (auto-download, cached) | free | Terrain in metres (EGM2008) |
| Elevation | **CartoDEM V3R1** (Bhuvan, upload) | NRSC licence | Indian national DEM |
| Geoid | **NGA EGM2008** 2.5′ grid | public domain | CartoDEM ellipsoid → sea level |
| Buildings | **OpenStreetMap** (Overpass or GeoJSON) | ODbL | Footprints and tagged heights |
| Test town | Our own synthetic generator | ours | Ground truth for accuracy tests |

---

# Appendix 2: Methods and algorithms

## Image and geometry
| Method | What it does | Where |
|---|---|---|
| UTM reprojection | Puts the image on a metric grid so 1 pixel = known metres | `scene.py` |
| Percentile stretch (2–98 %) | Converts 16-bit satellite bands to 8-bit | `scene.py` |
| Area-average downsampling | Shrinks huge images without aliasing | `scene.py` |
| NOAA solar position algorithm | Sun elevation/azimuth from date, time and location | `sun.py` |

## AI depth
| Method | What it does | Where |
|---|---|---|
| Depth Anything V2 (ViT encoder + DPT decoder) | Relative inverse depth: closer = taller from above | `depth.py` |
| Global pass + tiling, 50 % overlap | Full-resolution detail without memory limits | `depth.py` |
| Robust affine alignment (median/MAD start + 2 trimmed least-squares refits) | Makes every tile agree with the global pass | `depth.py` |
| Hann (sin²) window blending | Seam-free stitching; windows sum to 1 | `depth.py` |
| Test-time augmentation (flips, 90° rotation) | Per-pixel uncertainty | `depth.py` |

## Object detection
| Method | What it does | Where |
|---|---|---|
| Morphological opening (ground surface) | Separates terrain from raised objects (nDSM) | `calibrate.py` |
| Excess-Green index + local texture | Trees = green **and** leafy texture | `calibrate.py` |
| Roof segmentation (LAB Sobel gradient + uniform patches) | Whole roofs, clean outlines | `calibrate.py` |
| Shadow detection (Otsu + median threshold, morphology) | Finds shadows | `calibrate.py` |
| Water detection (HSV hue/saturation + flatness + area, per region) | Rivers, lakes; rejects blue roofs | `calibrate.py` |
| Connected-component labelling | One label per building | OpenCV |

## Metric heights
| Method | Formula / idea | Where |
|---|---|---|
| **Shadow height** | h = L · tan(sun elevation) | `calibrate.py` |
| **Slope correction** (ours) | h = L · (tan θ + s), s = terrain slope along the shadow | `calibrate.py` |
| **Off-nadir lean correction** (ours) | visible L = h · (1/(tan θ + s) − tan(off-nadir) · cos(view az − sun az)) | `calibrate.py` |
| Shadow outlier rejection | Drops measurements with a large spread or above median + 5·MAD | `calibrate.py` |
| Robust scale fit | Weighted median ratio, MAD rejection, leave-one-out error | `calibrate.py` |
| Spatial scale field | Nadaraya–Watson kernel smoothing of the scale across the scene | `calibrate.py` |
| **Shadow-first fusion** (ours) | Measured buildings keep their height; others = median + ρ·(AI − median) | `calibrate.py` |
| GCP scale (Method B) | Least squares of height above DEM on model detail | `calibrate.py` |
| **DEM-band scale** (Method B) | DEM detail regressed on model detail at the DEM's scale, used if r ≥ 0.1 | `calibrate.py` |
| OSM heights | `height` tag, or `building:levels` × 3 m + 1 m | `osm.py` |

## Terrain and datum
| Method | What it does | Where |
|---|---|---|
| Copernicus tile lookup + caching | Downloads the right 1°×1° tiles once | `dem.py` |
| Bilinear resampling + light Gaussian (0.4 cell) | DEM on the image grid without facets | `dem.py` |
| **EGM2008 geoid conversion** | H = h − N (N = −43.8 m at Namchi) | `dem.py` |
| Datum check vs Copernicus | Median difference; removed if large and uniform | `dem.py` |
| **Surface-DEM correction** (Method B) | ground = DEM − lowpass(heights, σ = k·DEM res / pixel) | `calibrate.py` |

## Hazards and routing
| Method | What it does | Where |
|---|---|---|
| **Priority-Flood** depression filling (Barnes et al. 2014) | Every cell drains to the edge | `hydrology.py` |
| **D8 flow direction** + flow accumulation | Water paths and streams | `hydrology.py` |
| **HAND** (Height Above Nearest Drainage; Nobre et al. 2011) | Flood depth for a water rise | `hydrology.py` |
| Height above **detected water body** (ours) | Flood starts from rivers/lakes in the image | `hydrology.py` |
| Binary propagation (connectivity) | Water only reaches connected cells | `hazards.py` |
| Landslide susceptibility | 0.6·slope + 0.25·flow convergence + 0.15·bare ground, plus a D8 run-out | `hazards.py` |
| Earthquake debris reach | Euclidean distance transform; reach = 0.5 × building height | `hazards.py` |
| Vertical-evacuation refuges | Buildings ≥ 9 m with roofs ≥ 3 m above water | `hazards.py` |
| **Tobler's hiking function** | v = 6·exp(−3.5·\|tan θ + 0.05\|) km/h | `routing.py` |
| **Multi-source Dijkstra** (reversed graph, virtual sink) | Time-to-safety and next step for every cell | `routing.py` |
| Flow accumulation on the route tree | People passing each cell → choke points | `routing.py` |
| Shelter capacity (Sphere standard, 3.5 m²/person) | Over-capacity warnings | `hazards.py` |
| Population from building volume | floors × footprint ÷ 15 m² per person | `hazards.py` |

## Validation
| Metric | Meaning |
|---|---|
| RMSE, MAE, bias, Pearson r | Standard accuracy |
| **NMAD** (1.4826 × median absolute deviation) | Robust to a few extreme outliers |
| **Offset-free RMSE** | Error after removing a constant datum offset |
| Object mask from the reference | Buildings and trees scored without using our own detections |

---

# Appendix 3: Likely questions

**Q: Isn't this still just Depth Anything?**
A: No. The AI gives the shape only. Metres come from shadow physics with slope and lean corrections, GCPs, OSM or the DEM band. The app measures how much to trust the AI on every image (ρ), and on our test town ρ was about zero, so it relied on physics. Escape routing is entirely our own.

**Q: How accurate is it?**
A: On the synthetic town, error on buildings and trees is 5.3 m vs 10.7 m for the DEM alone. On real Indian ground truth we have no figure yet; ICESat-2 and GEDI validation is the next step.

**Q: What if the image is off-nadir?**
A: Roofs shift away from the satellite and hide or lengthen their shadows. We correct with the off-nadir angle and view azimuth from the metadata. At Namchi (26°) this changes heights by more than 50%.

**Q: CartoDEM heights look 46 m wrong?**
A: CartoDEM is ellipsoidal. We convert it with the EGM2008 geoid (N = −43.8 m at Namchi). Without the grid file, the app measures the offset against Copernicus instead.

**Q: Does it need internet?**
A: Only the first time: to install packages, download the model, and fetch the DEM tile. After that it runs fully offline.

**Q: What hardware does it need?**
A: An ordinary laptop CPU: about 40 s for a 2048-pixel scene. A GPU is optional.
