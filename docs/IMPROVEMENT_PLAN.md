# DepthWizard v2: Improvement Plan

This plan answers the review critique point by point. It adds two things to the project:

1. **Better height estimation**, meaning metric heights that you can defend and that come with uncertainty.
2. **Disaster escape routing**, meaning evacuation directions that depend on the heights we estimate.

The goal is to stop being "a wrapper around Depth Anything" and become **a height-aware disaster-planning tool, validated on Indian data**.

---

## 0. The new one-line pitch

> **DepthWizard turns a single satellite image into a metric 3D surface model and tells a disaster manager where the water goes, which streets get blocked, and which way each neighbourhood should run.**

Every feature below has to support that sentence. If a feature doesn't support it, move it to "expert mode" or drop it.

---

## 1. Height estimation: from "depth picture" to defensible metres

### 1.1 What's wrong today

| Today | Why it's weak |
|---|---|
| Depth Anything V2 (DA-V2) outputs relative, affine-invariant depth | The model was trained mostly on ground-level perspective photos, not nadir satellite views. "Depth" is not "height". |
| Fusion = DEM + Gaussian high-pass of the network output | This is plain filtering (the reviewer is right). A single global scale smears errors everywhere. |
| Scale comes only from GCPs | A typical user has no GCPs, so they get a "relative DSM" they can't use. |
| No uncertainty | The user has no way to tell which heights to trust. |

### 1.2 The new height pipeline (four contributions you can defend)

```
 Image ──► [A] DEM-conditioned nDSM network ──► nDSM (height above ground, metres-ish)
   │                                              │
   ├──► [B] Shadow-based height anchors ──────────┤
   │        (sun angle + shadow length)           ▼
   │                                    [C] Robust, spatially varying calibration
 DEM ──► ground mask ──► DTM ─────────────────────┤   (RANSAC/Huber over anchors)
                                                  ▼
                                   DSM = DTM + calibrated nDSM
                                                  │
                              [D] Uncertainty map (TTA / MC dropout)
```

#### [A] DEM-conditioned nDSM network (the architectural change)

- **Change the target.** Predict *height above ground* (nDSM) instead of depth. GAMUS already provides nDSM labels.
- **Change the input.** Feed the network RGB plus a DEM-derived channel (normalised DEM, or slope/hillshade). On the DA-V2 Small encoder, replace the first patch-embedding layer with one that takes 4–5 channels. Initialise the RGB weights from the pretrained model and the new channels at zero, so training starts from the original behaviour.
- **Loss.** Use a combination of:
  - scale-and-shift-invariant loss (keeps the pretrained behaviour)
  - gradient-matching loss (sharp building edges)
  - an L1 loss on metric nDSM where labels exist
  - a small penalty for predicting non-zero height on pixels labelled ground or water. This directly fixes the "hallucinated relief on flat farmland" failure.
- **Novelty statement for the deck:** *"We condition a foundation depth model on the coarse national DEM and retarget it to predict height above ground. The DEM tells the network where the terrain is, so the network only has to learn what stands on it."*

#### [B] Shadow-based metric anchors (scale without GCPs)

Physics: on flat ground in a near-nadir image, **building height ≈ shadow length × tan(sun elevation)**.

- **Sun elevation and azimuth** come from the image metadata. Cartosat, Resourcesat, Sentinel-2 and PlanetScope all record them. If the metadata is missing, compute them from acquisition time and lat/lon (pvlib / pysolar).
- **Shadows**: detect them with a threshold on a shadow index (low brightness with high blue/(R+G) ratio), then clean up with morphology.
- **Shadow length**: measure it along the sun azimuth, starting from the building edge. Use the building mask from the network's nDSM > 2 m, or from OSM footprints.
- **Result**: dozens to hundreds of independent metric height samples per scene. The camera doesn't need to be calibrated and the user doesn't need GCPs.
- **Big consequence**: a *plain PNG* can get metric heights if the user enters only the pixel size (GSD) and the date/time. Label these outputs **"shadow-calibrated, ±X m"**, not "absolute", so the honesty principle is kept.

```python
# sketch: shadow height for one building
import numpy as np
def shadow_height(shadow_mask, bldg_edge_px, sun_az_deg, sun_el_deg, gsd_m):
    # walk from the building edge in the direction away from the sun
    d = np.deg2rad(sun_az_deg + 180)
    step = np.array([-np.cos(d), np.sin(d)])      # (row, col) in image coords
    p, n = np.array(bldg_edge_px, float), 0
    while shadow_mask[int(p[0]), int(p[1])]:
        p += step; n += 1
    return n * gsd_m * np.tan(np.deg2rad(sun_el_deg))
```

#### [C] Robust, spatially varying calibration (replaces the Gaussian trick)

Collect **anchors** from every source we have and fit `nDSM_metric = a(x,y) · nDSM_pred + b(x,y)`:

| Anchor source | What it gives | Weight |
|---|---|---|
| DEM at ground pixels (nDSM≈0, NDVI, water) | Terrain level (b) | high |
| Shadow heights | Building heights (a) | medium |
| User GCPs | Absolute points | highest |
| OSM `building:levels` × ~3 m | Building heights | low |
| ICESat-2 / GEDI LiDAR footprints in the scene (optional) | Ground + canopy heights | high |

- Fit with **RANSAC or Huber regression** so bad shadows and leaning buildings get rejected.
- Make `a` and `b` vary smoothly. Fit per 256 px block, then interpolate. This handles depth-model drift across a large scene.
- Report the **residual RMSE on held-out anchors** as an honest, built-in accuracy number for *every* output. This also replaces the "Validate" tool that needed a reference DSM the user doesn't have.

#### [D] Uncertainty map

- Use test-time augmentation: 4 rotations × 2 flips gives 8 passes. The **per-pixel standard deviation** is the uncertainty.
- Combine it with the anchor residuals into a **traffic-light confidence layer** (green / amber / red).
- Automatically flag water, deep shadow and very flat regions as low confidence. This is your "confidence flags" promise, now implemented.

### 1.3 Validation on Indian data (the biggest feasibility gap)

You do **not** need Indian airborne LiDAR. Use these instead:

| Source | Covers India? | What you validate |
|---|---|---|
| **ICESat-2 ATL08/ATL03** (NASA spaceborne LiDAR, free) | Yes | Ground elevation and canopy height along tracks; ATL03 photons show building tops |
| **GEDI L2A** (spaceborne LiDAR, 25 m footprints, free) | Yes (±51.6° latitude) | Ground elevation, canopy height (forests, Western Ghats, NE) |
| **Google Open Buildings 2.5D Temporal** | Yes | Building heights over Indian cities (check licence) |
| **GHS-BUILT-H / WSF-3D** | Yes (coarse, 90–100 m) | Average building height per block, as a sanity check |
| **Cartosat-1/3 stereo → DSM** via NASA Ames Stereo Pipeline (RPC) | Yes, if NRSC gives you a stereo pair | A full independent DSM. Ask your ISRO mentor for one stereo pair. |
| **Your own field survey** | Yes | 30–50 buildings on campus or in your town: laser rangefinder, clinometer app, or storey count × floor height. Cheap, real, and judges love it. |

**Test sites.** Pick one site per failure mode:

- Namchi or Gangtok (hills)
- Chengannur or Assam (flood plain, 2018/2022 floods)
- Bengaluru or Mumbai (dense urban)
- Western Ghats (forest)
- Punjab (flat farmland)

**Report a table** of RMSE / MAE / correlation for:

1. DEM only
2. DA-V2 + Gaussian (your v1)
3. v2 without shadows
4. v2 full

This ablation table *is* your proof of novelty, because it shows each contribution actually helps.

---

## 2. New feature: disaster escape-route direction

This is the part that turns "a cool 3D viewer" into "a decision tool". It also justifies the height work: **evacuation planning needs building heights**.

### 2.1 Inputs

- DSM, DTM, slope, building mask and building heights (from section 1)
- Road network: an offline OSM extract (Geofabrik India `.pbf`). If OSM is missing, use roads segmented from the image.
- Optional population: WorldPop, or **building volume ÷ floor area per person** (another use of height)
- Hazard type and scenario parameter, chosen by the user

### 2.2 Hazard models (simple, physically grounded, explainable)

| Hazard | Model | Where height matters |
|---|---|---|
| **Flood** | **HAND** (Height Above Nearest Drainage) from the DTM with pysheds/WhiteboxTools. For a water rise of *h* m, every cell with HAND < *h* is flooded. Slider: 0.5–10 m. | **Vertical evacuation**: buildings with roof height > water level + margin, and ≥ 3 storeys, become refuges |
| **Landslide** | Susceptibility from slope (> ~30°), curvature, aspect, NDVI and drainage proximity, plus a downslope runout buffer | Roads cut below steep slopes are marked unsafe |
| **Earthquake** | **Debris-blocking**: a street is likely blocked if building height > ~street width (collapse debris reaches about half the building height outward) | Needs per-building height. This is the novel part. |
| **Cyclone / storm surge** | Elevation threshold plus distance from the coast | Elevation |

### 2.3 Routing algorithm

1. **Safe zones** = (high-HAND, open, flat areas) ∪ (designated shelters, schools and hospitals from OSM) ∪ (tall, sturdy buildings for floods).
2. **Graph** = road network. Each edge has a cost:

   `cost = length / walking_speed(slope) × hazard_multiplier`

   - Walking speed comes from **Tobler's hiking function**: `v = 6·exp(−3.5·|tanθ + 0.05|)` km/h. In hill towns, uphill and downhill speeds are very different.
   - Edges inside the hazard zone are **removed** (flood) or **heavily penalised** (landslide/debris).
3. Run a **multi-source Dijkstra from all safe zones on the reversed graph**. One pass gives every node its time-to-safety and next step. That is an **evacuation direction field** for the whole town.
4. Where roads are sparse (hill towns), also run an off-road cost-distance with `skimage.graph.MCP_Geometric` on a slope-cost raster.
5. **Bottlenecks**: compute edge betweenness weighted by population. These are the roads that most people depend on and that need traffic marshals.
6. **Shelter capacity**: compare the people assigned to each safe zone against its capacity (area ÷ m² per person; cite the Sphere standard). Flag overloaded shelters.

```python
# sketch: evacuation direction field
import networkx as nx
G = build_road_graph(osm_pbf, dtm)          # edges carry length & slope
for u, v, d in G.edges(data=True):
    if flooded(d["geom"], hand, water_rise): d["w"] = float("inf")
    else: d["w"] = d["len"] / tobler(d["slope"]) * hazard_mult(d)
R = G.reverse()
R.add_node("SAFE"); R.add_edges_from(("SAFE", s, {"w": 0}) for s in safe_nodes)
dist, path = nx.single_source_dijkstra(R, "SAFE", weight="w")
next_hop = {n: p[-2] for n, p in path.items() if len(p) > 2}   # arrow direction
```

### 2.4 What the user sees

- **Arrows on every street** pointing toward safety, coloured by time-to-safety (green < 5 min, amber < 15, red > 15).
- **"What if water rises X m?"** slider that recomputes live. HAND is computed once, so this is cheap.
- **Plain-language summary**, for example: *"At a 2 m flood: 412 buildings inside the flood zone, about 3,100 people. Nearest high ground: Ward 4 playground, 600 m north-east, 9 min on foot. Choke point: MG Road bridge. Send marshals."*
- **Exports**: printable A3 PDF evacuation map, GeoJSON/KML for phones, CSV of at-risk buildings. These work offline and can be shared on WhatsApp.
- **Honesty note on screen**: *"Pre-disaster planning map from a single image. Not a live navigation system."*

---

## 3. Point-by-point response to the critique

| Critique | Action | Evidence to show |
|---|---|---|
| **Novelty**: "wrapper, Gaussian blur" | Contributions [A] DEM-conditioned nDSM network, [B] shadow self-calibration, [C] robust spatially varying fusion with uncertainty, plus height-aware evacuation (vertical refuges, debris-blocked streets). Drop "Gaussian blur" and "datum correction" from the novelty slide; they become plumbing. | Ablation table (section 1.3) |
| **Complexity** | The same four contributions, plus the hazard and routing engine | Architecture diagram and loss equation on one slide |
| **Feasibility in India** | ICESat-2, GEDI, Open Buildings 2.5D, Cartosat stereo, own field survey. 5 Indian test sites. | Per-terrain error table, Indian sites only |
| **4.7 GB package, CUDA fragility** | Export DA-V2 to **ONNX**, quantise to INT8 (~25–30 MB model), run with `onnxruntime` (CPU by default, CUDA/DirectML optional). Remove PyTorch from the runtime. Target an installer **< 500 MB**. Add a **LAN/web mode**: one office PC runs FastAPI and staff use a browser. | Benchmark table: CPU i5 vs GPU, seconds per km², RAM, installer size |
| **Practicability / UX** | **Guided mode** (default): 3 steps, *Load image → Pick hazard → Get evacuation map*. 2D map first, 3D optional. **Expert mode** keeps the 3D flythrough, heatmaps and profiles. Replace WASD with orbit/pan controls plus a "fly to" search. | Screenshots, 90-s demo video, usability test |
| **User testing** | Test with 5–8 non-technical people (NSS/NCC volunteers, a municipal engineer, a teacher). Use the System Usability Scale (SUS) questionnaire plus task time ("find the nearest safe place"). | SUS score, quotes, what you changed after the test |
| **"Validate needs a reference DSM"** | Built-in self-validation against held-out anchors and DEM, plus bundled ICESat-2 tracks | Accuracy number printed on every output |
| **Scale of impact** | Batch CLI (`depthwizard batch district/*.tif`), tile-parallel processing, district mosaics | Throughput: km² per hour on a laptop. How long a full district takes. |
| **No pilot / partners** | Email the **SDMA** (e.g. Sikkim SDMA for Namchi), the district disaster management officer, or your city municipal corporation. Show them a map of *their* town. Even a feedback session or a support letter counts. | Letter of support or feedback notes |
| **Cost-benefit** | Find **sourced** costs, for example aerial LiDAR tenders on CPPP/GeM (₹/km²) and commercial stereo DSM prices. Compare with your cost: free imagery + laptop time. **Never put an unsourced number on a slide.** | One table with citations |
| **Sustainability / business** | Open-core: free tool, plus paid services (district onboarding, custom calibration, training, annual maintenance). Grant routes: ISRO RESPOND, DST/NM-ICPS, Startup India Seed Fund, iDEX for defence use. ONNX + pinned dependencies + CI reduce breakage from library updates. | One business-model slide |
| **"Atmanirbhar" is a slogan** | Replace it with concrete facts: *runs fully offline on ISRO data (Cartosat, CartoDEM, Bhuvan), no foreign cloud, open-source stack, validated on Indian sites.* | — |
| **Failure gallery** | Build it: 6–8 cases (water, deep shadow, leaning tower, flat farmland, cloud, snow), each with image, output, confidence layer and an explanation | One slide plus an appendix |
| **Backup video** | Record a 2-minute screen capture of the full flow (image → DSM → flood slider → evacuation arrows → PDF export) | Link or QR code on the last slide |
| **Requirement matrix missing** | Table mapping each problem-statement line → feature → status → evidence | One slide |
| **Work plan has no risks** | Add a risk and fallback for each phase (section 5) | — |

---

## 4. Deck fixes (quick wins, do these first)

- [ ] Remove "TITLE PAGE" and the red "[ADD SOURCED STAT…]" placeholder.
- [ ] Use one font family (sans-serif) everywhere, including slide 1.
- [ ] Use one colour system: **blue = data/terrain, orange = model output, red = hazard, green = safe**. Use it in the slides and in the app.
- [ ] Use high-resolution exports for slide 2 images (render at 2× and crop instead of shrinking).
- [ ] Set a title hierarchy: titles ≥ 32 pt, body ≥ 18 pt, captions ≥ 14 pt.
- [ ] Use at most 1 idea per slide. Split slide 2. Fill the empty half of slide 6 with the requirement matrix.
- [ ] Put a "before/after" hero image on slide 1: satellite image → 3D model with evacuation arrows.
- [ ] Research slide: for each reference, add a line on *what we took from it and what we changed*.
- [ ] Impact slide: replace generic claims with numbers from your own runs (buildings at risk, minutes to safety, error in metres).
- [ ] Team slide: add GitHub/portfolio links.

Suggested slide order:

1. Hook: an Indian flood or landslide image and the question "where should people go?"
2. Problem: 30 m DEM can't see buildings, and plans need heights
3. Solution overview: one diagram
4. Height engine: [A]–[D]
5. Indian validation results and ablation
6. Escape routing
7. Live demo / screenshots
8. UX for non-experts and user-test results
9. Deployment (size, CPU, LAN)
10. Impact and cost-benefit (sourced)
11. Business and sustainability
12. Risks and failure gallery
13. Requirement matrix
14. Team and roadmap

---

## 5. Work plan with risks

| Phase | Work | Risk | Fallback |
|---|---|---|---|
| **Week 1** | Deck fixes; ONNX export + INT8; CPU benchmark; field survey of 30 buildings | Quantisation hurts accuracy | Ship FP16 ONNX (~50 MB) instead |
| **Week 2** | Shadow anchors + robust calibration [B][C]; TTA uncertainty [D] | Shadow detection noisy in dense areas | Use only isolated buildings as anchors; RANSAC rejects outliers |
| **Week 2–3** | HAND flood model + road graph + Dijkstra direction field; guided-mode UI | OSM roads incomplete in small towns | Road segmentation from the image, or off-road cost-distance |
| **Week 3** | Indian validation (ICESat-2, GEDI, Open Buildings, field data); ablation table | No Cartosat stereo access | Rely on spaceborne LiDAR + field survey; say so honestly |
| **Week 3–4** | DEM-conditioned fine-tune [A] on GAMUS (+ Indian samples) | GPU time and convergence | Keep v1 network + [B][C][D]; report [A] as in-progress with partial curves |
| **Week 4** | Landslide + earthquake debris models; PDF/KML export; user test; SDMA outreach; demo video | No reply from agencies | Use municipal engineers, NSS/NCC, or a college disaster club as test users |

**Order of priority if time is short:** deck fixes → escape routing (flood) → shadow calibration + uncertainty → Indian validation → ONNX → fine-tune [A].

Routing and validation are what the judges will remember. The fine-tune is the riskiest item, so do it last.

---

## 6. Answers to have ready for the judges

- **"Isn't this just Depth Anything?"**: "Depth Anything gives a unitless picture. Our contributions are: conditioning it on the national DEM to predict height above ground; physics-based shadow calibration that gives metres without GCPs; uncertainty-aware fusion; and using those heights for evacuation planning (vertical refuges and debris-blocked streets). The ablation table shows each part reduces error on Indian sites."
- **"How do you get metres from one image?"**: "On its own, you can't, and we say so. Metres come from the DEM, GCPs, or the sun-shadow geometry recorded in the image metadata. Every output carries its calibration source and an error estimate."
- **"Why would a disaster officer use this?"**: "Because they get an evacuation map of their own town in minutes, offline, on an office PC, from imagery ISRO already has. No LiDAR survey needed."
- **"How accurate is the escape route?"**: "It's a planning aid. Routes depend on the road data and the hazard scenario. We show confidence and recommend field verification, the same way a hazard zonation map is used today."
