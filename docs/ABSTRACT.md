# DepthWizard: abstract

Smart India Hackathon 2026 · Problem statement SIH26175 · ISRO

Free elevation data for India, such as CartoDEM, SRTM and Copernicus GLO-30, is posted at about 30 m. At that resolution hills and valleys appear, but individual houses, trees and streets do not. Disaster planners need exactly that missing detail. They need to know which buildings will flood, which are tall enough to shelter on, which streets may be blocked by debris, and which way people should walk to reach safety. LiDAR and stereo surveys can supply it, but they are slow and costly. AI depth models, for their part, return unitless numbers.

DepthWizard is a web application. It turns a single satellite image (PNG, JPG or GeoTIFF) into a digital surface model and then into an evacuation plan. The pipeline works as follows:

* **Relative height.** Depth Anything V2 runs on overlapping tiles that are aligned and blended without seams. Test-time augmentation provides a per-pixel uncertainty.
* **Metric building heights.** Heights are measured physically from shadows: shadow length × (tangent of the sun elevation + terrain slope along the shadow). The slope comes from the DEM, which corrects the bias that sloping ground causes in hill towns.
* **Fusion.** A shadow-first step keeps measured heights. The AI estimate is used only as far as it agrees with those measurements.
* **Terrain.** The DEM provides the terrain after an automatic datum correction (−46 m for CartoDEM at Namchi).
* **Honest labels.** Every output is labelled as absolute metres, metres above ground or relative 0–1, with an error estimate. No metres are claimed without a scale source.

On these heights the system models three hazards:

* **Flood:** Height Above Nearest Drainage, with tall buildings as vertical-evacuation refuges.
* **Landslide:** susceptibility and run-out.
* **Earthquake:** debris reach proportional to building height.

It then computes a walking-time field with Tobler's hiking function and Dijkstra's algorithm. The result is escape arrows, safe zones with capacity, choke points, a route from any tapped location, and a plain-language summary. Plans can be exported as a printable report, KML/GeoJSON for phones, or GeoTIFF.

On a synthetic Namchi-like town with known truth, DepthWizard reduced height error on buildings and trees by 39% compared with the 30 m DEM alone (RMSE 7.0 m vs 11.5 m). The slope correction lowered shadow-height error from 5.5 m to 3.1 m. The application installs in about 0.8 GB, needs no GPU, processes a 1024-pixel scene in about 8 seconds on a laptop CPU, and serves an entire office over the local network through a browser. Next steps are validation on Indian terrain using ICESat-2 and GEDI LiDAR and a field survey, integration of the road network, and a pilot with a district disaster management office.

**Keywords:** digital surface model, monocular depth estimation, shadow-based height, disaster management, evacuation routing, HAND, CartoDEM
