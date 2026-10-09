# DepthWizard: final presentation script

Smart India Hackathon 2026 · SIH26175 · ISRO · App version 2.4

**Total time:** about 15 minutes: 9.5 minutes of talk and a 5-minute live demo. Questions follow.

> *Italic notes are stage directions.* Everything else is spoken. Change the speaker names to your own team.

| Part | Slide | Speaker | Time |
|---|---|---|---|
| 1 | Title and hook | Speaker 1 | 0:45 |
| 2 | The problem | Speaker 1 | 1:00 |
| 3 | Our solution in one picture | Speaker 1 | 0:45 |
| 4 | How we get real heights | Speaker 2 | 1:30 |
| 5 | The system understands the ground | Speaker 2 | 1:00 |
| 6 | From heights to escape routes | Speaker 3 | 1:00 |
| 7 | Results | Speaker 2 | 1:00 |
| 8 | Live demo | Speaker 3 + 4 | 5:00 |
| 9 | Built for a district office | Speaker 4 | 0:45 |
| 10 | Impact and adoption | Speaker 4 | 0:45 |
| 11 | Honest limits and roadmap | Speaker 1 | 0:45 |
| 12 | Close | Speaker 1 | 0:20 |

---

## Before you start (checklist)

* Start `run_local.bat` 10 minutes early. The badge at the top must say **v2.4.0**.
* Open **Real: Namchi** once so it is already in **Previous results**. Also open **Try demo town** once.
* Put a phone selfie or any ordinary photo on the desktop for the input-check demo.
* Have the backup screen recording ready in case the demo fails.
* Close every other program and turn notifications off.

---

## Slide 1: Title and hook (45 s), Speaker 1

> Good morning. We are team DepthWizard, and we are working on problem statement SIH26175 from ISRO.
>
> Imagine a cloudburst over a hill town in Sikkim. The district officer has a satellite image of the town and a free 30-metre elevation map. Neither one can tell them which houses will flood, which buildings are tall enough to shelter on, or which way people should walk.
>
> DepthWizard answers those questions from **one satellite image**, on **one ordinary office computer**, in **under a minute**.

---

## Slide 2: The problem (60 s), Speaker 1

> India has free elevation maps: CartoDEM from ISRO, plus SRTM and Copernicus. They have one problem: each value covers a 30-metre square. You can see hills and valleys, but **a house, a tree or a street simply disappears**.
>
> Disaster planning needs exactly that missing detail:
> - which buildings will flood;
> - which are tall enough to be a refuge;
> - which streets debris will block;
> - how long people need to walk to safety.
>
> LiDAR and drone surveys give that detail, but they are slow and expensive, and they don't exist for most Indian towns. AI depth models can guess shape from a photo, but they give numbers with **no units**: not metres.
>
> So the problem is: **from a single satellite image, produce a 3D model in real metres that a disaster officer can use.**

---

## Slide 3: Our solution in one picture (45 s), Speaker 1

*Show the pipeline diagram: image → heights → hazards → escape plan.*

> DepthWizard does four things.
>
> **One:** it reads the image and checks that it really is a view of land from above.
> **Two:** it builds a 3D model of every building, tree and slope, **in metres**, and says how far each height can be trusted.
> **Three:** it simulates a **flood, a landslide or an earthquake** on that 3D model.
> **Four:** it draws **escape arrows, safe zones, refuge buildings and choke points**, and prints a one-page evacuation plan.
>
> It is a web app. One office PC runs it, and everyone else just opens a browser.

---

## Slide 4: How we get real heights (90 s), Speaker 2

*This is the core idea. Speak slowly.*

> The AI model we use is Depth Anything V2. It is good at **shape**: what is higher than what. But it cannot tell metres. So we don't trust it for metres. **Metres come from physics.**
>
> **Shadows.** The sun casts a shadow, and the length of a building's shadow tells you its height:
> **height = shadow length × tan(sun elevation).**
> The sun's angle comes from the image metadata, or we compute it from the date, time and place.
>
> We added two corrections that matter in India:
> - **Slope.** In a hill town, a shadow falling uphill is shorter and one falling downhill is longer. We take the slope from the elevation map and correct for it. This cut our shadow-height error from 5.5 metres to 3.1.
> - **Lean.** Satellites often look from the side. Our Namchi image was taken 26 degrees off vertical, so buildings lean and hide part of their own shadow. We correct for that too. Our original project listed this as an unsolved problem.
>
> **Fusion.** Buildings with a measured shadow keep that measured height. For the others, we check how well the AI agrees with the measured ones, and **trust the AI only as much as it has earned**.
>
> **Ground.** The 30-metre elevation map gives the ground. We convert ISRO's CartoDEM to sea-level heights. At Namchi that is a 44-metre correction. We also remove the blurred buildings it already contains, so nothing is counted twice.
>
> Every result carries a label: metres above sea level, metres above ground, or "relative". **If we have no scale, we don't claim metres.**

---

## Slide 5: The system understands the ground (60 s), Speaker 2

*New since Review 2. Show the Land cover layer image and the "Invalid input" screenshot.*

> A hazard model is only as good as its understanding of the scene. Since Review 2 we added two things.
>
> **First, land cover.** Every part of the image is classified as one of:
> - building;
> - hillside or mountain slope;
> - vegetated slope;
> - flat ground;
> - water.
>
> Earlier, building walls looked like cliffs, and our landslide map put landslides **on rooftops**. Now:
> - slopes are measured on the bare ground under the buildings;
> - **a landslide can only start on a natural slope**;
> - a building in the path of the debris is marked **at risk**, in purple.
>
> On the real Namchi image, landslide cells on buildings went from about **5,600 to zero**.
>
> **Second, an input check.** If someone uploads a selfie, a street photo, a document or a logo, DepthWizard stops in half a second and says **"Invalid input: not an image of land"**, with the reason.
> - We tested it on **72 real satellite images from 18 disaster zones**: it rejected **none**.
> - Of **93 ordinary photos and documents**, it rejected **73**.
> - If it is ever wrong, there is a **Process anyway** button.

---

## Slide 6: From heights to escape routes (60 s), Speaker 3

> Now the heights become decisions.
>
> **Flood.** The app finds rivers and lakes in the image itself. The water level starts at **zero**, and as you raise it, the water **spreads outward from those rivers** and only reaches places it can actually flow to. Buildings at least three storeys tall whose roofs stay above the water become **refuges**: people can go up instead of out.
>
> **Landslide.** Risk comes from slope, water convergence and bare soil, on natural slopes only. The debris flows downhill for up to 100 metres.
>
> **Earthquake.** A collapsing building can throw debris about half its height into the street. That is why **building height** matters here.
>
> **Escape routes.**
> - Walking speed uses **Tobler's hiking function**, so uphill is slower.
> - One shortest-path search from all safe zones gives **every spot in town its direction and walking time**.
> - We count how many people pass each point. That gives the **choke points** where marshals are needed.
> - We check whether each safe zone has enough room, using the international Sphere standard of 3.5 square metres per person.

---

## Slide 7: Results (60 s), Speaker 2

*Show the results table.*

> We tested on a town where we know the true height of every point.
>
> | | Whole scene | Buildings and trees |
> |---|---|---|
> | 30 m elevation map alone | 5.7 m error | 10.7 m error |
> | **DepthWizard** | **3.5 m error** | **5.3 m error** |
>
> That is **half the error on buildings and trees**, which is the detail disaster planning needs.
>
> On the **real Namchi image** (0.3 m WorldView-3):
> - it found **618 buildings** and measured about **200 of them from their shadows**;
> - the median height was **12 metres, about four storeys**, which is plausible for Namchi;
> - it took **about 40 seconds on a laptop CPU**.
>
> Namchi has no ground-truth survey, so **we do not claim an accuracy number there**. That is our next validation step.

---

## Slide 8: Live demo (5 min), Speakers 3 and 4

*Speaker 3 drives the mouse; Speaker 4 narrates.*

1. **Input check: upload a selfie and click Build 3D model.**
   > Watch what happens with a photo that isn't land. It stops in a second: *"Invalid input: not an image of land. It looks like a photo taken from the ground."* No garbage output.
2. **Open Real: Namchi from Previous results.**
   > This is a real 0.3-metre satellite image of Namchi in South Sikkim. The card shows:
   > - the elevation map was fetched automatically and converted to sea level;
   > - the 26-degree lean is corrected;
   > - about 200 heights were measured from shadows.
3. **Click Expert, then Map layer → Heights. Then Inspect point and click a building.**
   > Taller buildings are yellow. The panel gives this building's height in metres, its uncertainty, and the slope of the ground.
4. **Map layer → Land cover.**
   > This is what the system recognised:
   > - red for buildings;
   > - brown for hillside;
   > - green for vegetation;
   > - beige for flat ground.
5. **Click Landslide.**
   > Red is where a slide can start: only on natural slopes, never on roofs. Orange is the debris path. Purple marks the buildings in that path. Arrows show the way out.
6. **Switch to 3D. Press B, then B again. Then press F and fly with WASD; Esc to stop.**
   > Press B: this is all the 30-metre map knows. No buildings at all. Press B again: this is what DepthWizard adds.
7. **Open the demo town. Click Flood and drag the slider from 0 to 6 m.**
   > At zero, only the river. Now the water spreads out from it. The cyan buildings are refuges.
8. **Tap a house inside the flood.**
   > This is that family's walking route and the minutes it takes.
9. **Click Print evacuation plan.**
   > One page for the ward officer. It also exports to KML, which opens on a phone in Google Earth or any map app that reads KML.

*If anything fails, say "let me show you the recording" and play the backup video. Don't debug live.*

---

## Slide 9: Built for a district office (45 s), Speaker 4

> At Review 1 our tool was a **4.7 GB** desktop app that needed a gaming GPU. Reviewers said a district office can't use that, and they were right.
>
> Now:
> - it installs in **0.8 GB** and needs **no GPU**;
> - one double-click on `run_local.bat` installs and starts it;
> - after the first setup it works **fully offline**: important when the network goes down during a disaster;
> - one office PC serves **everyone on the office network** through a browser.
>
> **Guided mode** is three steps for non-experts. **Expert mode** adds:
> - layers;
> - 3D fly-through;
> - elevation profiles;
> - validation against a reference;
> - GeoTIFF downloads.
>
> It has **37 automated tests**.

---

## Slide 10: Impact and adoption (45 s), Speaker 4

> **Who uses it:**
> - district and state disaster management authorities;
> - municipal planners;
> - rescue teams preparing before the monsoon.
>
> **Why they would use it:**
> - the inputs are free or already available to government: satellite imagery, CartoDEM from Bhuvan, Copernicus;
> - there is **no per-town survey cost** and no special hardware;
> - the output is a **plan they can print and act on**, not just a 3D picture.
>
> **How it reaches them:**
> - open-source core, deployed on existing office PCs;
> - training for district staff;
> - a natural fit as a planning layer alongside **Bhuvan and NDMA's** existing systems;
> - the same pipeline works for any Indian town with a satellite image.

---

## Slide 11: Honest limits and roadmap (45 s), Speaker 1

> We would rather tell you our limits than have you find them.
>
> **Limits:**
> - Our accuracy number comes from a test town; we don't yet have ground truth in India.
> - In very dense areas, neighbouring buildings can merge.
> - Routes use open ground and streets as seen in the image, not yet a road map.
> - The input check catches about four in five non-map pictures, not all.
>
> **Next steps:**
> 1. **Validate on Indian terrain** with free NASA spaceborne LiDAR (ICESat-2 and GEDI), a field survey of 30 to 50 buildings, and a Cartosat stereo pair.
> 2. **Add the road network** from OpenStreetMap, and population data.
> 3. **Pilot with one district disaster office**: build a plan for their own town and measure how easily non-experts use it.

---

## Slide 12: Close (20 s), Speaker 1

> From **one satellite image**, on **one office PC**, DepthWizard tells a disaster officer **where the water goes, where the hillside can slide, which streets get blocked, and which way to run**, with heights you can check.
>
> Thank you. We are happy to take your questions.

---

# Likely questions and answers

**Q: Isn't this just Depth Anything with a nice interface?**
A: No. The AI gives only the shape. The metres come from shadow physics with our slope and lean corrections, from ground control points, OpenStreetMap, or the elevation map itself. The app measures how far to trust the AI on every image. Land-cover recognition, the input check, the hazard models and the escape routing are all our own work.

**Q: How accurate is it?**
A: On the test town, error on buildings and trees is 5.3 m against 10.7 m for the elevation map alone, about half. Shadow heights alone reach about 3 m error. For real Indian terrain we have no number yet; validating with ICESat-2 and GEDI is our next step.

**Q: How does it know a building from a mountain?**
A: Buildings come from height above the local ground plus roof colour and shape. Vegetation comes from greenness and texture. Water comes from colour, flatness and size. Slope is measured on the bare ground under the buildings, so a wall is never mistaken for a cliff. Natural ground steeper than 15 degrees counts as hillside, and only there can a landslide start.

**Q: What if someone uploads a wrong picture?**
A: It is checked in half a second. Bright phone-photo colours, flat drawing colours, a page of text, a black image, or the depth pattern of a ground-level photo all reject it with the reason. On our test set it rejected none of 72 real satellite images and 73 of 93 other pictures. If the check is wrong, "Process anyway" overrides it.

**Q: Why not just use LiDAR or drones?**
A: Where they exist, they are better, and DepthWizard can use them as a reference. But most Indian towns have neither, and surveys take time and money. A satellite image is usually already available, even right after a disaster.

**Q: Does it work on CartoDEM?**
A: Yes. CartoDEM heights are measured from an ellipsoid, not sea level. We convert them with the EGM2008 geoid: minus 43.8 m at Namchi. Without the geoid file, the app measures the offset against Copernicus instead.

**Q: Does it need the internet?**
A: Only for the first setup: packages, the AI model and the elevation tile. After that it runs offline.

**Q: What hardware does it need?**
A: An ordinary laptop: about 40 seconds for a 2048-pixel scene on the CPU. A GPU is optional.

**Q: Did you use Hugging Face?**
A: The AI model, Depth Anything V2 Small, was published on Hugging Face. We run an ONNX version with ONNX Runtime, which avoids PyTorch and keeps the install small. Our original desktop version loaded it with the Hugging Face libraries.

**Q: What about the imagery licence?**
A: The Namchi image is from the Maxar Open Data Program under CC BY-NC 4.0, credited and used non-commercially. CartoDEM files are never bundled; users download them from Bhuvan under NRSC's licence.
