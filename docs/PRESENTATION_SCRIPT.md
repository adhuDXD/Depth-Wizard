# DepthWizard: presentation script

**Total time:** about 10 minutes: 7 minutes of slides, 3 minutes of live demo, then questions.
**Deck:** `docs/DepthWizard.pptx` (12 slides).

> In this script, *italic notes* are stage directions. Everything else is spoken.

---

## Slide 1: Title (30 s)

> Good morning. We are team DepthWizard, working on problem statement SIH26175 from ISRO.
>
> Imagine you are a district disaster officer in a hill town like Namchi in Sikkim. The river is rising. You have a satellite image of your town on your screen, and three questions:
> **Which buildings will flood? Which buildings are tall enough to shelter in? And which way should people walk?**
>
> DepthWizard answers these from **one satellite image**, on **one office computer**, in under a minute.

---

## Slide 2: The problem (60 s)

> First, why is this hard today?
>
> India's free elevation data (CartoDEM from ISRO, plus SRTM and Copernicus) is posted at about **30 metres**. At that size you can see the hill, but not the town on it. Houses, trees and streets simply disappear.
>
> *(point to the middle column)* Disaster planners need exactly the detail that is missing: which buildings are in the water, which are tall enough to shelter on, which streets will jam.
>
> *(point to the right column)* Existing options don't close the gap. LiDAR and stereo surveys are slow and costly. AI depth models give numbers with no units. Most tools stop at a pretty 3D picture instead of a decision.
>
> **So the gap is: decisions need building heights in metres, and free data stops at the terrain.**

---

## Slide 3: Our pipeline (60 s)

> DepthWizard closes that gap in five steps.
>
> 1. We read the image, PNG, JPG or GeoTIFF, and if it is georeferenced we put it on a metric map grid.
> 2. An AI model, **Depth Anything V2**, gives us the *shape* of every building and tree. It processes the image in tiles and blends them so there are no seams.
> 3. Then comes the key step: **we measure real metres**. I'll explain how on the next slide.
> 4. We model three hazards: flood, landslide and earthquake.
> 5. Finally we route people: for every spot in the town, which way to walk and how long it takes.
>
> And every result tells you honestly what it is: real metres, metres above ground, or relative only.

---

## Slide 4: How we get real heights (90 s). The core idea, so speak slowly.

> An AI looking at one picture cannot know whether a building is 6 metres or 60 metres tall. Nobody can, from one image alone. So we don't let the AI guess the metres. **We use physics.**
>
> Every building casts a **shadow**. Satellite images record the sun's angle. The building's height equals the shadow length times the tangent of the sun's elevation. The relationship is simple and physical, and it is the same for every building.
>
> *(point to the green card)* We added something important for **hill towns**. On a slope, a shadow falling uphill looks shorter, and a shadow falling downhill looks longer. We take the slope from the DEM and correct for it. In our test this **cut the height error from 5.5 metres to 3.1 metres**.
>
> *(point to the orange card)* Then we combine: buildings with a clean shadow keep their *measured* height. For the rest we use the AI estimate, but only as far as the AI agrees with the measurements. The app actually computes that agreement and shows it.
>
> *(point to the red card)* Lastly, **we never claim metres we can't justify**. If an image has no location, no sun angle and no reference, the app labels its output "relative, 0 to 1" and does not pretend.
>
> **The AI gives the shape. Physics gives the metres.**

---

## Slide 5: Results (60 s)

> Does it work? We built a test town modelled on Namchi, where we know the true height of every building, so we can measure our error exactly.
>
> *(point to the chart)* On buildings and trees, the 30-metre DEM alone is off by **11.5 metres** on average. DepthWizard brings that down to **7.0 metres**, a **39% reduction**. Across the whole scene, error drops from 6.1 to 4.1 metres.
>
> *(point to the right)* Shadow-measured building heights alone are within **3.1 metres**, and the automatic datum correction removes CartoDEM's 46-metre offset at Namchi.
>
> To be clear: **this is a synthetic town**. Validating on real Indian ground truth is our next step, and I'll show how on slide 11.

---

## Slide 6: Three hazards (60 s)

> Heights are only useful if they change a decision. Here is where ours do.
>
> **Flood.** We compute how high every spot sits above its nearest stream. When the water rises by, say, 3 metres, everything lower floods. Because we know building heights, **buildings at least 9 metres tall whose roofs stay above the water become refuges**. People who cannot reach high ground in time can go *up* instead of out.
>
> **Landslide.** We find steep, bare, water-collecting slopes and trace where a slide would run downhill. Routes avoid both.
>
> **Earthquake.** Debris from a collapsing building reaches roughly half its height into the street. So a narrow street between tall buildings is dangerous, and open ground beyond that reach is where people should gather. Again, this needs building heights.
>
> Routing uses a walking-speed model that knows uphill is slower than downhill. That matters in a hill town. One calculation gives every spot its direction to safety and its walking time.

---

## Slides 7 and 8: What the officer sees (45 s)

*(If the demo works, skip straight to it. Otherwise talk over these slides.)*

> These are real screens from the app. Arrows show the fastest way out, coloured by minutes. Letters are safe zones with their capacity. A red exclamation mark is a choke point where marshals are needed.
>
> The officer does not have to read a heatmap. The app writes plain sentences like *"117 of 123 buildings are in the danger zone, about 6,700 people; half can reach safety within 1.6 minutes."*
>
> With one click it prints an evacuation plan, or exports KML for phones and GeoTIFF for GIS teams.

---

## Live demo (3 min)

*Before the talk: run `run_local.bat` so the app is already open at http://localhost:8000.*

1. **Click "Try demo town".** *(about 10 s)*
   > It reads the image, runs the AI, measures shadows and builds the model.
2. **Point at the height card.**
   > "Absolute DSM, metres above sea level." It found 119 buildings and measured 114 from their shadows. It also shows the pixel size and the sun angle it read from the image.
3. **Flood: move the slider from 3 m to 6 m.**
   > Watch the blue spread and the arrows update. The cyan buildings are refuges.
4. **Click a spot in the blue zone.**
   > The pink line is that household's route: "Go to zone G, 44 metres, about 2 minutes."
5. **Click "Earthquake".**
   > Now the danger is debris around tall buildings, and the open ground becomes the assembly area.
6. **Click "3D".** Rotate.
   > The same town in 3D, with the measured heights and the arrows draped on top.
7. **Switch to Expert, open Validation.**
   > For experts: our error against the reference, next to the DEM's error.
8. **Click "Print evacuation plan".**
   > This one page is what goes to the ward officer.

*If the demo fails: "Here's the recording," then go back to slides 7 and 8.*

---

## Slide 9: Built for district offices (45 s)

> Our earlier version was a 4.7 GB desktop app that wanted a gaming GPU. Reviewers rightly said a district office can't use that.
>
> Now it installs in **0.8 GB**, needs **no GPU**, and processes a scene in **about 8 seconds on an ordinary laptop**. **One PC runs it and everyone in the office uses it from a browser.**
>
> Guided mode is three steps. The expert tools (layers, profiles, validation) stay out of the way until you switch them on.

---

## Slide 10: Requirement matrix (30 s)

> Here is every requirement in the problem statement mapped to a feature. Relative DSM, absolute DSM in metres, building heights, 3D view, accuracy check: all done. Disaster routing goes beyond the brief. We have marked honestly what is still pending: a user test and validation on Indian terrain.

---

## Slide 11: Limits and plan (45 s)

> We'd rather tell you our limits than have you find them.
>
> Our accuracy numbers come from a synthetic town. The shadow method assumes a near-vertical view and flat roofs. Routes don't use the road map yet. And we haven't run a pilot.
>
> The plan:
> **One:** validate on real Indian ground using free NASA spaceborne LiDAR (ICESat-2 and GEDI, which both cover India), a field survey of 30 to 50 buildings, and a Cartosat stereo pair from NRSC.
> **Two:** add the OpenStreetMap road network and population data.
> **Three:** show a district disaster office a plan for *their own* town, and test the app with non-experts.

---

## Slide 12: Close (20 s)

> So: **where the water goes, which streets block, and which way to run**, from one satellite image, on one office PC, with heights you can check.
>
> Thank you. We're happy to take questions.

---

# What we changed, and why it matters

Use this section if a judge asks *"What's new since your last review?"*

| Review criticism | What we changed | Effect |
|---|---|---|
| "Just a wrapper around a pretrained model" | Metres now come from **physics** (shadow geometry with slope correction). The AI only provides shape, and its trust is measured per image. | An original, explainable method. Shadow-height error is 3.1 m (5.5 m without the slope correction). |
| "Gaussian blur is not novel" | Replaced with shadow-first fusion and robust calibration with leave-one-out error | Every output carries its own accuracy figure |
| "No practical impact / not for disaster managers" | Added **flood, landslide and earthquake escape routing**, refuges, choke points, capacity checks and a printable plan | The tool now supports a decision, not just a 3D picture |
| "4.7 GB package, needs CUDA" | Replaced PyTorch with **ONNX Runtime on CPU**, and made it a **web app** | 0.8 GB, no GPU, the whole office uses one PC through a browser |
| "UI too technical" | **Guided mode** (3 steps), plain-language summary, expert tools hidden | A non-technical officer can use it |
| "Validate tool needs a reference the user doesn't have" | A built-in accuracy figure from held-out shadow measurements | Users see reliability without any reference data |
| "No requirement matrix" | Added (slide 10) | The brief is visibly covered |
| Deck: placeholder text, mixed fonts, inconsistent colours | One font, one colour system (blue = water, orange = model, red = hazard, green = safe), no placeholders | A clean, consistent deck |

---

# Likely questions and answers

**Q: How can one image give real heights?**
A: On its own it can't, and we don't claim it does. Metres come from the shadow length and the sun angle (physics), the DEM, or ground control points. Without any of these, the app labels the output "relative".

**Q: What if there are no shadows, for example at noon or under clouds?**
A: Buildings without a usable shadow get the AI estimate, and the app reports how well the AI matched the measured buildings. If nothing gives a scale, the output stays relative. We never invent metres.

**Q: Your results are on synthetic data. Why trust them?**
A: Synthetic data lets us measure error exactly. It is a controlled test, not our final claim. Next we validate with ICESat-2 and GEDI (free spaceborne LiDAR that covers India), a field survey and a Cartosat stereo pair.

**Q: What is HAND?**
A: Height Above Nearest Drainage: how high each spot sits above the stream it drains to. If water rises h metres, everything with HAND below h floods. It is a standard, fast and explainable flood screening method.

**Q: Why not use real roads for routing?**
A: That's next: OpenStreetMap roads. Today routing works over open ground and streets as seen in the image, which still shows the safe direction and the choke points.

**Q: How accurate are the population numbers?**
A: They are estimates from building volume: floors × footprint ÷ 15 m² per person. They are labelled as estimates and used for relative planning, such as which shelter is overloaded, not for exact counts.

**Q: What does the AI actually contribute?**
A: It finds buildings and trees and their shape, and it fills in heights where no shadow is measurable. The app measures the AI's agreement with physical measurements on every image and trusts it only that much.

**Q: Can it run without internet?**
A: Yes. After the one-time install, everything runs offline on the local machine.

**Q: Who would use it, and how is it sustained?**
A: District and state disaster management authorities, municipal planners and NDRF teams. The core is open source. Training, onboarding and customisation for districts can fund maintenance, and we would also apply to schemes such as ISRO RESPOND.
