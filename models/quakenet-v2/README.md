<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# QuakeNet v2 <sub>v1.0</sub>

> **Several seismometers, one earthquake.**

*Where was that earthquake, how deep, how big, and when, from several stations together?*

QuakeNet v1.2 reads each station's 41-minute record; a graph network with stations as nodes and station pairs as edges combines those readings and the P-wave arrival times into an epicentre, depth, Mw and origin time, each with an error bar. It learned only from simulated earthquakes seen at 2-10 real stations at once, laid into real station noise, and was examined on 1,456 real earthquakes at 2 stations and 400 at up to 10.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/quakenet-v2/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/quakenet-v2/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | 2-10 stations, each with 41 minutes of three-component ground velocity (1 sample/s, response removed), its coordinates, the time of its first sample and a rough P time |
| **Returns** | Epicentre (latitude, longitude) with a 68% error ellipse (km)<br>Depth (km) with a bar on log depth<br>Moment magnitude Mw ± 68% bar<br>Origin time ± bar (s) |
| **Learned from** | QuakeNet v1.2's simulator run per station for one shared source: Syngine ak135f Green's functions with v1.2's measured corrections (physics v7), exact distances, ak135 P times plus station residuals, and real noise from the same station. Physics v2 and v3 add how v1.2 reads real records differently from their simulated twins, measured on dev. |
| **Noise** | 1,284 real quiet-time windows from the 127 IU/II stations (QuakeNet v1.2's pool), from the same station where possible (80%) |
| **Network** | Graph network, 115k weights: node and edge MLPs, 3 rounds of message passing (width 64), mean + max readout; corrects a classical anchor; bivariate Gaussian NLL on the epicentre offset, beta-NLL on log depth, Mw and origin time |
| **Physics also enters** | inputs: ak135 P travel times and slownesses at the anchor, station geometry in the anchor's frame<br>output: QuakeNet v1.2's per-station readings combined classically (least squares on distance and back-azimuth; inverse-variance Mw and depth; origin time from P picks) |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | QuakeNet v1.2's 40 dev events (seed 2026), each at its 2 original stations plus up to 8 more (399 records) |
| **Exam** | QuakeNet v1.2's exam events: 1,456 with 2+ records (original pairs); a 400-event subset (seed 7, depth strata) at up to 10 stations (3,523 records fetched for dev + subset) |

## Run it

```bash
cd models/quakenet-v2
pip install numpy scipy
python demo.py        # about 2 s
```

or from anywhere in the repo: `mlratchet demo quakenet-v2`.

```python
import numpy as np, json
from quakenet2 import QuakeNet2
net = QuakeNet2()
z = np.load("examples/us7000kulj.npz"); m = json.loads(str(z["meta"]))
st = [dict(zne=x, lat=s["lat"], lon=s["lon"], t_start=s["t_start"], p_guess=s["p_guess"]) for x, s in zip(z["zne"], m["stations"])]
r = net.locate(st)
print(r["lat"], r["lon"], r["err_ellipse"], r["depth"], r["mw"], r["t0"])
```

## The exam

Real earthquakes (2012-2025, M >= 5.5) graded against USGS: first 2 stations for 1,456 quakes, and first 3, 5 and all (up to 10) stations for 400. Every number here is on real data that never chose anything.

| Metric | QuakeNet v2 | Baselines | |
|---|---|---|---|
| Epicentre, 2 stations, median miss | **427 km** | v1.2 per station, combined classically (classical): 521 km<br>Travel-time locator (STA/LTA P + AIC S, ak135 grid search) (classical): 4139 km<br>Station centroid (dumb baseline): 6190 km | ✅ wins |
| Epicentre, all stations (up to 10), median miss | **127 km** | v1.2 per station, combined classically (classical): 240 km<br>Travel-time locator (classical): 173 km<br>Station centroid (dumb baseline): 4223 km | ✅ wins |
| Depth, all stations, median \|error\| | **17.6 km** | v1.2 combined (inverse-variance) (classical): 29.1 km<br>Travel-time locator (classical): 152.5 km<br>Always guess the dev median, 77 km (dumb baseline): 67 km | ✅ wins |
| Deep quakes (> 70 km), all stations, median depth \|error\| | **38.3 km** | v1.2 combined (inverse-variance) (classical): 27.7 km | ❌ loses |
| Mw, all stations, mean \|error\| | **0.097 Mw** | v1.2 combined (inverse-variance) (classical): 0.089 Mw<br>Always guess the dev median, 5.85 (dumb baseline): 0.315 Mw | ❌ loses |
| Origin time, all stations, median \|error\| | **4.3 s** | v1.2 combined (P picks at its epicentre) (classical): 5.8 s<br>Travel-time locator (classical): 14.4 s<br>Station centroid (dumb baseline): 72.3 s | ✅ wins |

Coverage of the 68% error bars on the exam: Epicentre ellipse (all stations) **62%**, Epicentre ellipse (2 stations) **65%**, Depth (all) **60%**, Mw (all) **64%**, Origin time (all) **74%**.

### Where it wins

- Epicentre at every station count: median miss 427 / 279 / 188 / 127 km at 2 / 3 / 5 / all stations, against 521 / 410 / 320 / 240 km for v1.2 combined classically and 4,139 / 669 / 276 / 173 km for the travel-time locator.
- Shallow depth (<= 70 km): median error 11.7 km with all stations against 29.3 km for v1.2 combined: the dev-twin depth calibration (r6) carried over.
- Origin time: median 4.3 s with all stations, against 5.8 s (v1.2 combined) and 14.4 s (travel-time locator).

### Where it loses

- Mw: 0.097 mean error against 0.089 for simply averaging v1.2's per-station Mw (0.114 against 0.109 at 2 stations).
- Deep quakes (> 70 km) read too shallow: median depth error 38 km against 28 km for v1.2 combined (all stations).
- Without catalogue help it does not locate well (supplementary exam, 27 Nov 2026, same 400 quakes, continuous records that know only the UTC hour of the origin): with P picked by a grid-search associator over each event's records, the median epicentre miss is 497 km with all stations (mean 1,346 km), against 123 km for the same records with the catalogue's predicted P as the guess. A plain per-record trigger picker gives 1,284 km. Only 40% of the associator's picks lie within 10 s of the true P.

## Limits

- Long-period (LH) records only, 25-95 degrees, M >= 5.5; 2 to 10 stations.
- The P pick searches +-30 s around a guess; in the exam (as in v1.2) the guess came from the catalogue origin. The headline exam numbers are for catalogue-assisted picking; with independent picks (supplementary exam, 27 Nov 2026) the median miss is 497 km, not 127 km. The travel-time locator baseline also picks inside the catalogue window.
- Two stations leave most of the location uncertain: median miss 427 km.
- Error bars: epicentre ellipses cover 62-68% and depth bars 60% for a nominal 68%.
- The dev-twin calibrations (r5, r6) are shifts fitted on 40 dev quakes; deep quakes are now over-corrected.

## The ratchet

**Objective:** Mean over epicentre (error capped at 2,000 km), log depth, Mw and origin time of 1 - mean error / mean error of rung 0 (QuakeNet v1.2 per station, combined classically) on the frozen simulated set  
**Rule:** Keep if the simulated objective beats the incumbent by more than 0.01 AND the real dev events get no more than 5% worse.

| Rung | Change | Frozen-set score | Decision | Why | Epicentre, 2 stations (km, median) | Epicentre, all stations (km, median) | Depth, all (km, median) | Mw, all (mean |error|) | Origin time, all (s, median) |
|---|---|---|---|---|---|---|---|---|---|
| r0 | QuakeNet v1.2 per station, combined classically: least squares on the sphere from distance + back-azimuth; inverse-variance Mw and log depth; origin time from P picks | 0.000 | ● kept | Rung 0, the classical combination every later rung must beat (dev: epicentre 533 km mean, origin time 12.7 s). | 521 | 240 | 29.1 | 0.089 | 5.8 |
| r1 | Graph network correcting rung 0 from v1.2's per-station readings and the station geometry, no pick times | 0.025 | ● kept | +0.084 on val1 (+0.025 on val3) over r0; dev error ratio 0.989. | 512 | 205 | 27.8 | 0.111 | 5.9 |
| r2 | + P pick times: each station's pick residual at the anchor and the ak135 slowness on the nodes, residual differences on the edges | 0.070 | ● kept | +0.053 on val1 (+0.045 on val3) over r1; dev error ratio 0.927 (dev epicentre 487 -> 375 km). | 452 | 114 | 25.3 | 0.104 | 5.7 |
| r3 | + v1.2's 256-number summary of each record (last hidden layer), projected to 32 per node | 0.071 | ○ rejected | -0.010 on val1, +0.001 on val3 over r2: short of the 0.01 margin. | 451 | 122 | 26.9 | 0.102 | 5.8 |
| r4 | + P-wave cross-correlation on the edges (best lag around the picks, and \|correlation\|) | 0.075 | ○ rejected | +0.000 on val1, +0.006 on val3 over r2: short of the 0.01 margin. | 446 | 117 | 26.7 | 0.098 | 6.1 |
| r5 | Physics v2 from the dev twins: simulated v1.2 distance readings +1.5 deg with 3 deg extra scatter, back-azimuth 6.5 deg extra scatter, simulated pick errors x0.75 (real picks are tighter). Fine-tuned from r2. | 0.108 | ● kept | +0.083 on val2 (+0.038 on val3) over r2; dev error ratio 0.971 (dev epicentre 375 -> 331 km). | 417 | 121 | 28.4 | 0.11 | 5.9 |
| r6 | Physics v3 from the dev twins: simulated depth readings of shallow quakes +0.37 in log10 depth (shared by the event's stations, plus scatter; deep +0.04), Mw readings +0.06 / +0.02. Fine-tuned from r5. | 0.219 | **◆ released** | +0.111 on val3 over r5; dev error ratio 0.952 (dev log-depth error 0.222 -> 0.190). | 427 | 127 | 17.6 | 0.097 | 4.3 |

<sub>Columns after *Why*: Real exam (1,456 quakes at 2 stations; 400 at all stations), reported after the fact; never used to choose.</sub>

val1: physics v1 (QuakeNet v1.2's physics v7 per station, shared source, P residual 1.5 s). val2 and val3 add reading-level mismatches measured on the dev twins. Every rung was re-scored on each new set and every decision re-taken in order; no decision flipped.

Raw per-rung logs are in [`training/`](training/).

## Physics

QuakeNet v1.2's simulator run per station for one shared source: Syngine ak135f Green's functions with v1.2's measured corrections (physics v7), exact distances, ak135 P times plus station residuals, and real noise from the same station. Physics v2 and v3 add how v1.2 reads real records differently from their simulated twins, measured on dev.

**Simulator**

- One source per event at 2-10 real GSN stations 25-95 degrees away: shared epicentre, depth, Mw, mechanism, source duration and ocean layer
- QuakeNet v1.2's per-station chain (physics v7 with layering), the nearest 1-degree Green's function stretched so S-P matches ak135 at the exact distance
- Absolute timing: ak135 P time + station residual N(0, 1.5 s) + record-start rounding; the simulated STA/LTA picker's own errors
- Epicentres from the 1990-2011 USGS catalogue (80%, jittered) or uniform on the sphere (20%); v1.2's world mix of depth and Mw
- v2/v3: simulated v1.2 readings get the dev-twin mismatches (distance +1.5 deg, extra distance and back-azimuth scatter, shallow-depth shift +0.37 in log10, Mw +0.02-0.06); pick errors x0.75

**Checked against independent references**

- ak135 travel-time table (bilinear) vs direct ObsPy TauP: max 0.078 s; vs iasp91: max 0.24 s (40 random points)
- Great-circle distance vs ObsPy locations2degrees: 6e-14 degrees
- Inherited: local Green's-function synthesis vs Syngine full synthetics, correlation 0.9999

## Help wanted

- EarthScope dataselect answered 'too many requests' for 51 of about 4,200 requests; events still reached 10 stations from other stations.
- Catalogue arrivals (audit, 25 Nov 2026): the exam gives each station the catalogue's predicted P arrival as its pick guess. Station coordinates and those arrivals alone locate all four 10-station packaged events within 4.24 m, so the arrivals carry the location. The network does read the waveforms (zeroed or noise waveforms put the epicentre 519-16,307 km off), but jittering the guess by 60 s moves it 241-955 km. The exam's location numbers therefore describe catalogue-assisted picking, not independent detection.

## Next rungs

- An independent picker that works at 1 sample/s for M 5.5-6 (only 40% of associated picks lie within 10 s), or broadband (BH) picking for the P time only; the supplementary exam (training/indep_exam.py) is the benchmark.
- Mw: let the classical inverse-variance Mw through unchanged (or learn a per-event bias only), and measure the deep-depth over-correction on dev by depth class.
- Bigger station sets and regional distances; broadband (BH) channels and a sharper P picker for timing.
- A second pass: re-linearise at the network's own epicentre and run the graph again.

## Files

| Path | What |
|---|---|
| `quakenet2.py`, `quakenet.py`, `pipeline.py` | Inference — numpy/scipy only |
| `quakenet2_weights.npz`, `quakenet_weights.npz`, `tt_ak135.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [USGS ComCat catalogue and moment tensors](https://earthquake.usgs.gov/fdsnws/event/1/) | truth | Public domain (U.S. Government) |
| [EarthScope / IRIS DMC waveforms (IU and II networks)](https://service.earthscope.org/) | exam, dev, noise | Open; cite the networks (IU doi:10.7914/SN/IU, II doi:10.7914/SN/II) |
| [EarthScope Syngine Green's functions (ak135f_2s), QuakeNet v1.2's grid](https://ds.iris.edu/ds/products/syngine/) | physics | Open; cite Syngine |
| [ak135 travel times (ObsPy TauP)](https://docs.obspy.org/packages/obspy.taup.html) | physics | ObsPy LGPL; model published (Kennett et al. 1995) |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
