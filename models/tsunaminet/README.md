<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# TsunamiNet <sub>v1.0</sub>

> **A flow trained on simulated tsunamis reads real DART buoys and forecasts the next ones.**

> [!WARNING]
> Research only. Not a tsunami warning system; official forecasts come from NOAA's tsunami warning centers.

*How big is the tsunami coming, from the first deep-ocean pressure readings?*

From up to three DART bottom-pressure records (a 90-minute window around each buoy's predicted arrival) after a Pacific subduction earthquake, a conditional RealNVP normalizing flow gives the posterior of the seafloor source on NOAA's SIFT unit sources: magnitude, where along the trench the slip is centred, how much sits on the deeper row, and how long the rupture is. Pushing posterior samples through linear shallow-water Green's functions forecasts the maximum wave at DARTs the network did not read. Trained only on simulated tsunamis laid into real quiet DART noise; graded once on 28 real DART-recorded tsunamis (2007-2025).

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/tsunaminet/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/tsunaminet/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Detided, high-passed 1-minute DART water-column height (NDBC historical files) for up to 3 DARTs, the USGS epicentre and origin time. |
| **Returns** | Posterior samples of Mw, along-trench centroid (km from the epicentre's unit-source column), deep-row slip share and log10 rupture length; 68 % and 90 % ranges widened on dev (Mw x3.18, centroid x4.40)<br>Forecast maximum \|eta\| at any other DART with Green's functions, in a 190-minute window after its predicted arrival: median and 68 % range (widened x2.08 on dev) |
| **Learned from** | Okada (1985) vertical seafloor displacement from 1 m of thrust slip on each of 1,317 Pacific SIFT unit sources (100 x 50 km, NOAA geometry), propagated by the linear shallow-water equations on a spherical C-grid over ETOPO 2022 at 8 arc-min (dt 20 s, 24 h). Green's functions come from one receiver-started run per DART (discrete reciprocity), so any slip pattern is a sum. Simulated physics version p1 adds what the dev twin test measured: arrivals 1.65 % of the travel time late, amplitudes x1.60 with 0.17 dex scatter per DART, and rupture spreading at 2.5 km/s. |
| **Noise** | Real DART bottom-pressure noise: 955 quiet 24-hour windows starting at event-mode (1-minute) triggers 2006-2025 that are more than 2 days from any NGDC tsunami (712 for training, 243 for the frozen sets), with the same detiding, gaps and glitches as real records. |
| **Network** | Per-DART 1-D CNN over 20 channels (record, mask, 18 unit-source responses) x 90 minutes, permutation-invariant mean/max pooling over up to 3 DARTs, plus 23 features of the SIFT-style NNLS solution on the same windows (physics anchor), conditioning an 8-layer RealNVP flow over 4 source parameters. 520k weights. |
| **Physics also enters** | inputs: Each DART's own Green's-function responses to the 18 unit sources around the epicentre, as input channels<br>output: The SIFT-style non-negative least-squares inversion on the same windows feeds the flow; forecasts are posterior samples pushed through the Green's functions |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | 5 events drawn with seed 20261122 (2 with Mw >= 8, 3 below) from the 33 eligible ones, the 5 named exam events excluded: Kuril 2006 (8.3), Kuril 2007 (8.0, outer rise), Solomon Sea 2016 (7.9), Shumagin Oct 2020 (7.6, strike-slip), Kamchatka Sep 2025 (7.8). Frozen in splits.json before training. |
| **Exam** | The other 28: every Pacific NGDC tsunami 2006-2025 with eq Mw >= 7.5 within 200 km of a SIFT unit source and at least one NGDC deep-ocean reading, including Tohoku 2011, Maule 2010, Iquique 2014, Illapel 2015 and Kamchatka Jul 2025 (named in advance). 19 interface thrusts, 9 other mechanisms. |

## Run it

```bash
cd models/tsunaminet
pip install numpy scipy
python demo.py        # about 2 s
```

or from anywhere in the repo: `mlratchet demo tsunaminet`.

```python
import numpy as np, tsunaminet as T
net = T.TsunamiNet(); gf = T.GreensFunctions(); gf.cal = net.cal
# records: {site: (eta_m[1441], mask[1441])} minute series from the origin, from pipeline.detided_minutes
ev = gf.event(lon, lat, records)
post = net.posterior(ev)                        # Mw, centroid, ranges, samples
fc = gf.forecast(ev, post, '46402_1', a, b, widen=net.cal['widen']['amp'])
```

## The exam

Run once on 28 real tsunamis (2007-2025): source magnitude and centroid against USGS, and the maximum wave at 119 held-out DARTs (15 events) against what they recorded, beside a SIFT-style NNLS inversion on the same windows, seismic W-phase scaling and a dumb baseline. 95 % intervals are bootstrap over events. Every number here is on real data that never chose anything.

| Metric | TsunamiNet | Baselines | |
|---|---|---|---|
| Mw error vs USGS W-phase (28 events, mean \|error\|)<br><sub>95 % interval 0.18-0.27 (NNLS 0.18-0.33). TsunamiNet reads low on average (bias -0.22); Tohoku 8.67 vs 9.1. Seismic W-phase Mw is the truth here, so it is not a rival on this row.</sub> | **0.23** | SIFT-style NNLS on unit sources (classical): 0.25<br>Dev-median Mw (7.9) (dumb baseline): 0.3 | ✅ wins |
| Centroid distance to the USGS finite-fault centroid (median, km)<br><sub>Mean 59 km (NNLS 112, epicentre 63).</sub> | **45** | SIFT-style NNLS (classical): 82<br>Epicentre's unit-source column (dumb baseline): 56 | ✅ wins |
| Centroid along the trench, mean error (km)<br><sub>95 % intervals 29-58 and 26-54 km: a tie with the epicentre, a clear win over NNLS.</sub> | **42** | SIFT-style NNLS (classical): 98<br>Epicentre's unit-source column (dumb baseline): 39 | ❌ loses |
| Held-out DART max amplitude, mean \|log10(forecast/observed)\|, all 119 forecasts<br><sub>95 % interval 0.16-0.27. Within a factor 2: 79 % (NNLS 71 %, seismic 56 %, dumb 52 %). USGS finite-fault twin (hindsight): 0.30.</sub> | **0.2** | SIFT-style NNLS (classical): 0.23<br>SIFT-style NNLS with dev-calibrated p1 Green's functions (classical): 0.21<br>Seismic W-phase Mw, standard rupture (classical): 0.28<br>Dev-median amplitude (2.8 cm) (dumb baseline): 0.34 | ✅ wins |
| Same, each event weighted once (15 events)<br><sub>The pooled win comes from the big, well-instrumented events (Tohoku 26 forecasts, Kamchatka 18, Illapel 16, Maule 15). On small events with one or two held-out DARTs a constant 2.8 cm does better.</sub> | **0.34** | SIFT-style NNLS (classical): 0.32<br>NNLS with p1 Green's functions (classical): 0.3<br>Seismic W-phase Mw (classical): 0.35<br>Dev-median amplitude (dumb baseline): 0.27 | ❌ loses |

Coverage of the 68% error bars on the exam: Mw (dev-widened x3.18; 90 % range covers 96 %) **68%**, Centroid along the trench (x4.40) **89%**, Held-out DART amplitude (x2.08) **78%**.

### Where it wins

- Beats the SIFT-style NNLS inversion on the same windows for Mw (0.23 vs 0.25), centroid (45 vs 82 km median) and held-out amplitudes (0.20 vs 0.23 dex), and beats seismic-Mw scaling on amplitudes (0.20 vs 0.28).
- 79 % of 119 held-out DART forecasts within a factor 2 of what the buoy recorded.
- Dev-widened Mw ranges cover 68 % of exam truths at nominal 68 %.
- Kamchatka 2025: 8.58 [8.34, 8.84] vs 8.8; Maule 2010: 8.84 vs 8.8; Illapel 2015: 8.21 vs 8.3.

### Where it loses

- Reads magnitudes low (mean bias -0.22). Tohoku 2011: 8.67 [8.44, 8.92] vs 9.1.
- The along-trench centroid ties the epicentre's own column (42 vs 39 km).
- Weighted by event, the dumb 2.8 cm amplitude wins (0.27 vs 0.34 dex).
- Non-interface events (outer-rise, strike-slip, intraslab): Mw 68 % coverage 44 % (interface 79 %).

## Limits

- Linear propagation at 8 arc-min (not NOAA's 4'): far-field window correlation 0.73-0.81 against 4'; no dispersion, Coriolis or Kajiura filter.
- Only thrust slip on the two shallowest SIFT rows; other mechanisms are read as if they were interface thrusts.
- Hindcast-quality data: NDBC real-time files with event-mode gaps; the detiding is causal but tuned on the 3 days before.
- Not strictly real-time: glitch removal and gap filling read up to 34 later observed samples (about 34 min at 1-minute sampling, longer across gaps), so a live input window is final only that long after its last minute. Training and the exam used the same processing.
- Dev is 5 events; the widening factors (x2-4.4) rest on them.
- The released widening factors were computed with the interval sides swapped (training/calibrate.py took the upper half-width for a truth below the median; fixed 22 Nov 2026, not re-run: the per-event dev rows are not shipped). The exam coverage above is for the factors as shipped.
- Green's functions exist for 68 DART sites (those that recorded the eligible events); no coastal tide gauges.

## The ratchet

**Objective:** Mean skill of the posterior-median Mw and along-trench centroid against guessing the frozen set's median  
**Rule:** Keep a rung if its frozen-set score beats the incumbent's by more than 0.01 AND its dev error (Mw MAE/0.3 + centroid MAE/100 km + held-out amplitude MAE/0.2 dex, on the 5 real dev events) is no more than 5 % worse.

| Rung | Change | Frozen-set score | Decision | Why | Sim Mw MAE | Sim centroid MAE (km) | Dev Mw MAE | Dev centroid MAE (km) | Dev amplitude MAE (dex) |
|---|---|---|---|---|---|---|---|---|---|
| r1 | p0 textbook physics (linear SWE 8', instantaneous rupture), 40k sims in real DART noise, CNN set encoder + RealNVP | 0.327 | ● kept | First rung (0.438 on val_p0). Its flow never learned the centroid: skill 0.006 on val_p0. | 0.155 | 82 | 0.21 | 91 | 0.4 |
| r2 | p1 physics from the dev twin test: arrivals 1.65 % of travel time later, amplitudes x1.60 (sd 0.17 dex) per DART, rupture at 2.5 km/s; forecasts with the same delay and gain | 0.663 | ● kept | +0.336 on val_p1; dev error x0.753 | 0.092 | 39 | 0.21 | 79 | 0.24 |
| r4 | Mixed bank: 20k p0 + 20k p1 simulations | 0.641 | ○ rejected | -0.022 vs r2 on val_p1; dev error x1.054 | 0.104 | 40 | 0.28 | 54 | 0.28 |
| r3 | Physics anchor: the SIFT-style NNLS solution on the same windows (Mw, centroid, deep share, misfit, 18 slip shares) joins the flow's context | 0.718 | **◆ released** | +0.054 vs r2; dev error x0.969 | 0.091 | 30 | 0.23 | 57 | 0.26 |
| r5 | Polish: r3 trained 3,000 more steps at a 3x lower learning rate | 0.724 | ○ rejected | +0.006, short of the 0.01 margin; dev error x0.987 | 0.09 | 29 | 0.22 | 59 | 0.25 |

<sub>Columns after *Why*: Frozen simulated set val_p1 (3,000 sims) and the 5 real dev events; never the exam.</sub>

Physics changed once (p0 -> p1, from the dev twin test), so a new frozen set val_p1 was built and r1 re-scored on it before any p1 rung was judged. r4 was trained and judged before r3; decisions were taken in that order.

Raw per-rung logs are in [`training/`](training/).

## Physics

Okada (1985) vertical seafloor displacement from 1 m of thrust slip on each of 1,317 Pacific SIFT unit sources (100 x 50 km, NOAA geometry), propagated by the linear shallow-water equations on a spherical C-grid over ETOPO 2022 at 8 arc-min (dt 20 s, 24 h). Green's functions come from one receiver-started run per DART (discrete reciprocity), so any slip pattern is a sum. Simulated physics version p1 adds what the dev twin test measured: arrivals 1.65 % of the travel time late, amplitudes x1.60 with 0.17 dex scatter per DART, and rupture spreading at 2.5 km/s.

**Simulator**

- Random interface earthquakes Mw 7.3-9.3 on any Pacific SIFT column; length from Strasser et al. (2010); heterogeneous slip along strike; random up-dip/down-dip share
- DART network of a random date 2006-2025 from NDBC deployment history; the 3 earliest-arriving DARTs with data are the input
- Laid into 712 real quiet DART noise windows (event-mode 1-minute data away from every NGDC tsunami), with their real data gaps
- The same causal detiding and 3-hour high-pass on records and on every Green's function

**Checked against independent references**

- Okada vs GeoClaw's independent implementation: largest relative difference 4e-13 over 6 unit sources and 3 rakes
- Okada vs NOAA MOST's own deformation for unit source acsza1: r = 0.995, peak uplift 0.352 vs 0.349 m, volume ratio 1.03
- 1-D flat basin: wave speed 198.5 vs sqrt(gh) = 198.1 m/s, rms error 0.1 % of the peak
- Green's law on a slope from 4,000 m: amplitude x1.416 at 1,000 m (law 1.414) and x1.868 at 281 m (law 1.943, -3.9 %)
- 2-D flat basin vs the exact Hankel-transform solution: front at 607 km in both, peak 0.0747 vs 0.0763 (-2.1 %)
- Reciprocity on the real 8' Pacific grid: forward run vs receiver-started run, relative difference 2.4e-6
- 8' vs 4' (NOAA MOST's resolution) Green's functions at DART 46409, all unit sources: 3-hour peak ratio median 0.99 within 1,500 km and 0.97 beyond; window correlation 0.98 near, 0.73-0.81 beyond 3,000 km

## Help wanted

- NOAA's compressed propagation database (sift.pmel.noaa.gov/ComMIT/compressed) answers, but its amplitude encoding is undocumented and the documented THREDDS/OpenDAP access returns 404, so MOST waveforms could not be compared directly.
- API (audit, 25 Nov 2026; fixed 27 Nov 2026): an event with no usable buoy reached the network as a pooling sentinel and returned Mw -2053; posterior() now refuses it (all 28 exam events had a buoy). GreensFunctions() now loads the dev calibration from the released weights by default; without it the first British Columbia forecast read 1.12 cm instead of 1.76 cm. Five of 119 exam target arrivals came at or before the end of the input window, so those are not forecasts ahead of the wave.

## Next rungs

- 4' Green's functions (NOAA's resolution) for the 68 sites: 10 min each on one core; check the far-field loss of correlation and the low magnitude bias.
- Teach the flow other mechanisms (outer-rise normal, strike-slip) with their own unit sources, or a mechanism prior from the W-phase.
- Coastal tide gauges (CO-OPS) with nested grids or a learned amplification per gauge.

## Files

| Path | What |
|---|---|
| `tsunaminet.py`, `pipeline.py` | Inference — numpy/scipy only |
| `tsunaminet_weights.npz`, `gf_compact.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [NDBC historical DART water-column height (820 Pacific station-year files)](https://www.ndbc.noaa.gov/data/historical/dart/) | exam | US Government work (public domain) |
| [NDBC station metadata (DART deployment positions and dates)](https://www.ndbc.noaa.gov/metadata/stationmetadata.xml) | physics | US Government work (public domain) |
| [NCEI/WDS Global Historical Tsunami Database (events, DART readings)](https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis/events) | truth | US Government work (public domain) |
| [USGS ComCat: W-phase moment tensors and finite-fault models](https://earthquake.usgs.gov/fdsnws/event/1/) | truth | US Government work (public domain) |
| [ETOPO 2022 15 Arc-Second Global Relief Model (60 arc-sec surface, NCEI THREDDS)](https://www.ncei.noaa.gov/products/etopo-global-relief-model) | physics | US Government work (public domain) |
| [NOAA NCTR SIFT unit-source table (info_sz.dat) and MOST deformation for unit source acsza1](https://sift.pmel.noaa.gov/ComMIT/compressed/info_sz.dat) | physics | US Government work (public domain) |
| [GeoClaw dtopotools Okada (reference implementation for the check only)](https://github.com/clawpack/geoclaw) | physics | BSD-3-Clause |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
