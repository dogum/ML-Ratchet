<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# FireCast <sub>v1.0</sub>

> **Where a wildfire will have burned by tomorrow afternoon, cell by 90 m cell**

> [!WARNING]
> Research model. It forecasts the satellite view of a fire (FEDS perimeters from 375 m VIIRS hot spots, 12-hourly), not ground truth, and it has never seen suppression tactics, plume-driven runs or wind-event nights. Not for evacuation, firefighting or any safety decision.

*Which ground will this fire have burned by the next afternoon satellite pass?*

FireCast gives every 90 m cell around a burning wildfire the probability that it has burned by the next afternoon VIIRS overpass, about 24 h later. It reads today's observed fire (FEDS burned area and active fire line, VIIRS S-NPP hot spots at the issue and the night overpass) and gridMET daily weather for today and tomorrow. It is a small U-Net (374k weights) trained only on 1,083 FireSim fires burned on real LANDFIRE landscapes with real HRRR weather and observed through an emulated satellite. Graded once on 3,030 real exam fires (25,105 growth fire-days, 2012-24), it beats every baseline: mean AP 0.323 [0.318, 0.328] and F1 0.239 against 0.252 and 0.184 for the best one, yesterday's growth measured from the active fire line. It wins in every period, region and size class, and on the 48 replay fires it more than doubles the calibrated physics replay (AP 0.351 against 0.165). It loses on total burned-area overlap, where persistence still wins (Dice 0.964 against 0.922), and its probabilities above about 0.35 are over-confident: a 0.9 forecast burned 61 % of the time.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/firecast/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/firecast/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | One fire-day on a 90 m grid, north up (the release used EPSG:5070 CONUS Albers windows from FireData), covering the burned area plus at least 10 km on each side: the burned area at the issue (afternoon) overpass (boolean grid); optionally the burned area 24 h earlier and the FEDS active fire line; VIIRS S-NPP detections at the issue overpass and the night overpass 12 h earlier (fractional row and column of the pixel centre, pixel size along scan and track in km, FRP in MW); gridMET daily values for the issue day and the next day: wind speed (m/s) and direction (deg, from), minimum relative humidity (%), maximum temperature (°C), VPD (kPa), precipitation (mm), 100-h fuel moisture (%), plus the grid convergence angle; whether this is the fire's first detected day. Optional (mode L): LANDFIRE FBFM40 fuel model, slope (deg), aspect (deg) and canopy cover (%). |
| **Returns** | P(cell burned by the next afternoon VIIRS overpass), every cell of the grid, calibrated on dev only (p = sigmoid(logit / 1.44 − 1.23), widen-only temperature); reliable up to about 0.35, over-confident above (forecast 0.9 → 61 % burned)<br>A yes/no map at the dev-fitted threshold 0.10 (it over-predicts the new area about 1.9× at the median exam fire-day) |
| **Learned from** | FireSim, the shared fire platform's simulator (shared/fire/firesim): Rothermel 1972 surface spread with Albini 1976, Scott & Burgan 40 fuel models, Anderson 1983 ellipses, Van Wagner 1977 crown initiation, Nelson 2000 dead-fuel moisture, minimum-travel-time spread, spotting and timed containment lines, with a VIIRS/MODIS/GOES detection emulator (firesim.observe). FireCast wraps it, without editing it, in two versioned physics files (training/phys/p0.json, p1.json; training/src/simbank.py). Each training fire burns 2-8 days on a real LANDFIRE landscape with real hourly HRRR weather, in 12 h segments, with a randomised rate-of-spread factor (lognormal, median 0.35 of the dev-calibrated replay level, sd 0.7), night slow-down, growing containment and spotting. p1 adds a fire-age ROS factor fitted to the p0 dev twin test (×2.6 on the first day down to ×0.55 after day 10). |
| **Noise** | No real recordings are laid into the training signals. Observation noise comes from firesim.observe (VIIRS detection against FRP, scan-angle footprints, overpass timing, clouds) and the dev-fitted FEDS emulator; the weather the network reads is the HRRR series pushed through a dev-fitted gridMET operator plus its residual noise; landscapes and HRRR weather are real, from dev fires and guard-passed non-exam windows. |
| **Network** | U-Net on 90 m patches: stem at 90 m (16 channels), encoder 32-48-64-64 to 1.44 km, decoder with skip connections, BatchNorm folded, 374k weights (373,122 in the weight file). 22 map channels: burned now, yesterday's growth, fire line, two VIIRS FRP rasters (issue and night), distances to the burn, the fire line and the nearest hot spot, the offset from the nearest fire-line cell along and across tomorrow's wind, the 10 km scoring zone, and 11 LANDFIRE channels (fuel groups, slope, uphill vector, canopy cover, a landscape flag; zero in the default exam mode F). 22 scalars (16 gridMET features for today and tomorrow, 6 fire features: area, yesterday's radial growth, flags, hot-spot count, fire-line share) through a 2-layer MLP that adds a per-channel bias (FiLM-style) at every level and to the head. Loss: BCE on unburned cells in the 10 km zone. 128-cell random crops × 12, dihedral augmentation, landscape dropout 0.5, Adam; 1,800 + 1,200 steps for the released rung. CPU only, one thread; a rung takes about 30 min. |
| **Physics also enters** | inputs: Distances to the burned area, the active fire line and the VIIRS hot spots; the offset from the nearest fire-line cell projected along and across tomorrow's wind; wind as a spread-toward vector in grid coordinates; optional LANDFIRE fuel groups, slope and uphill vector |
| **Real data's jobs** | backdrop, calibration, gate, exam (never training labels) |
| **Dev set** | FireData's frozen split (shared/fire/data/proc/splits.json, seed 20261118): 40 CONUS wildfires 2017-21, 535 growth fire-days. Used to fit the FEDS emulator, the HRRR-to-gridMET weather operator and the p1 fire-age factor (through the p0 twin test), to gate every rung and to fit the calibration (T, b, threshold). Dev landscapes and HRRR also seeded simulated fires |
| **Exam** | Every exam fire in the split with a growth fire-day: 3,030 of the 4,197 MTBS exam fires, 25,105 growth fire-days (observed new area > 0), in three periods: early 2012-16 (1,234 fires, 9,661 days), main 2017-21 (1,331, 12,049) and recent 2022-24 (465, 3,395; whole seasons held out). Replay subset: the 48 exam fires packaged with LANDFIRE and HRRR. Run once, after r2 and its calibration were frozen. Examples: three exam growth fire-days, #0, #4 and #8 of a 12-day draw with seed 20261119, 4 per size stratum (small < 5 km², medium 5-50, large > 50 km² burned at issue; training/src/exam.py examples) |

## Run it

```bash
cd models/firecast
pip install numpy scipy
python demo.py        # about 3 s
```

or from anywhere in the repo: `mlratchet demo firecast`.

```python
import firecast as fc
net = fc.FireCastNP()                       # firecast_weights.npz next to this file
prob = fc.predict(net, day, land=False)     # calibrated probability, same shape as day['now']
yes = prob >= net.thr                       # dev-fitted yes/no threshold (0.10)
win = fc.score_window(day['now'])           # cells within 10 km of the burned area
fc.nextday(day['now'], day['nxt'], prob, fc.CELL_KM2, thr=net.thr, window=win)   # AP, F1, Dice, area error
```

## The exam

Every exam growth fire-day, scored like FireData's baselines: unburned cells within 10 km of the burned area at issue time, AP over the cell ranking and F1 of the yes/no map on the newly burned cells, averaged over growth fire-days, with 95 % bootstrap intervals that resample whole fires (training/results/exam_summary_r2.json). Each method's interval is its own (not a paired interval on the difference: the per-fire-day rows were not shipped); on every row marked win the intervals do not overlap. Baselines (FireData, the same fire-days): persistence (no growth), the climatological median growth, yesterday's growth repeated from the whole perimeter, and yesterday's growth measured from the active fire line (the best); on the 48 replay fires also FireSim run forward with the dev-calibrated replay settings (replay_calibration.json as stored), deterministic and as a 12-member ensemble. The baselines' F1 is at 0.5 on their distance score; FireCast's at its dev-fitted threshold 0.10. Every number here is on real data that never chose anything.

| Metric | FireCast | Baselines | |
|---|---|---|---|
| All exam fire-days (3,030 fires, 25,105 growth fire-days), mean AP<br><sub>95 % fire-bootstrap intervals: FireCast 0.323 [0.318, 0.328], fl_prev 0.252 [0.247, 0.257], climatology 0.204 [0.201, 0.208], grow_prev 0.191 [0.187, 0.194], persistence 0.013 [0.012, 0.013]. Median AP 0.297 against 0.216. FireCast has the higher AP than fl_prev on 71.5 % of fire-days. It leads in all 9 regions (from SP 0.211 vs 0.173 to SW 0.366 vs 0.304) and all 5 size bins (500+ km²: 0.302 vs 0.238). Published context, not comparable: WildfireSpreadTS test AP 0.19-0.37 (375 m, next-day active-fire target, Gerard et al. 2023).</sub> | **0.323** | Yesterday's growth from the active fire line (fl_prev) (classical): 0.252<br>Climatological median growth (dumb baseline): 0.204<br>Yesterday's growth repeated (grow_prev) (classical): 0.191<br>Persistence (no growth) (dumb baseline): 0.013 | ✅ wins |
| All exam fire-days, mean F1 of the yes/no map on new burn<br><sub>95 % fire-bootstrap intervals: FireCast 0.239 [0.234, 0.244] at its dev-fitted threshold 0.10, fl_prev 0.184 [0.180, 0.188], grow_prev 0.140 [0.136, 0.144]; grow_prev × 3 (dev-chosen k) 0.146. At the untuned threshold 0.5 FireCast's F1 is only 0.058 [0.055, 0.061], below both growth baselines: the threshold, fitted on dev, does the work. FireCast has the higher F1 on 50.1 % of fire-days (many ties at 0).</sub> | **0.239** | Yesterday's growth from the active fire line (fl_prev) (classical): 0.184<br>Yesterday's growth repeated (grow_prev) (classical): 0.14<br>Climatological median growth (dumb baseline): 0.004<br>Persistence (no growth) (dumb baseline): 0 | ✅ wins |
| Early period 2012-16 (1,234 fires, 9,661 fire-days), mean AP<br><sub>FireCast 0.318 [0.310, 0.327], fl_prev 0.249 [0.241, 0.258]. F1 0.232 [0.224, 0.240] against fl_prev 0.181 [0.175, 0.187].</sub> | **0.318** | fl_prev (classical): 0.249<br>Climatological median growth (dumb baseline): 0.206<br>grow_prev (classical): 0.192<br>Persistence (dumb baseline): 0.013 | ✅ wins |
| Main period 2017-21 (1,331 fires, 12,049 fire-days), mean AP<br><sub>FireCast 0.327 [0.320, 0.334], fl_prev 0.257 [0.250, 0.263]. F1 0.249 [0.241, 0.255] against 0.190 [0.184, 0.195]. The dev fires come from the same years (2017-21), different fires.</sub> | **0.327** | fl_prev (classical): 0.257<br>Climatological median growth (dumb baseline): 0.201<br>grow_prev (classical): 0.189<br>Persistence (dumb baseline): 0.013 | ✅ wins |
| Recent seasons 2022-24, held out whole (465 fires, 3,395 fire-days), mean AP<br><sub>FireCast 0.318 [0.304, 0.332], fl_prev 0.242 [0.229, 0.256]. F1 0.222 [0.208, 0.234] against 0.175 [0.165, 0.186]. LANDFIRE fuels date from 2016-2023 releases; no 2024 disturbance layer.</sub> | **0.318** | fl_prev (classical): 0.242<br>Climatological median growth (dumb baseline): 0.209<br>grow_prev (classical): 0.192<br>Persistence (dumb baseline): 0.012 | ✅ wins |
| Replay subset (48 exam fires with LANDFIRE + HRRR, 627 replayable growth fire-days), mean AP<br><sub>The released model (mode F, the calibration shipped in firecast_weights.npz) 0.351 [0.326, 0.376]. Mode L (same network with LANDFIRE inputs and its own dev calibration T 1.57, b −1.20, not shipped) 0.355 [0.332, 0.378]; the handoff quotes mode L. Baselines: fl_prev 0.273 [0.252, 0.294]; replay ensemble 0.165 [0.150, 0.181], deterministic 0.150 [0.133, 0.172]. The replay uses replay_calibration.json as stored and over-predicts new area 3.5× (median log10 ratio +0.54; ensemble +0.59). Of the 713 growth days of these fires, 14 were skipped for crops over 500k cells and 72 have no HRRR coverage; on all 713, FireCast F 0.338 and L 0.340 against fl_prev 0.259. LANDFIRE adds +0.004 AP here and +0.001 on all 713 days.</sub> | **0.351** | Yesterday's growth from the active fire line (fl_prev), same days (classical): 0.273<br>FireSim physics replay, dev-calibrated, 12-member ensemble (classical): 0.165<br>FireSim physics replay, dev-calibrated, deterministic (classical): 0.15 | ✅ wins |
| Replay subset (627 fire-days), mean F1 on new burn<br><sub>FireCast L 0.271 [0.247, 0.291], F 0.265 [0.243, 0.285]; replay deterministic 0.217 [0.199, 0.233], ensemble 0.216 [0.199, 0.232]; fl_prev 0.210 [0.192, 0.227]. Per-method intervals, not paired: F's lower bound (0.243) clears the deterministic replay's upper bound (0.233) by 0.010.</sub> | **0.265** | FireSim physics replay, dev-calibrated, deterministic (classical): 0.217<br>FireSim physics replay, dev-calibrated, 12-member ensemble (classical): 0.216<br>fl_prev, same days (classical): 0.21 | ✅ wins |
| All exam fire-days, median Dice on the total burned area (burned now + predicted new, against the next day's)<br><sub>Most fire-days barely grow, so drawing no new burn scores best on total area. FireCast at its 0.10 threshold over-predicts the new area: median log10(predicted / observed new area) +0.28 (×1.9), against −0.30 for grow_prev, −0.38 for fl_prev and −2.20 for persistence. At threshold 0.5 its total-area Dice is 0.963, still below persistence, and its F1 collapses to 0.058. No interval is shipped for medians. By period: early 0.917, main 0.930, recent 0.907 (persistence 0.961, 0.967, 0.961).</sub> | **0.922** | Persistence (no growth) (dumb baseline): 0.964<br>Climatological median growth (dumb baseline): 0.964<br>fl_prev (classical): 0.947<br>grow_prev (classical): 0.924 | ❌ loses |
| Reliability, pooled exam cells forecast 0.90-0.95 (mode F): share that burned<br><sub>Nominal: the mean forecast in that bin, 92.3 % (123,653 cells). Forecasts are close to the diagonal up to 0.35 (bin 0.30-0.35: forecast 32.4 %, burned 31.8 %), then over-confident: 0.50-0.55 → 43.6 %, 0.70-0.75 → 52.0 %, 0.95-1.00 → 65.1 %. Slightly under-confident at 0.10-0.25 (0.12 → 13.5 %, 0.17 → 19.4 %). Calibration is two dev-fitted numbers with T ≥ 1 (widen only), which cannot reshape the curve; on dev the top bin (0.7-1.0) already read 79 % → 61 %. Mode L on the replay fires: 0.95-1.00 → 13 % (1,860 cells).</sub> | **61.1%** | — | ℹ️ |

### Where it wins

- Beats every baseline on all exam fire-days: mean AP 0.323 [0.318, 0.328] against 0.252 [0.247, 0.257] for yesterday's growth from the fire line, and F1 0.239 against 0.184; in every period (early, main, and the 2022-24 seasons held out whole), all 9 regions and all 5 size bins. Higher AP than the best baseline on 71 % of fire-days.
- More than doubles the dev-calibrated physics replay of its own simulator on the 48 replay fires: AP 0.351 (the released mode F; 0.355 with LANDFIRE inputs, mode L) against 0.165 for the 12-member ensemble, F1 0.265 (mode L 0.271) against 0.217. Trained on noisy, randomised simulations, the network hedges where the deterministic physics over-commits (the replay over-predicts new area 3.5×).
- Probabilities up to about 0.35, where 99.4 % of the scored exam cells sit, are close to the diagonal (slightly under-confident at 0.10-0.25: a 0.12 forecast burned 13.5 %).

### Where it loses

- Total burned-area overlap: persistence (draw no growth) wins, median Dice 0.964 against 0.922. At its threshold FireCast over-predicts the new area 1.9× at the median fire-day.
- Over-confident above about 0.35: a 0.9 forecast burned 61 % of the time, a 0.97 one 65 %.
- Its F1 depends on the dev-fitted threshold 0.10, at the low edge of the searched grid; at 0.5 it is 0.058, below the growth baselines.
- Sim AP 0.59 against real dev AP 0.32: the simulated fires are more predictable than real ones.

## Limits

- It forecasts the FEDS/VIIRS picture of a fire (375 m detections, 12-hourly perimeters), not ground truth.
- Probabilities above about 0.35 are over-confident; the widen-only calibration (2 numbers) cannot reshape the curve. The yes/no threshold (0.10) sits at the edge of the searched grid (0.10-0.60).
- Simulator gaps: no suppression tactics beyond growing containment, no plume-driven runs or wind-event nights (handoff: Riverside 2020 burned 496 km² overnight), no fuel change since the LANDFIRE year, and simulated fires grow on 91 % of half-days against 50 % for real ones. Sim AP 0.59 against real dev AP 0.32.
- LANDFIRE (mode L) adds little: +0.004 AP on the 627 replay days, +0.001 on all 713 growth days of the replay fires (the card says +0.002 to +0.004), and on dev mode L scores below mode F (0.308 vs 0.316). It exists for just the 48 replay fires; the shipped weights carry the mode F calibration only (mode L: T 1.57, b −1.20 in training/results/cal_r2_L.json).
- Only three trained rungs fit on 2 shared CPU cores; rung decisions within ±0.01 AP are at noise level, and no interval on a rung difference can be computed from the shipped files.
- Dev fires fed the physics calibration (FEDS emulator, weather operator, the p1 age factor through the twin test) and also gate every rung and fit the calibration: the gate is partly in-sample. The exam was untouched.
- CONUS 2012-24, MTBS-sized fires (≥ 1,000 acres in the West, 500 in the East), 90 m grid, daily step; the 12 h (morning overpass) product was not built.

## The ratchet

**Objective:** Mean AP over the growth fire-days of a frozen simulated set, exam mode F (fire + gridMET, no LANDFIRE): newly burned 90 m cells within 10 km of the burned area, ranked by the network's raw probability (training/src/evaluate.py). One frozen set per physics version: p0 (FireSim as wrapped by simbank.py: 113 fires, 495 fire-days) and p1 (p0 + the fire-age ROS factor: 149 fires, 562 fire-days).  
**Rule:** Keep if the sim mean AP (mode F) on the current frozen set beats the incumbent's on that same set by more than 0.01 AND the dev mean AP (mode F, 40 real dev fires, 535 growth fire-days) is at least 95 % of the incumbent's.

| Rung | Change | Frozen-set score | Decision | Why | Set | Incumbent on it | Sim AP L | Sim F1 @0.5 | Dev AP | Dev AP L | Dev F1 @0.5 | Dev F1 L | Dev log10 area | Dev Dice total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r0 | Base: p0 bank (668 fires, 2,710 fire-days: 2-8-day FireSim fires on real LANDFIRE landscapes with real HRRR weather, randomised ROS factor, night slow-down, containment, spotting, observed through firesim.observe and the dev-fitted FEDS emulator), U-Net 374k weights, 1,800 steps × 12 crops of 128 cells, lr 2e-3, landscape dropout 0.5 | 0.599 | ● kept | First rung (base). Sim AP 0.599 on p0; re-scored on p1 0.573. Dev AP 0.303 against the best real baseline's 0.252 (fl_prev, dev). | p0 | – | 0.606 | 0.359 | 0.303 | 0.291 | 0.158 | 0.168 | -0.6 | 0.971 |
| r1 | p1 physics: a fire-age ROS factor fitted to the p0 dev twin test (×2.6 on the first day down to ×0.55 after day 10; the p0 first-day boost folded in), init r0, 1,200 steps on the p1 bank (415 fires), lr 1e-3 | 0.582 | ○ rejected | Sim 0.582 vs r0's 0.573 on p1 (+0.009, needs > +0.01): failed. Dev 0.325 vs 0.303 (×1.074) passed; the best dev AP of all four rungs. | p1 | 0.573 | 0.593 | 0.354 | 0.325 | 0.31 | 0.176 | 0.169 | -0.48 | 0.971 |
| r2 | init r0, 1,200 steps on the p0 + p1 banks together (1,083 fires, 4,402 fire-days), lr 1e-3 | 0.587 | **◆ released** | Sim 0.587 vs r0's 0.573 on p1 (+0.014) passed; dev 0.316 vs 0.303 (×1.046) passed. Released as v1.0, with the dev calibration T 1.44, b −1.23, threshold 0.10 (training/results/cal_r2_F.json). | p1 | 0.573 | 0.596 | 0.37 | 0.316 | 0.308 | 0.181 | 0.176 | -0.43 | 0.97 |
| r3 | Longer training: init r2, +1,000 steps on the p0 + p1 banks, lr 5e-4 | 0.588 | ○ rejected | Sim 0.588 vs r2's 0.587 on p1 (+0.001, needs > +0.01): failed. Dev 0.327 vs 0.316 (×1.034) passed. | p1 | 0.587 | 0.598 | 0.375 | 0.327 | 0.314 | 0.188 | 0.175 | -0.42 | 0.97 |

<sub>Columns after *Why*: Sim: the rung's frozen set, its mean AP in mode L (with LANDFIRE) and its mean F1 at 0.5 on that set; dev (40 fires, 535 growth fire-days): mean AP in modes F and L, mean F1 at 0.5 (uncalibrated) in modes F and L, median log10 new-area ratio and median total-area Dice in mode F (training/ratchet.json, eval_*.json).</sub>

The p1 frozen set was built when the p1 physics arrived, and the incumbent r0 was re-scored on it before r1 was decided (0.573; on its own p0 set r0 scored 0.599). r1, r2 and r3 were scored on p1 only: no rung after r0 has a p0 score in the shipped files, although ratchet.py's docstring says both sets are re-scored. All decisions sit within 0.014 AP of each other on p1, and the card calls margins of ±0.01 noise-level (only three trained rungs fit on 2 shared cores); no interval on a rung difference can be computed from the shipped files (they hold means, not per-fire-day scores). r1 was the best rung on dev F and was rejected on the sim margin (+0.009); r3 was better on sim and dev than r2 and was rejected on the margin (+0.001). Dev data fed physics calibration (the FEDS emulator, the weather operator, the p1 fire-age factor from the p0 twin test) and gates every rung, so the gate is partly in-sample. configs/smoke.json was a 2-step plumbing test, not a rung. Calibration (widen-only temperature T ≥ 1 and offset b, then the yes/no threshold) was fitted on the 535 dev fire-days after r2 was chosen; the AP columns are rank-based and do not change with it.

Raw per-rung logs are in [`training/`](training/).

## Physics

FireSim, the shared fire platform's simulator (shared/fire/firesim): Rothermel 1972 surface spread with Albini 1976, Scott & Burgan 40 fuel models, Anderson 1983 ellipses, Van Wagner 1977 crown initiation, Nelson 2000 dead-fuel moisture, minimum-travel-time spread, spotting and timed containment lines, with a VIIRS/MODIS/GOES detection emulator (firesim.observe). FireCast wraps it, without editing it, in two versioned physics files (training/phys/p0.json, p1.json; training/src/simbank.py). Each training fire burns 2-8 days on a real LANDFIRE landscape with real hourly HRRR weather, in 12 h segments, with a randomised rate-of-spread factor (lognormal, median 0.35 of the dev-calibrated replay level, sd 0.7), night slow-down, growing containment and spotting. p1 adds a fire-age ROS factor fitted to the p0 dev twin test (×2.6 on the first day down to ×0.55 after day 10).

**Simulator**

- Bank: p0 668 fires (2,710 fire-days) and p1 415 fires (the p1 bank stopped at 35 of 72 shards); the released rung trained on both (1,083 fires, 4,402 fire-days). Landscapes: 70 % real (dev-fire windows and guard-passed non-exam LANDFIRE windows), 30 % parametric. Ignition between 10:00 and 17:00 in a 40-cell box
- Randomised per fire: ROS factor (lognormal, median 0.35, sd 0.7, clipped 0.03-2.5) with per-fuel-group scatter (sd 0.3); night slow-down ×0.15-1.0; growing containment on 70 % of fires (final fraction 0.2-0.95, time scale 1-4 days); spotting on 30 % (ember factor 0.5-5); wind speed (sd 0.2 per fire, 0.15 per day), wind direction (sd 15° per day) and RH errors; live herbaceous 40-150 %, woody 60-150 %, foliar 80-130 %
- Observation: firesim.observe gives S-NPP VIIRS detections at real overpass times with scan-angle footprints, detection against FRP and clouds on 30 % of fires (10-60 % cover); a FEDS emulator fitted on the 40 dev fires turns them into FEDS-style perimeters and active fire lines (training/results/calib_feds.json: buffer 190 m, closing 1,000 m, line distance 500 m; dev mean Dice 0.950, median log area ratio −0.005, fire-line F1 0.62)
- Weather: the network reads gridMET-like daily features. In simulation they come from the HRRR series through a dev-fitted linear operator with its residual noise (calib_wx.json, 687 dev fire-days: correlation with gridMET 0.40 wind speed, 0.78 min RH, 0.92 max temperature, 0.91 VPD, 0.73 100-h fuel moisture; wind direction offset −10.7°, sd 57°), so the network sees the same kind of weather it gets on exam fires, where HRRR does not exist before Aug 2016
- Tried and not kept: p1 alone (rung r1); see ratchet.yaml

**Checked against independent references**

- FireSim vs published BehavePlus/Rothermel tables (GTR-371, GTR-143): 436 of 436 within half the last published digit; vs pyretechnics on 6,720 flat and slope cases: ROS median relative difference 4.9e-5, maximum 5.2e-5 (shared/fire/checks)
- Uniform-fuel fires against analytic ellipses (55 cases, 30-90 m, 7 wind directions): mean area ratio 0.981, minimum 0.940 (level_set solver 0.992 / 0.973); grid-orientation bias ≤ 3.4 %
- Dev physics replays (40 fires, 470 growth days): next-day F1 on new burn 0.153 uncalibrated, 0.194 dev-calibrated in sample (0.184 leave-one-out); new area over-predicted 4-12× (shared/fire/checks/replay_dev.json)
- Twin test p0 (training/results/twin_p0.json; 470 dev fire-days × 3 simulated members from the observed start): median log10(simulated / real new area) +0.16 (quartiles −0.58, +0.82); 34 % of real days fell below all three members. The status doc's by-age split (first day −0.44, days 6-10 +0.33, days > 10 +0.35) is not in the shipped file. It led to p1
- Twin test p1, partial (stopped at 126 dev fire-days for memory; twin_compare_p0_p1.json): on the same 126 days the first-day median went from −0.82 to +0.20 and days > 10 from +0.44 to +0.01; days 1-10 now about −0.34
- FireData's dev baselines reproduced with this pipeline: fire-line baseline dev AP 0.252 (eval_baselines.json)
- numpy release == torch to 2.5e-6 in the logits (status doc; export.py checks three random grids); the published page's JavaScript == numpy to 2.1e-6 with float16 weights; the site instrument (TypeScript, float16 weights, inputs rebuilt in the browser) == firecast.py on the same weights to 3.9e-6 on 35 cases (logits and the 22 scalar inputs; site/public/instruments/firecast/reference.json)

## Help wanted

- The per-fire-day exam rows (FireCast's out/exam/firecast_r2.jsonl and FireData's spread_exam.jsonl, 89 MB, rebuildable with shared/fire/data/scripts) to put paired intervals on FireCast's margins over the baselines; the shipped summary carries each method's own fire-bootstrap interval only.
- LANDFIRE landscapes and HRRR for more exam fires (only the 48 replay fires are packaged), to test mode L and the physics replay beyond 48 fires.
- Dead-fuel moisture observations (the NFMD moved into FEMS, which has no public download) and ICS-209 suppression records: the twin tests point at persistence and suppression as the largest physics gaps.
- Memory and cores: the p1 twin test stopped at 126 of 470 dev days (1.3-1.45 GB), the p1 bank at 35 of 72 shards, and a 12-member replay ensemble on crops over 500k cells needs more than 1.2 GB.
- Exam grids (audit, 25 Nov 2026): the archived grids are cropped to the final perimeter and exclude 45-59 % of the nominal 10 km window on two packaged days. Padding 112 unburned cells outward put no exterior cell above 0.10 and moved AP by +0.005, -0.003 and 0.000 on the three packaged Mode-F days, so the network does not lean on the grid edge there. Mode FL with real exterior fuels and issue-time grids remain untested.

## Next rungs

- Reliability shape: a monotone (isotonic or beta) calibration on dev, which needs a method decision to replace the widen-only rule; or a distribution-matching term in training, since simulated growth is more predictable than real growth.
- Finish twin p1 with lower memory (stream the dev cache), fit a p2 fire-age factor for days 1-10, then re-score r1 (the best rung on dev).
- Wind-event nights and suppression: condition the night slow-down on gridMET night wind and RH, add a suppression-effort prior by region and size, and run the twin test by wind tercile.
- A lower threshold grid (0.03-0.10) and a 12 h product (the morning overpass as target).
- A bigger bank (finish p1: 35 of 72 shards) and the platform's physics p2 (the diagonal line-leak fix U1 and a persistence module), then re-score every rung.

## Files

| Path | What |
|---|---|
| `firecast.py` | Inference — numpy/scipy only |
| `firecast_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [FEDS-MTBS fire progression (Zenodo 20187963; FEDS method: Chen et al. 2022, Scientific Data), through FireData](https://zenodo.org/records/20187963) | exam (12-hourly perimeters and active fire lines: the inputs and the truth of 3,030 exam fires), dev (40 fires: FEDS emulator fit, weather operator, twin tests, dev gate, calibration); examples/ excerpts | CC BY 4.0 |
| [VIIRS S-NPP 375 m active fire detections (NASA LANCE / FIRMS), as shipped inside FEDS-MTBS](https://firms.modaps.eosdis.nasa.gov/) | dev, exam (hot-spot inputs); physics (the VIIRS emulator was calibrated on dev counts and FRP); examples/ excerpts | NASA open data (no restrictions; cite LANCE/FIRMS) |
| [MTBS burned-area boundaries (USGS / USDA Forest Service), inside FEDS-MTBS](https://www.mtbs.gov/) | dev, exam (the fire set and each fire's grid: final perimeter + 30 %) | Public domain (U.S. Government) |
| [gridMET daily surface meteorology (Abatzoglou 2013, Int. J. Climatology 33:121)](https://www.climatologylab.org/gridmet.html) | dev (weather-operator fit, gate, calibration), exam (the 16 weather inputs of every fire-day); examples/ excerpts | Public domain |
| [LANDFIRE FBFM40, canopy and topography (USGS / USDA Forest Service; LF2016, LF2020, LF2022, LF2023)](https://landfire.gov/) | physics (landscapes for every real-landscape training fire, dev and guard-passed non-exam windows), exam (mode L and the physics replay on the 48 replay fires) | Public domain (U.S. Government) |
| [HRRR analyses (NOAA; AWS Open Data, University of Utah hrrrzarr)](https://registry.opendata.aws/noaa-hrrr-pds/) | physics (hourly weather driving every simulated fire), dev (weather-operator fit, twin tests), exam (the physics replay baseline on the 48 replay fires) | NOAA open data (public) |
| [FireData splits, fire-day tables and baselines (shared/fire/firedata; Wildfire Phase 1)](https://github.com/dogum/ML-Ratchet/tree/main/shared/fire) | dev, exam (frozen splits, score window, metrics, the persistence / climatology / growth / fire-line baselines, the replay subset and replay_calibration.json) | MIT (code); scores derived from the CC BY 4.0 / public-domain data above |
| [WildfireSpreadTS (Gerard et al. 2023, NeurIPS Datasets and Benchmarks; Zenodo 8006177)](https://zenodo.org/records/8006177) | context only: its published test AP (0.19-0.37, a different target at 375 m) is quoted; no WildfireSpreadTS data is used | CC BY 4.0 |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
