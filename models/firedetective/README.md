<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# FireDetective <sub>v1.0</sub>

> **What drove a wildfire, read from its satellite record**

> [!WARNING]
> Research model. On 944 real exam fires it ties the best fire-danger indices at ranking forest burn severity and beats them on its error, ranks fire-day wind better than a Rothermel back-calculation but misses its level (worse than guessing the dev median), and fails on live fuel moisture. Its other outputs (spread-rate factor, spotting, initial attack) are simulator readings with no real truth, and several are not identifiable even in simulation.

*Given a wildfire's first six days from space (burned area every half day, VIIRS fire pixels and their radiative power, the weather and the fuel map), how much of it burned as crown fire, how windy was it, and how wet were the live fuels?*

FireDetective reads a wildfire's first six days (up to 12 half-day steps): the FEDS-style burned-area mask at each VIIRS overpass on a 180 m grid, the VIIRS 375 m fire pixels and their fire radiative power, daily gridMET weather, and LANDFIRE fuels, canopy and terrain. A small 1-D CNN (3 seeds, 189k weights) says what drove the fire, with error bars: the share of the burned area MTBS maps as high severity (through a 2-number map from the simulated crown-fire share, fitted on dev), the daily wind on the fire days read with the wind inputs blanked, and live fuel moisture. It was trained only on about 4,100 FireSim fires on real dev landscapes and real HRRR weather, seen through a simulated satellite, and graded once on 944 real exam fires from 2017-23. In forests its severity rank (rho 0.58) ties FWI and ERC (0.54, 0.53) and its error is the lowest (0.088 vs 0.092); across all fuels Van Wagner's crown-fire potential ranks higher (0.55 vs 0.48, not decisive). Blind wind ranks at rho 0.30 against 0.13 for a Rothermel back-calculation, but its error is worse than the dev median. Live fuel moisture fails (MAE 27.7 vs 21.3 % for a climatology).

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/firedetective/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/firedetective/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | One fire's first 12 half-day steps (S-NPP night ~01:30 and day ~13:30 overpasses), starting at the first step with burned area: cumulative burned-area masks on a 180 m EPSG:5070 grid (real fires: rasterised FEDS-MTBS perimeters), the VIIRS 375 m detections of each step (grid row and column, FRP in MW), a day/night flag, the gridMET day driving each step's growth (tmmx, rmin, rmax, vs, th, fm100, fm1000, erc, bi, pr, vpd), LANDFIRE FBFM40 fuel, canopy cover, height, base height and bulk density, slope and elevation at 180 m, the day of year and the grid convergence angle (pipeline.py turns these into 12 x 25 sequence and 17 static features). |
| **Returns** | hs ± hs_sd: share of the burned area MTBS maps as high severity: hs = 0.017 + 0.271 x simulated crown-fire share (fitted on 30 dev fires), 1-sigma through the same map<br>wind, wind_lo, wind_hi: daily-mean 10 m wind on the fire days (m/s, gridMET definition), inferred with the wind inputs blanked; 68 % range (log-normal)<br>lfmc ± lfmc_sd: live fuel moisture (% dry mass) the fire behaviour implies; fails on the exam<br>Simulator readings, no real truth: crown (simulated crown-fire share), ros_adj (spread-rate factor vs uncalibrated Rothermel), m1 (afternoon 1-h dead moisture, %), wind_regime (wind- vs slope-driven share), p_spot, p_ia (spotting, initial-attack line)<br>Error bars are the 3-seed ensemble's sigma (mean variance + spread of means) times a widen-only dev factor: severity x2.53, wind x1.98, LFMC x1.09, fitted so the 80 % band covers 80 % of dev truths (weights/calib.json); FireDetective.raw() gives the unwidened sigmas |
| **Learned from** | FireSim, the shared fire platform's simulator (ML Ratchet Wildfire Phase 1, shared/fire/firesim, not edited): Rothermel 1972 + Albini 1976 surface spread on the Scott & Burgan 40 fuel models with wind adjustment and slope, Van Wagner 1977 crown-fire initiation and Cruz 2005 / Rothermel 1991 crown spread, Albini spotting with stochastic embers, Nelson-type dead fuel moisture, minimum-travel-time spread on 180 m grids, and its VIIRS detection emulator. FireDetective's bank generator (training/simbank.py) wraps it with randomised drivers and three physics versions found by dev twin tests, then sees each fire through a FEDS-like perimeter emulator tuned on real dev VIIRS pixels. |
| **Noise** | Not real recordings. The weather is real (HRRR drives the simulator, the matching gridMET days are the network's input), the landscapes are real dev LANDFIRE windows, and the satellite view is FireSim's VIIRS emulator plus a FEDS-like perimeter emulator tuned on real dev VIIRS pixels. Wind is blanked in half the training samples (wind-blind mode). |
| **Network** | Three 1-D CNNs (seeds), 62,898 weights each, 188,694 in all. Per member: a 25 → 48 linear step embedding (ReLU), two 3-tap convolutions (48 → 64 → 64, ReLU, time mask), mean and max pooling over the valid steps, a 17 → 32 static branch (ReLU), 160 → 128 → 128 (ReLU), Gaussian heads (mean and log-sigma, clipped to [-5, 3]) for 8 regression targets and 2 logistic heads (spotting, initial attack). Gaussian NLL + BCE, Adam, 60 epochs, one CPU thread, about 4 minutes per seed. Ensemble: mean of means; variance = mean member variance + spread of means. |
| **Physics also enters** | inputs: gridMET fire-danger fields (ERC, BI, 100-h and 1000-h dead fuel moisture) and LANDFIRE fuel groups and canopy fuels, beside the observed growth, shape and FRP |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | FireData's frozen split (shared/fire/data/proc/splits.json): 40 dev fires, CONUS 2017-21; 30 have MTBS severity (16 forest), all 40 gridMET wind, 9 Globe-LFMC samples. Dev is used for the FEDS emulator tuning, the twin tests behind physics v2/v3, the ratchet gate, the severity map and the widen-only factors, and R5's rejected re-weighting: the gate is partly in-sample |
| **Exam** | Exam fires 2017-23 with at least one FEDS growth day: every fire whose dominant fuel group is timber (TU/TL) plus a seeded draw of 250 others (seed 20261119): 944 fires (718 main 2017-21, 226 recent 2022-23). 2024 is excluded because LANDFIRE has no disturbance layer for it yet. Built (inputs and truths) before the ratchet closed, scored once after it (rule note in the exam summary) |

## Run it

```bash
cd models/firedetective
pip install numpy scipy
python demo.py        # about 1 s
```

or from anywhere in the repo: `mlratchet demo firedetective`.

```python
import firedetective as fdet, pipeline, demo
model = fdet.load()                                     # weights/firedetective_v1.npz + weights/calib.json
fire, meta = demo.load_fire('examples/WA4757812313020180804.npz')
out = pipeline.run([fire], model)                       # fire = dict(masks, dets, day, wx, layers, doy, gamma)
out['hs'], out['hs_sd'], out['wind'], out['wind_lo'], out['wind_hi'], out['lfmc']
```

## The exam

944 exam fires, read from their first six days and graded on measurements the network never saw: MTBS high-severity share (357 forest fires with severity, 531 in all fuels), gridMET wind on the fire days (944, the network reads with the wind inputs blanked) and Globe-LFMC samples (175). Run once with R6 after the dev calibration (training/results/exam.json). Classical rivals: max ERC, BI and FWI over the fire days and Van Wagner crown-fire potential (each through the same kind of 2-number dev map for MAE), a Rothermel wind back-calculation from the observed head spread (dev-mapped for MAE), an LFMC climatology (fuel type x month from dev sites) and dev medians. Intervals at import (21 Nov 2026, training/import_check.py, from shipped numbers only): a paired bootstrap over the shipped per-fire rows (10,000 resamples, seed 20261119) where the rows carry the baseline (crown potential, ERC, back-calculation, dev medians), else Zou's interval from exam.json's own 95 % bootstrap intervals (independent: conservative). Verdict: loss if a baseline is better with an interval excluding 0, win only if every baseline is beaten that way, else tie. Result: severity ranks tie (forests: level with FWI and ERC; all fuels: crown potential ahead, interval just includes 0), severity errors win narrowly, wind ranks win, wind level loses to the dev median, live moisture loses on error and ties on rank. Error bars: the 80 % bands cover 69 % of forest severity truths (too narrow), 77 % across fuels, 83 % of wind and 90 % of LFMC. Rule note: the exam input build (exam.py build), including the truth values stored in the same records, ran before the ratchet closed, while R4 was the incumbent; nothing was predicted, scored or opened before the close (model card, ratchet.yaml). Dev twins fed both the physics calibration and the gate. Every number here is on real data that never chose anything.

| Metric | FireDetective | Baselines | |
|---|---|---|---|
| Burn severity, forest fires (forest share ≥ 0.5, n = 357): Spearman rho with the MTBS high-severity share<br><sub>exam.json: model 0.580 [0.510, 0.642]. Difference, model minus baseline: vs FWI +0.044 [-0.058, +0.153] (Zou; FWI is not in the per-fire rows); vs ERC paired +0.046 [-0.041, +0.134] (Zou [-0.062, +0.160]); vs crown potential paired +0.117 [+0.034, +0.205]; vs BI Zou +0.155 [+0.048, +0.267]. Highest rho, but level with FWI and ERC. The status doc and the page call it the best.</sub> | **0.58** | FWI, maximum over the fire days (classical): 0.535<br>ERC, maximum over the fire days (classical): 0.534<br>Van Wagner crown-fire potential (classical): 0.465<br>BI, maximum over the fire days (classical): 0.425 | ➖ tie |
| Burn severity, forest fires (n = 357): mean absolute error of the high-severity share (each method through a 2-number dev map)<br><sub>exam.json: 0.0880 [0.079, 0.098]. Paired difference: vs ERC -0.004 [-0.008, -0.001]; vs crown potential -0.009 [-0.013, -0.005]; vs dev median -0.019 [-0.023, -0.015]. A narrow win: 4 thousandths against ERC. FWI and BI have no MAE in exam.json. Truth median 0.073.</sub> | **0.088** | ERC max, dev-mapped (classical): 0.092<br>Van Wagner crown-fire potential, dev-mapped (classical): 0.097<br>Dev median (0.043) (dumb baseline): 0.107 | ✅ wins |
| Burn severity, all fuels (n = 531): Spearman rho<br><sub>exam.json: 0.484 [0.417, 0.548]; crown potential 0.551 [0.484, 0.608]. Paired difference vs crown potential -0.067 [-0.146, +0.013] (Zou [-0.155, +0.026]): crown potential ranks higher, but the interval includes 0, so by the rule this is a tie, not the loss the card and the status doc describe. Beats ERC (paired +0.096 [+0.021, +0.173]), FWI and BI (Zou, both exclude 0).</sub> | **0.484** | Van Wagner crown-fire potential (classical): 0.551<br>ERC max (classical): 0.388<br>FWI max (classical): 0.308<br>BI max (classical): 0.261 | ➖ tie |
| Burn severity, all fuels (n = 531): mean absolute error (dev-mapped)<br><sub>exam.json: 0.0798 [0.073, 0.087]. Paired difference: vs crown potential -0.006 [-0.009, -0.002]; vs ERC -0.009 [-0.012, -0.006]; vs dev median -0.010 [-0.014, -0.007]. Truth median 0.028: most fires have little high severity, so small errors dominate.</sub> | **0.08** | Van Wagner crown-fire potential, dev-mapped (classical): 0.085<br>ERC max, dev-mapped (classical): 0.089<br>Dev median (dumb baseline): 0.09 | ✅ wins |
| Wind on the fire days, read with the wind inputs blanked (n = 944): Spearman rho with gridMET wind<br><sub>exam.json: 0.298 [0.240, 0.354]; back-calculation 0.129 [0.064, 0.191]. Paired difference (923 fires with both) +0.162 [+0.071, +0.251]; Zou +0.169 [+0.084, +0.255]. Main 2017-21 (718): 0.350 vs 0.106. Recent 2022-23 (226): 0.135 vs 0.207, paired -0.083 [-0.260, +0.098] (tie); the card and the status doc round the recent rho to 0.14, exam.json gives 0.1349. The rho uses the raw back-calculated wind (36 of 923 hit its 30 m/s cap); on dev it correlated -0.12 with gridMET wind.</sub> | **0.298** | Rothermel back-calculation from the observed head spread (n = 923) (classical): 0.129 | ✅ wins |
| Wind on the fire days (n = 944): mean absolute error<br><sub>exam.json: 1.079 [1.005, 1.159]; dev median 0.962 [0.911, 1.021]. Paired difference vs dev median +0.116 [+0.058, +0.175]: the level is worse than a constant. vs the dev-mapped back-calculation +0.033 [-0.034, +0.102]. The dev map of the back-calculation has a negative slope (-0.049 per m/s), so the mapped baseline is nearly a constant that falls with the back-calculated wind.</sub> | **1.079 m/s** | Rothermel back-calculation, dev-mapped (n = 923) (classical): 1.052 m/s<br>Dev median (3.75 m/s) (dumb baseline): 0.962 m/s | ❌ loses |
| Live fuel moisture (n = 175): Spearman rho with the median Globe-LFMC sample within 25 km and ±10 days<br><sub>exam.json: 0.168 [0.022, 0.310]; climatology 0.339 [0.177, 0.466]. Zou difference -0.171 [-0.365, +0.044] (the per-fire rows do not carry the climatology, so no paired interval): not decisive on rank. On the main 2017-21 fires alone the climatology wins, -0.263 [-0.466, -0.036]. The error row below is the failure.</sub> | **0.168** | Climatology: fuel type x month (dev sites) (classical): 0.339 | ➖ tie |
| Live fuel moisture (n = 175): mean absolute error<br><sub>exam.json: 27.7 [24.7, 30.7]; climatology 21.3 [18.4, 24.5]; dev-site median 22.5 [19.7, 25.7]. Difference from the shipped intervals (Zou-style, conservative): vs climatology +6.4 [+2.0, +10.6]; vs dev-site median +5.2 [+0.8, +9.3]. Fails: worse than a constant. Even in simulation the network reads LFMC poorly (R2 0.14); the dev rho of 0.55 was on 9 fires.</sub> | **27.7 % dry mass** | Climatology: fuel type x month (dev sites) (classical): 21.3 % dry mass<br>Dev-site median (dumb baseline): 22.5 % dry mass | ❌ loses |
| Burn severity, recent 2022-23 forest fires (n = 15): Spearman rho<br><sub>Only 15 recent forest fires have MTBS severity in LANDFIRE (it codes severity with a lag). Model 95 % interval [0.50, 0.98]; every difference interval includes 0.</sub> | **0.84** | ERC max (classical): 0.792<br>Van Wagner crown-fire potential (classical): 0.735<br>FWI max (classical): 0.695 | ℹ️ |

Coverage of the 68% error bars on the exam: Severity, forest fires, ±1 sigma (from the shipped per-fire rows, n = 358; the 80 % band covers 69.5 %) **60.6%**, Severity, all fuels, ±1 sigma (rows, n = 531; 80 % band 77.2 %) **69.1%**.

### Where it wins

- Burn severity error: lowest mean absolute error in forests (0.088 vs 0.092 for ERC, 0.097 crown potential, 0.107 dev median) and across fuels (0.080 vs 0.085), every paired interval excluding 0, but by only 0.004-0.006 against the best index.
- Wind read blind: ranks fire-day wind at rho 0.30 against 0.13 for a Rothermel back-calculation from the same observed spread (paired +0.16 [+0.07, +0.25]); 0.35 vs 0.11 on 2017-21 fires.
- Forest severity rank 0.58: the highest of every method, level with FWI (0.54) and ERC (0.53), from a network that never saw a real fire while learning.

### Where it loses

- Live fuel moisture fails: MAE 27.7 % against 21.3 % for a fuel-type x month climatology and 22.5 % for a constant; rho 0.17 vs 0.34.
- Wind level: MAE 1.08 m/s against 0.96 for the dev median (paired +0.12 [+0.06, +0.18]). It ranks windy fires but does not measure wind.
- On recent 2022-23 fires the blind wind rank drops to 0.13, below the back-calculation's 0.21 (interval includes 0).
- Across all fuels Van Wagner's crown-fire potential ranks severity higher (0.55 vs 0.48; paired -0.07 [-0.15, +0.01], not decisive).
- Severity error bars are too narrow: the 80 % band covers 69 % of forest fires (±1 sigma covers 61 %) after a dev widening of 2.5.

## Limits

- Simulated crown-fire share and MTBS 'high severity' are different things (a burnt-out surface fire can be high severity). The 2-number map between them was fitted on 30 dev fires.
- Simulator gaps (twin test): simulated fires grow on 91 % of half-days against 50 % real, are rounder (elongation 0.40 vs 0.70), and have too much night FRP. The persistence gap is shared by every wildfire model here.
- Not identifiable from these inputs even in simulation: spread-rate factor (R2 0.14, 0.01 blind), fuel-model mismatch (0.01), spotting (AUC 0.47). Those outputs are simulator readings, not measurements.
- The dev set is small (40 fires; 16 forest with severity, 9 with LFMC) and fed the emulator tuning, the physics knobs, the gate, the severity map and the widening: the gate is partly in-sample.
- Inputs follow FEDS-MTBS (MTBS-sized fires, ≥ 1,000 acres in the West, 500 in the East), CONUS 2017-23, S-NPP overpasses on a 180 m grid; LANDFIRE fuels can be stale or post-fire for later fire years.
- Rule note: the exam input build, with the truths in the same records, ran before the ratchet closed (nothing was scored or opened before the close).
- Blind mode without fire-danger indices: the matched retrain of 27 Nov 2026 showed that BI (computed from gridMET wind) and ERC carry the wind into blind mode once the flag is scaled normally. Blank BI (and test ERC) in blind mode, then repeat the matched pair R6c / R7f on the same banks; this changes the blind task's definition and needs the owner's decision.

## The ratchet

### FireDetective v1.0 (FireSim physics v1-v3, 18-19 Nov 2026)

**Objective:** Sim primary: mean of max(R2, 0) over three targets on the frozen simulated set of the current physics version: simulated crown-fire share, wind read with the wind inputs blanked, and live fuel moisture (val_v1, then val_v3; about 650 simulated fires each, same seeds)  
**Rule:** Keep if the sim primary beats the incumbent, re-scored on the same frozen set, by more than 0.01 AND the real dev composite (mean of the Spearman rho of predicted crown share vs MTBS high-severity share on the 30 dev fires with severity, and of blind wind vs gridMET wind on the 40 dev fires) is no more than 0.05 below the incumbent's. When the physics (frozen set) changes, the incumbent is re-scored on the new frozen set first.

| Rung | Change | Frozen-set score | Decision | Why | Frozen set | Sim primary | Incumbent sim (same set) | Sim R² crown share | Sim R² blind wind | Sim R² LFMC | Dev ρ severity, all fuels (30) | Dev ρ severity, forest (16) | Dev ρ blind wind (40) | Dev ρ LFMC (9) | Dev composite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 1-D CNN (63k weights) over up to 12 half-day steps x 25 features plus 17 statics, Gaussian heads, half the samples wind-blind; bank v1 (1,340 sims: FireSim + VIIRS and FEDS emulators, real dev HRRR weather, gridMET inputs) | 0.240 | ● kept | First rung (baseline network). | val_v1 | 0.2399 | – | 0.464 | 0.196 | 0.06 | 0.182 | 0.492 | 0.024 | -0.033 | 0.103 |
| R1 | R0 without the shape-elongation input (twin test: simulated fires are rounder than real ones, elongation 0.40 vs 0.70; the FEDS emulator preserves shape, so the gap is physics) | 0.254 | ● kept | sim 0.2542 vs 0.2399 (+0.0143, need > +0.01); dev composite 0.097 vs 0.103 (-0.006, need >= -0.05). | val_v1 | 0.2542 | 0.2399 | 0.459 | 0.194 | 0.109 | 0.221 | 0.434 | -0.027 | 0.133 | 0.097 |
| R2 | Physics v3 in the bank (from dev twin tests): night fine-fuel recovery, anchor-and-flank control lines, and a wind-sensitive WAF in half the sims; trained on v1 + v2 + v3 sims (2,121); new frozen set val_v3, incumbent re-scored on it | 0.261 | ● kept | sim 0.2614 vs R1 re-scored on val_v3 0.2246 (+0.0368); dev composite 0.256 vs 0.097 (+0.159). The v3 physics was fitted to the same dev fires that gate it (rule note). | val_v3 | 0.2614 | 0.2246 | 0.464 | 0.242 | 0.078 | 0.347 | 0.532 | 0.164 | 0.35 | 0.256 |
| R3 | More v3 simulations: the bank grows to 2,548 (v1 1,340 + v2 131 + v3 1,077), same network and seed | 0.307 | ● kept | sim 0.3072 vs 0.2614 (+0.0457); dev composite 0.282 vs 0.256 (+0.026). | val_v3 | 0.3072 | 0.2614 | 0.58 | 0.239 | 0.102 | 0.438 | 0.695 | 0.126 | 0.617 | 0.282 |
| R4 | 3-seed ensemble of the R3 network (mean of means; variance = mean variance + spread of means) | 0.338 | ● kept | sim 0.3383 vs 0.3072 (+0.0312); dev composite 0.348 vs 0.282 (+0.066). The exam input build ran while R4 was the incumbent (rule note). | val_v3 | 0.3383 | 0.3072 | 0.609 | 0.29 | 0.116 | 0.497 | 0.711 | 0.2 | 0.667 | 0.348 |
| R5 | Dev-informed prior: training sims re-weighted toward the dev replay rate-of-spread factors (median 0.45) and fire-season live moisture (100 ± 30 %); 3-seed ensemble | 0.274 | ○ rejected | sim 0.2739 vs 0.3383 (-0.0644, need > +0.01); dev composite 0.269 vs 0.348 (-0.080, need >= -0.05). Fails both. | val_v3 | 0.2739 | 0.3383 | 0.582 | 0.24 | -0.058 | 0.479 | 0.705 | 0.059 | 0.667 | 0.269 |
| R6 | The R4 recipe (3-seed ensemble) on the full v3 bank: train_v3 grows from 25 to 60 shards (about 4,100 sims in all: v1 1,340, v2 108, v3 2,652) | 0.376 | **◆ released** | sim 0.3759 vs R4 0.3383 (+0.0376); dev composite 0.344 vs 0.348 (-0.004, inside 0.05). Released as v1.0; the ratchet closed at 11:53 on 19 Nov 2026, before any exam scoring. On val_v3: crown share R² 0.68 (0.51 wind-blind), blind wind 0.31, LFMC 0.14, 1-h moisture 0.72 (0.22 blind), spread-rate factor 0.14 (0.01 blind), fuel-model mismatch 0.01, spotting AUC 0.47, initial attack AUC 0.81. | val_v3 | 0.3759 | 0.3383 | 0.682 | 0.307 | 0.139 | 0.581 | 0.753 | 0.108 | 0.55 | 0.344 |
| calib | Dev calibration of R6 (training/calibrate.py, 40 dev fires): severity map hs = 0.017 + 0.271 x crown share (least squares on 30 dev fires; the same 2-number map is fitted for every classical index), and widen-only sigma factors so the 80 % band covers at least 80 % of dev truths | — | ● kept | Factors: severity x2.53, wind x1.98, LFMC x1.09 (9 dev fires). No effect on the ranks; the dev fires that gate the rungs also fit the map and the factors. | – | – | – | – | – | – | – | – | – | – | – |

<sub>Columns after *Why*: training/ratchet.json: sim primary on the rung's frozen set and the incumbent's on the same set, then the real dev gate on 40 dev fires (Spearman rho; raw network outputs).</sub>

Physics versions (training/simbank.py wrappers around FireSim, no edit to the shared package): v1 FireSim as shipped; v2 + night fine-fuel moisture recovery and anchor-and-flank control lines; v3 = v2 + a fixed wind adjustment factor U(0.25, 0.6) in half the runs. v2 and v3 came from twin tests on the dev fires (training/twin.py, results/twin_v1.json and twin_waf.json), so dev data fed the physics AND the gate: the gate is partly in-sample. The FEDS perimeter emulator was also fitted on real dev VIIRS pixels (results/calib_feds.json). A frozen set val_v2 was built (results/compare_val_v2.json) but no rung was scored on it: R2 went straight to the v3 bank and val_v3. Each gate compares a rung with the current incumbent only. The dev composite uses the all-fuel severity rho (30 fires), not the forest one. R6's dev wind rho (0.11) and dev wind MAE (1.21 m/s) are worse than R4's (0.20, 1.10); its composite still passed (-0.004). The bank counts in ratchet.json disagree a little: R3 says v2 has 131 sims, the status doc says v2 has 108 in the final bank (the banks are not shipped). Rule note (disclosed in the model card and in ratchet.json 'closed'): `exam.py build` ran on 19 Nov 10:15-10:23 local, while R4 was the incumbent and before R6 was trained and the ratchet closed (11:53). It fetched LANDFIRE for the 944 listed exam fires, built their input records, which hold the truth values in the same files, and computed the classical index values. Nothing was predicted, scored, compared or opened before the close; R6 (the R4 recipe on more simulations) was decided on the frozen sim set and dev only, and `exam.py score` ran once after calibration.

### FireDetective audit follow-up: blind-wind flag scaling, matched retrain (27 Nov 2026)

**Objective:** As above, on val_v3  
**Rule:** As above: keep if the sim primary beats R6 (released) on val_v3 by more than 0.01 AND the dev composite is no more than 0.05 below R6's. The exam is opened once, as a new version, only for a promoted rung.

| Rung | Change | Frozen-set score | Decision | Why | Frozen set | Sim primary | Incumbent sim (same set) | Sim R² crown share | Sim R² blind wind | Sim R² LFMC | Dev ρ severity, all fuels (30) | Dev ρ severity, forest (16) | Dev ρ blind wind (40) | Dev ρ LFMC (9) | Dev composite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R6c | Control: R6's recipe on the rebuilt bank, flag scaled as shipped (sd floor 1e-3) | 0.390 | ○ rejected | Not a candidate: it measures the rebuild. sim 0.3899 vs R6 0.3759 (+0.014); dev composite 0.325 vs 0.344 (-0.019). | val_v3 | 0.3899 | 0.3759 | 0.674 | 0.357 | 0.138 | 0.536 | 0.744 | 0.115 | 0.683 | 0.325 |
| R7f | R6's recipe with the blind-wind flag stored at sd 1 (train.py --flag-sd1) | 0.612 | ○ rejected | Not promoted, held for the owner although it passes the rule's numbers (sim 0.6122 vs 0.3759, +0.236; dev composite 0.730 vs 0.344) but held, exam not opened: every blind output now equals the visible one (sim blind-wind R² 0.990 vs visible 0.994; crown 0.707 vs 0.706). BI, computed from the gridMET wind that is blanked in blind mode and is also the dev wind truth, stays in the blind input; with the flag at z = 1000 the network could not use it, at z = 1 it does. Neutralising BI drops R7f's blind-wind correlation with the sim truth from 0.995 to 0.224 (R6: 0.556 to 0.554). The gain is a leak, not a better blind estimate. | val_v3 | 0.6122 | 0.3759 | 0.706 | 0.99 | 0.14 | 0.551 | 0.742 | 0.908 | 0.867 | 0.73 |

<sub>Columns after *Why*: training/rung_R7f.json.</sub>

R6's training bank was not shipped; it was rebuilt with the shipped simbank.py commands, cut to the shards R6's log implies (train_v1 shards 0-30 = 1,340 sims, train_v2 0-2 = 108, train_v3 0-59 = 2,652: all three counts exact). Both arms are R6's recipe (3 seeds 0-2, elongation dropped); checkpoints now also store the torch and numpy RNG states. R6c (the shipped 1e-3 floor) is the control: it does not reproduce R6 exactly (sim primary 0.390 vs 0.376, dev composite 0.325 vs 0.344), so the rebuild alone moves these scores by about that much. R7f stores the flag with sd 1. Log: training/rung_R7f.json.

Raw per-rung logs are in [`training/`](training/).

## Physics

FireSim, the shared fire platform's simulator (ML Ratchet Wildfire Phase 1, shared/fire/firesim, not edited): Rothermel 1972 + Albini 1976 surface spread on the Scott & Burgan 40 fuel models with wind adjustment and slope, Van Wagner 1977 crown-fire initiation and Cruz 2005 / Rothermel 1991 crown spread, Albini spotting with stochastic embers, Nelson-type dead fuel moisture, minimum-travel-time spread on 180 m grids, and its VIIRS detection emulator. FireDetective's bank generator (training/simbank.py) wraps it with randomised drivers and three physics versions found by dev twin tests, then sees each fire through a FEDS-like perimeter emulator tuned on real dev VIIRS pixels.

**Simulator**

- Bank: about 4,100 accepted simulations (burned area ≥ 2 km²; v1 1,340, v2 108, v3 2,652), and frozen validation sets val_v1, val_v2, val_v3 of about 650 fires each with the same seeds. Each runs on a coarsened 180 m LANDFIRE window of one of the 40 dev fires or on a parametric landscape, under real hourly HRRR weather from dev fire dates; the network reads the matching real gridMET days, so its weather carries real analysis error
- Randomised drivers: spread-rate factor, live and dead moisture factors, crown fire on/off and Cruz vs Rothermel crown spread, spotting, initial-attack lines. Physics v2 adds night fine-fuel moisture recovery and anchor-and-flank control lines; v3 adds a fixed wind adjustment factor U(0.25, 0.6) in half the runs (twin test: sim-vs-real per-step growth rho 0.36 → 0.44)
- Observations: FireSim's VIIRS emulator (detection probability vs FRP, scan angle, cloud omission, commission), then a FEDS-like perimeter emulator (pixel footprints, closing radius 3 cells, hole filling; feats.feds_emulate) tuned on real dev VIIRS pixels against real FEDS perimeters: median log10 area ratio +0.012, MAD 0.020 on 1,646 dev half-day steps (training/results/calib_feds.json)
- Labels: simulated crown-fire share (passive + active) of the burned area, mean daily wind, live fuel moisture, spread-rate factor, 1-h moisture, wind-driven share, fuel-model mismatch, spotting and initial attack, straight from the simulator

**Checked against independent references**

- FireSim Phase 1 (shared/fire/checks): 436 of 436 published BehavePlus / Rothermel cases within tolerance (published_examples.json); WAF vs pyretechnics max difference 3e-8 and crown fire-type agreement 1.0 (pyretechnics_crosscheck.json); two-fuel strip arrival time vs the hand solution 3e-9 relative (analytic.json); Van Wagner I0 example 2,476 vs quoted 2,500 kW/m (crown.json)
- Twin test (training/twin.py, results/twin_v1.json): each dev fire re-simulated from its first mapped perimeter with its own fuels and weather. Sims grow on 91 % of half-days vs 50 % real, have 3x the VIIRS pixels (log10(1 + pixels) 1.38 vs 0.85), more night FRP (0.86 vs 0.50) and rounder shapes (elongation 0.40 vs 0.70)
- numpy == torch: max difference 6e-7 (means), 9e-6 (log-sigma), 9e-8 (probabilities) on 40 dev fires (weights/export_check.json). The site instrument recomputes the features from the raw record and runs the ensemble in TypeScript: equal to feats.py + firedetective.py to 8e-16 on 40 cases (4 fires x 10 what-ifs, float16 weights); float16 weights move the outputs by ≤ 3e-4 relative
- Import check (21 Nov 2026, training/import_check.py): the shipped per-fire exam rows re-derive exam.json's headline numbers (differences ≤ 0.003, from the rows' rounding), and the four example fires' raw observations through pipeline.py reproduce their exam rows exactly to the shipped digits

## Help wanted

- Per-fire MTBS burn-severity mosaics (dNBR thematic rasters) are behind an e-mail order queue; LANDFIRE Annual Disturbance codes were used instead, which lag (only 15 recent forest fires have severity).
- Real dead-fuel moisture sticks: NFMD moved into FEMS and has no public download.
- More dev fires with severity and LFMC truth, for the severity map, the widening and the gate.
- Inputs (audit, 25 Nov 2026): an empty burned-area history overflowed to infinite wind; predict() now refuses fires with no observed step (every exam fire had 3-12). The blind-wind flag is scaled by a 0.001 sd floor (z = 1000 in blind mode). The matched retrain (27 Nov 2026; rebuilt bank, 3 seeds per arm; training/rung_R7f.json) answered the audit's question the other way round: with the flag at sd 1 (R7f) the blind outputs equal the visible ones (sim blind-wind R² 0.99 vs 0.31 for R6; dev blind-wind ρ 0.91 vs 0.11), because BI, computed from the gridMET wind that is both the blanked input and the dev truth, stays in the blind input; neutralising BI drops R7f's blind-wind correlation with the truth from 0.995 to 0.224 and R6's from 0.556 to 0.554. The z = 1000 scaling kept the released model off that path. R7f passes the rule's numbers but was held, and the exam was not opened.

## Next rungs

- A FireSim persistence knob (burnout, stalls, night inactivity): sims grow on 91 % of half-days vs 50 % real. The same gap drives FireCast's over-prediction.
- Simulate a dNBR-like severity label (heat per unit area plus crown) instead of crown share, so the dev map is not needed.
- Conformal or heteroscedastic severity bands per fuel group, fitted on more dev fires.
- Drop live fuel moisture or replace it with a phenology input.
- Validate growth against GOFER hourly perimeters.

## Files

| Path | What |
|---|---|
| `firedetective.py`, `feats.py`, `pipeline.py` | Inference — numpy/scipy only |
| `weights/firedetective_v1.npz`, `weights/calib.json` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [FEDS-MTBS fire event data set (Chen et al. 2022, Scientific Data, FEDS method; MTBS-constrained release, Zenodo 20187963) with VIIRS S-NPP 375 m fire pixels, via FireData](https://doi.org/10.5281/zenodo.20187963) | dev (40 fires: emulator tuning, twins, gate, calibration), exam (944 fires: inputs) | CC BY 4.0. The example fires' masks and VIIRS pixels in examples/ are excerpts |
| [MTBS burn severity (USGS/USFS), via LANDFIRE Annual Disturbance (MTBS codes 11-14 and 111-114), fetched per fire from LFPS](https://lfps.usgs.gov) | truth (high-severity share), dev, exam | Public domain (U.S. Government) |
| [LANDFIRE fuels (FBFM40), canopy (CC, CH, CBH, CBD) and topography, via the LANDFIRE Product Service](https://landfire.gov) | physics (simulated landscapes), dev, exam (inputs) | Public domain (U.S. Government) |
| [gridMET daily surface meteorology and fire-danger indices (Abatzoglou 2013)](https://www.climatologylab.org/gridmet.html) | dev, exam (inputs; wind truth with the wind inputs blanked) | Public domain |
| [HRRR hourly analyses (NOAA, via hrrrzarr on AWS Open Data)](https://registry.opendata.aws/noaa-hrrr-pds/) | physics (weather driving the simulated dev-landscape fires) | NOAA, public |
| [Globe-LFMC 2.0 live fuel moisture samples (Yebra et al. 2024)](https://doi.org/10.6084/m9.figshare.25413790) | truth (median sample within 25 km and ±10 days), dev (9 fires), exam (175 fires); climatology baseline from dev sites | CC0 (the shipped card credits the 2019 Globe-LFMC paper as CC BY; FireData used version 2.0, which is CC0) |
| [FireData Phase 1 (frozen splits, fire tables, exam guard) and FireSim (ML Ratchet Wildfire Phase 1)](https://github.com/dogum/ML-Ratchet/tree/main/shared/fire) | physics, dev, exam (fire lists and splits) | MIT (code); derived from the sources above |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
