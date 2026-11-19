# FlowCast v1.0: model card

**What it does.** It predicts daily streamflow (mm/day and m³/s) for an **ungauged** basin from basin-average daily
weather and 11 static attributes, with 5/25/50/75/95 % quantiles. It never reads streamflow from the basin it
predicts. It was trained **only on simulations** (WaterSim, the water platform's simulator) and graded once on 619 real
CAMELS-US exam basins.

Part of ML Ratchet ("Trained on physics. Graded on reality."), water platform (Watershed Phase 2, project 1).
Imported into the repository on 18 Nov 2026 (see `NOTEBOOK.md`); weights and inference numbers unchanged.

> **Read before comparing.** All forcing is ERA5-Land through Caravan. ERA5-Land under-catches mountain and Pacific
> Northwest precipitation (50 of the 671 CAMELS-US basins have a validation-period runoff ratio above 1) and drifts
> about 6 % between decades. That caps every method in this card, the baselines included. Published LSTMs driven by
> Daymet or NLDAS (median NSE about 0.69-0.76 on CAMELS) are not comparable with these numbers.

## Inputs and outputs
- **Forcing** (daily, UTC day, Caravan/ERA5-Land conventions): `P` mm/d, `PET` mm/d (FAO-56 Penman-Monteith),
  `T_mean`, `T_min`, `T_max` °C, `SRAD` W/m² (**net** shortwave), `vp` Pa, `wind` m/s. It needs 365 days before
  the first predicted day (warm-up) and one day after the last.
- **Attributes:** area km², elevation mean/min/max m, slope (m/km, or HydroATLAS degrees with
  `slope_unit='degrees'`), forest fraction, soil porosity, soil conductivity (cm/h, or mm/h with
  `soil_conductivity_unit='mm/h'`), carbonate (karst) fraction, reservoir storage / mean annual flow, irrigated
  fraction, urban fraction. Seven climate indices are computed from the forcing itself (mean P and PET, aridity,
  snow share of P, high- and low-precipitation day frequencies, mean T), over the window being predicted plus its
  warm-up.
- **Outputs:** `Q` (the mean, trained with an NSE-type loss), `q05 … q95` (pinball loss, widened on dev),
  `Q_m3s` when the area is given.

## Network
One LSTM layer (64 units; 24,710 weights) over 11 daily weather features plus the 18 static features repeated at
each step; a linear head gives the mean and five monotone quantiles, all in units of the basin's mean
precipitation. Inference runs in 365-day blocks, each with its own 365-day warm-up (as in training). numpy release
equals torch to 1.7e-7 relative (`training/results/export_check.json`). The site instrument's TypeScript port equals
numpy on the float16 weights to 5.3e-8 on 54 cases (the shipped page's JS port was reported at 7.6e-6).
The shipped card gave 24,966 weights; the shipped weight files hold 24,710 (256 × 29 input, 256 × 64 recurrent,
2 × 256 biases, 6 × 64 + 6 head; plus 36 standardisation constants). The nowcast copy has 25,478.

## Simulator and training data
- **WaterSim** (`shared/water/watersim`): HBV, GR4J, SAC-SMA and VIC-lite; degree-day or energy-balance snow on
  elevation bands; unit-hydrograph or Muskingum-Cunge routing; reservoirs and irrigation. Checks in
  `shared/water/checks/` (GR4J vs airGR max |ΔQ| 1.0e-7 mm; HBV vs TUWmodel ≤ 1e-12 mm on the stores; water balance
  closes to 1.4e-15 relative over 121 runs).
- **Bank:** 12,000 runs per physics version (+800 frozen validation runs). Each run: a random basin (50 % jittered
  dev-basin attributes, 50 % parametric prior), random latent and per-parameter scatter, one of four models, one of
  two snow routines, 25 % Muskingum-Cunge routing; 5 years of real ERA5-Land forcing from a random CAMELS basin in a
  non-exam window (1951-1988, 1999-2015; 1 year spin-up, 4 years stored).
- **Real noise:** the network reads forcing after WaterSim's forcing-error model (P bias and daily scatter, snow
  undercatch, T error); the target is gauge-like flow (AR(1) rating error, revisions, low-flow error).
- **Physics rung kept (p1, water yield):** the basin's true precipitation is ERA5-Land × m, with
  log m = −0.088 + 0.058 elev_km + 0.153 snow share + 0.397 log(P/3 mm) + N(0, 0.146), fitted on the 52 dev basins
  by matching each basin's observed mean flow with the attribute-prior simulator (`training/src/calib_water.py`).
  m > 1 in wet mountain and Pacific Northwest basins (ERA5-Land undercatch), m < 1 in dry plains basins (losses).

## Ratchet (frozen simulated set = median NSE of 800 sim basins; dev = 52 real dev basins, WY2001-2014, ungauged)
Rule: keep if sim > incumbent + 0.01 on the current frozen set AND dev median NSE ≥ 95 % of the incumbent's.

| Rung | Change | Frozen set | Sim NSE (vs incumbent) | Dev NSE | Dev KGE | Dev FHV % | Kept |
|---|---|---|---|---|---|---|---|
| r0 | WaterSim as shipped | p0 | 0.441 | 0.390 | 0.319 | −51 | base |
| r1 | p1 water yield, old loss | p1 | 0.332 (0.397) | 0.331 | 0.259 | −59 | no |
| r0b | loss scaled by Fu runoff ratio of the forcing (input-only) instead of the target's own std | p0 | 0.476 (0.441) | 0.436 | 0.442 | −27 | **yes** |
| r1b | p1 water yield + new loss | p1 | 0.425 (0.395) | 0.425 | 0.424 | −32 | **yes, released** |
| r2 | p2: full dev prior shift (flashiness 0.62× → 1.00× of observed) | p2 | 0.463 (0.446) | 0.380 | 0.490 | −16 | no (dev NSE −11 %) |
| r3 | p3: forcing error cv 0.4 + timing jitter 30 % | p3 | 0.326 (0.335) | 0.406 | 0.430 | −38 | no |
| r4 | p4: p3 + p2 | p4 | 0.404 (0.380) | 0.378 | 0.490 | −23 | no (dev NSE −11 %) |
| r5 | r1b fine-tuned 5,000 more steps | p1 | 0.419 (0.425) | 0.434 | 0.441 | −28 | no |

Dev reference on the same window (as reported; `devref.parquet` was not shipped): U-dev regionalisation 0.466, NWM
0.573, GR4J calibrated 0.586 (partly in-sample). Every rung was re-scored on every frozen set
(`training/results/sim_<rung>_<set>.json`; matrix in `ratchet.yaml`). A paired bootstrap over the 800 frozen basins,
added on import, puts r1b's gain over r0b on p1 at +0.030 (95 % CI +0.001 to +0.052) and r5's change at −0.006
(−0.019 to +0.015): the 0.01 margin is at the noise level.

**Twin test (r1b, dev, `twin.py`):** median NSE of the same prediction against the real record 0.425, the
calibrated twin 0.595, the prior twin 0.380. Flashiness ratio (prediction / target) real 0.79, calibrated twin 0.75:
the network reads real and calibrated twin alike. No same-day/next-day timing offset was found (best mixing 0).

## Error bars
Pinball-loss quantiles, widened only, one factor per band fitted on dev: 50 % band × 1.00, 90 % band × 1.08 about
the median (pooled dev coverage 0.50 and 0.95; the 1.08 mostly lowers the bound to 0 on zero-flow days).

## Final exam (run once; medians over basins)
| Method | Protocol | NSE main 1990-99 | KGE main | FHV % | NSE < 0 | NSE recent 2016-23 | KGE recent |
|---|---|---|---|---|---|---|---|
| **FlowCast v1.0** | ungauged | **0.341** | 0.367 | −31 | 14 % | **0.378** | 0.397 |
| NN regionalisation, dev donors (U-dev) | ungauged | 0.372 | 0.433 | −27 | 14 % | 0.433 | 0.469 |
| NN regionalisation, 12-fold (U-k12) | ungauged | 0.385 | 0.462 | −21 | 14 % | 0.443 | 0.469 |
| GR4J + snow, calibrated per basin | gauged | 0.473 | 0.602 | −18 | 6 % | 0.517 | 0.604 |
| NOAA NWM v3.0 | operational | 0.550 | 0.596 | −13 | 9 % | 0.687 | 0.802 |
| Persistence, 1 day | gauged forecast | 0.475 | 0.738 | 0 | 15 % | 0.480 | 0.740 |
| Climatology | gauged | 0.065 | −0.054 | −81 | 16 % | 0.053 | −0.065 |

n = 619 (main), 603 (recent); NWM 600 in both. CAMELS-531 subset: 0.357 (U-dev 0.402).

### Verdicts and intervals (both exams and the flood peaks re-scored 18 Nov 2026; seen)
On 18 Nov 2026 the released weights were re-scored on the main exam (WY1990-99) from the rebuilt Caravan data
(`training/src/rescore_main.py`; the 619 basins are exactly `splits.json` camels_us exam, whose hashes verified). It
reproduces every claimed median (NSE 0.34117 vs 0.34117 claimed; KGE, FHV, peaks, coverage, CAMELS-531, strata and the
nowcast at all three leads to within 4e-7 relative, the largest gap FLV 59.11915 vs 59.11918 %; the win
fractions against every baseline exactly). The exam had already been opened, so this is a seen reproduction; nothing was
changed because of it. Main-exam intervals: paired bootstrap over basins of the difference in medians (FlowCast −
baseline) on the basins both have, 10,000 resamples, seed 20261118, using WaterData's per-basin baseline scores
(`training/results/rescore_main_summary_2026-11-18.json`; per-basin FlowCast scores in
`rescore_main_2026-11-18.json`). Win or loss only if the 95 % interval excludes 0. The recent exam (WY2016-23) and the
instantaneous flood peaks were re-scored the same way from the NWIS data later the same day (`training/src/rescore_recent.py`, `rescore_recent_summary_2026-11-18.json`); they too
reproduce every claimed median.

| Row | FlowCast | Baseline | Median difference (95 % CI) | Verdict |
|---|---|---|---|---|
| Main NSE, ungauged | 0.341 | U-dev 0.372 | −0.031 (−0.056 to −0.014) | loss |
| | | U-k12 0.385 | −0.044 (−0.067 to −0.028) | loss |
| | | climatology 0.065 | +0.276 (+0.251 to +0.294) | win |
| Main NSE, other protocols | 0.341 | GR4J (gauged) 0.473 | −0.132 (−0.157 to −0.109) | loss |
| | 0.345 (600) | NWM 0.550 | −0.205 (−0.235 to −0.182) | loss |
| | | persistence 0.475 | −0.134 (−0.182 to −0.087) | loss |
| Main KGE | 0.367 | U-dev 0.433 / U-k12 0.462 | −0.067 (−0.087 to −0.039) / −0.095 (−0.118 to −0.069) | loss |
| Main median \|FHV\| | 39.1 % (signed −30.9 %) | U-dev 34.3 % / U-k12 31.3 % | +4.8 (+2.9 to +7.8) / +7.7 (+5.6 to +11.6) | loss |
| Main annual daily peaks, abs. error | 52.2 % | U-dev 47.3 % / U-k12 45.5 % | +4.9 (+2.7 to +6.9) / +6.7 (+4.4 to +9.2) | loss |
| Main peak timing, days | 0.85 | U-dev 0.82 / U-k12 0.82 | +0.03 (−0.03 to +0.06) / +0.03 (0.00 to +0.06) | tie |
| CAMELS-531 NSE (487 exam basins) | 0.357 | U-dev 0.402 / U-k12 0.401 | −0.045 (−0.063 to −0.017) / −0.044 (−0.068 to −0.023) | loss |
| Recent NSE, ungauged | 0.378 | U-dev 0.433 | −0.054 (−0.068 to −0.034) | loss |
| | | U-k12 0.443 | −0.064 (−0.094 to −0.046) | loss |
| | | climatology 0.053 | +0.325 (+0.307 to +0.341) | win |
| Recent NSE, other protocols | 0.378 | GR4J (gauged) 0.517 | −0.139 (−0.159 to −0.120) | loss |
| | 0.378 (600) | NWM 0.687 | −0.309 (−0.329 to −0.287) | loss |
| | | persistence 0.480 | −0.101 (−0.146 to −0.055) | loss |
| Recent KGE / median \|FHV\| / daily peak error vs U-dev | 0.397 / 40.1 % / 52.7 % | 0.469 / 35.3 % / 47.7 % | −0.073 (−0.089 to −0.046) / +4.8 (+2.8 to +7.5) / +5.0 (+2.6 to +7.4) | loss |
| Recent CAMELS-531 NSE (475 exam basins) | 0.393 | U-dev 0.448 / U-k12 0.475 | −0.056 (−0.073 to −0.039) / −0.082 (−0.102 to −0.056) | loss |
| Flood peaks (instantaneous), main, abs. error | 59.8 % (bias −45.8 %) | own-record median 42.2 % | +17.6 (+14.2 to +20.2) | loss |
| | | GR4J × 1.38 46.5 % | +13.2 (+10.2 to +15.1) | loss |
| | | NWM daily × 1.38 42.4 % (588) | +17.2 (+13.3 to +20.4) | loss |
| Flood peaks, recent, abs. error | 59.7 % (bias −48.5 %) | own-record 43.5 % / GR4J 47.8 % | +16.2 (+13.2 to +19.4) / +11.9 (+9.2 to +14.3) | loss |
| | | NWM daily 34.1 % / NWM hourly 30.6 % | +25.7 (+22.6 to +27.9) / +29.1 (+26.0 to +31.8) | loss |
| 90 % band coverage (median basin) | 97.3 % | nominal 90 % | — | info (too wide) |

Cross-check, main exam (exact binomial on the share of basins where FlowCast's NSE is higher): U-dev 34.1 %
(30.4-38.0), U-k12 32.6 % (28.9-36.5), GR4J 19.5 % (16.5-22.9), NWM 21.3 % (18.1-24.8), persistence 44.4 %
(40.5-48.4), climatology 80.6 % (77.3-83.7). Both methods give the same verdicts. Against persistence the FHV and
peak comparisons are degenerate (persistence copies the observed series: 0 by construction). Recent exam:
U-dev 34.0 % (30.2-37.9), U-k12 34.2 % (30.4-38.1), GR4J 20.1 % (16.9-23.5), NWM 15.5 % (12.7-18.6), persistence
47.9 % (43.9-52.0: a tie on that count, a loss on the paired median), climatology 82.6 % (79.3-85.5).

By basin type (main exam, median NSE: FlowCast / U-dev / NWM / GR4J; breakdowns, `info`): snow (frac_snow > 0.3,
n 118) 0.40 / 0.56 / 0.57 / 0.67; arid (aridity > 1, n 174) 0.21 / 0.20 / 0.32 / 0.30; humid (aridity < 0.6, n 128)
0.49 / 0.62 / 0.64 / 0.66; regulated (n 37) 0.30 / 0.25 / 0.46 / 0.56; large > 2000 km² (n 48) 0.23 / 0.22 / 0.42 /
0.38. Recent: snow 0.41 / 0.56 / 0.70 / 0.70; arid 0.19 / 0.20 / 0.48 / 0.29; humid 0.49 / 0.61 / 0.72 / 0.69;
regulated 0.38 / 0.46 / 0.70 / 0.59; large 0.34 / 0.32 / 0.52 / 0.44. The regulated "win" of the 1990s reverses on
the recent exam.
Paired against U-dev on the main exam (same bootstrap): snow −0.154 (−0.210 to −0.067), arid +0.009 (−0.040 to
+0.033), humid −0.124 (−0.166 to −0.086), regulated +0.054 (−0.127 to +0.130), large +0.006 (−0.113 to +0.121).
Recent exam: snow −0.150 (−0.193 to −0.067), arid −0.008 (−0.052 to +0.058), humid −0.123 (−0.155 to −0.084),
regulated −0.079 (−0.127 to +0.074), large +0.022 (−0.052 to +0.141).

**Flood peaks** (USGS annual instantaneous peaks; annual max daily × 1.38; median abs error / median bias):
FlowCast mean 60 % / −46 % (main), 60 % / −49 % (recent); FlowCast q95 64 % / +31 % (main), 61 % / +28 % (recent) — the RAW pinball q95 (evalreal.py stored
unwidened quantiles; the released, widened q95 gives 66 % / +38 % and 63 % / +35 %, re-score 18 Nov);
baselines (main): own-record median 42 / −14, GR4J 47 / −31, NWM daily 42 / −24; recent: own-record 43 / −8, GR4J
48 / −32, NWM daily 34 / −12, NWM hourly 31 / −2.

**Coverage:** the 90 % band holds 97 % of days at the median basin (95 % averaged over basins); the 50 % band 56 %.
Bands are too wide in most basins and too narrow in some dry ones.

## Optional gauged nowcast mode
`flowcast_nowcast_weights.npz` (see `NOWCAST.md`): r1b fine-tuned to also read the observed flow up to 1-7 days
before each day. Gauged protocol, hindcast weather. Medians over 619 (main) and 603 (recent) exam basins:

| Lead | Main NSE | Persistence | Beats persistence at (95 % CI) | GR4J / NWM | Verdict | Recent NSE | Persistence | Beats persistence at | NWM | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 day | **0.569** | 0.475 | 65.4 % (61.5-69.2) | 0.473 / 0.550 | tie (NWM, no interval) | **0.591** | 0.480 | 66.3 % (62.4-70.1) | 0.687 | loss |
| 3 days | **0.474** | −0.190 | 84.8 % (81.7-87.6) | 0.473 / 0.550 | loss (NWM) | **0.511** | −0.210 | 85.2 % (82.2-88.0) | 0.687 | loss |
| 7 days | **0.426** | −0.465 | 90.0 % (87.3-92.2) | 0.473 / 0.550 | loss | **0.466** | −0.458 | 88.7 % (85.9-91.1) | 0.687 | loss |

Main-exam re-score (18 Nov, paired bootstrap of the median difference as above): vs persistence +0.093
(+0.060 to +0.133), +0.665 (+0.630 to +0.705), +0.892 (+0.859 to +0.923) at 1 / 3 / 7 days; vs GR4J +0.095
(+0.073 to +0.122), +0.001 (−0.016 to +0.027), −0.047 (−0.064 to −0.025); vs NWM (600 basins) +0.025 (−0.011 to
+0.058), −0.072 (−0.100 to −0.034), −0.121 (−0.147 to −0.090). Recent exam re-score (NWIS): vs persistence +0.112 (+0.077
to +0.152), +0.721 (+0.655 to +0.777), +0.924 (+0.869 to +0.961); vs GR4J +0.074 (+0.055 to +0.097), −0.006
(−0.027 to +0.010), −0.052 (−0.071 to −0.036); vs NWM −0.096 (−0.119 to −0.065), −0.176 (−0.201 to −0.152),
−0.222 (−0.244 to −0.201).
It beats persistence clearly at every lead. Against the fair gauged and operational references (GR4J calibrated on
the gauge: 0.473 main, 0.517 recent; NWM: 0.550, 0.687) it ties NWM at one day in the 1990s and loses otherwise. KGE
at one day is below persistence's (0.635 vs 0.738 main).

## Where it fails
- Overall it does not beat classical ungauged regionalisation; U-dev borrows GR4J parameters calibrated on the real
  dev gauges, information the network only gets through the simulator.
- Snowy and humid basins: −0.13 to −0.16 NSE against U-dev. No snow rung was tried; only 10 dev basins are snowy.
- Peaks are 46 % low (a conditional mean under parameter uncertainty is smooth). Flashier physics fixed peaks on dev
  but cost NSE (rungs r2, r4).
- Wet Pacific Northwest basins can be over-predicted (the water-yield multiplier is a regional average). The demo
  shows one: South Prairie Creek, WA, predicted at 1.76 × the observed volume over WY1995-96 (NSE 0.24); the
  regionalisation baseline also fails there (decade NSE −0.44; GR4J calibrated on the gauge 0.66, NWM 0.76).

## Limits
ERA5-Land forcing only (Caravan conventions; net shortwave). CONUS-like attribute ranges. Daily step. Not a
forecast: it is a simulation from observed weather. The water-yield correction is fitted on the dev basins that also
gate it (4 coefficients).

## Rule notes (from the build handoff, section 6, as they concern FlowCast)
- **Dev data used both to fit physics and to gate rungs (partly in-sample):** FlowCast's 4 water-yield coefficients
  were fitted on the 52 dev basins that also gate every rung. The exam was untouched.
- **Free-window forcing from exam basins** was used in training banks, as the brief allows. No exam flow was used for
  training.
- **Climate statics at exam time:** FlowCast computes them from the forcing of the window being predicted, which is
  exam-period forcing but contains no flow.
- **Ratchet noise:** sim-metric differences of about 0.01 are noise-level for FlowCast; a 300-basin subset and the
  full 800 disagreed on rung r5.

## Live instrument and lab
The site instrument (`site/src/instruments/flowcast/`) and the offline lab (`lab/index.html`) run the released
network in the browser on the three exam rivers of `examples/`: the observed hydrograph against the mean and its
bands, with what-ifs on the input (precipitation × 0.5-1.5, temperature ± 3 °C, PET, a storm added or removed,
elevation, area, forest, statics frozen) and the gauged nowcast with an issue-day slider, lead and gauge outage.

## Data credits and licences
Caravan v1.6 (Kratzert et al. 2023, CC BY 4.0) with CAMELS-US (Newman et al. 2015; Addor et al. 2017, CC BY 4.0),
ERA5-Land (Muñoz-Sabater et al. 2021, Copernicus licence), HydroATLAS (Linke et al. 2019, CC BY 4.0); USGS NWIS
daily values and peaks (public domain); NOAA NWM v3.0 retrospective (public domain). None is share-alike; the
weights are MIT like the rest of the repository. Splits and baselines: WaterData; simulator: WaterSim (water
platform, `shared/water`). Method context: Kratzert et al. 2019 (HESS, WRR). Example credits: `examples/README.md`.
