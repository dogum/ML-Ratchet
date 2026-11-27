# LawFinder v1.1: what each simulator believes, in one line

*v1.1 (27 Nov 2026) changes only the road law (section "Road law v1.1" below); the other five laws are v1.0.*

LawFinder fills the symbolic-regression slot on the Model families page. It asks four of the collection's shared
simulators thousands of questions, searches for the shortest equation that answers them, keeps an equation only if
it passes the ratchet, and grades it once on the platform's real exam beside the textbook law, the collection's
network and a dumb baseline. The outputs are equations people can read, plot and check.

Built 22 Nov 2026 in the B batch, on a shared 2-core machine. Page: the "LawFinder" artifact (interactive: equation,
sliders, simulated cloud, real points, law ladder, textbook rival, exam).

## What it does

| Law | Simulator | Inputs | Output | Exam |
|---|---|---|---|---|
| Long-term runoff ratio | WaterSim | aridity φ = PET/P, snow fraction f_s, P-PET seasonal synchrony δ, soil index S | Q/P over a decade | 619 CAMELS-US basins, WY1990-99 |
| Daily fire growth | FireSim | perimeter P (km), daily 10 m wind U (m/s), 100-h fuel moisture M (%), fuel group (R₀) | log₁₀ next-24 h new area (km²) | 38,599 FEDS fire-days 2012-24 (3,081 fires) |
| GB system inertia | GridSim | demand D (GW), inverter-based share s | inertia E (GVA·s) | 410 GB events 2018-25 |
| GB frequency nadir | GridSim (GridPulse sampler) | MW lost ΔP (GW), inertia E, demand D | drop to the nadir Δf (Hz) | 148 loss-labelled GB events |
| Home energy signature | BuildingSim | HDD and CDD base 18 °C (1000 K·d), floor area A (100 m²), year built Y | site energy per m² (kWh/m²/yr) | 6,019 RECS 2020 single-family homes |
| Trip energy per km | RoadSim | mean speed v̄ (m/s), positive kinetic energy per metre PKE (m/s²), net and absolute grade (%), mass m (t), CdA (m²), Crr ×1000, ambient T (°C), hybrid and electric flags | energy per km at the tank or battery (MJ/km) | 22,877 VED trips, 52,522 EPA tests |

The RoadSim trip-energy law (fifth in the brief's order) was not built in the cloud batch; it was added in the
repository on 22 Nov 2026 (section "The road law" below).

## The laws (released rung, constants fitted on simulations only)

- **Water** (w2-c13): Q/P = 1.84 · (3.733 + δ)^(0.364 f_s − φ − 0.299). WaterSim's simplest strong belief (w2-c3): Q/P = 0.338 / φ.
- **Fire** (f2-c3): log₁₀ G = −0.559 / P. FireSim's own belief (f2-c6): log₁₀ G = 0.698 ln(0.320 P), i.e. G ≈ (0.32 P)^1.61 km²/day.
- **Inertia** (i1-c7): E = D · (6.603 − 5.617 s). Same as GridSim's own pick.
- **Nadir** (g1-c12): Δf = (ΔP + 2)² / (52,940 E − 3.54·10⁶)^0.208. GridSim (g2, GridPulse's release physics) alone (g2-c3): Δf = 0.433 ΔP.
- **Homes** (b2-c13): EUI = 186.5 / 1.926^A + HDD · (2022 − Y). BuildingSim's simplest strong belief (b2-c9): EUI ≈ HDD · (2021 − Y).
- **Road** (v1.1, p1-c17): E = m · exp[(0.3353·√(Crr×1000) − hybrid) · (2 / v̄^0.6288 − 2·EV)]. It does not read grade.
  v1.0 (p1-c23, until 27 Nov 2026): E = m · exp[(2 / (v̄ + 1.48)^0.595 − 2·EV) · (2·PKE + 0.0698·(Crr×1000 + g_net) − hybrid)].
  RoadSim's simplest strong belief (p2-c10): E = √PKE · (m − hybrid) / (EV + 0.238).

"Simulator's own law" = PySR's model-selection rule applied to the final frozen set only: among rungs with loss within
1.5× of the best, the largest drop in log loss per added node. It uses no real data.

## Method

1. **Simulators, unchanged.** WaterSim v1 (four runoff models, two snow routines, real ERA5-Land forcing from free
   windows only, 10-year runs). FireSim v1 (Rothermel + Van Wagner crown, Nelson moisture, replay settings, guarded real
   dev LANDFIRE windows or parametric landscapes, real dev HRRR weather, 3-6 day runs, burned masks read at the 13:30
   overpass). GridSim: `generate_states` GB fleets (inertia); GridPulse's GB sampler `gpsim` p0/p1 for trips, measured
   like GridData's catalogue (1 s means, kept if the 10 s fall is ≤ −0.15 Hz). BuildingSim 1.0 and 1.4 (single-family
   homes from the ResStock-bootstrapped generator, TMYx weather shifted by U(−3, +3) °C).
2. **Banks** (sim only; frozen val sets per physics version):
   water w1/w2 8,000 train + 3,000 val basins each; fire f1/f2 600 + 240 fires (≈2,000 + 750 fire-days);
   inertia i1 6,000 + 2,000 states; nadir g1/g2 8,000 + 3,000 states (≈20-30 % pass detection); homes b1/b2 1,000 + 300.
3. **Physics versions (rungs).** w2 = w1 + FlowCast's dev-fitted ERA5-Land water-yield multiplier; f2 = dev-calibrated
   ROS factors (`replay_calibration.json`); g2 = GridPulse's dev-calibrated GB response (its release physics, with
   Dynamic Containment) vs g1 = generic; b2 = BuildingSim physics 1.4 (dev-fitted intensities, heat-pump signature).
4. **Search.** `lawsearch.py` (numpy/scipy, written here): 4 islands × 120 trees, regularised evolution, mutation +
   crossover, Levenberg-Marquardt constants, a hall of fame per complexity (nodes ≤ 25), migration every 20 generations.
   Operators + − × ÷ ^ exp log sqrt inv sq. 4-7 minutes per physics version on one shared core (25k-90k evaluations).
   The water search was seeded with Fu's curve. **PySR was not used:** Julia plus its packages is about 1 GB and needs a
   long compile on the shared cores (disk budget 1.5 GB); gplearn has no constant fitting. Check: lawsearch recovers a
   planted Budyko-Fu law with 1 % noise down to the noise floor.
5. **Ratchet.** Rungs = the Pareto front in order of complexity (r0 = the simulator mean), then the next physics
   version's front. Keep if R² on the frozen set beats the incumbent by > 0.01 AND the dev error is ≤ 1.05× the
   incumbent's. After a physics change every rung is re-scored on the new frozen set and every decision re-taken.
   Dev errors: water MAE of Q/P (52 basins); fire mean |log₁₀ ratio| (791 fire-days of 40 fires); inertia mean APE
   (4 events); nadir median |error| (4 events; median because 9 Aug 2019 cascaded from 1,131 to 1,878 MW); homes MdAPE
   (1,504 RECS dev homes).
6. **Error bars.** σ = residual sd of the released law on the final frozen set, widened once by the 68th percentile
   of |dev residual| / σ (widen only).
7. **Exam, once.** `exam_<law>.py`, each refusing to run twice. Every rung's exam score is reported after the fact
   (ratchet.yaml, last column); nothing was chosen on it. Each script was first run on dev units only (`--dry-dev`)
   to test the code.

## Ratchet summary

| Law | Rungs tried | Kept | Released | Frozen sets | Why the release |
|---|---|---|---|---|---|
| Water | 27 | 5 | w2-c13 (R² 0.609, dev MAE 0.130) | val_w1, val_w2 | w2 rungs beyond c13 gained < 0.01 or failed the gate |
| Fire | 28 | 2 | f2-c3 (R² 0.408, dev 1.17) | val_f1, val_f2 | Every law with growth rising in P (R² up to 0.86) made dev worse than the constant (1.39) |
| Inertia | 10 | 5 | i1-c7 (R² 0.964, dev 37 %) | val_i1 | Bigger laws gained < 0.001 |
| Nadir | 16 | 3 | g1-c12 (R² 0.259 on g1, −0.005 on g2) | val_g1, val_g2 | g2 laws (R² 0.12-0.20 on g2) read the 4 dev events 2.5-4.5× worse |
| Homes | 37 | 8 | b2-c13 (R² 0.694, dev MdAPE 39 %) | val_b1, val_b2 | c15+ gained < 0.01 |
| Road | 53 | 7 | p1-c17 (R² 0.754 on val_p2, dev MdAPE 17.5 %; v1.1) | val_p1, val_p2 | Sign gate (27 Nov 2026): p1-c23 (0.767) and the grade-reading p2 laws let energy fall with grade; nothing that passes beats p1-c17 by 0.01 |

Full ladders with every decision: `ratchet.yaml`, `training/results/<law>_ratchet.json`.

## Exam (real, run once)

| Law | Metric | LawFinder | Simulator's own law | Textbook, tuned on dev | Network | Dumb | Verdict |
|---|---|---|---|---|---|---|---|
| Water (619 basins) | MAE of Q/P | 0.129 | 0.140 | Fu ω = 2.38: 0.146 (ω = 2.6: 0.141) | FlowCast **0.112** | 0.211 | loss (to FlowCast) |
| Fire, growth days (25,122) | mean \|log₁₀ ratio\| | **0.693** | 1.132 | Rothermel × P, k = 0.0152: 0.870 | FireCast 0.757 | clim 2.07 | win |
| Fire, all days (38,599) | mean \|log₁₀ ratio\| | 1.153 | 1.800 | 1.061 | (growth days only) | clim 1.371; yesterday's growth **0.671** | loss |
| Inertia (410) | MdAPE | 33.0 % | 33.0 % | 7.00 s × D: **15.1 %** | GridPulse 16.4 % | 24.0 % | loss |
| Nadir (148) | MAE | **0.060 Hz** | 0.106 | k ΔP²/E: 0.173 | GridPulse (5 s of trace) 0.080 | 0.129 | win |
| Homes (6,019) | MdAPE | 39.3 % | 39.7 % | a + b HDD + c CDD: 35.6 % | none in the collection | **33.4 %** | loss (to dumb) |
| Road, VED trips (22,877) | MdAPE | 18.7 % (v1.0: 18.6) | 20.9 % | road-load / η: 18.3 % | EnergyNet **10.1 %** | 21.3 %; MOVES-like 11.8 % | loss |
| Road, EPA tests (52,522) | MdAPE | 19.4 % (v1.0: 17.3) | 15.1 % | road-load / η: 23.8 % | EnergyNet **7.5 %** | 24.6 % | loss |

Also: water R² 0.594 (Fu 0.43, FlowCast 0.64); inertia rank correlation 0.975 (7 s × D 0.60, GridPulse 0.61);
homes: BuildingData-style log-linear fit on dev 26.6 % (best).

**Coverage of the 68 % bars on the exam:** water 83 %, fire 69 %, inertia 91 %, nadir 100 %, homes 69 %, road 72 % (VED)
and 80 % (EPA; v1.0 83 %). Water,
inertia and nadir bars are too wide (inertia's ×6.7 widening was fitted on 4 dev events).

## What the simulators believe, and whether it is true

- **WaterSim** believes runoff falls exponentially with aridity, more slowly where rain comes in the cool season, and
  rises with snow. That beats the Budyko-Fu curve on real basins (MAE 0.129 vs 0.141-0.154) but not FlowCast (0.112).
  Twin test (dev): the law reads WaterSim's ungauged twins of the dev basins at MAE 0.091 and the real basins at 0.130;
  the ungauged twins themselves are 0.111 from reality, the calibrated twins 0.059. Most of the gap is basin detail the
  simulator's attribute prior does not know.
- **FireSim** believes a fire adds G ≈ (0.32 P)^1.6 km² a day: growth keeps rising with perimeter. On real fires that
  runs a median 36× too high (bias +1.56 decades). FireSim's growth rate in Rothermel × perimeter terms is k = 0.175;
  real dev fires give k = 0.015, 11× lower. This is the platform's known missing process (fires stall and burn out).
- **GridSim** believes inertia = 6.6 s × synchronous demand, falling as inverter-based share rises. Real GB has the same
  shape (ρ 0.975) at a higher level: GridSim's raw GB fleets commit ~4.5 s per MW where NESO's outturn shows ~7 s.
  Bias by year −83 to −50 GVA·s, shrinking as GB inertia per MW falls (2023-25).
- **GridSim's nadir** with generic response: Δf grows roughly with the square of the loss and falls with inertia;
  it beats the swing-equation ramp law (0.060 vs 0.173 Hz) and GridPulse's 5 s forecast (0.080). With GridPulse's
  Dynamic Containment physics the simulator's nadirs are shallower than the 2019 dev events, and the gate refused it.
- **BuildingSim** believes home energy per m² is heating-dominated: degree days × the age of the house, plus a base
  load for small homes. On RECS homes it loses to the dev median (39.3 % vs 33.4 %): home-to-home variation (fuel,
  occupants, setpoints) is larger than what these inputs carry. At physics 1.0 the homes used 1.4× too much energy.

## The road law (added 22 Nov 2026)

Built in the repository after the import, following `law_bldg.py` / `exam_bldg.py`. Scripts: `training/road_bank.py`,
`law_road.py`, `fit_law.py road`, `exam_road.py`, `exam_road_errata.py`, `add_road_release.py`.

- **Bank.** 16,000 RoadSim trips: 6,000 train + 2,000 frozen per physics version (p1 = RoadSim physics v1, p2 = v2, the
  dev-calibrated default; same vehicles and routes in both). Light-duty generator vehicles (petrol 50 %, hybrids 28 %,
  battery-electric 22 %; car, SUV, pickup, van), spread as EnergyNet's fleet is. Routes: RoadData's 1,092 train_noise
  VED trips (real routes, lidar grade, airport weather; no labels exist) 60 %, dynamometer schedules 40 %. The
  simulator drives the true weather; the law reads the observed one. 0.04 s per trip on one core.
- **Dev.** 1,143 VED dev trips (7 petrol cars, 5 hybrids, 1 Leaf; plug-ins left out) and 1,178 EPA dev tests (a seeded
  draw of the same size). Dev error = MdAPE.
- **Search and ratchet.** 600 s per physics version. 40 rungs: p1's front kept 8 on val_p1; on val_p2 every p2 rung
  failed the 0.01 margin against p1-c23 (best p2-c24 0.764 vs 0.767), although several read dev better (p2-c22: 12.9 %
  against 16.8 %). The rule decided; the exam was not consulted. Error bar ±0.78 MJ/km; dev widening 1.0 (raw dev
  coverage 84 %, widen-only).
- **A first ratchet run** used a dev set that dropped 35 trips of vehicles with no recorded mass (it released p1-c18,
  dev 16.6 %). Before the exam those vehicles got their class-default mass, as EnergyNet does, and the ratchet was
  re-scored on the corrected dev set (same fronts). Both runs are logged (`training/results/superseded/`).
- **Textbook rival.** Road-load energy at the wheels, (m g Crr + ½ ρ CdA v̄² + m·PKE + m g g_net) per km, over one η
  per powertrain fitted on dev (petrol 0.185, hybrid 0.280, electric 1.15; on RoadSim 0.189, 0.280, 1.08). The electric
  η is above 1 because the sum counts no regenerative braking.

**Exam (run once, 22 Nov 2026), MdAPE %:**

| Stratum | n | Law | RoadSim's law | Road-load (dev η) | EnergyNet v2.1 | Label × distance | MOVES-like | Dev median |
|---|---|---|---|---|---|---|---|---|
| VED petrol | 14,469 | 19.3 | 14.0 | 20.0 | **8.6** | 13.0 | 10.3 | 22.4 |
| VED, all three | 22,877 | 18.6 | 20.9 | 18.3 | **10.1** | 14.1 | 11.8 | 21.3 |
| VED hybrids | 8,241 | 16.8 | 40.4 | 15.7 | **13.4** | 16.2 | 15.3 | 19.0 |
| VED electric (2 Leafs) | 167 | 43.3 | 21.9 | 21.7 | 23.2 | 21.5 | **19.0** | 28.0 |
| EPA petrol | 46,876 | 17.3 | 14.5 | 25.0 | **7.5** | – | – | 25.3 |
| EPA hybrids | 4,154 | 13.1 | 22.3 | 19.6 | **7.8** | – | – | 21.9 |
| EPA battery-electric | 1,492 | 43.1 | 26.8 | 11.9 | **6.3** | – | – | 15.8 |

- The law loses every row. It beats road-load and the dumb guess on EPA petrol and hybrid tests, and loses badly on
  electric cars (bias −42 % on the VED Leafs, +42 % on EPA BEVs): one exponent for every electric car, learnt on
  generator EVs heavier than the real ones.
- RoadSim's own shorter law beats the released one on VED petrol and on EPA petrol: the dev gate preferred the longer law.
- **Errata, exam run 1.** `exam_road.py` multiplied EnergyNet's VED frame by its unit factor although the frame is
  already in MJ, so EnergyNet's VED rows came out ×0.042 (petrol, hybrids) and ×3.6 (electric): 95.8 %, 95.6 % and
  187 % MdAPE. The run's output is kept as written (`training/results/exam_road_v10.json`, renamed for v1.1). With the owner's agreement,
  `exam_road_errata.py` re-computed EnergyNet's VED rows only, from its stored per-trip predictions against run 1's
  stored truth (which agrees with EnergyNet's own to 6e-8): petrol 8.6 %, hybrids 13.4 %, electric 23.2 % (its published
  values exactly) and 10.1 % on the pooled trips. The law's exam was not re-run; every law, rung, road-load,
  RoadSim-law, RoadData and dumb row is unchanged. The page, release and card read `exam_road_reported.json` (v1.0's: `exam_road_reported_v10.json`).

## Road law v1.1 (27 Nov 2026)

The audit (`docs/audits/lawfinder-v1.0.md`) found that v1.0's road law gave a battery-electric car less energy on a
steeper road. On 27 Nov 2026 the owner adopted a release rule for the road ladder: a law may not be released if its
energy falls as net grade rises at any val_p2 row. Tank and battery energy rise with uphill grade. p1-c23 failed on 21 %
of val_p2 rows and on every BEV row. The rule came from the audit, not from an exam result.

The walk was re-run with no new search (`training/road_gate.py`) over the 40 v1.0 rungs and the 13 sign-checked p2s
rungs. p1-c23, p1-c25 and the grade-reading p2 laws were rejected by the gate, and **p1-c17** was released (R² 0.754,
dev 17.5 %, error bar ±0.80 MJ/km). The choice was committed before the exam ran. p1-c17 does not read grade at all.
v1.0's exam had already scored p1-c17 after the fact, at 18.67 % on VED against p1-c23's 18.60 %. The exam was run once
more for v1.1, and every non-law row reproduced v1.0's exactly.

| Stratum | n | v1.0 p1-c23 MdAPE (bias) % | v1.1 p1-c17 MdAPE (bias) % | Road-load (dev η) | EnergyNet v2.1 |
|---|---|---|---|---|---|
| VED petrol | 14,469 | 19.3 (-19) | 19.0 (-18) | 20.0 | 8.6 |
| VED hybrids | 8,241 | 16.8 (-7) | 17.7 (-7) | 15.7 | 13.4 |
| VED electric (2 Leafs) | 167 | 43.3 (-42) | 33.7 (-32) | 21.7 | 23.2 |
| VED, all three | 22,877 | 18.6 (-16) | 18.7 (-15) | 18.3 | 10.1 |
| EPA petrol | 46,876 | 17.3 (+11) | 19.6 (+14) | 25.0 | 7.5 |
| EPA hybrids | 4,154 | 13.1 (+6) | 15.8 (+8) | 19.6 | 7.8 |
| EPA battery-electric | 1,492 | 43.1 (+42) | 20.0 (+11) | 11.9 | 6.3 |
| EPA, all | 52,522 | 17.3 (+11) | 19.4 (+14) | 23.8 | 7.5 |

- Electric cars improve most: the EPA BEV error halves (43.1 → 20.0 %) and the VED Leafs go from 43.3 to 33.7 %. Both
  still lose to road-load.
- Petrol and hybrid EPA tests get worse (17.3 → 19.6 %, 13.1 → 15.8 %), with more over-prediction on the highway
  (ICE HWY 24.0 → 30.4 %).
- The law still loses every row. Records: `training/results/road_ratchet.json` (v1.1; v1.0 in `road_ratchet_v10.json`),
  `exam_road.json` / `exam_road_reported.json` (v1.1; v1.0 in `*_v10.json`).

## Limits

- One search per physics version, time-boxed on a shared core; another seed may give a different front. Fronts are
  in `training/results/<law>_search_<ver>.json`.
- Dev sets are small (4 grid events), so the gate is coarse; the grid widening factors come from 4 events.
- All water rows use ERA5-Land forcing (mountain under-catch). Fire laws have no slope (exam fires have no landscape)
  and FireCast's rows exist on growth days only. Nadir loss labels are noisy REMIT/FUELINST matches.
- Inputs are computed the same way for simulated and real units, but some come from different products (gridMET
  fm100 vs FireSim's Nelson m100h; RECS 2020 degree days vs TMYx-shifted runs; NESO generation as demand).
- Road: real CdA and Crr are class defaults on VED (EPA: from the road-load coefficients); VED fuel is MAF-estimated
  for most cars; plug-in hybrids are left out; cold start and trip length are not inputs.

## Rule notes

- Dev data entered the simulators through physics versions built by earlier models (FlowCast's water multiplier,
  FireSim's replay calibration, GridPulse's GB response, BuildingSim 1.4) and then gated the same rungs, so gates
  are partly in-sample, as in those models.
- The textbook rivals' constants were fitted on the dev sets (the "tuned on dev" lesson), and also reported fitted on
  the simulators.
- GridData's 14 Nov 2026 correction of dev event GB_20191203T200030 (1,788 → 854.7 MW) was applied in LawFinder's
  loader because the sandbox copy of GridData predates it.
- The fire exam's primary metric is reported both ways: growth days (FireData's convention, where FireCast is scored)
  and all fire-days. The law wins one and loses the other; the verdict says so.

## Use

```python
from lawfinder import LawFinder
lf = LawFinder()
mu, sigma = lf['water'].predict(phi=0.8, fs=0.1, sync=-0.2, S=0.14)
lf['grid_inertia'](D=30, s=0.4)                # GVA s
```
Inputs as in the table above (fire: R0 = 2.34 GR, 0.99 GS, 12.7 SH, 1.45 TU, 0.76 TL m/min). `python demo.py` prints
each law on 5 real exam examples (fixed seed within terciles of the first input) in about 0.2 s.

## Data and credits

Caravan/CAMELS-US (CC BY 4.0) with ERA5-Land; USGS NWIS (public domain); FEDS-MTBS (CC BY 4.0), MTBS, LANDFIRE, HRRR,
gridMET (US public data); NESO frequency, inertia and generation mix (NESO Open Data Licence); Elexon REMIT/B1610 via
GridData; EIA RECS 2020 (public domain); TMYx weather (climate.onebuilding.org). Splits: each platform's frozen split,
copied in `splits.json` (with the platform files' sha256).
