> **Lab notebook.** EnergyNet was built in a claude.ai cloud session (12–13 Nov 2026) on the shared RoadSim + RoadData
> platform, now in `shared/vehicle/`. The first entry is that session's status document as written; its paths
> (`/home/claude/…`, `/mnt/user-data/…`, `src/`) refer to the cloud sandbox. `MODEL_CARD.md` holds every number.

## 12–13 Nov 2026 — build notes (energynet-status.md, as written)

- **What it is:** predicts the energy a trip will take, before it happens, with an error bar. It covers fuel for conventional cars and hybrids, battery kWh for EVs, and the total for plug-ins. It works for every class (car through Class 8) and every powertrain. It was trained only on RoadSim, and it is Phase 2 project 1 of the vehicle work in ML Ratchet.
- **Page:** https://claude.ai/artifact/UYg6maW3hrMNAG3h8h6jRg
  - Hero: one real exam trip.
  - In-browser lab: 9 vehicles × 8 routes, with sliders for temperature, wind, mass, grade, start charge, and warm/cold start. It shows the component split and a temperature sweep. JS matches numpy to 1e-14.
  - Explorer: 39 real exam trips (seeded draw), scatters, ZTBus.
  - Also: exam tables, twin-calibration chart, ratchet, limits, credits.
- **Release:** `/mnt/user-data/outputs/energynet-v1.0.zip` (3.9 MB). It contains:
  - numpy-only `energynet.py`, `features.py`, `model_np.py`
  - 5-network ensemble weights and dev sigma factors
  - `demo.py` on 5 real exam trips, tested from an unzipped copy
  - `MODEL_CARD.md`, `README.md`, `training/`
- **Workspace:** `/home/claude/road/energynet` (ephemeral).

### Recipe as built

- **Physics:** frozen RoadSim (Phase 1 checks). Fixes are wrappers in `src/wrappers.py`, ready for the coordinator to upstream.
- **Data:** 2,400 training / 400 validation generator vehicles, widened in three ways:
  - engine efficiency 0.68–1.08, because the dev twins needed 0.72–1.07
  - older automatics
  - 35% articulated buses
- **Simulated labels:** window-sticker labels from FTP/HWFET plus the EPA 5-cycle formulas.
- **Routes:**
  - the 1,092 VED train_noise routes (real speed, lidar grade, ASOS weather)
  - EPA and heavy-duty cycles
  - synthetic drives
- **Real-noise analogue:** the network sees station weather, while the simulated car drives T ± 1.5 °C with wind × 0.4–0.9.
- **Bank sizes:** 28.8k training and 3,991 frozen validation examples.
- **Spec masking:** redrawn every epoch, from EPA-like specs down to class + powertrain only.
- **Network:** MLP with 149 inputs → correction to a generic road-load energy estimate, plus σ and 5 energy components. The ensemble has 5 members.
- **Dev twins** (16 dev cars on their own 1,498 trips), bias by stage:

  | Stage | Petrol | Hybrid |
  |---|---|---|
  | Generic spec | −28% | −22% |
  | Label-matched twin | −17% | −20% |
  | + MAF label model | −6% | — |
  | + on-road loads | 0% | 0% |

### Calibration rungs from dev

- **MAF label model:** during fuel cut-off, VED's MAF-estimated fuel reads about 0.2–0.3 g/s instead of zero.
- **On-road loads:** A/C, accessories, and real-world hybrid operation:
  - conventional: 1380 W + 133 W/°C above 15 °C
  - hybrid: 2831 W + 204 W/°C
- **Plug-in hardware patch:** pack power ≥ 1.2 × motor, R0 / 2.5, MG2 torque × 1.8. Real dev plug-ins burned no fuel on 95% of trips above 30% charge; the simulated ones took 31% of their energy from fuel.

### Ratchet

v4 val skill and dev D (mean of 15 real dev MdAPEs):

| Rung | Change | Skill | Dev D | Decision |
|---|---|---|---|---|
| r1_abs | Predicts absolute log energy | .544 | .152 | Kept |
| r2_corr | Correction to the physics estimate | .556 | .160 | Kept |
| r3_maf | MAF label model | .580 | .152 | Kept |
| r4_rw | On-road loads | .641 | .142 | Kept |
| r5_phev | Plug-in hardware patch | .648 | .129 | Rejected: +0.007, below the 0.01 margin |
| r6_ens | 5-member ensemble on v4 physics | .656 | .129 | **Kept, released** |

### Exam (MdAPE)

VED trips, EnergyNet vs label / MOVES-like / class-median baselines:

| Fleet | Trip MdAPE | Car total |
|---|---|---|
| ICE | **8.9%** vs 13.0 / 10.3 / 22.8 | **6.8%** |
| HEV | 16.7% vs 16.2 / **15.3** / 17.2; bias +12% | 16.2% vs label 8.9 |
| EV (2 Leafs) | 22.8% vs 21.5 / **19.0** / 27.1; bias −20% | — |
| PHEV | **20.1%** vs 46.1 / 41.3 / 42.7 (uses logged start SOC) | **8.9%** |

- P2 hybrids: 9.0% vs 18.3%.

EPA exam (cycle labels corrected; regression refit on dev):

| Group | Test | EnergyNet | Regression |
|---|---|---|---|
| Conventional | FTP | 9.6% | **7.1** |
| Conventional | HWY | 15.6% (+15% bias) | **5.0** |
| Conventional | US06 | 11.5% | **6.9** |
| HEV | FTP | **8.7%** | 16.3 |
| HEV | HWY | **5.9%** | 21.5 |
| HEV | US06 | **14.9%** | 15.8 |
| BEV | UDDS | **7.9%** | 9.8 |
| BEV | HWY | **5.0%** | 6.5 |

Heavy duty:

| Set | EnergyNet | Baseline |
|---|---|---|
| ZTBus legs | **16.4%** | 22.1 |
| ZTBus missions | 16.2% | **14.5** |
| Mersin diesel | 14.8% | **8.6** (same-route median) |
| Mersin CNG | 15.5% | **7.1** |
| ICCT | 11.4% | **7.0** |

Coverage on VED is too low even after dev widening:

| Fleet | Inside 68% band | Inside 95% band |
|---|---|---|
| ICE | 55% | 85% |
| HEV | 65% | 88% |
| EV | 47% | 84% |
| PHEV | 65% | 89% |

EPA coverage is near nominal.

### Limits

- Hybrids and EVs lose to simple baselines. Their calibration rests on 5 dev hybrids and 1 dev Leaf.
- Conventional EPA highway is biased: the fleet prior was widened toward old engines.
- Error bars are too narrow on VED.
- Heavy duty is thin and partly synthetic.

### Data bug found (for the coordinator)

- RoadData's EPA wide table swaps cycles:
  - 'HWFE' is labelled FTP
  - 'Federal fuel 2-day exhaust' is labelled HWY
- As a result, conv FTP_mpg/HWY_mpg medians are 35.6/24.5.
- `src/epa_fix.py` rebuilds them in the EnergyNet folder; data/ is untouched.
- The published EPA baseline numbers (8.8/7.0%) refer to the swapped labels.
- The VED spec also double-counts 136 kg (mass_kg already includes it, and payload_kg adds it again).

### Next rungs

1. A cold-start calibration from EPA dev FTP_cold.
2. An HEV/EV twin rung with more dev units, or a per-architecture σ model.
3. A "model-era" input so EPA-modern specs do not inherit the old-fleet prior.

### Sandbox gotchas

- The bank takes about 2 h on 2 shared cores (hybrids run at ~1,500× real time with record=True).
- pandas groupby.apply drops the group column.
- json NaN breaks JS JSON.parse.

## 13 Nov 2026 — integrated into ML Ratchet

- Moved from `energynet-v1.0.zip` into `models/energynet/`. Sandbox paths now go through `training/_paths.py`
  (work folder `models/energynet/work/`, the vehicle platform at `shared/vehicle/` or `$ML_RATCHET_VEHICLE`).
- The demo reproduces the release's output exactly from the repository (checked against the zip's own run).
- Caveat carried from the handoff: built before RoadData's 13 Nov fixes (EPA city/highway labels were swapped; VED
  mass double-counted the 136 kg driver). Re-score after the fixes are upstreamed into RoadSim.

## 13 Nov 2026 — original lab and live instrument

- The original claude.ai page is now `lab/index.html`, unchanged.
- The site's model page has a live instrument (`web/export_instrument.py`, `site/src/instruments/energynet/`). Five real exam trips (one per powertrain); temperature, mass, grade, wind, start charge and cold start change the input and the 5-network ensemble reruns. Parity with Python: 32 cases, worst 1.6e-7.

## 14 Nov 2026 — Phase D, step 1: v1 re-scored on the ZTBus by-month exam

Before anything is rebuilt on RoadSim physics v2, the released v1 ensemble (`weights/`, `sigma_factors.json`) is
re-scored on ZTBus mission totals under the split in force since 13 Nov (`proc/hd/ztbus_full_splits.json`: 294 dev /
1,115 exam missions; V2_PLAN decisions log). Same method as v1's `exam.py` (route proxy: v1's traced legs of the same bus
route, run at the mission's mean temperature; baseline: v1's dev kWh/km x distance). Script:
`training/rescore_v1_ztbus.py` -> `work/rescore_v1_ztbus.json`. Nothing is chosen with it.

| missions | n | EnergyNet v1 MdAPE | bias | baseline MdAPE | inside 68% / 95% band |
|---|---|---|---|---|---|
| v1 rule exam (reproduces the card) | 1,407 | 16.2% | -16.2% | 14.5% | 59% / 97% |
| **by-month exam (the v1 number from now on)** | **1,115** | **16.0%** | **-16.0%** | **14.6%** | **60% / 97%** |

v1 still loses to the dev kWh/km baseline on mission totals (16.0 vs 14.6%) and is biased low by 16%. The v1 card's
missions row and the v2 comparison use the 1,115-mission numbers.

Fix needed to run the training scripts in this repository: `common.EN` pointed at `shared/vehicle/energynet` (a sandbox
path); it is now `work/` (`_paths.WORK`), so `realfeat.py`, `bank.py` and `fleet.py` write where `evaluate.py` reads.

## 14 Nov 2026 — Phase D, step 2: EnergyNet rebuilt on RoadSim physics v2, every decision re-taken

**Physics.** Every RoadSim call in training goes through `training/physics.py`, pinned to `roadsim-physics-v2`
(`options='roadsim-physics-v2'`, generator donors `'v2'`). Physics v2 carries what v1 kept in `wrappers.py`: an on-road
extra load per architecture (`onroad_aux_W`) and the OBD MAF label model (`roadsim.obd.label_fuel`, now with the missed
enrichment at high load). `wrappers.py` stays as v1 ran it (pinned to v1), and is only used to re-take v1's rungs on top
of v2. Dynamometer trips run on a dyno spec (no tyre-temperature effects, so no on-road load; HVAC only for SC03 and the
20 F cold FTP), as RoadSim's EPA checks do.

**Fleet and banks.** 2,400 training and 400 frozen-validation vehicles (seeds 201 / 202), 12 and 10 trips each:
`bank_train2` 28,800 trips (later `train2b`, `train2c`: the same fleet, new trips, for r11-r13), `val2` 4,000 trips (the frozen set). New:
- **Model year.** Each light-duty vehicle gets a model year (2005-2026) and an engine efficiency from the era prior
  (`training/era.py`, fitted on fueleconomy.gov **dev** families only, 4,146 conventional records): a MY2005 engine uses
  21% more fuel than the same engine in MY2021, MY2011 10%, MY2017 0.5%; residual sd 0.09 (drawn as exp(N(0, 0.08))).
  This replaces v1's flat 0.68-1.08 widening. The dev twins' label-matched efficiencies agree: MY2005 cars 0.72-0.84,
  2007-2012 0.86-0.93, a 2015-18 car 0.99. Hybrids use the conventional curve (their own fit, 168 records, is not
  monotone); BEVs get none (140 records, no trend that size does not explain).
- **Cold dyno starts.** v1's bank passed no cycle name to `init_state`, so no dynamometer draw ever started cold. Now
  cold UDDS / FTP / 20 F FTP draws start soaked 60% of the time.
- Target columns per trip: raw, + v1's on-road wrapper, + OBD label model, both; and the same four with v1's plug-in
  patch for plug-ins.
- The generic physics-estimate constants (`features.PHYS_K`) are kept from v1 (the network corrects that estimate).
Build: 4 worker processes, 45 min for the training bank.

**Real records.** VED dev/exam with the model year (median year of the fueleconomy.gov models RoadData matched to the
car); EPA dev/exam with the model year, plus the 20 F cold FTP and SC03 tests; full ZTBus: all 7,567 legs of the 1,409
missions (split by month) plus a standstill layover record per mission, so a mission total is the sum of its records
(layovers are ~1% of a mission's energy).

**Dev score D2** (fixed before any v2 rung was scored): mean of 18 dev MdAPEs: VED ICE / HEV / EV / PHEV trips; EPA
conventional FTP / HWY / US06 / 20 F cold FTP / SC03, EPA hybrids FTP / HWY, EPA BEV city / HWY; full-ZTBus dev legs
(1,641) and dev mission totals (294); Mersin diesel / CNG; ICCT. The v1 release scores D2 = 0.1307.

**Rule change, in the open.** v1's rule judged a physics rung (one that changes the simulated targets) on the frozen set
of its own physics, with the incumbent re-scored on it. That compares a network trained on the new targets with one
trained on the old: r4 (v1's on-road wrapper on top of v2, which double-counts v2's own on-road load) gained +0.097 skill
that way while D2 got worse (+0.005), and would have been kept. From r4 on (the rule changed after r4 was scored, and r1-r4
were re-judged with it; only r4's decision changed): **a physics rung must improve D2 by more than 0.01**, and its skill on
its own version may not fall more than 0.01 below the incumbent's on its own. Model rungs keep v1's rule. Rungs that only
change what real data tells the network (gate-only) must improve D2 by more than 0.01. `ratchet2.json` logs each physics
rung's first-rule decision too.

| rung | change | judged on | val2 skill | val2 MdAPE | dev D2 | decision |
|---|---|---|---|---|---|---|
| r1 | MLP predicts log energy | a | 0.614 | 9.0% | 0.1376 | kept |
| r2 | correction to the physics estimate | a | 0.629 | 8.6% | 0.1327 | kept |
| r3 | RoadSim's OBD label model (MAF estimate) + flag | b | 0.630 | 8.6% | 0.1234 | rejected |
| r4 | v1's on-road wrapper on top of v2 | c | 0.631 | 8.3% | 0.1378 | rejected (first rule: kept) |
| r5 | v1's plug-in hardware patch | f | 0.626 | 8.7% | 0.1228 | rejected |
| r5b | r3 + r5 together | e | 0.632 | 8.6% | 0.1225 | kept |
| r6 | plain 5-network ensemble | e | 0.636 | 8.4% | 0.1259 | rejected |
| r7 | 5 networks reading the model year | e | 0.651 | 8.1% | 0.1260 | kept |
| r8 | EPA hybrids told their hybrid system (same networks; gate-only) | e | 0.651 | 8.1% | 0.1195 | rejected |
| r9 | ZTBus told its 19 t curb mass (gate-only) | e | 0.651 | 8.1% | 0.1239 | rejected |
| r9b | r8 + r9 together (gate-only) | e | 0.651 | 8.1% | 0.1174 | rejected |
| r10 | window-sticker labels from RoadSim EPA procedure emulation | e+L | 0.640 | 8.3% | 0.1277 | rejected |
| r11 | 2x data, 512-wide nets, 100 epochs (5 nets) | e | 0.632 | 8.6% | 0.1219 | rejected |
| r11b | 2x data (57,600 trips), 5 nets as r7 | e | 0.660 | 7.9% | 0.1195 | rejected |
| r13 | 3x data (86,400 trips), 5 nets as r7 | e | 0.670 | 7.7% | 0.1195 | **kept, released** |

Notes on the ladder:
- r3 (RoadSim's own label model) and r5 (plug-in patch) each missed the D2 margin by a hair (-0.0093, -0.0100), and v1
  had kept both. Together (r5b) they pass (-0.0102). The twin read (`twin_read.py`, dev cars against their own RoadSim v2
  twins from the calibration agents' check) showed why the patch matters: before it, the network put the dev Volt and
  Energi 33% above their own twins (the generated plug-ins' packs are energy cells at 3 C, so the engine starts); after,
  6%.
- r6 (plain 5-network ensemble) gained only +0.004 skill and was rejected; r7 (5 networks reading the model year) gained
  +0.019 and was kept. On dev it moved the petrol bias from -5% to -3% and the EPA petrol highway / US06 bias from +4 / +10%
  to +1 / +6% (v1 release: +14 / +11%).
- r8 (EPA hybrids told their hybrid system: 34 of 73 dev hybrids are P2s that v1 called power-splits) cut the EPA hybrid
  FTP error from 24.7 to 16.9% and HWY from 12.9 to 9.1%, but D2 by only 0.0065: rejected. r9 (ZTBus told its 19 t curb
  mass) -0.0021, rejected; both together -0.0086, rejected.
- r10 (window-sticker labels from RoadSim's EPA procedure emulation instead of fleet.py's shortcut: P2 city labels +8%,
  power-split highway -4%, plug-in city +12-18%) made D2 slightly worse and was rejected.
- GPU rungs. r11 (twice the data, 512-wide networks, 100 epochs) lost skill (-0.019); r11b (twice the data, r7's
  networks) gained +0.009 skill and -0.0065 D2: rejected by 0.001. r13 (three banks, 86,400 trips, r7's networks) gained
  +0.019 skill and -0.0065 D2: kept and released. Training: 60 epochs, about 5 minutes a network on the RTX 3080
  (under 1.5 GB). More simulated trips helped; wider networks did not.

**Error bars (rung r12, dev only).** Three ways to widen the bands, judged by leave-one-car-out on the 16 dev VED cars
(fit on the other cars, count the held-out car's trips inside the 68% band): v1's one factor per group; plus a car-level
term per group; plus a per-architecture factor that first brings the simulated val2 set to 68% (the networks are
overconfident on simulated trips too: a single r2 network has 52% of val2 inside its 1-sigma band). On the release
candidate r13, mean |held-out coverage - 68%| over the four groups: 0.072 (v1's factors), 0.074 (+ car term), **0.064
(simulation factor, then v1's factors): kept** (`training/sigma_rung.json`). On r7 the three had tied (0.068). The
simulation factors are 0.99 (conventional) to 1.22 (BEV); the dev factors on top: petrol 1.46, hybrid 1.0, EV 1.39,
plug-in 2.36, EPA 1.30-1.38, ZTBus 1.0, Mersin 1.46, ICCT 2.64. Twin read of r13: the network is within 0-5% of the
dev cars' own twins in every group (petrol -1%, hybrid +2%, Leaf +4%, plug-ins +4%).

**Release choice (before the exam):** r13 (5 networks, 151 -> 256 -> 256 -> 256 -> 128 -> 7, trained on 86,400
simulated trips) with the r12 error bars. `training/release.py r13_data3` copied the networks to `weights/`, fitted
`sigma_factors.json` on dev and wrote `ratchet.yaml`.
