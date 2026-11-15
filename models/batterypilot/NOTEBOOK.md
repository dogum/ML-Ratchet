> **Lab notebook.** BatteryPilot was built in a Claude cloud session (13–14 Nov 2026) as Phase 2 of the grid work, on
> the shared GridSim + GridData platform, now in `shared/grid/`. The first entry is that session's status document as
> written; its paths (the sandbox workspace `grid/batterypilot/`, its `src/`, `data/`, `web/`) refer to the cloud
> sandbox. `MODEL_CARD.md` holds every number.

## 14 Nov 2026 — build notes (batterypilot-status.md, as written)

**What it is.** A causal grid-battery arbitrage policy, trained only on simulated price paths and graded on real prices. Before each hour a small network returns a value curve: what one stored MWh is worth at 0/25/50/75/100 % full. The battery uses that curve as its offer for the hour, cleared against the real-time price. It works for 1, 2 and 4 h batteries, in ERCOT (primary) and CAISO (secondary).

- Page: the original page, now `lab/`. It has a lab: pick a hub, year and battery, and see prices, state of charge, the money and the value curve for the hovered hour. A what-if mode re-runs the network live in JS with a different efficiency or wear cost. The page also has the exam tables, the twin and calibration charts, the ratchet, limits and credits.
- Zip: `batterypilot-v1.0.zip` (0.7 MB): numpy/scipy model, weights, a demo on 3 real exam node-years (~6 s), the model card and all training scripts.
- Folder: the sandbox workspace `grid/batterypilot` (STATUS.md, ratchet.json, src/, data/, web/, release/).

### Recipe as built
- **Battery physics:**
  - 85 % round trip, power and SOC limits.
  - Rainflow wear with a Wöhler depth curve plus LiFePO4 temperature and C-rate dependence (Wang 2011, CellSense's chemistry), at $130/kWh.
  - Checks: reproduces GridData's PF, spread and threshold baselines on dev days to 1e-13; rainflow matches the `rainflow` package exactly.
- **Simulator:** a wrapper around GridSim (unmodified). It uses the real ERCOT/CISO fleet and inputs of 2018, 2020 and 2022 (EIA-930, NASA POWER), plus:
  - merit stack with a must-run floor
  - gas-turbine heat derate
  - cold outages (fitted on Uri)
  - storage fleet
  - ERCOT-style ORDC, μ −4,000 MW and σ 5,000 MW (fitted on dev 2019 and 2021)
  - RT noise
  - day-ahead prices from forecasts plus the expected adder
  - scenario randomisation
  - a CAISO-like family (solar ×1.6 and fixed imports ×0.6, checked on Aug–Dec 2023)
- **Banks:** 120 ERCOT + 45 CAISO-like path-years for training; 24 + 9 frozen.
- **Training:** direct policy search, with the battery physics inside the loss in torch. MLP 135→128→128→5. JS matches numpy to 2e-6.
- **Dev gate:** ERCOT 2019 and 2021 at 4 hubs, and CAISO Aug–Dec 2023 at NP15 and SP15.

### Rungs
Frozen score = share of perfect-foresight net revenue. Keep a rung if it gains more than 0.01 and dev drops less than 0.01.

| Rung | What | Frozen | Dev | Decision |
|---|---|---|---|---|
| R0 | Rolling day-ahead LP (classical) | 0.447 (v3) | 0.593 | Incumbent |
| R1 | Net trained on GridSim prices with no scarcity (sim v1) | 0.570 (v3) | 0.483 | Rejected: dev gate |
| R2 | Sim v2: dev-calibrated scarcity | 0.747 (v3) | 0.598 | Kept |
| R3 | + CAISO-like family, all data | 0.749 | 0.623 | Rejected: +0.002 |
| R4 | Fine-tune R3, 21-day windows | 0.758 | 0.637 | **Kept, released** |

### Exam (real, run once)
Mean share of perfect hindsight, net of wear:

| Market | BatteryPilot | Day-ahead optimiser | Spread rule | Threshold |
|---|---|---|---|---|
| ERCOT 2023–25 (36 node-years) | 0.667 | 0.685 | 0.561 | 0.114 |
| CAISO 2024–25 (12) | 0.642 | 0.681 | 0.589 | 0.158 |

- **Against the rules:** beats the spread rule in 89 % (ERCOT) and 81 % (CAISO) of cases, and the threshold rule in all.
- **Against the optimiser:** beats it in 57 % and 19 % of cases.
- **By duration (ERCOT):** 1 h 0.56 vs 0.63 (loses); 4 h 0.77 vs 0.74. ERCOT 2024 4 h: +$3.3/kW-yr, 95 % CI [+1.1, +5.4].
- About 245 cycles a year vs 300 for the optimiser. No month worse than −$0.21/kW.
- A comparison net trained on real 2018/2020/2022 ERCOT prices (not released) scored ERCOT 0.690 and CAISO 0.564.

### Limits
- Ties rather than beats the classical day-ahead optimiser.
- There is a sim-to-real gap. On dev twins it captures 91–94 % vs 80 % on real 2019; for Uri, 64–68 % vs 38 %.
- Energy only, at hourly RT prices. No ancillary services and no congestion.
- The CAISO family is hand-set.
- The calibration hit its grid edges.

### Next rungs
1. Make simulated real-time spikes less predictable from day-ahead prices: unforecast outage events and 15-minute ramp scarcity, fitted so the dev twin capture matches the real one.
2. A 1-hour specialist, or a 15-minute execution head.
3. Nodal basis: a West Texas congestion model, plus a real CAISO import and hydro supply curve.

### Upstream candidates for GridSim
- ORDC adder on reserves
- Cold-outage curve
- Gas-turbine ambient derate
- Day-ahead price generator
- Single-LP daily perfect foresight

### Sandbox gotchas
- With 4 parallel builds on 2 cores, a simulated path-year takes 20–30 s.
- Bash calls need an explicit timeout for sleeps over 2 minutes.
- The log file can appear late after `setsid nohup`.

## 14 Nov 2026 — brought into the repository

**Layout.** The release zip became this folder: `batterypilot.py` (the whole model, numpy + scipy), `batterypilot_weights.npz`
(rung R4), `demo.py`, the three real exam node-years in `examples/` (ERCOT HB_HOUSTON 2023, HB_WEST 2024, CAISO SP15
2025, hourly CSVs), `MODEL_CARD.md`, and every training script in `training/` as run. The release's `checks/` JSONs
are in `training/checks/` and its raw ratchet log is `training/ratchet.json`. The release's `web/page.html` and
`web/bp.js` are the original page, kept untouched as `lab/index.html` and `lab/bp.js` (with its data in `lab/data.js`).

**Code changes.** One in the model: `PolicyNet()` now loads `batterypilot_weights.npz` next to `batterypilot.py` when no
path is given (the demo still passes the path). None in the maths. In `training/`, `_paths.py` is new and every
sandbox path now goes through it: the sandbox workspace → `WORK` (`models/batterypilot/work/`, git-ignored, or
`$ML_RATCHET_WORK/batterypilot`), the GridSim and GridData folders → `GRID` (`shared/grid/`, or `$ML_RATCHET_GRID`),
the fleet parquet files → `GRID/data/proc/fleet/`. `jscheck.py` reads `lab/bp.js` and writes its scratch file to
`WORK/web/`. `report.py` gained the `import _paths` it needed, and `check_battery.py` now writes to `WORK/checks/`
instead of the sandbox's `../checks/`. Outputs go to `WORK` only.

**Battery checks re-run** (`training/check_battery.py` on the shared data): GridData's PF, spread and threshold
baselines reproduced on the 89 dev days at both hubs to 1.1e-13 $; the HB_WEST spread day that differed by $1.65 in
the release (a price tie) now matches too. Rainflow vs the `rainflow` package: 0.0 on 200 series; annual LP chunked vs
single $61,491.70 both; rainflow wear of the summer-2019 PF schedule $2,414.37 vs the linear proxy $2,416.26 — all as
released.

**Demo (reproduced).** `python demo.py` from this folder prints the same numbers as the release demo run from the
extracted zip, line for line (only the timings differ): 2 h battery, $/kW-yr net of wear, perfect / BatteryPilot /
DA optimiser / spread = 18.1 / 11.0 / 11.9 / 10.6 (CAISO SP15 2025), 135.5 / 93.7 / 91.4 / 82.4 (ERCOT HB_HOUSTON
2023), 51.1 / 38.7 / 37.6 / 30.7 (ERCOT HB_WEST 2024). About 15 s on a laptop. Golden: `tests/golden/batterypilot.txt`.

**Exam re-scored** with `training/exam.py` and `training/report.py` on the shipped weights and the shared data in
`shared/grid/data` (48 node-years × 3 batteries × 5 strategies; 6 min 20 s single-threaded; outputs in
`work/data/exam/`). Released → re-scored:

| | BatteryPilot | DA optimiser | Spread rule | Threshold | Beats optimiser | Beats spread |
|---|---|---|---|---|---|---|
| ERCOT 2023–25 (36) | 0.667 → 0.667 | 0.685 → 0.685 | 0.561 → 0.561 | 0.114 → 0.114 | 57.4 % → 57.4 % | 88.9 % → 88.9 % |
| CAISO 2024–25 (12) | 0.642 → 0.641 | 0.681 → 0.681 | 0.589 → 0.589 | 0.158 → 0.158 | 19.4 % → 19.4 % | 80.6 % → 77.8 % |

- Every baseline (perfect foresight, DA optimiser, spread, threshold) reproduces in all 144 node-year-battery rows to
  the printed $0.01/kW-yr, and so do BatteryPilot's cycle counts. ERCOT's headline reproduces to the printed
  precision. CAISO's BatteryPilot share is 0.641 against the released 0.642 (close, not exact).
- BatteryPilot itself differs in 88 of 144 rows, by at most $0.29/kW-yr (CAISO SP15 2025, 2 h: 11.01 released, 10.72
  re-scored). One CAISO case flips against the spread rule (DLAP SCE 2024, 2 h: 22.45 released, 22.26 re-scored, spread
  22.27), so "beats the spread rule" is 28 of 36 here, not 29.
- Why: the policy is sensitive to the last decimals of its inputs. The shared data carry CAISO day-ahead LMPs at 5
  decimals (float32); the example CSV holds them at 3. On SP15 2025 2 h, BatteryPilot earns 10.72 on the shared
  frame and 11.02 with only the DA column swapped for the CSV's 3-decimal values (the released exam: 11.01). Rounding
  every exam frame's prices to 3 decimals does not bring back the released rows exactly either (94 rows still differ,
  by up to $0.24/kW-yr; CAISO 0.641, ERCOT 0.667), so the sandbox's frames differed from today's shared data in small
  ways beyond rounding. A hold-or-trade decision near a tie can flip on a $0.0005 change, and the state of charge then
  follows a different path for days. Read single node-years as ±2-3 %; the pooled shares move by 0.001 at most.
- The released numbers in `model.yaml` and `MODEL_CARD.md` are unchanged.

**Live instrument.** `web/export_instrument.py` writes `site/public/instruments/batterypilot/` (1.0 MB): the network in
float16, the three example node-years (hourly f32 prices, demand forecast, temperature, local calendar), the
perfect-hindsight net revenue of every month for every duration × efficiency (75-95 %) × wear cost ($0-260/kWh) on
the slider grid (an LP in Python, labelled as precomputed on the page), and the whole-year Python results at the
as-built settings. The browser port (`site/src/instruments/batterypilot/net.ts`) rebuilds the 135 features, runs the
network every hour of the chosen month, runs the offer rule against the real RT price, runs GridData's spread rule
and settles both with rainflow wear. Parity (`npm test`): 9 cases (3 node-years × as built, 1 h, and 4 h at 75 % with
$260/kWh), raw value-curve knots at 17 hours plus the state of charge over a month; worst relative difference 8.4e-6.
Float16 weights move the raw knots (asinh of value / price scale) by at most 2.9e-3 against the full network.

**Not fixed in this pass.** GridSim's outage-rate defaults cite Texas RE's 2024/2025 reliability-assessment aggregates,
published during the exam years: a possible mild exam leak, noted in the card and in `limits`. The platform plan is to
replace them with pre-2023 NERC GADS values and re-score.

## 14 Nov 2026 — v1.1: GridSim outage-rate leak removed (HANDOFF P0.5)

**What leaked.** GridSim physics 1.0 filled the forced and planned outage shares of coal, CC, CT and gas-steam units
from Texas RE's 2024/2025 assessments: exam-period data. GridSim physics 1.1 (commit a4a24d4) takes them from PJM's
2017-2021 class average EFORd and Texas RE's 2017 assessment (coal 0.15/0.11 → 0.136/0.083, CC 0.08/0.11 →
0.036/0.083, CT 0.08/0.06 → 0.058/0.083, gas steam 0.13/0.19 → 0.136/0.083). It re-fits the dev-2019 coal offer adder:
−6.62 → −5.73 $/MWh.

**Where it reached BatteryPilot.** `pricesim.py`: `prepare_fleet` (outage shares), `availability(mode='random')`, and
GridSim's coal must-run and adder, hard-coded as −6.62. From there: `calib.py` (ORDC μ/σ, RT noise, DA scale, the Uri
cold-outage curve), `simbank.py` (every bank and frozen set), `twin.py`, every rung trained or scored on simulated
paths, and the released network. Not reached: battery physics, the dev gate and the exam (real prices only).

**Code.** `pricesim.Engine` now reads the coal must-run and adder from `gridsim.fleet.calibration()` of the active
`GRIDSIM_PHYSICS` and raises if the key is missing. With 1.0 it gives −6.62. `simbank.calib_file()` picks the scarcity
calibration of the physics version: 1.0 → the release's `training/checks/calibration.json`; 1.1 →
`work/data/calibration_v1_1.json`. `simbank.py` gained `--calib`, `--part k/n` (parallel builds, same files) and
`--prefix`, and stores the physics version, adder and calibration file in each path's meta. `calib.py` writes per
version and has a `wide` grid. New: `leakcheck.py` (the measurement) and `ladder_v1_1.py` (the v1.1 decisions →
`training/ratchet_v1_1.json`).

**Reproduction first.** With `GRIDSIM_PHYSICS=1.0` and the release calibration, `leakcheck.py` reproduces the release's
stage-A loss (3.6695), every dev target statistic, the Uri hours (180, 166; overlap 0.948, 0.915) and the twin
captures (0.937/0.911 on 2019, 0.681/0.641 on 2021) exactly.

**Measured (dev only).** Physics 1.0 → 1.1, both with the v1.0 calibration:
- Mean thermal capacity on random outage (ERCOT 2019 fleet, 74.1 GW): 14.8 → 10.8 GW.
- 2019 twins: hours above $1,000 22.5 → 0 (real 26); above $300 120 → 20 (75); daily-max p90 $212 → $67 ($232);
  2 h PF value $84.0 → $27.9/kW-yr ($103.3); RT mean $38.6 → $22.8 ($35.8).
- 2021 (Uri masked): above $300 53.5 → 1 (47); PF $61.1 → $32.9 ($54.4).
- Calibration loss 3.67 → 21.07. Uri hours above $1,000: 180/166 → 156/149 (real 153).
- Frozen sets with the same seeds (24 ERCOT + 9 CAISO-like): v1.0 network 0.798 → 0.771; rolling LP 0.516 → 0.421.
  Dev capture does not move (real prices): 0.6373 and 0.5927.

The v1.0 calibration had soaked up the leaked ~4 GW of extra outages. Without them, its scarcity curve almost never
binds.

**Recalibrated on dev (physics 1.1).**
- On the release's grid the best point sat on the opposite edges: μ −2,000, σ 6,500, loss 4.253.
- Widened once, as in v1.0 (μ −4,000…0, σ 3,500…9,500; it contains the old grid): μ 0 MW, σ 5,000 MW, loss 4.165.
  Both files are in `work/data/`. μ 0 is a grid edge again.
- RT noise 0.1, ramp_k 1.5 and DA scale 0.7 are unchanged. Uri: cold f_max 0.40 (was 0.55), T50 −3 °C, w 3 K; 158 and
  150 hours above $1,000 vs 153 real.
- 2019 twins now: 32.5 hours above $1,000, PF $75.6.
- Released v1.0 network on the rebuilt v4 frozen set: 0.778.

**Banks and frozen set (physics 1.1, v1.1 calibration).** Seeds 8101 (ERCOT 120), 8102 (CAISO-like 45), 8103 (sim v1,
45). The frozen set v4 has 24 ERCOT + 9 CAISO-like paths, seeds 7001/7002, the same seeds as the measurement sets
m10_frozen and m11rc_frozen. Each path took 13-45 s, 16 builds in parallel.

**v1.1 ladder** (rule as written: frozen > incumbent + 0.01 and dev ≥ incumbent − 0.01; torch CPU, about 0.5 s/step):

| Rung | v4 frozen | Dev | Decision |
|---|---|---|---|
| R0 rolling LP | 0.5042 | 0.5927 | incumbent |
| R1 sim v1, 45 paths, 1,500 steps | 0.6074 | 0.4888 | rejected: dev −0.104 |
| R2 sim v2, 80 ERCOT paths, 1,500 steps | 0.7792 | 0.6382 | kept (+0.275, dev +0.046) |
| R3 120 ERCOT + 45 CAISO-like, 2,500 steps | 0.7877 | 0.6437 | rejected: +0.0085 |
| R4 fine-tune R3, 504 h windows, batch 64, lr 5e-4, 900 steps | 0.7890 | 0.6530 | rejected: +0.0098 |

Two things are not recorded from the sandbox: the release's bank seeds and R4's learning rate. Those above are this
run's. R4 missed by 0.0002 with a better dev score, and it is rejected as the rule says. No further rung was tried
after seeing that. R2 is released as v1.1.

**Shipped.** `batterypilot_weights.npz` = v1.1 (R2); `batterypilot_weights_v1_0.npz` = v1.0 (identical to the
release's R4); `PolicyNet()` loads v1.1, `bp.WEIGHTS_V1_0` names v1.0. The demo now prints the v1.1 numbers, so the
golden (`tests/golden/batterypilot.txt`) was regenerated. 2 h, $/kW-yr, BatteryPilot v1.0 → v1.1: CAISO SP15 2025
11.0 → 10.2; ERCOT HB_HOUSTON 2023 93.7 → 93.9; HB_WEST 2024 38.7 → 39.4. The live instrument was re-exported with
v1.1. Parity: 9 cases, worst relative difference 6.5e-6. Float16 moves z by at most 3.4e-3.

**Exam, second look** (v1.1, run once, 48 node-years, `training/checks/exam_report_v1_1.json`). Mean share of perfect
hindsight, first look (v1.0) → second look (v1.1):
- ERCOT: 0.667 → 0.679 (optimiser 0.685). 1 h 0.559 → 0.578; 2 h 0.671 → 0.680; 4 h 0.769 → 0.780 (optimiser 0.742).
- CAISO: 0.642 → 0.614 (optimiser 0.681). 4 h 0.730 → 0.719 (optimiser 0.723): the CAISO 4 h win becomes a loss.
- Beats the optimiser in 63 % / 11 % of cases (was 57 % / 19 %); the spread rule in 89 % / 58 % (was 89 % / 81 %).

The pooled verdicts stay losses. CAISO is worse because R2 never saw the CAISO-like family. The model.yaml exam rows
keep the v1.0 values; the second look is in each row's note.

**Not done.** The CAISO-like family check (hourly medians against Aug-Dec 2023) was not re-run on physics 1.1. ERCOT's
own pre-2023 outage rates by technology are still wanted (help_wanted).

## 14 Nov 2026 — wording corrected (site-wide honesty audit)

- MODEL_CARD limits said BatteryPilot "roughly ties, does not beat" the day-ahead optimiser. Both pooled rows are losses
  (ERCOT 0.667 vs 0.685, CAISO 0.642 vs 0.681). It now says it lost to the optimiser: it wins with 4 h batteries and
  loses with 1 h and in CAISO. model.yaml had no "ties" wording.
- Cycles a year: model.yaml said 244 in one place and 245 in another, and the card said 245. The exam output
  (`training/checks/exam_report.json`) gives 244 in ERCOT and 248 in CAISO (245 is the mean of both, set against
  ERCOT's 300). Both files now say 244 (ERCOT) and 248 (CAISO) vs 300 and 282 for the optimiser. The first notebook
  entry is left as written.

## 15 Nov 2026 — v1.2 attempt: a day-ahead value-curve backbone with a learned residual (not released)

**Goal.** v1.1 loses every pooled exam row to the rolling day-ahead LP. The directions: keep the LP as the backbone
and let the network supply what it lacks, or train on the CAISO-like family. Decide everything on dev and the v4 frozen
set with the rule unchanged; examine only a released version.

**Setup.** The work folder of the grid-follow-up worktree (banks, frozen set v4, calibration, checkpoints, dev paths)
was copied into this checkout's `work/` (git-ignored). Physics 1.1, calibration v1.1, frozen set v4 as for v1.1.

**Dev first.** On the 10 dev node-years (path-mean share of perfect foresight, R2 = v1.1 vs the rolling LP):
ERCOT 1 h 0.470 vs 0.491 (the LP wins), 2 h 0.566 vs 0.540, 4 h 0.669 vs 0.575; CAISO 1 h 0.774 vs 0.657, 2 h 0.792 vs
0.785, 4 h 0.836 vs 0.825; CAISO SP15 0.842 vs 0.856 (the LP wins at that node). So dev shows the 1 h loss in ERCOT and
a loss at SP15, not the pooled CAISO loss of the exam.

**The backbone** (`training/backbone.py`). The rolling LP's plan written as a value curve: at each re-plan (first hour,
then 14:00 local, as the LP), a backward dynamic programme over the known DA horizon (DA prices as the forecast, the
battery's wear proxy, 85 % round trip, no value left at the horizon end, as the LP) on a 201-point SOC grid gives the
marginal value of stored energy at every hour; the offer rule then trades it against the RT price. Same information as
the LP, but it responds to RT deviations instead of following a fixed schedule. 1.7 s per node-year and duration.
Alone (reference R5, no network): dev 0.630 (LP 0.593, v1.1 0.638), frozen 0.684 (LP 0.504, v1.1 0.779). It gains in
2021 (0.51-0.52 vs LP 0.28-0.32 and v1.1 0.37-0.38: it sells into the RT spikes the DA plan did not see) and loses in
2019 (0.67 vs 0.78-0.79 for the LP and 0.81-0.82 for v1.1) and at 1 h in ERCOT (0.35 vs LP 0.49). 5, 9 or 41 SOC knots give the same dev result
(0.630 / 0.624 / 0.622), so the 5-knot curve stays.

**Hybrid rungs** (`train.py --hybrid`): the network reads the 135 v1.1 features plus the backbone curve
(asinh(λ/ps) at the 5 knots) and outputs a residual on it; the last layer starts at zero, so training starts from the
backbone. Backbone curves for every bank path are cached by `training/bbcache.py` (`work/data/bb/`). Scored with
`ratchet.py score`, decided by `training/ladder_v1_2.py` (`training/ratchet_v1_2.json`); rule as released (frozen
> incumbent + 0.01 and dev ≥ incumbent − 0.01); incumbent R2 (v1.1): frozen 0.7792, dev 0.6382.

| rung | change | v4 frozen | dev | decision |
|---|---|---|---|---|
| R5 | backbone alone (reference, not a candidate) | 0.6836 | 0.6295 | — |
| R6 | hybrid, curve kept non-increasing by a running minimum; 80 ERCOT path-years, 1,500 steps | 0.7024 | 0.6614 | rejected: frozen −0.077 |
| R7 | R6 on 120 ERCOT + 45 CAISO-like path-years, 2,500 steps | — | — | stopped at step 775 (training capture 0.21 vs R3's 0.37 on the same windows); not scored |
| R8 | R6 with each knot at least 0.01 below the last (the v1.1 floor); 80 ERCOT, 1,500 steps | 0.7769 | 0.6150 | rejected: frozen −0.002, dev −0.023 |
| R9 | R8 on 120 ERCOT + 45 CAISO-like, 2,500 steps | 0.7869 | 0.6327 | rejected: frozen +0.0077 (margin 0.01) |

R6 and R7 trained badly because the running minimum leaves flat pieces in the curve (the backbone has many), and the
offer rule's gradient divides by the drop between knots; R8 fixed that with the same 0.01 floor the v1.0/v1.1 softplus
steps use. Training captures of R8 and R9 then track R2 and R3 step by step on the same windows. R9 scores like R3 on the frozen
set (0.7869 vs 0.7877) but below it on dev (0.633 vs 0.644). The CAISO-like family helps the hybrid no more than it
helped the plain network.

**What was learned.** The simulator and the real dev years disagree about the backbone. R6, which stayed close to the
backbone, is the best policy on real dev so far (0.661 vs v1.1's 0.638; 2021 0.45-0.48 vs 0.37-0.38) and the worst on
the frozen set (0.702 vs 0.779). R8 and R9, free to move away from it, learn what the simulator rewards and lose the
dev gain. This is the twin-test gap again (94 % capture on twins of 2019 vs 82 % on real 2019): simulated RT spikes are
more predictable from DA prices than real ones, so a DA-anchored policy that trades RT deviations looks worse in
simulation than on real prices. The fix belongs in the simulator (unforecast scarcity), not in the policy, and it must
be calibrated until the twin gap closes; only then can the frozen set judge the hybrid fairly.

**Outcome.** No rung beat v1.1 by the rule; nothing released, no exam run. v1.1 stays. Code kept in `training/`
(`backbone.py`, `bbcache.py`, `ladder_v1_2.py`, `--hybrid` and `--min-step` in `train.py`, the hybrid path in
`model.py`, the `bb` reference in `evaluate.py` and `ratchet.py`); `batterypilot.py` is unchanged.
