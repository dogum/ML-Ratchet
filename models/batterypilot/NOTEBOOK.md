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
