# PlantDetective v1.1 — model card

Part of ML Ratchet ("Trained on physics. Graded on reality."), built on the shared grid platform (GridSim + GridData,
`shared/grid/`).

## What it does
Reads a fossil unit's hourly output (MW) and infers its operating character, each with a 68 % and 95 % interval:
technology (coal steam, gas combined cycle, gas combustion turbine, gas steam), full-load and part-load heat rate
(CAMPD gross basis, MMBtu/MWh; part load = max(min stable load, 40 %)), minimum stable load, ramp (99th percentile
hour-to-hour change while running), minimum up / median run / minimum down time, starts per year and their
hot (< 8 h off) / warm (8–48 h) / cold (≥ 48 h) shares, capacity factor and an operating-flexibility class
(baseload, load-following, cycling, peaking; a rule on the estimates). Several years are pooled window by window.

## Inputs
- hourly output in MW, hour-beginning UTC timestamps, ≥ 30 days (NaN = missing hour); gross basis expected (a `net` flag exists);
- optional: hourly electricity price and gas price (the lab page and the exam used the ERCOT day-ahead hub average and Henry Hub);
- optional EIA-860/923 context: prime mover, CHP flag, nameplates, EIA-860 minimum load, cold-start time, online year,
  EIA-923 annual heat rate of the linked generators. Same EIA-860 values as `gd.fleet(..., plant_safe=True)`; no CAMPD data.
The model never sees CAMPD heat input or operating time.

Windows: calendar years with ≥ 300 days; else about 91-day pieces (≥ 80 days); else the whole series (≥ 30 days).

## Physics and simulator
1. **Systems**: GridSim `generate_fleet` + unit-level MILP `dispatch` (cluster=False) driven by real EIA-930 demand, wind and
   solar shapes and Henry Hub gas of the training-input years 2018/2020/2022 (5 systems × 92 days; slow: 12–21 min each).
2. **Units**: self-commitment of each unit against a price background, by an exact dynamic programme (on/off and time-in-state,
   7 output levels, convex fuel curve F(P) = P·HR_full·(1 + β(1/f − 1) + c(f − 1)²), start costs by hot/warm/cold type,
   minimum up/down, ramp and start-up/shut-down capability, Markov forced outages and a planned outage). Backgrounds:
   GridSim system prices (30 %) or real ERCOT day-ahead hub prices of 2018/2020/2022 (70 %). Cogeneration units run near
   flat with trips. **Check: DP profit equals an independent MILP (scipy/HiGHS) on 20 random 96-h cases to 1.0e-14**
   (`training/checks/check_dp.json`; re-run in this repository: 1.0058963487522703e-14, the same to every digit).
   GridSim's own physics checks are in `shared/grid/checks/`.
3. **Measurement emulator** (CAMPD-like record): partial start and stop hours, warm-up ramps measured on the dev starts
   (coal: first hour ~1 % of capacity, then 2–8 %/h), AR(1) load noise, summer derate of gas turbines, trips and aborted
   starts, CEMS heat-input noise, combined cycles that report only the gas turbine (shared steam part), cogeneration
   steam fuel; and an EIA-923 emulator (linked-generator accounting, CHP fuel allocation, gross/net ratio fitted on dev).
4. Real background: real ERCOT prices and gas of training years; no real labelled unit is ever a training sample.

`dispatch(outages='random')` goes infeasible when an outage starts while a unit is on (the minimum-down row);
`training/gswrap.py` turns outages into derates instead (an upstream candidate for GridSim).

## Network
3 × MLP (170 features + a net-output flag → 256 → 256 → 128 → 4 + 2×13; leaky ReLU), ensemble mean, variance = mean member
variance + spread. 440,797 weights. Features: load histogram and quantiles of interior hours, first/second/last-hour loads,
ramp quantiles, on/off spell-length histograms, start rates and types, daily and weekly pattern, on-probability and load
versus the market implied heat rate (price/gas, 12 bins), context. Heteroscedastic Gaussian NLL + cross-entropy.
Bank v2: 1,829 simulated units, 20,580 windows (30–365 days), 13 % frozen validation by unit.
Numpy inference (`plantdetective.py`, `pdcore.py`); weights in `plantdetective_weights.npz` (the release's `weights.npz`,
renamed). The browser runs a float16 copy: estimates within 2.1e-4 of the float32 weights; JS == numpy to 8.8e-7.

## v1.1 (15 Nov 2026): run statistics as corrections to the counts

v1.1 predicts starts per year, minimum up time, median run and minimum down time as **corrections to the statistic
counted in the window itself** (the features already hold it: log1p of starts per year, log median and log 5th
percentile of complete runs and off spells). Where the window has no such count (no complete run, fewer than 3 spells)
the network predicts the value on its own, as before. Everything else is unchanged. It was trained on bank v2 rebuilt
with GridSim physics 1.1 (the HANDOFF P0.5 rebuild, frozen set v2p11) and released by the unchanged rule.

Why, on dev first: on the 40 dev units (2019 + 2021), the count in a one-year window beat v1.0 for starts per year
(17 % vs 33 % median error), median run (13 % vs 44 %), minimum up (35 % vs 44 %) and minimum down (25 % vs 37 %). The
same failure was in the published exam rows.

| rung | change | frozen val (v2p11) | dev | decision |
|---|---|---|---|---|
| R4 | v1.0 as released, re-scored | 0.501 | 0.215 | incumbent |
| R5 | R4 recipe retrained on the physics-1.1 bank | 0.514 | 0.209 | rejected: dev −0.006 (tol 0.005) |
| R6 | R5 + count skips | 0.532 | 0.225 | **released as v1.1** (+0.031, dev +0.010) |

Error bars refitted on dev (`training/sigma_factors_v1_1.json`, 1.00–2.07). Log: `training/ratchet_v1_1.json`
(`training/ladder_v1_1.py`); exam tables: `training/exam_report_v1_1.txt`, rows by `training/exam_rows.py`.
v1.0's weights are kept as `plantdetective_weights_v1_0.npz` (`Detective(WEIGHTS_V1_0)`).

**v1.1 exam (run once, 15 Nov 2026, the same 395 units; v1.0 in brackets).** Starts per year from one year 27.6 %
(31.0 %) against 27.8 % for the count: a win by 0.2 points. Starts per year pooled 8.5 % (23.3 %); median run pooled
9.8 % (26.3 %), one year 19.4 % (33.9 %) against the count's 18.2 %: still a loss, and one month 45.1 % (42.2 %) is worse.
Minimum stable load pooled 4.7 pt (5.7), one month 10.1 pt (10.8) vs the dev median 8.0. Ramp up pooled 14.9 % (13.7 %).
Full-load heat rate pooled 11.6 % (13.7 %) vs the technology default 10.3 %; coal 5.0 % (4.4 %), **gas steam 6.3 %
(4.7 %), now a loss to the dev median 5.8 %**; combined cycle 23.5 % (20.7 %); gas turbine 8.8 % (16.2 %) vs EIA-923
6.5 %. Part load 14.5 % (14.5 %). Heat input per unit-year: coal 6.4 % (5.7 %), gas steam 11.5 % (10.1 %), turbine 11.3 %
(17.3 %), combined cycle 26.5 % (22.5 %). Technology from output only 61 % / 74.7 % (66 % / 76.5 %). The heat-rate
changes do not come from the count skips (they touch only the four run-statistic heads); they come from retraining.
**The run-statistic error bars are now too narrow**: 17 % (starts), 31 % (min down), 33 % (median run) and 40 % (min up)
of year windows fall inside the 68 % band (v1.0: 69 %, 63 %, 71 %, 48 %). The dev factors did not catch it: dev truth
spans 2 years, exam truth 8.

**Combined cycle and gas turbine heat rates (not changed in v1.1).** The 3 dev combined-cycle plants are all CHP in
EIA-860. Two have truths of 10–13 MMBtu/MWh (process steam or gas-turbine-only reporting); one (55327) reports the whole
block (truth 7.0, which v1.0 reads as 9.9–10.7). One whole-block plant cannot support a CC-specific fit, so none was
tried. The 8 dev gas turbines do not show
the exam's turbine bias (median +1 %; 6.0 % error vs 5.1 % for EIA-923 on the 4 with a figure), so it cannot be fixed from dev.

## Ratchet v1.0 (`training/ratchet.json`; keep if frozen-val skill +0.01 and dev gate not worse than −0.005)
| rung | change | sim val | dev | decision |
|---|---|---|---|---|
| R0 | GridSim unit-level MILP units only | 0.021 | 0.015 | first |
| R1 | + self-committed and cogeneration units, GridSim and ERCOT-train-year price backgrounds | 0.491 | 0.117 | kept |
| R2 | bank v2: dev twin calibration (start/stop shapes, trips, offer slope, start costs, CHP steam credit, EIA-923 gross ratio) | 0.475 (R1: 0.266) | 0.200 | kept |
| R3 | skip connection: heat rate as a correction to EIA-923 | 0.470 | 0.191 | rejected |
| R4 | 3-network ensemble | 0.501 | 0.215 | kept (released as v1.0) |

Twin test (40 dev units, 2019+2021, twin vs real pattern mismatch): v1 0.211 → v2 0.128.
Error bars: widen-only factors per output, fitted on the dev windows so that ≥ 68.3 % fall within 1 σ
(`training/sigma_factors.json`, 1.00–1.90).

## Exam, v1.0 (407 exam CAMPD units, 395 with enough full-load hours; all years 2018–2025; gross basis; never used for choosing)
Median error vs the unit's whole-history CAMPD truth (fractions in absolute points, others relative). With context and price:

| quantity | 1 month | 1 year | all years | best baseline (all years) |
|---|---|---|---|---|
| min stable load | 10.8 pt | 6.8 pt | **5.7 pt** | dev median 8.0 pt; output p1 37.9 pt; tech default 14.0 pt |
| ramp up | 25 % | 18 % | **14 %** | dev median 39 %; output p99 70 % |
| starts / year | **45 %** | 31 % | 23 % | output count 57 % (1 month), 28 % (1 yr), 0 % (all: it is the truth's definition) |
| median run | 42 % | 34 % | 26 % | output stats 23 % / 18 % / 0 % |
| full-load heat rate | 15 % | 14 % | 14 % | tech default (oracle tech) 10 %; EIA-923 14 % |
| part-load heat rate | 17 % | 16 % | 15 % | EIA-923 12 %; tech default 13 % |
| capacity factor | 5.3 pt | 3.8 pt | 2.1 pt | output stats 0 pt (all years) |
| technology, output only | 66 % | 67 % | 77 % | — |
| technology, with EIA-860 prime mover | 95 % | 96 % | 93 % | — |

By technology (all years, full-load heat rate): coal 4.4 % (EIA-923 10.3 %, default 7.5 %), gas steam 4.7 % (EIA-923 22 %,
default 11 %), **gas CC 21 % (default 7.0 %, EIA-923 22 %), gas CT 16 % (EIA-923 6.5 %)**.
Annual heat input per unit-year (GridData's baseline metric, 2,499 unit-years): coal 5.7 % vs EIA-923 8.8 %; gas steam
10.1 % vs 11.8 %; CT 17.3 % vs 5.1 %; CC 22.5 % vs 22.0 %; all 17.1 % (on the 2,255 with EIA-923: 16.1 % vs 13.7 %).
Output only (no price, no context) is similar for load, ramps and starts; heat rate is worse (22 % full load).

Coverage (year windows, after dev-fitted widen-only factors): 68 % band covers 48–79 % (median ~65 %); 95 % band 79–98 %.
All-years pooled: 68 % band 53–90 %.

The full tables are in `training/exam_report.txt` (all technologies, every window length, both modes); the per-technology
rows above come from the exam tables `training/webdata.py` wrote from the same grade file into the lab page.

**Price context in the exam.** The exam fed the real ERCOT day-ahead hub prices of 2023–2025 (exam years) as optional
context at test time. No choice was made from them: training used only 2018/2020/2022 prices, and the dev gate and the
error-bar factors used the dev years 2019 and 2021. Whether to keep this or restrict the context to pre-exam years is an
open decision (see `model.yaml` `next`).

**Reproduction in this repository (14 Nov 2026).** The full 395-unit exam was re-run from the shipped weights on CAMPD hourly data rebuilt that day from EPA's bulk files (`mlratchet data grid --only campd`; the release ships no CAMPD tables): `cache_real.py dev|exam`, `dev_medians.py`, `exam.py R4_ens exam`, `report.py`, with the ensemble unpacked from `plantdetective_weights.npz` into `work/`. The report is identical to `training/exam_report.txt` on all 190 lines. The demo prints the release's numbers on its three exam units; its pooled readings agree with the lab page's stored readings to 1.7e-4 relative (the examples store prices as float32); `check_dp` gives 1.0058963487522703e-14.

## Limits
- Combined-cycle heat rate biased +21 %: the 3 dev CC plants (two cogeneration, one gas-turbine-only reporting) pushed the
  simulator's priors toward CT-only reporting and process steam; most exam CC units report the whole block (~7.3).
- Gas-turbine heat rate biased +15 %: the simulated CT range sits above the Texas fleet; EIA-923 alone is better for CTs.
- Starts and run lengths from ≥ 1 year of history: plain counts are better. One-year windows: starts per year 27.8 %
  median error for the count vs 31.0 % for the network; median run 18.2 % vs 33.9 % (the count wins at every window
  length). Start shares are about even (hot 4.8 vs 5.3 pt, warm 7.3 vs 7.2 pt, cold 6.4 vs 6.0 pt). With all years
  pooled the counts are the truth's own definition.
- Minimum up/down (5th percentile of spells) are noisy targets; worst coverage.
- Calibrated on 10 dev plants; ERCOT hub prices used for every unit, including SPP/MISO/El Paso ones.
- Ramp truth is the observed 99th-percentile hourly change, a lower bound on capability.
- CAMPD gross load is 0 for some combined-cycle units (South Houston EPN801/803): those units have no truth.
- Bank v2 was built with GridSim physics 1.0, whose coal, CC, CT and gas-steam outage defaults came from Texas RE's
  2024/2025 assessments (exam-period data). They set the outages of the 69 GridSim-dispatched units and the GridSim prices
  behind 37 of 110 self-commitment shards (about 29 % of the frozen-set windows). The bank was rebuilt with physics 1.1
  (pre-2023 sources). Scarcity hours fell (for example 209 -> 94 hours above $1,000 in one system), the ERCOT-price
  shards were byte-identical, the frozen-set skill of R4 moved 0.492 -> 0.501, and R4's lead over a single network
  moved +0.025 -> +0.029 (margin 0.01). The dev gate is unchanged (0.215). No rung decision changes, so the release
  stands (NOTEBOOK, 14 Nov 2026; `training/checks/outage_physics_1_1.json`).
- Self-committed units use their own outage ranges (`training/bank_dp.py`: EFOR 2-15 %, planned 0-12 % by technology).
  No source is recorded for these ranges.
- Real units' EIA-860 facts come from each generator's latest report (up to report year 2025). For the dev gate,
  restricting to reports up to 2022 changes 4 dev units' facts and moves the score by 1e-6.

## Files
`plantdetective.py` (API: `Detective().infer(x, t, price, gas, context)`), `pdcore.py` (features, truth statistics),
`pipeline.py` (CSV in, JSON report out), `plantdetective_weights.npz` (v1.1: 3-network ensemble with count skips +
dev-fitted interval factors), `plantdetective_weights_v1_0.npz` (v1.0),
`demo.py` and `examples/` (three real exam units, EPA CAMPD 2018–2025, public domain), `training/` (every script used to
build the banks, train, calibrate, grade and build the page, paths through `training/_paths.py`; `training/pd.js` is the
lab page's own JavaScript port; `training/checks/` the DP-vs-MILP check), `lab/index.html` (the original page),
`web/export_instrument.py` (the live instrument on the site).

## Data and licences
EPA CAMPD hourly emissions (public domain); EIA-860/923 via Catalyst Cooperative PUDL (CC-BY 4.0); EIA-930 (public domain);
ERCOT day-ahead settlement point prices (public ERCOT data); Henry Hub daily (EIA, public domain); RTS-GMLC defaults
(Barrows et al. 2020, BSD-3). Platform: GridSim and GridData (`shared/grid/gridsim`, `shared/grid/data`).
