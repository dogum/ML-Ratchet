> **Lab notebook.** PlantDetective was built in a claude.ai cloud session (13–14 Nov 2026) on the shared GridSim + GridData
> platform, now in `shared/grid/`. The first entry is that session's status document as written, except that the page
> link now points at `lab/index.html` and the sandbox paths are named rather than spelled out; its `src/`, `out/` and
> `checks/` refer to the cloud sandbox workspace (`grid/plantdetective/`). `MODEL_CARD.md` holds every number.

## 14 Nov 2026 — build notes (plantdetective-status.md, as written)

# PlantDetective — status (v1.0, 14 Nov 2026)

GridSim phase 2 project in ML Ratchet. Reads a fossil unit's hourly output (MW) and infers technology, full- and part-load heat rate (CAMPD gross basis), minimum stable load, ramps, minimum up/down, starts (hot/warm/cold), capacity factor and flexibility class, each with error bars. Trained only on simulated units; graded on 407 real Texas CAMPD units.

- Page: the original page, now `lab/index.html` (explorer of 16 exam units with the network live in JS, lab with sliders, exam, ratchet, twin test)
- Release: `plantdetective-v1.0.zip` in the sandbox outputs (numpy only; demo runs in 0.3 s from an unzipped copy)
- Code and data: the sandbox workspace `grid/plantdetective` (src/, STATUS.md, ratchet.json, out/)

## Recipe as built
- Physics: GridSim generated systems + unit-level MILP dispatch (5 systems x 92 days, train-year EIA-930 inputs); per-unit self-commitment DP (start costs by type, min up/down, ramps, outages) against GridSim prices (30 %) or real ERCOT day-ahead prices of 2018/2020/2022 (70 %); cogeneration mode. DP verified against an independent MILP: max relative difference 1.0e-14 on 20 cases (checks/check_dp.json).
- Real background: ERCOT train-year prices + Henry Hub; measurement emulator (partial start/stop hours, warm-ups, load noise, derates, trips, CT-only CC reporting, CHP fuel, EIA-923 accounting).
- Dev: 10 dev plants (40 units), years 2019 and 2021 only. Twin test mismatch v1 0.211 -> v2 0.128.
- Network: 3 x MLP (170 features), 440,797 weights; heteroscedastic outputs; dev-fitted widen-only interval factors.

## Rungs (frozen sim val skill / dev gate skill)
| R0 GridSim MILP units only | 0.021 / 0.015 | first |
| R1 + self-committed and cogen units | 0.491 / 0.117 | kept |
| R2 bank v2, dev twin calibration | 0.475 / 0.200 | kept |
| R3 skip connection to EIA-923 heat rate | 0.470 / 0.191 | rejected |
| R4 3-network ensemble | 0.501 / 0.215 | kept (final) |

## Exam (395 units with enough full-load hours, all years, with price + context; median error)
- Min stable load: 5.7 pt pooled (1 month 10.8, 1 yr 6.8) vs dev median 8.0, output 1st percentile 37.9, tech default 14.0.
- Ramp up p99: 14 % pooled (1 yr 18 %) vs dev median 39 %, output p99 70 %.
- Starts/yr: 45 % from 1 month vs counted 57 %; from 1 yr 31 % vs 28 %; pooled the count is the truth itself. Run lengths: plain counts win.
- Full-load heat rate: 14 % overall vs tech default 10 %, EIA-923 14 %. Coal 4.4 % (EIA-923 10.3 %), gas steam 4.7 % (22 %); CC 21 % biased high (default 7 %); CT 16 % (EIA-923 6.5 %).
- Heat input per unit-year: coal 5.7 % vs EIA-923 8.8 %; steam 10.1 % vs 11.8 %; CT 17.3 % vs 5.1 %; CC 22.5 % vs 22.0 %.
- Technology: output only 66 % (1 month) to 77 % (pooled); with EIA-860 prime mover 93-96 %.
- Coverage (year windows): 68 % band 48-79 %, 95 % band 79-98 %.

## Limits
CC heat-rate bias (dev's 3 CC plants were cogeneration / CT-only reporting; exam CC mostly report whole blocks); CT heat rate too high (sim prior); start/run statistics no better than counts with >= 1 yr; min up/down noisy; 10 dev plants; ERCOT hub prices used for all units.

## Next rungs
1. Re-centre CC/CT priors with non-CAMPD evidence (EIA-860 CC configuration, CT vintage/model) and make EIA-923 trust learnable per technology (a skip that passes the dev gate).
2. Unit-specific prices (ERCOT settlement point of the unit's node) and coal delivered prices from EIA-923 receipts.
3. Learn the run-statistics heads as corrections to the counted statistics (helps short windows, never worse for long).

## Sandbox gotchas
- gridsim dispatch(outages='random') goes infeasible when an outage starts while a unit is on (min-down row); src/gswrap.py turns outages into derates (upstream candidate). Unit-level MILP: 12-21 min per 92-day system on the shared CPU.
- pandas 3 timestamps are microseconds: use epoch helpers (src/common.py).
- CAMPD gross load is 0 for some CC units (South Houston EPN801/803): no truth.

## 14 Nov 2026 — brought into the repository

**Layout.** The release zip became `models/plantdetective/` under the model contract: `plantdetective.py`, `pdcore.py`,
`pipeline.py`, `demo.py` and `examples/` (the three exam units of the release, unchanged) at the top; `weights.npz` renamed
to `plantdetective_weights.npz` (the one code change in inference: the default path in `Detective.__init__`); every
training script as run in `training/`, with the raw ratchet log (`ratchet.json`), `exam_report.txt`, `sigma_factors.json`,
the twin summaries (`twin_v1.json`, `twin_v2.json`) and the page's JS port (`pd.js`); the release's `checks/` in
`training/checks/`. The original page is `lab/index.html`, untouched. `model.yaml`, `ratchet.yaml` and this notebook are new;
`MODEL_CARD.md` is the release card with every number kept, the paths fixed, and three additions (the price-context note,
the reproduction below, the files list).

**Paths.** `training/_paths.py` (new) defines `WORK` (`models/plantdetective/work/` or `$ML_RATCHET_WORK/plantdetective`),
`GRID` (`shared/grid/` or `$ML_RATCHET_GRID`; puts `GRID/gridsim` and `GRID` on `sys.path` for `gridsim` and `griddata`)
and `MODEL`. `training/common.py` now takes `ROOT = WORK` and `DATA = WORK/data` from it, so banks, sets, networks, cached
real units, grade files, tables and page builds all land in `work/`. Mechanical replacements: the sandbox `sys.path` lines
(`grid/gridsim`, `grid`) -> `_paths`; `realdata.PUDL` -> `GRID/data/raw/pudl`; `exam.py` and `webdata.py` import the release
from `MODEL` (not `ROOT/release`), and `webdata.py` reads `MODEL/plantdetective_weights.npz`; `examples.py` and
`save_release.py` still write to `WORK/release/...` (they create the folder first); `checks/check_dp.py` imports from
`training/` and writes `WORK/checks/check_dp.json`. Every training module imports from the repository with the shared
platform in place; the scripts that need CAMPD stop at `griddata.campd_units()` (see below).

**Reproduced.**
- Demo: the repository demo prints exactly what the release demo printed from the unzipped release (every line
  identical except the timing line, 0.2 s vs 0.3 s). Golden: `tests/golden/plantdetective.txt`.
- The demo's three exam units against the lab page's stored Python readings (all years pooled, made by
  `webdata.unit_payload` in the sandbox): every estimate within 1.7e-4 relative, the same flexibility classes and
  technologies. The page computed from float64 prices; the examples store prices and gas as float32. The page's copy of
  each unit also held its 2026 hours (the reserve year): the lab explorer's truth and pooled readings included them
  (starts per year truth 69.328 vs 69.333 for Gregory Power Facility 101, 72.076 vs 72.082 for Decordova CT2). The exam
  (`exam.py`, years 2018–2025) and the release examples (`examples.py`, `year <= 2025`) did not; `lab/` stays as it was.
- `check_dp.py` re-run here on the ERCOT 2022 prices in `shared/grid/data/proc/prices`: max relative difference
  1.0058963487522703e-14, identical to the shipped `check_dp.json` (20 s).

**Not reproduced: the 395-unit exam.** `exam.py` needs the CAMPD hourly unit data (`griddata.campd_units()` reads
`shared/grid/data/proc/campd/units.parquet`), and `shared/grid/data/proc/campd/` is not in the shipped platform data. It
rebuilds with `mlratchet data grid --only campd` (about 6 GB of EPA bulk downloads), after which
`python training/cache_real.py exam`, `python training/dev_medians.py`, `python training/exam.py R4_ens exam` and
`python training/report.py WORK/out/grade_exam_R4_ens.pkl training/sigma_factors.json` would re-score it (`exam.py` loads
the networks from `WORK/data/models/R4_ens_s*.npz`; the release ships them packed in `plantdetective_weights.npz`, so they
would need unpacking first). Every exam number in `model.yaml` and `MODEL_CARD.md` is the release's, from
`training/exam_report.txt` and, for the per-technology rows, from the exam tables embedded in the lab page (written by
`webdata.py` from the same grade file; they agree with `exam_report.txt` to its 3 decimals). The GridSim simulated banks
and the dev twin test were not re-run either (hours of MILP; the dev units also need CAMPD).

**Price context.** The exam fed the real ERCOT hub prices of 2023–2025 as optional context; no choices were made from them
(training used 2018/2020/2022 prices; the dev gate and error-bar factors used 2019 and 2021). The card and `model.yaml`
say so; whether to keep it or restrict the context to pre-exam years is in `next`.

**Live instrument.** `web/export_instrument.py` writes `site/public/instruments/plantdetective/` (2.47 MB): the three
networks in float16 (normalisation in float32), the three units' hourly output 2018–2025 with the shared ERCOT hub price
and Henry Hub series, and 18 reference cases (3 units x all years, all years output-only, calendar year 2024, 91 days
from July 2023, 30 days from January 2025, 30 days from April 2019 without EIA facts). `site/src/instruments/plantdetective/net.ts`
ports `pdcore.features`, `window_vector`, `Detective.windows`, the ensemble, the pooling and `infer`; the feature edges
(`np.linspace` load bins) come from Python as exact float64 so histogram counts match. Parity: features identical, worst
relative difference over all outputs 8.8e-7. Float16 weights move the estimates by at most 2.1e-4 relative. The first
export used float16-rounded normalisation constants in the Python reference while the browser read float32 ones, which
gave 2.3e-3; the reference now uses the same float32 values.

## 14 Nov 2026 — the 395-unit exam re-run from the shipped weights

The CAMPD hourly tables were rebuilt from EPA's no-key bulk files (`mlratchet data grid --only campd`: 8 Texas state-years,
about 10 minutes of downloads on the repository's machine; `units.parquet` has 452 units, as in the build). The ensemble's
three members were unpacked from `plantdetective_weights.npz` into `work/data/models/R4_ens_s{0,1,2}.npz`, then
`cache_real.py exam`, `cache_real.py dev`, `dev_medians.py`, `exam.py R4_ens exam` and `report.py` ran as shipped (26 min
for the exam step, single thread). The report is identical to `training/exam_report.txt` on all 190 lines (line endings
aside): every exam number in `model.yaml` and the card reproduces. The paragraph above saying the exam was not reproduced
is superseded; it stays as the record of that pass.


## 14 Nov 2026 — GridSim outage defaults from the exam years (HANDOFF P0.5): checked, no decision changes

**What leaked.** GridSim v1.0's default forced and planned outage shares for coal, CC, CT and gas steam came from Texas
RE's 2024/2025 assessments (exam-period data). GridSim physics 1.1 (default since commit a4a24d4) uses pre-2023 sources:
forced/planned coal 0.15/0.11 -> 0.136/0.083, CC 0.08/0.11 -> 0.036/0.083, CT 0.08/0.06 -> 0.058/0.083, gas steam
0.13/0.19 -> 0.136/0.083. The handoff said PlantDetective did not use outage rates; it did, in two places.

**How PlantDetective touched them.**
- `simbank_gs.py` dispatches 5 generated systems (job seeds 0-4) with `dispatch(outages='random')`; `gswrap.py` turns the
  outages into derates. The outage hours of every unit in those systems follow `tech_default('efor'/'planned_frac')`.
- That reached bank v2 in two ways. (1) The 69 GridSim-dispatched units (19, 24, 12, 7, 7 per system). (2) The GridSim system price is the
  price background of 37 of the 110 self-commitment shards (592 units; `bank_dp.shard` draws it with probability 0.3).
  The price is also a network input for those windows. Bank v2 = 1,760 shard units + 69 GridSim units = 1,829, as
  logged; the job seeds are not recorded, but seeds 0-4 give exactly this count.
- Bank v0 (rung R0) was GridSim units only (3 systems); bank v1 (R1) also re-emulated each job's units inside every
  shard on a GridSim background. Neither bank nor their networks ship, so those were not rebuilt.
- Not touched: `bank_dp.py`'s own outage ranges for self-committed units (EFOR coal 0.03-0.12, CC 0.02-0.08, CT
  0.02-0.08, gas steam 0.03-0.15; planned 0-0.10, 0-0.08, 0-0.05, 0-0.12) do not read GridSim. No source is given for
  them in the code, the card or the build notes, so they cannot be shown to be pre-exam. They are wide uniform priors and
  are not copies of the v1.0 values: the v1.0 coal, CC and CT forced shares sit at or above their upper ends. The
  ERCOT-price shards, the twin test (fixed EFOR = planned = 0.04), the dev calibration, the dev gate and the exam do not
  call GridSim's outage model.
- Other inputs checked. `generate_fleet` sizes the GridSim units from `generator.SIZE` (EIA-860 2019-2024; the
  coordinating check found the exam years move its quantiles by at most 3 %). Its heat-rate medians (EIA-923 2017-2023)
  are overwritten by PlantDetective's own ranges for the four target technologies. `simunits.py` gives "EIA-860 sizes"
  and "EIA-923 operating heat rates" without years. `prepare_fleet`'s heat-rate window is set from the dispatch dates
  (train years), not the default. Prices, gas and EIA-930 shapes are train years only (asserted in code).
  `realdata.context` takes each generator's latest EIA-860 report (up to report year 2025) for real units, including dev
  units. With reports up to 2022 only, 4 dev units' facts change: 2 minimum loads (130 -> 122 MW, 300 -> 290 MW), and 2
  CTs that came online in 2024 lose their facts. The dev gate moves from 0.2147030 to 0.2147042.

**How much it mattered.** Bank v2 was rebuilt twice with the release's own scripts (same seeds), once per physics
version, in `work/p05/` (`training/outage_check.py`; results in `training/checks/outage_physics_1_1.json`). Each
92-day MILP took 17 s to 10 min here.
- GridSim prices (mean $/MWh, v1.0 -> v1.1): 29.5 -> 28.1, 81.8 -> 19.0, 26.6 -> 18.2, 285 -> 60, 338 -> 102. Hours
  above $1,000: 0 -> 0, 19 -> 4, 2 -> 0, 83 -> 22, 209 -> 94. Medians move by at most $1.1/MWh. Fewer outages mean far
  fewer scarcity hours in 3 of the 5 systems.
- The 73 ERCOT-price shards are byte-identical under both versions. The 37 GridSim-price shards and the 5 GridSim-unit
  shards differ. Usable units (ran at least once): 1,710 -> 1,678. Windows: 20,520 -> 20,136. Frozen-set windows:
  2,712 -> 2,700. Of these, 60 are GridSim units and 720 -> 708 are shard units on GridSim prices (29 % of the frozen
  set together).
- Targets (one row per unit): ERCOT-price units identical. GridSim units (50 -> 47): median starts/yr 53.6 -> 59.5, CF
  0.15 -> 0.19, every target's KS distance <= 0.13. Shard units on GridSim prices (492 -> 463): median starts/yr
  29.8 -> 23.8, full-load heat rate 11.29 -> 11.29, KS <= 0.13.
- Input features (standardised mean difference): ERCOT-price windows <= 0.02 sd. GridSim-price windows: mean 0.06 sd,
  largest 0.67 sd (top load-histogram bin) and 0.46 sd (an implied-heat-rate bin, price / gas).
- Frozen-set skill of the shipped networks (rebuilt v1.0 set -> v1.1 set): R4 ensemble 0.492 -> 0.501; members (the R2
  recipe, seeds 0-2) 0.467 -> 0.472, 0.476 -> 0.488, 0.473 -> 0.483. The logged values on the original set were 0.501
  (R4) and 0.475 (R2). The rebuild differs from the build because of the MILP time limits; it also has 5 fewer usable
  units.
- Dev gate (40 real dev units): 0.21470 under both versions, the logged value. It cannot depend on GridSim's outage
  model.

**Decisions.**
- R4 vs R2: ensemble minus seed-0 member +0.025 (v1.0) and +0.029 (v1.1), margin 0.01. Ensemble minus the best member
  +0.016 and +0.013. Kept either way.
- R3: rejected by the dev gate (-0.009, tolerance 0.005), which is unchanged. On val it needed a +0.015 relative swing.
  The largest swing between same-recipe networks here is 0.007.
- R2 vs R1 (+0.210) and R1 vs R0 (+0.470): an order of magnitude beyond any shift seen here.

No rung changes, so the release stands. No bank was rebuilt for training, no network was retrained, and the exam was
not re-run. The 1.0-built bank v2 remains the training record. A future bank should be built with physics 1.1.

## 14 Nov 2026 — wording corrections (honesty audit)

- `MODEL_CARD.md` limits and `model.yaml` losses said that from a year of history the plain counts are "as good" as the
  network for starts and run lengths. They are better (`training/exam_report.txt`, mode ctx, year windows): starts per
  year 27.8 % median error for the count vs 31.0 % for the network; median run 18.2 % vs 33.9 % (the count wins at every
  window length). Start shares are about even (hot 4.8 vs 5.3 pt, warm 7.3 vs 7.2, cold 6.4 vs 6.0). Reworded to say so.
- `MODEL_CARD.md`: the line after the ratchet table had no blank line before it and rendered as a table row. Fixed.
- `model.yaml` `next`: "re-run the 395-unit exam in this repository" was removed. It was done (commit d73d2c7, identical
  on all 190 lines; entry above).

## 15 Nov 2026 — v1.1: run statistics as corrections to the counts

**Goal.** Turn the exam losses in model.yaml into wins honestly: show each failure on dev first, fix it, show the dev
gain, then examine once.

**Setup.** The work folder of the grid-follow-up worktree (banks, cached dev and exam units, the physics-1.1 rebuild in
`work/p05/`) was copied into this checkout's `work/` (git-ignored). `shared/grid/gridsim/refdata/` (Henry Hub and other
reference files, git-ignored) was missing here and copied from the same worktree; nothing else under `shared/` changed.
The frozen set is `set_v2p11`: bank v2 rebuilt with GridSim physics 1.1 by the P0.5 check (`work/p05/v1_1`, 20,136
windows, 2,700 frozen). The v1.0 release re-scores 0.5008 on it, dev 0.2147 (both as logged on 14 Nov).

**Dev evidence (40 dev units, 2019 + 2021, mode ctx; scratch script, numbers below).** Median error of v1.0 vs the
statistic counted in the same window (where the window has one):

| target | month: net / count | quarter: net / count | year: net / count |
|---|---|---|---|
| starts per year | 0.60 / 0.82 | 0.44 / 0.45 | 0.33 / 0.17 |
| min up | 0.38 / 0.37 | 0.33 / 0.47 | 0.44 / 0.35 |
| median run | 0.28 / 0.33 | 0.42 / 0.36 | 0.44 / 0.13 |
| min down | 0.37 / 0.55 | 0.40 / 0.43 | 0.37 / 0.25 |

So dev shows the exam's failure: from a year of output the count beats the network for all four run statistics. The
window features already hold each count (feature 87: log1p starts per year; 70/71/72: log median, log 5th percentile
and its flag for complete runs; 85/86: the same for off spells).

Heat rates on dev (year windows, v1.0): combined cycle +12 % median bias, but the three plants differ. All three are CHP in
EIA-860. 50815 and 55470 have truths of 10-13 MMBtu/MWh (process steam or gas-turbine-only reporting); 55327 reports the
whole block (truth 7.0) and the network reads 9.9-10.7. One whole-block plant cannot support a CC-specific fit; none
was tried. Gas turbines: median
bias +1 %, error 6.0 % vs 5.1 % for EIA-923 (8 units, 4 with an EIA-923 figure): dev does not show the exam's +15 %
turbine bias, so it cannot be fixed from dev. Minimum stable load: the dumb baseline is the dev median itself
(in-sample on dev), so dev cannot show whether the network loses to it.

**Code.** `training/model.py`: `COUNT_SKIPS` and `count_offsets`; `train(..., cskip=True)` adds the counted statistic
to the mean head of starts per year, min up, median run and min down (gated where the count is undefined) and saves the
rows as `cskip` in the weights; `np_forward` applies them. `plantdetective.py` gained the same `count_offsets`
(inference stays numpy-only). `training/ladder_v1_1.py` scores and decides; `training/exam_rows.py` writes the
model.yaml rows from a grade file; `save_release.py` takes a version.

**Rungs** (rule unchanged: val > incumbent + 0.01, dev not below incumbent − 0.005; `training/ratchet_v1_1.json`):

| rung | change | val (v2p11) | dev (month / quarter / year) | decision |
|---|---|---|---|---|
| R4 | v1.0 as released | 0.5008 | 0.2147 (0.134 / 0.177 / 0.333) | incumbent |
| R5 | R4 recipe retrained on the physics-1.1 bank (seeds 0-2, 40 epochs, about 80 s each) | 0.5142 | 0.2089 (0.129 / 0.168 / 0.329) | rejected: dev −0.006 |
| R6 | R5 + count skips | 0.5317 | 0.2250 (0.128 / 0.171 / 0.376) | kept, released as v1.1 |

On val, the count skips raise starts per year 0.66 → 0.71, min up 0.39 → 0.47, median run 0.61 → 0.66, min down
0.32 → 0.36 (skill vs R5). On dev year windows: starts per year 0.57 → 0.75, median run 0.52 → 0.66, min down
0.29 → 0.35, but min up 0.18 → 0.07; on dev month windows median run fell 0.38 → 0.32.

Error-bar factors refitted on the dev grade of R6 (`exam.py R6_cs dev`; `training/sigma_factors_v1_1.json`, 1.00-2.07).
The release choice was committed (79fd39c) before the exam ran.

**Exam, run once** (`exam.py R6_cs exam`, 395 units, about 45 min; `training/exam_report_v1_1.txt`). v1.0 re-scored
from its grade file reproduces `training/exam_report.txt` exactly. Rows, v1.0 → v1.1 (median error):
- Min stable load pooled 5.7 → 4.7 pt (win); one month 10.8 → 10.1 pt vs dev median 8.0 (loss).
- Ramp up pooled 13.7 → 14.9 % (win, worse).
- Starts per year: one month 44.9 → 44.1 % (win); one year 31.0 → 27.6 % vs the count's 27.8 % (loss → win, by 0.2
  points); pooled 23.3 → 8.5 %.
- Median run: one month 42.2 → 45.1 % vs 23.1 % (loss, worse); one year 33.9 → 19.4 % vs 18.2 % (loss); pooled
  26.3 → 9.8 %.
- Full-load heat rate pooled 13.7 → 11.6 % vs default 10.3 % (loss). Coal 4.4 → 5.0 % vs dev median 5.2 % (win,
  worse). Gas steam 4.7 → 6.3 % vs dev median 5.8 % (**win → loss**). CC 20.7 → 23.5 % (loss). GT 16.2 → 8.8 % vs
  EIA-923 6.5 % (loss). Part load 14.5 → 14.5 % (loss).
- Capacity factor one month 5.3 → 5.4 pt (win).
- Heat input per unit-year: coal 5.7 → 6.4 % (win), gas steam 10.1 → 11.5 % vs 11.8 % (win, thin), GT 17.3 → 11.3 %
  (loss), CC 22.5 → 26.5 % (loss).
- Technology output only 66 / 76.5 % → 61 / 74.7 %; with the prime-mover rule 95 / 96 / 93 % → 95 / 95 / 91 %.
- Coverage of the 68 % band (year windows) collapsed for the skipped heads: starts 69 → 17 %, min up 48 → 40 %,
  median run 71 → 33 %, min down 63 → 31 %. The networks learned small corrections with small sigmas from simulated
  units whose truth spans little more than the window; the dev factors (1.05-1.56 for these heads) were fitted on
  2-year truths. On the exam the truth spans 8 years.

Wins: 8 → 8 (starts per year from one year gained; gas-steam heat rate lost). The heat-rate changes are not from the
count skips (they touch only the four run-statistic heads); they come from retraining R6 on the physics-1.1 bank.

**Shipped.** `plantdetective_weights.npz` = v1.1 (R6, 3 members with `cskip`, factors v1.1);
`plantdetective_weights_v1_0.npz` = v1.0. Golden demo regenerated (`tests/golden/plantdetective.txt`). Live
instrument re-exported with `cskip` in the pack's meta; `net.ts` applies it. Parity: 18 cases, worst 9.7e-7; float16
moves the estimates by at most 2.8e-4.

**Next.** Fit the run-statistic error-bar factors against truths that span more than the window. Unit-specific
heat-rate evidence for combined cycles from sources other than CAMPD.

## 27 Nov 2026 — audit fix: missing hours inside a run are not starts (no retrain, exam unchanged)

The two-reviewer audit (`docs/audits/plantdetective-v1.1.md`) found that a missing hour inside a run read as a stop and a
restart: hiding one interior hour a day took 6179_3 from 2.76 to 107.58 predicted starts a year. Run detection now
bridges a missing stretch with running hours on both sides (6.42 on the same test; 55086_101 87.74 vs 85.35; 8063_CT2
27.95 vs 27.11). The site port follows. The 407 exam unit files have no missing hours, so the exam is unchanged.
