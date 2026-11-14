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
