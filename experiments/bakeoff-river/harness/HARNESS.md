# River Bake-off harness (Phase 1)

One problem, many learner families, equal terms, one real exam. This folder holds everything the arms share: the
frozen simulated bank, the data API, the scorer, the arm contract, registration and the once-only exam. Phase 2 arm
agents should be able to start from this page alone. Status: see `STATUS.md`.

```
harness/
  env.sh            source it first (one thread, data root, numba cache)
  config.py         paths, windows, constants (single source of truth)
  bake.py           the data API: sim bank, dev gate, guarded exam loaders, the batch format
  score.py          metrics, summaries, paired bootstrap, per-class breakdowns
  refs.py           reference rows (U-dev, U-k12, GR4J, NWM, climatology, persistence)
  evaluate.py       score an arm on the frozen sim val set and the dev gate (what rung decisions may use)
  register.py       freeze an arm (hash + smoke run + file audit + causality test)
  exam.py           the once-only exam for all registered arms (and --check / --rehearsal)
  showcase.py       the seeded showcase draw -> showcase.json (frozen)
  build_bank.py     builds bank/ (done; do not re-run)
  features.py       FlowCast's input helpers (static vector, climate indices, classes)
  armlib.py         loading/hashing arms
  bank/             the frozen bank (806 MB) + MANIFEST.json
  vendor/           byte copies of the repo's shared/water/watersim + waterdata (VENDOR.json has the hashes)
  arms_ref/         reference arms: flowcast_v1 (FlowCast v1.0 as released), climatology (dumb baseline)
  registry/         one JSON per registered arm + log.jsonl
  results/          dev reference table, checks, rehearsal output; the exam writes here
  tests/            test_guard.py (27 guard tests), check_refs.py
```

## 1. Rules (from the brief; the harness enforces what it can)

1. **Same everything.** Same bank, same inputs (daily forcing + the same static attributes), same targets, same dev
   gate, same exam rows and metrics. Any feature engineering is allowed if it uses only those inputs; write it in
   `ratchet.json` (`feature_engineering`) and `card.md`.
2. **Budget.** At most about 4 single-thread CPU-hours of training per arm, at most 3 ratchet rungs after the base.
   Rungs are hyperparameters or architecture variants, chosen on the frozen sim val set + the dev gate only. Record
   the compute in `ratchet.json` (`compute.train_cpu_hours`, all rungs; measure with `time.process_time()` or
   `/usr/bin/time`).
3. **Zero real labels in training.** Training targets are simulated (`qo` in the bank). Real dev flow is for the
   gate and for widen-only / conformal calibration only. In the gauged track, observed flow up to day t − lead is an
   input, never a training label.
4. **One exam, run once, for all arms together**, by `exam.py`, after every arm is registered. It chooses nothing.
   It refuses an arm whose files changed after registration.
5. **Two protocols, two leaderboards.** *Ungauged*: forcing + attributes only. *Gauged*: also observed flow up to
   day t − lead (leads 1, 3, 7 days). Weather of the target day is known in both (a hindcast, as FlowCast).
6. Never read exam basins or exam windows (WY1990–99, WY2016–23 on the 619 exam basins) outside `exam.py`. `bake`
   refuses (PermissionError); don't go around it with `waterdata` directly.

## 2. Quick start for an arm agent

```bash
source /home/claude/bakeoff/river/harness/env.sh        # then set your own numba cache if you use numba:
export NUMBA_CACHE_DIR=/home/claude/bakeoff/river/arms_seq/.numba
```
```python
import sys; sys.path.insert(0, '/home/claude/bakeoff/river/harness')
import bake, score, config as C
B = bake.load_bank('train')                       # memory-mapped; 12,000 runs x 1462 days
T = bake.train_arrays(B, idx=[0, 1], offs=[0, 300])   # forcing (2, 731, 8), doy, target (2, 365), static, attrs,
                                                       # states (2, 365, 5), theta (2, 70)
b = bake.sim_window(B, 0, 0)                      # one batch in the standard format (see section 4)
for b in bake.dev_batches('ungauged'): ...        # 52 dev basins, WY2001-2014
```
Then write `arms_xxx/<arm>/predict.py`, `card.md`, `ratchet.json` (section 5) and run
```bash
python3 harness/evaluate.py arms_xxx/<arm>                  # frozen sim val + dev gate -> <arm>/work/eval/
python3 harness/register.py arms_xxx/<arm>                  # when the released rung is frozen
```

## 3. The frozen simulated bank

**Recipe.** FlowCast's released bank recipe, copied line for line from `models/flowcast/training/src/simbank.py`
(physics **p1**, `models/flowcast/training/phys/p1.json`), run with the repo's latest WaterSim
(`shared/water/watersim`, version 1.0.0, vendored byte for byte in `vendor/`) and FlowCast's own seeds (val 7,
train 1). Each run:
- a generated basin: 50 % jittered dev-basin attributes (`prior='camels'`, the 52 dev basins only), 50 % parametric
  prior; random latent factors and per-parameter scatter;
- one of four runoff models (HBV, GR4J, SAC-SMA, VIC-lite), one of two snow routines (degree-day, energy balance),
  25 % Muskingum-Cunge channel routing, else a unit hydrograph;
- 5 years of real ERA5-Land forcing from a random CAMELS-US basin in a free window (1951-01..1988-09 or
  1999-10..2015-09): 1 spin-up year (not stored) + 4 stored years + 1 day = 1462 stored days;
- **water-yield multiplier (p1):** the basin's true precipitation is ERA5-Land x m with
  log m = −0.088 + 0.058 elev_km + 0.153 frac_snow + 0.397 log(P/3 mm) + N(0, 0.146), clipped to 0.4–2.5
  (median m 0.95; 5–95 % 0.65–1.35). The network still sees ERA5-Land;
- **inputs** = ERA5-Land through WaterSim's forcing-error model (P bias sd 0.08, daily cv 0.25, 10 % snow
  undercatch, T bias 0.5 °C + AR(1) 0.7 °C); **target** `qo` = simulated flow through the gauge-noise model
  (AR(1) rating error, revisions, low-flow error).

**Size and time.** train 12,000 runs, val 800 runs (frozen). Built in 7.5 min on one core (train 398 s, val 50 s),
806 MB on disk. Composition (train): HBV 2,989, GR4J 3,067, SAC-SMA 2,954, VIC-lite 2,990; degree-day 6,055 /
energy balance 5,945; Muskingum 25 %; camels prior 50 %; reservoirs in 17 %. Classes (from the run's own forcing and
attributes): arid 4,102, humid 2,437, regulated 1,723, snow 1,594, large 1,129, other 3,689.

**Verification.**
- The val set reproduces FlowCast's frozen p1 set: FlowCast v1.0 scores per-run NSE equal to FlowCast's stored
  `sim_r1b_p1.json` (median |Δ| 1.3e-5; the rest is torch float32 vs numpy float64), median 0.426 vs 0.425 stored.
- `bake.bank_rerun(split, i)` re-runs WaterSim from the stored spec and reproduces `q` and `qo` exactly (runs 0, 5,
  333 of val checked; states to 3e-5 mm, float32 storage).

**Fields** (`bake.load_bank(split, fields=None)`, read-only memory maps, rows = runs; `bank/MANIFEST.json` has
names and sha256):

| Field | Shape, dtype | What |
|---|---|---|
| `F` | (n, 1462, 8) float16 | network-visible forcing, columns `C.DYN` = P, PET, T_mean, T_min, T_max, SRAD (net), vp, wind |
| `qo` | (n, 1462) float32 | **the target**: gauge-like flow, mm/day |
| `q` | (n, 1462) float32 | true simulated outlet flow (no gauge noise), mm/day |
| `states` | (n, 1462, 5) float32 | swe, soil, groundwater, routing_store, channel_storage (mm, end of day) |
| `attrs` | (n, 11) float32 | `C.ATTR_NAMES` (same 11 attributes the real basins have) |
| `st` | (n, 18) float32 | FlowCast's static vector `C.STATIC_NAMES` (attributes + climate indices of `F`, 4 stored years) |
| `theta` | (n, 70) float32 | true parameters: all four models' draws, both snow routines, routing, latent factors, pmult (`bank_info()['theta_cols']`) |
| `model`, `snow`, `routing`, `prior`, `reg` | (n,) int8 | indices into `bank_info()['models']`, `['snows']`, uh/muskingum, prior camels=1, reservoirs=1 |
| `pmult`, `lat`, `lon` | (n,) float32 | water-yield multiplier m; the forcing donor's location |
| `doy0`, `start`, `fbid` | (n,) | day of year and date of stored day 0; forcing donor basin |

Per-run labels beyond the arrays: `bake.bank_spec(split, i)` (full WaterSim spec incl. elevation bands, the model,
snow and routing choice, seeds), `bake.bank_true_forcing(split, i)` (the forcing WaterSim really used: ERA5-Land x m,
spin-up included), `bake.bank_rerun(split, i, **simulate_kwargs)` (e.g. other parameters, `return_bands=True`).

**Memory.** `F` is 281 MB (float16) and `states` 351 MB for train. Keep memory maps; cast windows to float32 as
you draw them (`train_arrays` does). Loading everything as float32 takes ~1.4 GB: over the per-process limit.

**Notes.** `routing_store` (the model's own unit-hydrograph buffer) can be slightly negative when a flow component
is, as WaterSim reports it. Train and val draw forcing from the same donors and free windows (different seeds), as
FlowCast's did. The static vector of a sim run is computed once over its 4 stored years; for real basins over the
batch's forcing (warm-up + target + 1 day), as FlowCast does.

## 4. The batch (what `predict` receives)

Every arm gets the same dict, from the bank, the dev gate or the exam:

| Key | Value |
|---|---|
| `uid` | opaque id of the basin-window, the same for all leads (cache on it; never look anything up with it) |
| `source`, `period` | `'sim'`/`'real'`; `'simval'`, `'simtrain'`, `'dev'`, `'main'`, `'recent'` |
| `protocol`, `lead` | `'ungauged'` (lead None) or `'gauged'` (lead 1, 3 or 7 days) |
| `forcing` | (T, 8) float32, `C.DYN` columns; T = `n_warm` + `n_target` + 1 (one extra day at the end) |
| `dates`, `doy` | (T,) datetime64[D] and day of year (sim: the forcing donor's real dates) |
| `n_warm`, `n_target` | 365 warm-up rows, then the target rows `forcing[n_warm : n_warm + n_target]` |
| `attrs` | (11,) `C.ATTR_NAMES`: area km², elevation mean and range m, slope m/km, forest fraction, soil porosity, ksat cm/h, carbonate fraction, reservoir storage / annual flow, irrigation, urban |
| `static` | (18,) FlowCast's static vector (the 11 attributes transformed + p_mean, pet_mean, aridity, frac_snow, high/low-P frequencies, T mean, from the forcing) |
| `lat`, `lon` | degrees |
| `classes` | list from `snow`, `arid`, `humid`, `regulated`, `large`, or `['other']` (from static attributes) |
| `qobs` | gauged only: (T,) float32 observed mm/day over every row, NaN = missing |

**Causality (gauged).** The prediction for target row t may use `qobs[: t − lead + 1]` only (the flow observed
`lead` days before). `register.py` checks it: it replaces `qobs` after a row r0 with noise and requires the
predictions before r0 + lead to stay identical.

Real-data conventions (WaterData): forcing days are UTC days, flow days are local standard-time days (5–8 h
behind), so a same-day response is real. SRAD is **net** shortwave. ERA5-Land under-catches mountain and Pacific
Northwest precipitation (that is what p1's multiplier models). Main-exam flow is Caravan Q; recent-exam flow is
NWIS; dev flow is Caravan Q.

## 5. The arm contract

```
arms_seq/<arm>/            (or arms_tab/, arms_filt/; one folder per arm)
  predict.py               predict(batch) -> {'mean', 'q05', 'q25', 'q50', 'q75', 'q95'}: (n_target,) mm/day each
                           optional predict_many(list_of_batches) -> list of outputs (used when present)
  card.md                  what it is, inputs, features, training, rungs, compute, limits
  ratchet.json             metadata + every rung (schema below)
  <weights, code>          everything predict.py reads must be in this folder or in "extra_files"
  work/                    NOT part of the artefact: training outputs, checkpoints, eval/ (ignored by the hash)
```
- **Load weights relative to the file** (`os.path.dirname(os.path.abspath(__file__))`); the exam runs from another
  directory. Cache models at module level. Shared code for several arms (e.g. `arms_seq/common/`) goes in
  `"extra_files": ["../common"]`.
- **numpy-only at inference if feasible.** If not (e.g. an EnKF or a physics arm that calls WaterSim with numba),
  set `"numpy_only": false` and say why in `"numpy_only_reason"`. Arms may import the harness's vendored WaterSim
  (`sys.path.insert(0, '<harness>/vendor')`); it is hashed with the harness.
- Outputs must be finite and of length `n_target`. Crossing quantiles are sorted before scoring and counted
  (`n_cross`). An arm without native bands may return `q* = mean` (then its CRPS equals its MAE), but report it.
- Deterministic: the same batch must give the same output.
- **Wrapper arms** (e.g. arm C's conformal layer around arm X): `armlib.load_predict('<X dir>')` returns X's
  predict; list X's folder in `extra_files`. Each wrapper is its own registered arm (`x_conformal`), so every arm
  is reported raw and conformalised. Fit on dev residuals only: `evaluate.py <X> --save-preds` writes daily dev
  predictions (obs, mean, q05..q95 per dev basin).

**`ratchet.json`** (required keys checked by `register.py`):
```json
{
 "arm_id": "seq_tcn",                       "label": "TCN (dilated 1-D CNN)",
 "family": "cnn1d",                         "family_label": "1-D convolutional network",
 "group": "seq",                            "protocols": ["ungauged"],
 "numpy_only": true,                        "numpy_only_reason": "",
 "compute": {"train_cpu_hours": 2.4, "notes": "3 rungs, one thread"},
 "feature_engineering": "FlowCast's 11 daily features + static vector; nothing else",
 "extra_files": ["../common"],
 "margin": 0.01,
 "rule": "keep if sim val median NSE > incumbent + 0.01 AND dev median NSE >= 0.95 x incumbent's",
 "rungs": [
  {"id": "r0", "change": "base", "decision": "base", "why": "first rung", "simval_nse": 0.40, "dev_nse": 0.41},
  {"id": "r1", "change": "kernel 7 -> 5, 8 layers", "decision": "kept", "why": "+0.02 sim, dev 0.43", "simval_nse": 0.42, "dev_nse": 0.43},
  {"id": "r2", "change": "dropout 0.2", "decision": "released", "why": "+0.012 sim, dev 0.44", "simval_nse": 0.432, "dev_nse": 0.44}
 ],
 "released": "r2"
}
```
`family` should be an id from the Model families page (`site/src/lib/model-families.ts`: cnn1d, attn, ssm, gbm,
gp, diffsim, symbolic, assim, conformal, ensemble, ...). Decisions: `base`, `kept`, `rejected`, `released` (exactly
one released). Log rejected rungs too.

## 6. Scoring rung decisions: `evaluate.py`

```bash
python3 harness/evaluate.py <arm_dir> [--sets simval,dev] [--protocols ungauged,gauged] [--tag r1] [--save-preds]
```
- **Frozen sim val** (FlowCast's protocol): each of the 800 val runs gives two windows (rows 0–730 and 731–1461;
  the last 365 days of each scored), scored on the 730 concatenated days against `qo`. Reported: median over runs
  of NSE, KGE, FHV, 50/90 % coverage, CRPS/mean flow. Gauged: the same windows at leads 1, 3, 7 with `qobs = qo`.
  About 20 s for FlowCast. **Use all 800 runs for decisions** (a 300-run subset flipped one of FlowCast's rungs;
  differences of ~0.01 are at the noise level: report the paired bootstrap if close).
- **Dev gate:** the 52 dev basins, WY2001–2014 (2000-10-01..2014-09-30), warm-up from 1999-10-01, Caravan Q,
  basins with ≥ 365 observed days. Per basin: all metrics of section 8; medians; per-class medians; paired
  bootstrap against every dev reference and against FlowCast v1.0.
- Output: `<arm>/work/eval/<tag>_summary.json` + per-unit parquet tables.
- Suggested rule (state yours in `ratchet.json`): keep a rung if its sim val median NSE beats the arm's incumbent
  by > 0.01 AND its dev median NSE is ≥ 95 % of the incumbent's. Gauged arms: use lead-3 NSE (or say otherwise).

**Numbers to beat** (all scored by this harness; dev = 52 basins, WY2001–2014; NSE / KGE medians):

| Row | Protocol | Sim val NSE | Dev NSE | Dev KGE | Dev FHV % | Dev cov50 / cov90 | Dev CRPS / mean Q |
|---|---|---|---|---|---|---|---|
| FlowCast v1.0 (reference arm) | ungauged | 0.426 | **0.425** | 0.424 | −32.3 | 0.58 / 0.97 | 0.380 |
| Climatology (reference arm) | ungauged | −0.049 | −0.033 | −0.444 | −88.6 | 0.59 / 0.95 | 0.593 |
| U-dev regionalisation | ungauged | | 0.466 | 0.477 | | point | 0.573 (MAE) |
| U-k12 regionalisation | ungauged | | 0.457 | 0.505 | | point | |
| GR4J calibrated per basin (1999–2008, overlaps dev) | gauged | | 0.586 | 0.650 | | point | 0.492 |
| NWM v3.0 (50 basins) | operational | | 0.573 | 0.690 | | point | 0.510 |
| Climatology, own day-of-year mean (1999–2008) | gauged | | 0.092 | 0.005 | | point | |
| FlowCast nowcast, lead 1 / 3 / 7 | gauged | 0.709 / 0.631 / 0.564 | 0.633 / 0.567 / 0.520 | 0.674 / 0.612 / 0.594 | | 0.65/0.96 (L1) | 0.225 / 0.246 / 0.281 |
| Persistence, lead 1 / 3 / 7 | gauged | | 0.549 / −0.139 / −0.515 | 0.775 / 0.430 / 0.242 | | point | |

FlowCast's dev NSE by class: snow 0.45 (n 10), arid 0.18 (14), humid 0.56 (11), regulated 0.28 (5), large 0.35.
FlowCast vs U-dev on dev: −0.042 (95 % CI −0.149 to +0.008), a tie; vs NWM −0.148 and GR4J −0.161, losses. These
reproduce FlowCast's published dev numbers (0.425; nowcast 0.633 / 0.567 / 0.520) and WaterData's references.

## 7. Registration and the exam

**Register** when the released rung is frozen: `python3 harness/register.py arms_xxx/<arm>`. It validates
`ratchet.json` and `card.md`, hashes every file of the artefact (folder minus `work/`, caches, logs; plus
`extra_files`), runs a smoke test (3 dev basins + 6 sim val windows per protocol/lead), audits that every file and
module read from the bake-off tree is hashed, checks gauged causality and determinism, and writes
`registry/<arm_id>.json` with a CPU estimate for the exam. After registration **do not touch the arm folder**
(except `work/`). A changed arm needs `--replace` (logged) before the exam; `register.py --verify <arm_id>` and
`--list` show the state.

**The exam** (Phase 3, the coordinator, once):
```bash
python3 harness/exam.py --check        # preflight, reads no exam data
python3 harness/exam.py --run          # the exam; resumable with --resume after a crash/restart
```
It refuses if it already ran (`results/EXAM_DONE.json`), if a run is in progress, if any registered arm changed,
if an arm folder (`arms_*/<arm>/` or `arms_*/<group>/<arm>/` with predict.py + ratchet.json) is not registered
(exclude one deliberately by listing its path in `registry/excluded.json` with a reason), or if the reference arm
`flowcast_v1` is missing. Each arm runs in its own process; every exam basin with ≥ 365 observed days is scored
(main WY1990–99: 619 basins, Caravan Q; recent WY2016–23: ~603, NWIS Q); ungauged arms once per period, gauged arms
at leads 1, 3, 7. Arms are re-hashed before and after. Exam data passes only through `bake`'s guard (lock file +
run id), which is closed whenever arm code runs.

Outputs: `results/exam_<id>.json` (per period: an ungauged board with the arms + U-dev, U-k12, GR4J, NWM,
climatology; gauged boards per lead with the gauged arms + persistence, GR4J, NWM; per row the medians, CAMELS-531
median, per-class medians; per arm the paired bootstrap against FlowCast v1.0, U-dev or persistence, NWM and GR4J,
also by class; hashes and exam CPU per arm), `results/exam_<id>/scores.parquet` (per basin x period x protocol x
lead x row), `results/exam_<id>/showcase.npz` (daily series of every row for the showcase basins, both periods).
`python3 exam.py --rehearsal` runs the identical pipeline on the dev gate (tested: `results/rehearsal_*.json`).
Estimated exam time: references ~6 min, FlowCast ~5 min (both protocols), climatology seconds; arm estimates are in
the registry.

## 8. Metrics (score.py, using waterdata.metrics)

Per basin-period: NSE; KGE (2009; r, alpha, beta); FHV (% bias of the top 2 % of flows; median |FHV| also
reported); FLV; annual-maximum daily flow error (`peak_bias_pct` mean, `peak_abs_err_pct_median`); peak timing
(days); 50 % and 90 % interval coverage (25–75 %, 5–95 % bands); CRPS approximated from the five quantiles
(2 x mean pinball loss, mm/day; equals the MAE for a point forecast) and CRPS / mean observed flow; 90 % band
width / mean flow; MAE. Summaries are medians over basins; comparisons are paired bootstraps over basins of the
difference in medians (10,000 resamples, `default_rng(20261118)`); a win or loss only when the 95 % interval
excludes 0. Point references have no bands (coverage n/a). NWM's recent exam is WY2016–2022 (NWM ends Jan 2023).

**Classes** (FlowCast's strata, WaterData attributes; a basin can be in several): snow = frac_snow > 0.3; arid =
aridity (PET/P) > 1; humid = aridity < 0.6; regulated = reservoir storage ≥ 10 % of annual flow (reg_class mid or
high); large = area > 2,000 km². Exam counts (main): snow 118, arid 174, humid 128, regulated 37, large 48.

## 9. Showcase (for the page)

`showcase.json`, drawn once by `showcase.py` (seed 20261118) from static attributes, NWIS coverage counts and the
forcing only (no streamflow, no results): 2 exam basins per class (10 basins), 3 storms each (2 from the main window,
1 from the recent one), drawn among each window's 10 largest 3-day precipitation totals; storm window = peak −10 to
+30 days. Basins: 01078000, 09066200 (snow); 08158810, 13161500 (arid); 01054200, 12411000 (humid); 03237280,
05120500 (regulated); 04056500, 12414500 (large). The exam stores every row's full daily series for them.

## 10. Machine rules

- 2 cores, ~6.3 GB RAM shared by three arm agents. One thread per job (`source env.sh`; `torch.set_num_threads(1)`),
  `nice -n 5`, `setsid nohup ... &`, resumable checkpoints, ≤ 1.2 GB per process. Check memory before big jobs:
  `/sys/fs/cgroup/memory/process_api/*/claude-code-bash/memory.usage_in_bytes`.
- Disk: each arm agent ≤ 0.8 GB. Do not copy the bank; memory-map it.
- Keep your own `NUMBA_CACHE_DIR` in your folder. Never `pkill -f`; kill only your own PIDs.
- Don't write in `harness/` (except by running `evaluate.py`, which writes into your arm's `work/`, and
  `register.py`, which writes `registry/`). Read-only trees: `/home/claude/mlr-repo`, `/home/claude/water`,
  `/home/claude/bb`. Don't write to the claude.ai project.
- Keep a `STATUS.md` in your group folder; the container can restart.

## 11. Checks done in Phase 1

| Check | Result | File |
|---|---|---|
| Vendored WaterSim/WaterData = repo files | 31 files, sha256 identical | `vendor/VENDOR.json` |
| splits.json hash | 33f88ca0… (checked on every real-data load) | `config.py` |
| Bank val = FlowCast's frozen p1 set | per-run NSE |Δ| median 1.3e-5; median 0.426 vs 0.425 | section 3 |
| Bank re-run from stored spec | q, qo exact; states ≤ 3e-5 mm | section 3 |
| FlowCast arm = released `flowcast.predict()` | 1.3e-6 mm/day (float32 rounding of log p_mean) | STATUS.md |
| FlowCast arm dev gate = FlowCast's published dev | 0.425; nowcast 0.633 / 0.567 / 0.520 | section 6 |
| Reference rows = WaterData flow_scores on dev basins | NSE, KGE, FHV identical (max diff 0.0), main + recent | `results/check_refs.json` |
| Guard tests | 27 / 27 pass | `results/test_guard.json` |
| Exam pipeline rehearsal on dev | all rows, boards, bootstrap, showcase | `results/rehearsal_*.json` |
