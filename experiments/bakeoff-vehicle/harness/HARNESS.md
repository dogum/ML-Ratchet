# Vehicle Bake-off harness (Phase 1)

One problem, many learner families, equal terms, one real exam. The problem is EnergyNet's: the energy (fuel or
electricity) a vehicle uses on a trip, from the vehicle spec, the route and the weather. Two input forms, two
leaderboards: the **table track** (EnergyNet's 151 features) and the **trace track** (the 1 Hz trip trace + a spec
vector). This folder holds everything the arms share: the frozen simulated bank, the data API, the scorer, the arm
contract, registration and the once-only exam. Phase 2 agents should be able to start from this page alone.
Status: `STATUS.md`.

```
harness/
  env.sh            source it first (one thread, vendored platform, local data, numba cache, PYTHONPATH)
  config.py         paths and constants (single source of truth)
  bake.py           the data API: sim bank, tiers, table / trace batches, dev gate, guarded exam
  score.py          output check, metrics, real-set units, summaries, D16, paired bootstrap
  refs.py           reference rows: E_phys alone, RoadData / EnergyNet classical baselines (dev-fitted)
  evaluate.py       score an arm on the frozen sim val set + the dev gate (what rung decisions may use)
  register.py       freeze an arm (hash + smoke run + file audit + determinism / row-independence)
  exam.py           the once-only exam for all registered arms (and --check / --rehearsal)
  showcase.py       the seeded showcase draw -> showcase.json (frozen: 8 exam trips)
  build_bank.py     built bank/ (done; do not re-run)        run_bank.sh  its driver
  build_real.py     real records: dev (done: real/dev/); exam only from exam.py
  armlib.py         loading / hashing arms
  bank/             val/ and train/: rows.npz, trace_v.npy, trace_x.npy, fleet.json, MANIFEST.json; dumb_rates.json
  real/dev/         dev records in EnergyNet's record format (+ Mersin synthetic traces)
  vendor/           byte copies of the repo's shared/vehicle (roadsim, realdata, checks/hybrid_specs.py) and
                    models/energynet (v2.1 code + weights); vendor/VENDOR.json has the hashes (140 files, all equal)
  data/raw/ved/     VED static vehicle tables (xlsx), fetched from the URL in the repo's data/build.py (see section 2)
  arms_ref/         reference arms: enet_v21 (EnergyNet v2.1 as released), climatology (dumb baseline)
  registry/         one JSON per registered arm + log.jsonl
  results/          checks, dev reference tables (results/dev/), rehearsal output; the exam writes here
  tests/            check_enet_dev.py, check_bank_val.py, check_trace.py, vendor_check.py, test_guard.py
```

## 0. Quick start and the ten things to know

```bash
source /home/claude/bakeoff/vehicle/harness/env.sh      # one thread, PYTHONPATH=harness; set your own NUMBA_CACHE_DIR
```
```python
import sys; sys.path.insert(0, '/home/claude/bakeoff/vehicle/harness')
import numpy as np, bake, score
B = bake.load_bank('train')                         # 28,800 simulated trips (EnergyNet v2.1's recipe, physics v2)
T = bake.draw_tiers(B, np.random.default_rng(0))    # what is known about each vehicle (redraw per epoch)
D = bake.sim_table(B, T)                            # table track: D['X'] (28800, 151), D['E_phys'], D['y'] (MJ)
S = bake.sim_spec(B, T); tr = bake.sim_trace(B, 0)  # trace track: spec (n, 49), rl (n, 5); trace (L, 8) at 1 Hz
blocks = bake.block_means(tr, 5)                    # the shared 5-s resampling (optional)
for b, y in bake.simval_batches('table'): ...       # frozen sim val (3,985 trips), batches as predict() gets them
for b in bake.dev_batches('trace'): ...             # the dev gate (labels stay in the harness)
```
```bash
python3 harness/evaluate.py arms_table/<arm> [--track table]   # frozen sim val + dev gate -> <arm>/work/eval/
python3 harness/register.py arms_table/<arm>                   # when the released rung is frozen
```
1. Output = trip energy in **MJ** (`pred` + q05..q95), one row per input row, > 0. Log errors are unit-free.
2. Table track input = EnergyNet's 151 features; trace track = 1 Hz trace (8 channels) + spec (49) + rl (5). No
   E_phys in the trace batch, but `bake.e_phys_from_trace(batch)` gives it (declare if used).
3. Targets depend on the tier (MAF flag): always `bake.target(B, T)` after drawing tiers.
4. Rung decisions: full sim val skill (EnergyNet's definition) + dev **D16**. EnergyNet v2.1: skill 0.680, D16 0.1133.
   The bank's train split is ONE of v2.1's three train banks: v2.1 saw 3x the data (86,400 trips, on a GPU).
5. Dev = VED 16 cars, EPA dev families, Mersin dev week, ICCT dev truck, 5 ZTBus legs. Never the exam.
6. EPA rows: 5 distinct cycle traces x ~10k test cars on dev (~93k on the exam): cache trace encodings on
   `trace_key`. Real heavy-duty traces reach 37,919 s (sim max 7,200 s): chunk them.
7. Conformal / widen-only layers fit on `<arm>/work/eval/<tag>_<track>_dev_preds.parquet` (multiply pred and
   quantiles by `vr` before comparing with `y`, as the scorer does for VED rows).
8. EnergyNet's own training code (torch MLP, `train.py`, `dataset.py`) is vendored in `vendor/energynet/training/`;
   RoadSim in `vendor/vehicle/roadsim` (physics pinned through `vendor/energynet/training/physics.py`).
9. ZTBus full-set rows are out of scope (no traces here); the exam has 78 ZTBus legs + 8 missions instead.
10. 2 cores / 6.3 GB shared by three agents: one thread, nice, setsid nohup, <= 1.2 GB per process, <= 0.6 GB disk.

## 1. Rules (from the brief; the harness enforces what it can)

1. **Same everything per track.** Same bank, same inputs, same targets, same dev gate, same exam rows and metrics.
   Feature engineering is allowed if it uses only the track's inputs; write it in `ratchet.json`
   (`feature_engineering`) and `card.md`.
2. **Budget.** At most about 4 single-thread CPU-hours of training per arm, at most 3 ratchet rungs after the base.
   Rungs are decided on the full frozen sim val set (3,985 trips) plus the dev gate. Record compute in
   `ratchet.json` (`compute.train_cpu_hours`, all rungs; measure with `time.process_time()`; there is no /usr/bin/time here).
3. **Zero real labels in training.** Targets come from the simulated bank. Real dev labels are for the gate and for
   widen-only / conformal calibration only. `ratchet.json` must not say `"real_labels_in_training": true`.
4. **One exam, run once, for all arms together**, by `exam.py`, after every arm is registered. It chooses nothing and
   refuses an arm whose files changed after registration.
5. Never read exam records. `bake` refuses (PermissionError) and `real/exam/` does not exist until `exam.py --run`
   builds it under the lock. Do not go around it with `realdata` on exam trips, EPA exam families, Mersin exam trips,
   ICCT exam trucks or ZTBus exam legs.

## 2. The data: reconciliation (step 1 of Phase 1)

**Local vs repo.** The local processed RoadData (`/home/claude/road/data/proc`, cloud build of 12-13 Nov) was compared
with the repo's `shared/vehicle/data` (tracked tables, build scripts, DATA.md):
- Every proc table the repo tracks is byte-identical locally (sha256): `splits.json` (d917df78...), baselines,
  checks, all 24 cycles, EPA test cars (vehicles, tests, split), fueleconomy tables, the 83 ZTBus v1 legs and mission
  table, Mersin, ICCT.
- The local build already has the two 13 Nov fixes: EPA cycle labels un-swapped (conventional medians FTP 24.5 /
  HWY 37.6 mpg) and VED mass = curb only (the 136 kg is added once by RoadSim; it is applied in `realdata.vedspec`,
  which is vendored from the repo).
- The repo's build scripts differ from the local copies only by path handling (`ML_RATCHET_VEHICLE`), an OS-neutral sort
  in `ved_index.py` (same order on Linux), and additions: Argonne D3 (`d3_*.py`) and the full ZTBus set (`hd.py`).
- **Missing locally:** `proc/d3/` (Argonne D3; EnergyNet does not use it) and `proc/hd/ztbus_full/` (the full ZTBus
  traces, 1.3 GB, git-ignored; its tables and split file are tracked in the repo). Also `raw/` is absent: the VED
  static vehicle tables (two small xlsx files that `realdata.vedspec` reads) were fetched from the documented source in
  the repo's `data/build.py` (raw.githubusercontent.com/gsoh/VED, Apache-2.0) into `data/raw/ved/`.

**EnergyNet v2.1 reproduces exactly on the local data** (`tests/check_enet_dev.py` -> `results/check_enet_dev.json`).
Real dev records rebuilt with EnergyNet's own `realfeat.py` helpers (`build_real.py`), scored with EnergyNet's own
`dataset.assemble` and `evaluate.py` metrics, from the released weights: all 21 dev terms of `ratchet2.json`'s r17
record (16 D2 terms + 3 EPA hybrid extras + the 5 v1 ZTBus legs and 2 traced missions) are equal, max |diff|
5.6e-16; same n in every term. The two D2 terms on the full ZTBus set (1,641 dev legs, 294 dev missions) cannot be
computed here. The bake-off's dev score is therefore **D16** = the mean of the 16 other D2 terms: EnergyNet v2.1
**0.1133** (published terms, same mean).

**Exam scope (decided here, before any exam data is read):**
| EnergyNet exam row | in the bake-off exam | note |
|---|---|---|
| VED trips: petrol 14,469 / hybrid 8,241 / Leaf 167 / plug-in 2,916; car totals; by architecture | yes, all | traces local |
| EPA exam families: petrol FTP / HWY / US06 / 20 F cold FTP / SC03, hybrid FTP / HWY / US06 / SC03 (+ cold FTP), BEV city / HWY | yes, all | cycle traces |
| Mersin diesel / CNG trips | yes | EnergyNet's synthetic routes (same generator draws) |
| ICCT tractor-trailers (6) | yes | |
| ZTBus full: 5,847 legs, 1,115 missions | **no** | traces not on this machine (git-ignored 1.3 GB build) |
| ZTBus v1 traced legs | **78 legs of 8 exam missions** (+ those 8 mission totals) | a subset of the full exam, no published v2.1 number |
The exam checks that EnergyNet v2.1 (the reference arm) reproduces its published exam numbers (`training/exam3.json`)
on every in-scope row and writes the comparison into the exam result (`reproduction_enet_v21`).

## 3. The frozen simulated bank

**Recipe: EnergyNet v2.1's, on RoadSim physics v2 (pinned, `options='roadsim-physics-v2'`, generator donors 'v2').**
EnergyNet built its v2.1 banks in three passes (bank.py, then bank_dyno.py for rung r14, then bank_idlestop.py for
r15). `build_bank.py` builds the end state in one pass with the same random draws:
- fleet = `training/fleet.py`: generated vehicles of all 7 classes x 7 powertrains, model year 2005-2026 with the era
  prior, road-load / transmission / BEV-motor / articulated-bus widening, simulated window-sticker labels (FTP-75
  approximation + HWFET, EPA 5-cycle formulas); r15: conventional light-duty cars with idle-stop get it switched off and
  their labels recomputed;
- trips = `training/bank.py`: VED train_noise routes (unlabelled cars, real speed / lidar grade / ASOS weather) 45 %,
  synthetic drives 35 %, dynamometer cycles 13 %, cycles on the road 7 % (light duty; heavy duty 80 % synthetic);
  the network sees station weather, the car drives T + N(0, 1.5 C) and wind x U(0.4, 0.9); start states (cold soak /
  warm, start SOC); r14: SC03 and 20 F cold-FTP dyno draws run under the EPA test conditions (`dynoproc.run`);
- targets: 8 versions per trip (`tg`); the bake-off target is EnergyNet v2.1's version (`bake.target`: the OBD MAF
  label model where the tier says the fuel is a MAF estimate, v1's plug-in hardware patch for plug-ins).

| split | fleet | trips | seeds (fleet / trips) | = EnergyNet's |
|---|---|---|---|---|
| `val` (frozen) | 400 vehicles (val2) | 4,000 (10 per vehicle); 3,985 scored (y > 0.02 MJ) | 202 / 777 | val2di (frozen set val2[h]) |
| `train` | 2,400 vehicles (train2) | 28,800 (12 per vehicle) | 201 / 101 | train2di, the first of v2.1's three train banks |

**Verification.** The val bank reproduces EnergyNet's frozen set: EnergyNet v2.1 scored on it with EnergyNet's own code
(frozen tiers seed 777) gives n 3,985, MdAPE 0.0746, E_phys MdAPE 0.2104, 68 % band coverage 0.6469 and every
per-architecture and per-class MdAPE equal to `ratchet2.json`'s val2[h] record (max |diff| 9e-10)
(`tests/check_bank_val.py` -> `results/check_bank_val.json`).
EnergyNet v2.1 was trained on three such banks (86,400 trips, seeds 101-103); the bake-off train bank is one of them
(28,800 trips), so v2.1 saw 3x the data of every arm (EnergyNet's r11b/r13 rungs: 2x and 3x data gave +0.009 and
+0.019 skill over 1x).

**Size and time.** Built on 2 worker processes, one thread each: val 4,000 trips in ~5 min, train 28,800 trips in 38 min (about 1.3
CPU-hours in all, no failed run). Disk: val 86 MB, train 595 MB (traces 33.2 M seconds), 681 MB in all.
Composition (train): conv 13,356, P2 hybrid 2,580, series hybrid 1,152, power-split 2,412, plug-in power-split
1,320, plug-in series 864, BEV 7,116 trips; car 8,016, SUV 8,508, pickup 4,128, van 2,040, bus 1,836, medium truck
2,136, heavy truck 2,136; routes VED 10,155, synthetic 12,778, dyno 3,030, cycle on the road 2,837; 450 SC03 /
cold-FTP dyno rows under the test conditions; 247 cars (2,964 trips) with idle-stop switched off (EnergyNet: 8,892
rows over its three banks = 3 x 2,964).

**Fields** (`B = bake.load_bank(split)`; rows in memory, traces memory-mapped):

| Field | Shape | What |
|---|---|---|
| `R` | (n, 90) | EnergyNet's route vector of the observed trip (`features.route_vec`) |
| `PH` | (n, 4, 8) | physics integrals (MJ) for 4 road-load sets: true m + true road load; true m + class road load; reported m; class m |
| `tg` | (n, 8) | energy targets (MJ): raw, + v1 on-road wrapper, OBD label model, both; the same 4 with v1's plug-in patch |
| `comp` | (n, 9) | RoadSim's energy ledger (MJ): aero, rolling, friction brake, aux / HVAC, powertrain losses, grade (net), kinetic change, fuel (chemical), battery chemical out |
| `lab` | (n, 4) | simulated window-sticker labels, MJ/km: city fuel, hwy fuel, city elec (wall), hwy elec |
| `arch`, `cls` | (n,) | indices into `features.ARCHS` (conv, hev_parallel_p2, hev_series, hev_powersplit, phev_powersplit, phev_series, bev) / `CLASSES` (car, suv, pickup, van, bus, truck_md, truck_hd) |
| `m_true`, `m_rep`, `cda`, `crr`, `cv`, `disp`, `power`, `diesel`, `trans`, `ngears`, `cap`, `artic`, `my` | (n,) | vehicle parameters (true mass incl. payload, reported mass, CdA, Crr, Crr speed coef., displacement, power kW, diesel, transmission index, gears, battery kWh, articulated bus, model year) |
| `soc`, `cold`, `warm`, `dyno`, `src`, `cycle`, `dur`, `dist`, `T_true`, `trace_miss`, `idle_off`, `vid` | (n,) | start SOC (usable fraction), cold / warm start, dynamometer run, route source (0 VED, 1 synthetic, 2 dyno, 3 cycle on road), cycle index (`bank.LD_CYC`, -1 none), duration s, distance m, true mean temperature, RoadSim's trace-following rms, idle-stop switched off (r15), vehicle id |
| `trace_off`, `trace_len` | (n+1,), (n,) | offsets into the trace arrays |
| `trace_v`, `trace_x` | (sum L,), (sum L, 7) | the observed trip at 1 Hz: v float32; grade, elev - elev[0], T_amb, wind along / across the heading, rho_air, surface float16 |

`bake.fleet(B)` gives the vehicle specs (RoadSim contract dicts; `spec_sim` = what RoadSim drove, idle-stop off), labels
and model years; row i belongs to `bake.vehicle_index(B)[i]`. Traces in the bank: median 862 s (val 892 s), 90 % 2,234 s, max 7,200 s.

## 4. Inputs, tiers and the two tracks

**Tiers (spec masking).** Each trip is seen with a "tier": what is known about the vehicle (exact / reported / unknown
mass, road load, engine size, power, transmission, gears, battery size, window-sticker labels, start SOC, warm / cold,
MAF-estimated fuel, model year). EnergyNet redraws tiers every epoch from `dataset.draw_tiers`; `bake.draw_tiers(B,
rng)` is that function. Draw fresh tiers per epoch (or per tree / per fit) as you see fit, and say so. The frozen val set
uses `bake.val_tiers()` (seed 777). Real records carry the tier of what the real data provides (VED: reported curb
mass, engine size, label, start SOC for plug-ins / EV, warm if the trip starts moving, MAF flag; EPA: exact test mass
and road-load coefficients, engine, transmission, model year, cold / hot bag; ZTBus: the bus's 19 t curb mass as a
reported mass; Mersin / ICCT: class-level).
**Important:** the target depends on the tier's MAF flag (`bake.target(B, T)`): draw tiers, then targets.

**Table track** (`bake.sim_table(B, T)` -> `X` (n, 151) float32, `E_phys`, `y`): EnergyNet's inputs,
`[R (90) | physics vector (12) | V (49)]` (`vendor/energynet/features.py`). X[:, 90] = log E_phys (MJ), the generic
road-load estimate EnergyNet corrects. V: one-hot architecture (102-108) and class (109-115), log mass and level
flags, road load, engine, transmission, battery, label terms (relative to E_phys), start SOC, warm / cold, MAF flag,
dyno flag (V[46] = X[:, 148]), model year (X[:, 149], flag X[:, 150]).

**Trace track** (`bake.sim_trace(B, i)` -> (L, 8) float32; `bake.sim_spec(B, T)` -> `spec` (n, 49), `rl` (n, 5)):
- channels `config.TRACE_CH`: v m/s, grade (rise / run, clipped +-0.2), elevation relative to the start m, T_amb C,
  wind along the heading m/s (+ = headwind), crosswind m/s, rho_air kg/m3, surface code (0 dry, 1 concrete, 2 wet,
  3 gravel, 4 snow). Exactly the per-second values EnergyNet's features are computed from (`features.prep`).
- `spec`: EnergyNet's V (49) with the two label entries 35 and 38 made absolute (log label energy per km, MJ/km);
  everything else as V (one-hots, mass, flags, SOC, model year, ...).
- `rl`: the road-load numbers the physics estimate uses at this tier: mass kg, CdA m2, Crr, Crr speed coefficient,
  rotating-mass fraction (class priors where unknown).
- E_phys is not an input of the trace track, but it is a function of these inputs: `bake.e_phys_from_trace(batch)`
  recomputes it (equal to the table's to 2e-4 in log; `results/check_trace.json`). Using it is allowed feature
  engineering; declare it.
- **Shared resampling / cropping rule** (use it, or state yours): feed 1 Hz, or 5-s blocks from
  `bake.block_means(trace, 5)` -> (ceil(L / 5), 11): the block means of the 8 channels, mean v^2, mean v^3 and the summed
  positive kinetic-energy increment per unit mass (these keep the energy physics that plain means lose); for traces longer than 7,200 s (only real heavy-duty records: ZTBus legs up to 37,919 s, Mersin up to
  ~6,000 s), split into chunks of <= 7,200 s, encode each chunk, and pool (sum for extensive quantities, mean for
  intensive ones). Sim traces: median 862 s, max 7,200 s. Real dev: VED 120-2,797 s (median 439), EPA cycles
  601-1,370 s, ICCT up to 879 s. EPA rows repeat 5 traces (`trace_key` lets you cache an encoder's output).

## 5. The batch (what `predict` receives)

| Key | Value |
|---|---|
| `uid` | (n,) opaque row ids (stable; cache on them; never look anything up with them) |
| `trace_key` | (n,) opaque id of the trace content (rows with the same trace share it) |
| `source`, `period`, `track` | 'sim' / 'real'; 'simval', 'simtrain', 'dev', 'exam'; 'table' / 'trace' |
| `n`, `arch`, `cls` | number of rows; (n,) int indices (also one-hot inside X / spec) |
| table: `X`, `E_phys` | (n, 151) float32; (n,) MJ |
| trace: `traces`, `spec`, `rl` | list of n float32 arrays (L_i, 8); (n, 49) float32; (n, 5) float32 |

`predict(batch)` returns `{'pred', 'q05', 'q25', 'q50', 'q75', 'q95'}`: (n,) trip energy in **MJ**, finite and > 0.
Energy = fuel chemical energy (fuel g x 0.042 MJ/g) for conventional and hybrid vehicles, battery DC net discharge for
battery-electric, the sum for plug-ins (RoadData's / EnergyNet's label definition; the per-powertrain units g, kWh and
MJ are this divided by 0.042 or 3.6; log errors are unit-free). `pred` is the point estimate (EnergyNet: the median,
exp of the mean log). Optional: `'components'` (n, 5) MJ: aero, rolling, friction brakes, aux / HVAC, powertrain
losses (EnergyNet gives them; the page shows them). Predict the whole trip; the harness scales VED predictions by the
trip's valid-energy share (RoadData's label counts valid seconds only), converts EPA tests to gal/mi or wall kWh/mi,
averages Mersin's two route draws and converts CNG, sums ZTBus legs into missions.

Each row must be predicted from its own inputs (register.py warns if reversing a batch changes outputs).

## 6. The arm contract

```
arms_table/<arm>/  (or arms_trace/, arms_phys/; one folder per arm, or arms_x/<group>/<arm>/)
  predict.py       predict(batch) -> dict above; optional predict_many(list_of_batches) -> list
  card.md          what it is, inputs, features, training, rungs, compute, limits
  ratchet.json     metadata + every rung (schema below)
  <weights, code>  everything predict.py reads must be in this folder or in "extra_files"
  work/            NOT part of the artefact: training outputs, checkpoints, eval/ (ignored by the hash)
```
- Load files relative to `__file__`; the exam runs from another directory. Cache models at module level.
- numpy-only at inference if feasible; otherwise `"numpy_only": false` + `"numpy_only_reason"` (e.g. TabPFN).
- Deterministic; quantiles that cross are sorted and counted; a point arm may return q* = pred (CRPS = |log error|).
- **Wrapper arms** (arm C's conformal layer): `armlib.load_predict('<X dir>')` gives X's predict; list X's folder in
  `extra_files`; register each wrapper as its own arm (`x_conformal`). Fit on dev residuals only: `evaluate.py <X>`
  writes `<X>/work/eval/<tag>_<track>_dev_preds.parquet` (per dev record: meta, y, vr, pred, q05..q95, MJ for the
  whole trip; multiply by `vr` before comparing with `y` on VED rows, as score.real_units does). Copy what the wrapper
  needs into its own folder (hashed).

**`ratchet.json`** (required keys checked by `register.py`):
```json
{
 "arm_id": "tab_gbm", "label": "Gradient-boosted trees (LightGBM)",
 "family": "gbm", "family_label": "Gradient-boosted trees",
 "group": "table", "tracks": ["table"],
 "numpy_only": true, "numpy_only_reason": "", "real_labels_in_training": false,
 "compute": {"train_cpu_hours": 2.1, "notes": "3 rungs, one thread"},
 "feature_engineering": "the 151 inputs; target log(y / E_phys)",
 "extra_files": ["../common"],
 "margin": 0.01,
 "rule": "keep if sim val skill > incumbent + 0.01 AND dev D16 not worse than the incumbent's + 0.01",
 "rungs": [
  {"id": "r0", "change": "base", "decision": "base", "why": "first rung", "simval_skill": 0.62, "simval_MdAPE": 0.085, "dev_D16": 0.125},
  {"id": "r1", "change": "monotone constraints on the grade-work inputs", "decision": "released", "why": "+0.012 skill, D16 0.121", "simval_skill": 0.632, "simval_MdAPE": 0.081, "dev_D16": 0.121}
 ],
 "released": "r1"
}
```
`family`: an id from the Model families page (`site/src/lib/model-families.ts`: mlp, gbm, gp, cnn1d, rnn, attn,
ssm, symbolic, conformal, ensemble, ...). `tracks`: ["table"], ["trace"] or both (an arm on both tracks is scored on
both boards). Decisions: base, kept, rejected, released (exactly one released). Log rejected rungs too.

## 7. Scoring rung decisions: `evaluate.py`

```bash
python3 harness/evaluate.py <arm_dir> [--track table|trace] [--sets simval,dev] [--tag r1] [--nmax N]
```
- **Frozen sim val** (EnergyNet's protocol): the 3,985 val trips with y > 0.02 MJ, frozen tiers. Reported: MdAPE,
  median |log error|, **skill** = 1 - median |log error| / median |log error| of the dumb baseline (median energy per km
  per architecture x class on the train bank: `bake.dumb_rates()`), bias, cov50 / cov90, crps_log, per architecture /
  class / route source. A full pass takes ~2 s for EnergyNet (table); building all 3,985 trace batches takes < 1 s.
  The dev gate: EnergyNet 6 s; trace batches ~4 s to build (13.6 M trace seconds, 11.8 M of them EPA cycles).
- **Dev gate** (never exam): VED 16 dev cars (1,498 trips: ICE 645, HEV 275, Leaf 258, plug-in 318 scored), EPA dev
  families (10,345 test legs -> conv FTP 2,877 / HWY 2,835 / US06 371 / cold FTP 280 / SC03 302 vehicles, hybrid
  70 / 72 / 29 / 29 / 29, BEV 98 / 97), Mersin dev week (56 diesel + 56 CNG trips, 2 route draws each), ICCT dev truck
  (3 cycles), 5 ZTBus legs (2 missions). Per set: MdAPE, bias, cov50 / cov90, crps_log, car totals, conditions;
  **D16** = mean of the 16 dev MdAPEs of EnergyNet's D2 that exist here (VED ICE / HEV / EV / PHEV trips; EPA conv FTP /
  HWY / US06 / cold FTP / SC03, hybrid FTP / HWY, BEV city / HWY; Mersin diesel / CNG; ICCT); paired bootstrap
  against EnergyNet v2.1 and the reference rows per set.
- Output: `<arm>/work/eval/<tag>_<track>_summary.json` + `_simval_preds.parquet` + `_dev_preds.parquet`.
- Suggested rule (EnergyNet's model-rung rule; state yours): keep a rung if its sim val skill beats the incumbent by
  > 0.01 AND its dev D16 is not worse than the incumbent's by more than 0.01. D16 moves by ~0.005 between two seeds of
  the same network (EnergyNet's notes): a dev "gain" under 0.01 is noise.

**Numbers to beat** (scored by this harness; dev D16 and sets = MdAPE):

| Row | Sim val skill | Sim val MdAPE | Dev D16 | VED ICE / HEV / EV / PHEV | EPA conv FTP / HWY | Mersin diesel / CNG | ICCT |
|---|---|---|---|---|---|---|---|
| EnergyNet v2.1 (reference arm) | **0.680** | 0.0746 | **0.1133** | 0.066 / 0.126 / 0.140 / 0.163 | 0.093 / 0.070 | 0.130 / 0.170 | 0.190 |
| E_phys alone | 0.048 | 0.2104 | 0.2278 | 0.299 / 0.202 / 0.153 / 0.225 | 0.190 / 0.093 | 0.171 / 0.304 | 0.865 |
| Climatology (reference arm) | -0.002 (0 by definition) | 0.2338 | 0.3093 | 0.285 / 0.198 / 0.426 / 0.678 | 0.181 / 0.706 | 0.061 / 0.076 | 0.391 |
| MOVES-like (VED, dev-fitted, in-sample on dev) | | | | 0.065 / 0.133 / 0.195 / 0.414 | | | |
| EPA regression (dev-fitted, in-sample on dev) | | | | | 0.060 / 0.043 | | |

EnergyNet v2.1's dev coverage (cov50 / cov90): VED ICE 0.49 / 0.89, HEV 0.52 / 0.88, EV 0.49 / 0.88, PHEV 0.53 / 0.84;
EPA conv FTP 0.46 / 0.82. Its dev EPA terms: conv US06 0.053, cold FTP 0.109, SC03 0.096; hybrid FTP 0.157, HWY 0.100;
BEV city 0.089, HWY 0.059 (`results/dev/enet_v21_table_summary.json`). The classical rows are fitted on dev, so their
dev numbers are in-sample (on the exam they are honest).

## 8. Registration and the exam

**Register** when the released rung is frozen: `python3 harness/register.py arms_x/<arm>`. It validates ratchet.json
and card.md, hashes the artefact (folder minus work/, caches, logs; plus extra_files), runs a smoke test per track
(24 records of each dev set + 48 sim-val trips), audits that every file read from the bake-off tree is hashed, checks
determinism and row independence (warnings), estimates the exam CPU and writes `registry/<arm_id>.json`. After
registration **do not touch the arm folder** (except work/). A changed arm needs `--replace` (logged) before the exam;
`register.py --verify <arm_id>` and `--list` show the state.

**The exam** (Phase 3, once):
```bash
python3 harness/exam.py --check        # preflight, reads no exam data
python3 harness/exam.py --run          # the exam; resumable with --resume after a crash / restart
python3 harness/exam.py --rehearsal    # the identical pipeline on the dev gate (tested)
```
It refuses if it already ran (`results/EXAM_DONE.json`), if a run is in progress, if any registered arm changed, if an
arm folder (`arms_*/<arm>/` or `arms_*/<group>/<arm>/` with predict.py + ratchet.json) is not registered (exclude one
deliberately in `registry/excluded.json` with a reason), or if a reference arm (enet_v21, climatology) is missing.
It builds the exam records under the lock (~10-15 min), runs each arm in its own process on each of its tracks
(re-hashed before and after), computes the reference rows, and writes `results/exam_<id>.json`: per track a board
(rows = that track's arms + EnergyNet v2.1 + climatology + E_phys + the classical baselines; per set the summary, VED
condition strata, car totals; paired bootstrap of every arm against EnergyNet v2.1, E_phys and the set's classical
baseline: MOVES-like and label x distance for VED, the regression for EPA, the dev median / dev rate for HD), the EnergyNet reproduction check, compute and hashes; `results/exam_<id>/scores.parquet` (per
unit x set x row x track) and `preds/`; `showcase.json` + `showcase.npz` (every row's prediction, band and components
on the 8 showcase trips, and each trip's route profile).

**Exam sets** (units; cluster for the bootstrap): VED_ICE / _HEV / _EV / _PHEV trips (cluster = car; y > EnergyNet's
floors), VED_*_cars (car totals), VED_arch_* (by architecture); EPA_{conv,hev,bev}_{FTP,HWY,US06,FTPcold,SC03}
(per test vehicle: FTP = 0.43 cold + 0.57 hot UDDS per mile; gal/mi or wall kWh/mi = DC / 0.88; hybrids are all
'hev', whatever system the input says); Mersin_diesel / _cng (per trip, mean of 2 route draws; CNG at 0.0471 MJ/g);
ICCT (6 tests); ZTBus_legs (78; cluster = mission), ZTBus_missions (8). VED conditions (inputs only): city < 40 km/h,
highway > 55 km/h, cold < 0 C, hot > 25 C, hilly >= 15 m climb per km, short < 2 km, long > 15 km.

## 9. Metrics (score.py)

Per unit: ape = |pred / y - 1|; sgn = pred / y - 1; in50 / in90 = y inside q25-q75 / q05-q95; crps_log = 2 x mean
pinball loss of log y against the 5 log quantiles (unit-free; |log error| for a point forecast); width90_log.
Summaries: MdAPE (EnergyNet's headline), MAPE, bias (median sgn), median |log error|, cov50, cov90, crps_log (mean),
width90_log. Comparisons: paired bootstrap over clusters of the MdAPE difference and the crps_log difference
(`default_rng(20261123)`; 2,000 resamples in evaluate.py and in the exam, to keep the exam's aggregation near 30 min);
a win or loss only when the 95 % interval excludes 0.
EnergyNet reports 68 / 95 % band coverage; the bake-off uses the 5 quantiles' 50 / 90 % bands for every row.

## 10. Showcase (for the page)

`showcase.json`, drawn once by `showcase.py` (seed 20261123) from trip attributes only (architecture, distance,
duration, mean speed, mean temperature, climb; never a label or result), one exam trip per stratum:
city VED_575_974 (conv, 28 km/h), highway VED_147_949 (conv, 86 km/h), hilly VED_374_1346 (hybrid, 15.5 m/km),
cold VED_230_1265 (conv, -13 C), hybrid VED_549_727, ev VED_10_1737 (Leaf, -24 C), plugin VED_492_2410 (Volt-type),
bus ZTBus_B208_2021-05-27_..._leg02 (14.8 km). The exam stores every row's prediction, band and components for them.

## 11. Machine rules

- 2 cores, ~6.3 GB RAM shared by three arm agents. One thread per job (`source env.sh`; `torch.set_num_threads(1)`),
  `nice -n 5`, `setsid nohup ... &`, resumable checkpoints, <= 1.2 GB per process. Check memory before big jobs:
  `/sys/fs/cgroup/memory/process_api/*/claude-code-bash/memory.usage_in_bytes`.
- `bake.load_bank('train')` holds ~30 MB of rows in memory; traces are memory-mapped (do not np.array() them all:
  train traces are 1.06 GB as float32). `sim_table` for all 28,800 trips takes < 1 s; a process with the bank and
  one table draw uses ~0.2 GB.
- Disk: each arm agent <= 0.6 GB. Do not copy the bank. Keep your own `NUMBA_CACHE_DIR` in your folder.
- Never `pkill -f`; kill only your own PIDs. Don't write in `harness/` (except `evaluate.py` into your arm's work/ and
  `register.py` into registry/). Read-only: /home/claude/mlr-repo, /home/claude/road, /home/claude/bb,
  /home/claude/priorwars, /home/claude/bakeoff/river. Don't write to the claude.ai project.
- Keep a `STATUS.md` in your group folder; the container can restart.
- RoadSim (vendored, `import roadsim` after `import config`) is available for physics arms: `physics.sim(spec, trip)`
  pins v2 (`vendor/energynet/training/physics.py`); bank specs are in `bake.fleet(B)`.

## 12. Checks done in Phase 1

| Check | Result | File |
|---|---|---|
| Vendored code = repo files | 140 files, sha256 identical | `vendor/VENDOR.json` |
| Local proc tables = repo tracked tables | all equal except d3/, hd/ztbus_full/ (absent) | section 2 |
| splits.json hash | d917df78... (checked on every real-data load) | `config.py` |
| EnergyNet v2.1 dev (21 published terms) | equal, max diff 5.6e-16; D16 0.1133 | `results/check_enet_dev.json` |
| Val bank = EnergyNet's val2[h] | MdAPE 0.0746, skill 0.6797, n 3,985, all per-arch / class terms, max diff 9e-10 | `results/check_bank_val.json` |
| Trace track: E_phys and R from the trace | E_phys log diff <= 2e-4 (sim, VED, EPA, HD) | `results/check_trace.json` |
| Reference arms on sim val + dev | see section 7 | `results/dev/`, `arms_ref/*/work/eval/` |
| Guard tests | 30 / 30 pass | `results/test_guard.json` |
| Exam pipeline rehearsal on dev (both tracks) | all rows, boards, bootstrap, showcase | `results/rehearsal_*.json` |
| Exam-mode pipeline, mock (dev records stand in for the exam: lock, guard, workers, aggregation, once-only) | passes; real results/ and real/exam untouched | `tests/test_exam_mock.sh`, `results/mock_exam_check.json` |
| Train bank dumb rates = EnergyNet's (skill denominator) | EnergyNet v2.1 val skill 0.6797 = published | `results/check_bank_val.json` |
