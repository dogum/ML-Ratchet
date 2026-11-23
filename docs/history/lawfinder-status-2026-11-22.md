# LawFinder — status (v1.0, 22 Nov 2026)

> Kept as written on 22 Nov 2026 in the cloud project, with names of people and build roles made neutral and the claude.ai page links replaced by each model's `lab/` in this repository. Paths and status refer to that project.

Symbolic regression on the collection's shared simulators: one short, readable law per platform, chosen on simulations
and a small real dev set, then graded once on that platform's real exam beside the textbook law (its constant fitted on
the same dev set), the collection's network and a dumb baseline. B batch, project 3. Fills the symbolic-regression slot
on the Model families page. Question it answers: what does each simulator believe, in one line, and is it true?

- Page (equation in big type, sliders for every input, simulated cloud + real exam points + law curve, clickable law
  ladder, textbook rival, exam table; laws evaluated live in JS): `models/lawfinder/lab/`
- Release: `lawfinder-v1.0.zip` (0.84 MB):
  - `lawfinder.py` (numpy only) + `lawfinder_laws.json` (trees, constants, error bars, textbook constants, exam rows)
  - `demo.py` on 5 real exam examples per law (seeded draw within terciles), about 0.2 s
  - MODEL_CARD, NOTEBOOK, model.yaml, ratchet.yaml (both pass the repo schemas; `mlratchet validate` passes in a scratch copy)
  - `training/` (every script as run, search fronts, ratchet logs, exam JSONs), `splits.json`, `lab/index.html`
  - `for_repo/models/lawfinder/` + `for_repo/tests/golden/lawfinder.txt` + MERGE_NOTES.md

## Honest headline

| Law (simulator) | Released law | Exam | Law | Textbook (dev-tuned) | Network | Dumb | Verdict |
|---|---|---|---|---|---|---|---|
| Runoff ratio (WaterSim) | Q/P = 1.84 (3.73 + δ)^(0.364 f_s − φ − 0.299) | 619 CAMELS basins, MAE | 0.129 | Fu 0.146 (ω 2.6: 0.141) | FlowCast **0.112** | 0.211 | loss |
| Fire growth (FireSim) | log₁₀ G = −0.559 / P | 25,122 growth fire-days, mean \|log₁₀\| | **0.693** | Rothermel × P 0.870 | FireCast 0.757 | clim 2.07 | win |
| | | all 38,599 fire-days | 1.153 | 1.061 | – | yesterday's growth **0.671** | loss |
| GB inertia (GridSim) | E = D (6.60 − 5.62 s) | 410 events, MdAPE | 33.0 % | 7 s × D **15.1 %** | GridPulse 16.4 % | 24.0 % | loss |
| GB nadir (GridSim) | Δf = (ΔP + 2)² / (52,940 E − 3.54·10⁶)^0.208 | 148 events, MAE | **0.060 Hz** | k ΔP²/E 0.173 | GridPulse (5 s of trace) 0.080 | 0.129 | win |
| Home energy (BuildingSim) | EUI = 186.5 / 1.926^A + HDD (2022 − Y) | 6,019 RECS homes, MdAPE | 39.3 % | 35.6 % | none | **33.4 %** | loss |

- Two wins (nadir; fire on growth days), four losses. The law beats its dev-tuned textbook rival for water, nadir and
  fire on growth days, and loses to it for inertia, homes and fire on all days. The dumb baseline is the best row only
  for homes; the inertia law is also worse than the dumb guess (33.0 % vs 24.0 %).
- **What the simulators believe and where it is false:**
  - FireSim's own law, G ≈ (0.32 P)^1.6 km²/day, over-predicts real next-day growth a median 36× (+1.56 decades).
    The released law exists only because the dev gate refused every law that kept growth rising with perimeter.
  - GridSim's raw GB fleets give ~4.5 s of inertia per MW of demand; NESO shows ~7. The law has the best ranking of
    all rows (ρ 0.975 vs 0.60) but sits 33 % low.
  - BuildingSim 1.0 homes used 1.4× the energy per m² of RECS dev homes; physics 1.4 fixed the median, not the spread.
  - WaterSim's belief beats Budyko-Fu on real basins (MAE 0.129 vs 0.141-0.154), and loses to FlowCast.
- **Coverage of the 68 % bars:** water 83 %, fire 69 %, inertia 91 %, nadir 100 %, homes 69 %.

## Recipe as built

- **Physics:** the shared simulators, unchanged (read-only copies in `vendor/`). Checks are the platforms' own.
  Engine check: lawsearch recovers a planted Budyko-Fu law (1 % noise) to the noise floor.
- **Simulated banks** (split rules respected: free forcing windows, guarded landscapes and weather, train-input-year
  operating points, the BuildingData prior split):
  - water w1/w2: 8,000 train + 3,000 frozen basins each (10-year runs, 4 runoff models, 2 snow routines, real ERA5-Land)
  - fire f1/f2: 600 + 240 fires each (≈2,000 + 750 fire-days; real dev LANDFIRE/HRRR or parametric)
  - inertia i1: 6,000 + 2,000 GridSim GB states; nadir g1/g2: 8,000 + 3,000 gpsim states (≈20-30 % pass detection)
  - homes b1/b2: 1,000 + 300 single-family homes each
- **Physics versions (rungs):** w2 = FlowCast's ERA5 water-yield multiplier; f2 = dev-calibrated ROS factors;
  g2 = GridPulse's dev-calibrated GB response (its release physics) vs g1 generic; b2 = BuildingSim 1.4.
- **Search:** own `lawsearch.py` (PySR-style, numpy/scipy). PySR rejected (Julia ~1 GB + compile on shared cores),
  gplearn rejected (no constant fitting). 4-7 min per physics version, 25k-90k evaluations.
- **Ratchet:** rungs = Pareto front by complexity, then the next physics version; keep if frozen-set R² beats the
  incumbent by > 0.01 and dev error ≤ 1.05×; after a physics change all rungs re-scored and all decisions re-taken.
- **Dev sets:** 52 CAMELS basins; 40 fires / 791 fire-days; 4 GB 2019 events; 1,504 RECS dev single-family homes.
- **Twin test (water, dev):** the law reads WaterSim's ungauged twins at MAE 0.091, real basins at 0.130; ungauged
  twins are 0.111 from reality, calibrated twins 0.059.
- **Parity:** release numpy == search engine exactly; page JS == numpy to 2e-15 on 2,360 cases.

## Rungs

| Law | Tried | Kept | Released | Frozen sets | Note |
|---|---|---|---|---|---|
| Water | 27 | 5 | w2-c13 (sim R² 0.609, dev 0.130) | val_w1, val_w2 | w1 released c18 on val_w1; re-taken on val_w2 |
| Fire | 28 | 2 | f2-c3 (0.408, dev 1.17) | val_f1, val_f2 | f1: every law failed the gate vs the constant (dev 1.39) |
| Inertia | 10 | 5 | i1-c7 (0.964, dev 37 %) | val_i1 | |
| Nadir | 16 | 3 | g1-c12 (0.259 on g1) | val_g1, val_g2 | all g2 rungs failed the gate (dev 2.5-4.5× worse) |
| Homes | 37 | 8 | b2-c13 (0.694, dev 39 %) | val_b1, val_b2 | |

Every decision with its reason: `ratchet.yaml`, `training/results/<law>_ratchet.json`. Each rung's exam score is
reported after the fact.

## Limits

- Searches are time-boxed on a shared core; another seed may give another front.
- Grid dev = 4 events (one a cascade); grid widening factors from 4 events (inertia ×6.7, so its bars are too wide).
- ERA5-Land caps the water law (mountain under-catch). No slope in the fire law (no exam landscapes). FireCast has
  growth-day rows only. Nadir labels are noisy REMIT/FUELINST matches.
- Inputs computed identically for sim and real, but from different products in places (gridMET fm100 vs Nelson m100h;
  RECS 2020 degree days vs shifted TMYx).
- RoadSim trip-energy law not built (fifth in the brief's order; out of time).

## Rule notes

- Physics versions carry dev calibrations built by earlier models, which then gate here: partly in-sample gates.
- Textbook constants fitted on dev (and reported fitted on the simulator).
- GridData's 14 Nov 2026 dev label correction (GB_20191203T200030 → 854.7 MW) applied in LawFinder's loader: the
  sandbox's GridData copy predates it.
- Each exam script ran once (logs/exam_*.started), after a `--dry-dev` run on dev units to test the code.
- Fire primary metric shown both ways (FireData's growth-day convention, and all fire-days); verdict "mixed".
- No writes to the cloud project; read-only folders untouched (vendored copies; GridPulse `_paths.py` patched in
  the copy only).

## Next rungs

1. FireSim persistence (stall/burnout) as physics f3, then re-run the fire ladder: the law ladder shows exactly where
   simulated growth departs from real.
2. GridSim GB inertia rescaled to NESO's s per MW (GridPulse's wrapper fix) as i2; the (1 − 0.85 s) shape should survive.
3. RoadSim trip-energy law vs road-load on the VED/EPA exam; water recession law −dQ/dt = a Q^b.

## Sandbox gotchas

- Load 5-7 on 2 cores for most of the run: banks took 3-4× their solo time; time-boxed searches did fewer evaluations.
- `python3 -I` drops the script directory from sys.path; `demo.py` adds its own folder.
- watersim/firesim write numba caches next to their source unless `NUMBA_CACHE_DIR` is set; GridPulse's `_paths.py`
  prepends the platform folder to sys.path (patched in the vendored copy).
- A tie in dev residuals (all zero-growth fire-days) makes the 68th-percentile widening factor jump; it still gives 68 %.
