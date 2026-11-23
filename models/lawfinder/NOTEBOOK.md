# LawFinder notebook

## 2026-11-22 (B batch, v1.0)

- Folder `/home/claude/bb/lawfinder` on a shared 2-core machine with three other B-batch builds (load 5-7 most of the
  day). One thread, `nice`, every job resumable (`STATUS.md` kept current).
- **Engine.** PySR would need Julia (~1 GB with packages) and a long compile on the shared cores; gplearn has no
  constant fitting. Wrote `lawsearch.py` (numpy/scipy): island populations, regularised evolution, Levenberg-Marquardt
  constants, hall of fame per complexity. Test: planted Budyko-Fu with 1 % noise reached the noise floor in 90 s.
- **Water** (WaterSim): banks w1 (as shipped) and w2 (+ FlowCast's dev-fitted ERA5 water-yield multiplier),
  8,000 + 3,000 basins each, ~0.1-0.2 s per basin-decade. The forcing-error model on the inputs makes the simulated
  runoff ratio noisy (Fu reaches only R² 0.46 on the bank). w1 released c18 on val_w1; after w2 every rung was
  re-scored on val_w2 and w2-c13 released. Twin test: law vs ungauged twins 0.091 MAE, vs real 0.130.
- **Fire** (FireSim): 600 + 240 fires per version, 256² cells at 90 m, 3-6 days; ~2-3 s per fire. Simulated fires
  grow on 98 % of days (real dev 68 %). Every law where growth keeps rising with perimeter failed the dev gate against
  the constant; f2-c3 (saturating) passed. Dev residuals have a spike at the zero-growth days (32 %), so the widening
  factor is large (×3.75).
- **Grid**: GridSim's raw GB `generate_states` gives inertia ≈ 4.5 s × demand, far below NESO (~7 s): the inertia
  law has the right shape and the wrong level. Nadir via GridPulse's `gpsim` (copied unchanged; `_paths.py` patched to
  import the vendored GridSim). g2 (GridPulse's release physics, Dynamic Containment in 60 % of states) makes every g1
  rung's R² negative on val_g2; g2 rungs failed the dev gate (2019 events, before DC). Released g1-c12.
- **Homes** (BuildingSim): 1,000 + 300 single-family homes per version, ~1.7 s each under load. b1 homes 1.4× too
  energy-intensive vs RECS dev; b2 (physics 1.4) fixed the median; b2-c13 released.
- **Exams** run once each (logs/exam_*.started), after a `--dry-dev` run of the same script on dev units.
- Not done: RoadSim trip-energy law; water recession law.

## 2026-11-22 — imported into ML Ratchet

- Copied from `for_repo/models/lawfinder/` of `lawfinder-v1.0.zip` (22 Nov 2026) with `tests/golden/lawfinder.txt`. Every `.npz` loads with
  `allow_pickle=False` and holds no object arrays; inference imports only numpy and scipy.
- `demo.py` run from this folder reproduces the shipped golden output line for line (timing aside), so
  `tests/golden/lawfinder.txt` is the zip's own file, unchanged.
- No training script was re-run, and nothing was fetched.
- `order: 33`; `accent` changed from `#1d6fb8` to `#6741d9` (the zip's value was a placeholder, almost AirfoilNet's
  `#1d4fb8`); `exam.short` changed from "Five platforms' real exams" to "Four platforms' real exams, five laws" (the
  RoadSim law was not built).
- Two findings recorded as platform issues, not fixed: FireSim's implied next-day growth is a median 36× the real one
  (`shared/fire/STATUS.md`, backlog `firesim-growth`), and GridSim's GB fleets commit about 4.5 s of inertia per MW
  against NESO's ~7 s, about 35 % low (`shared/grid/STATUS.md`, backlog `gridsim-gb-inertia`). The BuildingSim 1.0
  finding (homes 1.4× RECS dev energy per m²) was already fixed in physics 1.4.
- Backlog: `lawfinder` replaced by `lawfinder-road` (the unbuilt fifth law). Model families: `symbolic` is now a trait
  in use.
- Training scripts ported to `training/_paths.py` (the shipped ones wrote next to the model folder and read sandbox
  paths). `ROOT` is now the work folder (`models/lawfinder/work/` or `$ML_RATCHET_WORK/lawfinder`), seeded on first
  use with the shipped `training/results/` and page inputs, so a re-run never overwrites a shipped record and the
  exam scripts still refuse to run twice. Platform code comes from `shared/`; data roots from `$ML_RATCHET_WATER`,
  `_FIRE`, `_GRID`, `_BUILDING`, `_VEHICLE`. The network rows' inputs come from FlowCast's, FireCast's and GridPulse's
  work folders. GridPulse's `gpsim` is imported through `_paths.import_gpsim()` (both modules are named `_paths`).
  The shell wrappers `cd` to `training/` and source a portable `env_sandbox.sh`. Only `fit_law.py`-level imports were
  smoke-tested; no shipped bank or search was re-run.

## 2026-11-22 — the fifth law: RoadSim trip energy per km (built in the repository)

Asked for by the owner after the import ("Follow the existing pipelines exactly"). Details and the exam table are in
`MODEL_CARD.md` ("The road law"); the steps, as run:

- **Data.** Nothing downloaded. RoadData's processed VED trips, EPA tables and cycles were read from an existing data root
  (`ML_RATCHET_VEHICLE` pointed at another checkout's git-ignored `shared/vehicle/data`). EnergyNet's exam frames
  (`exam2_ved_v2.parquet`, `exam2_epa_v2.parquet`) and its corrected EPA labels (`epa_fixed.parquet`) came from that
  checkout's `models/energynet/work/` (`ML_RATCHET_PEER_REPO`, a new switch in `training/_paths.py`). RoadData's splits
  were not changed.
- **Bank** (`road_bank.py`, `run_road_banks.sh`): 2 physics versions × (6,000 train + 2,000 frozen) trips, 6 workers,
  about 4 minutes. 0 trips failed. Simulated energy per km sits above the dev cars' (petrol 3.4-3.5 against 2.6-2.7
  MJ/km; electric 0.6-0.75 against 0.44-0.47): the generator's cars are heavier (median 2.2 t, electric 2.65 t, against 1.3-1.5 t for the VED dev cars
  and 1.9 t for the EPA dev tests).
- **Search + ratchet** (`fit_law.py road`, 600 s per version): released p1-c23, R² 0.767 on val_p2, dev MdAPE 16.8 %.
  No p2 rung cleared the 0.01 margin. First run on a dev set missing the 35 trips of mass-less VED cars released
  p1-c18; the missing masses were set to class defaults (EnergyNet's rule) and the ratchet re-scored before the exam.
- **Dry run** on dev units (`exam_road.py --dry-dev`), then **the exam, once** (logs/exam_road.started). The law loses
  every stratum: VED trips 18.6 % (MOVES-like 11.8 %, road-load 18.3 %), EPA tests 17.3 % (EnergyNet 7.5 %, road-load
  23.8 %). Electric cars are its worst case (±42 % bias).
- **What failed.** Exam run 1 mis-scaled EnergyNet's VED rows (its frame is already in MJ; the script multiplied by the
  unit factor again). A corrected re-score was not run: the exam runs once, and the re-run was stopped for the owner to
  decide. `exam_road_errata.py` reports EnergyNet's published values on the same trips instead and marks its pooled VED
  row not scored. The law's own numbers do not depend on the bug.
- **Release.** `add_road_release.py`: the road entry merged into `lawfinder_laws.json`, `examples/road.csv` (fixed
  draw by terciles of v̄, seed 20261122), the road block added to the lab page (JS == numpy to 6e-16 on 1,500 points,
  `training/results/js_parity_road.json`). The other four laws were not rebuilt (their banks and point files are not
  shipped); their entries are unchanged. `lawfinder.py` gained the `road_load` textbook and the missing
  `degree_day_signature` form (`lf['bldg'].textbook()` raised KeyError before). The ratchet ladder was appended to
  `ratchet.yaml` by `build_card.py`'s own ladder code; `model.yaml` was edited by hand (as the shipped one had been).
- Windows note: the scripts need Python's UTF-8 mode (`PYTHONUTF8=1`) for the JSON and HTML they read.
- Backlog: `lawfinder-road` removed (built).

## 2026-11-22 — road exam: EnergyNet's VED rows re-computed (owner's OK)

- **What was wrong.** `exam_road.py` (exam run 1) multiplied EnergyNet's VED frame (`exam2_ved_v2.parquet`) by its
  `conv` column. That frame already stores `y` and `pred` in MJ (EnergyNet's `realfeat.py` multiplies before saving), so
  EnergyNet's VED predictions came out ×0.042 for petrol and hybrids and ×3.6 for electric cars. Only those comparison
  rows were wrong; the law's predictions, every rung and every other row never touched the factor.
- **What was re-computed.** EnergyNet's VED rows only, in `exam_road_errata.py`: its stored per-trip predictions / trip
  km (MJ/km), against run 1's stored truth for the same 22,877 trips (`exam_road_points.parquet`). EnergyNet's own
  truth agrees with it to 6e-8 (max relative difference); run 1's EnergyNet column is exactly 0.042× the corrected one
  at the median. The law's exam was not re-run, and `exam_road.json` is left as run 1 wrote it.
- **Before → after (MdAPE, bias):** petrol 95.8 % (−95.8 %) → 8.6 % (+0.3 %); hybrids 95.6 % (−95.6 %) → 13.4 %
  (+3.8 %); electric 187.0 % (+187.0 %) → 23.2 % (−20.3 %); all VED trips 95.8 % → 10.1 % (+1.1 %). The three
  per-powertrain values equal EnergyNet's published exam values. The interim report (published values, pooled row "not
  scored") is replaced.
- **What changed downstream.** The pooled VED headline now shows EnergyNet at 10.1 %; the page verdict reads "loses
  to EnergyNet" (was "loses to MOVES-like", 11.8 %). Every road verdict is still a loss. `exam_road_reported.json`,
  `lawfinder_laws.json`, the lab page, `model.yaml` (rows, limits, help_wanted, next) and `MODEL_CARD.md` updated.
