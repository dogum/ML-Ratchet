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
