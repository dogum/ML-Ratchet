# River Bake-off harness: STATUS

**READY** (2026-11-22 23:25 UTC). Phase 2 arm agents can start. Read `HARNESS.md`.

The exam has NOT been run. Do not run `exam.py --run` before every arm is registered (Phase 3, coordinator).

## Done
- Vendored the repo's latest `shared/water/watersim` + `waterdata` into `vendor/` (31 files, byte-identical,
  `vendor/VENDOR.json`). The repo copy differs from `/home/claude/water/watersim` in 6 files; the bank uses the repo's.
- `config.py`, `features.py` (FlowCast's input helpers), `build_bank.py` (FlowCast's simbank recipe, physics p1,
  seeds val 7 / train 1, plus labels).
- Bank built: train 12,000 + val 800 runs x 1462 days, 806 MB, 7.5 min (train 398 s, val 50 s, one core).
  `bank/MANIFEST.json` has sha256 of every array.
  - val reproduces FlowCast's frozen p1 set (FlowCast's per-run NSE median |diff| 1.3e-5; median 0.426 vs 0.425).
  - `bank_rerun` reproduces q, qo exactly from the stored spec (val runs 0, 5, 333).
- `bake.py` (data API; exam guarded), `score.py`, `refs.py`, `evaluate.py`, `armlib.py`, `register.py`, `exam.py`
  (+ `--check`, `--rehearsal`), `showcase.py` -> `showcase.json` (10 basins, 30 storms; seed + class + forcing only).
- Reference rows reproduce WaterData's flow_scores on the dev basins exactly (`results/check_refs.json`).
- `results/dev_refs.parquet`: dev references (U-dev 0.466, U-k12 0.457, GR4J 0.586, NWM 0.573, clim 0.092,
  persistence 0.549 / -0.139 / -0.515).
- Test arms in `arms_ref/` (registered; both are also exam reference rows):
  - `flowcast_v1` (FlowCast v1.0 as released, ungauged + nowcast): equals `flowcast.predict()` to 1.3e-6 mm/day.
    Sim val NSE 0.426; dev 0.425 (KGE 0.424); nowcast dev 0.633 / 0.567 / 0.520 = FlowCast's published numbers.
  - `climatology` (constant Budyko-Fu flow, fitted on the sim train bank): sim val -0.049, dev -0.033.
  - Their dev tables are in `results/dev/` (used by evaluate.py for paired comparisons with FlowCast).
- `tests/test_guard.py`: 27/27 pass (data guard, once-only, in-progress lock, unregistered/changed/added-file arms,
  file audit, gauged causality). `exam.py --check` passes. `exam.py --rehearsal` ran the full pipeline on dev
  (`results/rehearsal_20261122T232527.json`).

## If the container restarted
- Nothing is running. All outputs are on disk. Bank: if `bank/MANIFEST.json` is missing, re-run `./run_bank.sh`
  (resumable per shard).

## Open items / notes for Phase 2-3
- Before the exam: every arm registered; `python3 exam.py --check` must pass; then `python3 exam.py --run`
  (resumable with `--resume`). Estimated: references ~6 min + FlowCast ~5 min + each arm's estimate in registry/.
- If harness .py files change after an arm registered, `--check` only notes it; re-register to record the new hash.
