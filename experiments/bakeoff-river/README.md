# River Bake-off (experiment, 22-23 Nov 2026)

One physics problem, every learner family, equal terms, one real exam. Daily streamflow for ungauged CAMELS-US basins
(as FlowCast), plus a gauged nowcast track. Write-up: [`docs/experiments/bakeoff-river.md`](../../docs/experiments/bakeoff-river.md).

```
experiments/bakeoff-river/
  harness/          the shared harness: bank recipe (build_bank.py) + bank MANIFEST.json (sha256 of every array),
                    data API (bake.py), scorer (score.py), references (refs.py), registration (register.py),
                    the once-only exam (exam.py), showcase draw (showcase.json), HARNESS.md, registry/
                    (the 21 frozen arm hashes) and the vendored-code hashes (vendor/VENDOR.json)
  arms_seq/         LSTM (control), transformer, S4D state space, TCN: shared pipeline common/seqlib.py, train/
  arms_tab/         gradient-boosted trees, sparse GP, physics in the loop (HBV), formulas (LawFinder search)
  arms_filt/        ensemble Kalman filter on WaterSim (gauged) + the Mondrian conformal layer for every arm
  results/          exam_20261123T074907.json (boards, paired bootstrap, classes, compute, hashes),
                    exam_20261123T074907/scores.parquet (per basin x period x protocol x lead x row),
                    analysis.json + leaderboards.md (phase3/analyze.py), showcase_page.json (the 30 storm
                    windows the page uses; the full 43 MB showcase.npz is in the release zip, not here)
  phase3/           analyze.py, showcase_data.py, the page builder, STATUS.md (exam log)
  page/             river-bakeoff.html (the self-contained page)
```

**Not included:** the 806 MB simulated bank (rebuild with `harness/run_bank.sh`, ~7.5 min on one core; check against
`harness/bank/MANIFEST.json`), the vendored WaterSim/WaterData (byte copies of `shared/water/`, hashes in
`vendor/VENDOR.json`; copy them back to run), torch checkpoints beyond the exported numpy weights, the LightGBM venv,
and the dev prediction caches of the conformal layer (regenerate with `harness/evaluate.py <arm> --save-preds`).

**Rules kept.** Zero real labels in training; dev used for rung gates and conformal calibration only; each arm at most
about 4 single-thread CPU-hours and 3 rungs; one exam (`exam.py --run`, run id 20261123T074907) for all 21 registered
arms at once, which re-hashed every arm before and after scoring it. The exam chose nothing.

**Reproduce the scoring** (needs the real-data tree that WaterData reads): `python3 harness/exam.py --check`, then the
exam refuses to run again (`results/EXAM_DONE.json`). `python3 phase3/analyze.py` rebuilds `analysis.json` and
`leaderboards.md` from the exam outputs alone.

**In this repository (imported 24 Nov 2026).** Every file is as shipped (SHA256SUMS.json of the release zip checked on
import), except the paths: `harness/config.py` and `phase3/{analyze,showcase_data,build_page}.py` now resolve the repo
root, the water data root (`$ML_RATCHET_WATER`, default `shared/water`) and the code (`shared/water` itself: all 31
vendored files were byte-identical to it on import) instead of `/home/claude/...`. Results stay in `results/`; re-runs
write to `work/` (git-ignored). The arms' own training scripts still name their cloud paths: they are the record of what
ran. Check on import: `phase3/analyze.py` rebuilt `analysis.json` (0 differences) and `leaderboards.md` (identical)
from the shipped exam outputs. NOAA's NWM daily table (`data/proc/nwm/`) is not in this checkout's water data, so the
NWM reference cannot be re-scored here; the exam's stored NWM rows stand.
