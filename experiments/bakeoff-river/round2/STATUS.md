# River Bake-off round 2: STATUS

Dated entries (local time, Windows 11, 16 cores; at most 6 processes). Plan and pass marks: `PREREG.md`.
Exam 2 has NOT been run and must not be run by an agent: the owner runs `exam2.py --run` once.

## Layout
- `r2.py` path layer (harness on sys.path, `config.BANK` -> `work/banks/<name>`); `bank2.py` bank builder (p1 via
  harness `build_bank.make_sample`, b2 options in `make_sample_b2`); `phys/<bank>.json` recipes.
- `train_seq.py` (path-only copy of `arms_seq/train/train.py`), `eval2.py` (harness `evaluate.py` on a round-2 bank).
- `work/` (git-ignored): banks, checkpoints, candidates, eval outputs, logs.

## If the session restarted
- Banks: `python bank2.py build <name>` is resumable per shard. Trainings resume from `work/ckpt/<run>.resume.pt`.

## Log
- 2026-11-24 22:45 worktree fast-forwarded to 5a4b6da (bake-off import); water data junctions (raw, work, logs, proc/*)
  point at the main checkout's `shared/water/data` (git status clean).
- 23:20 `bank2.py selftest` OK (b2 code with every option off = p1 code). p1 rebuilt in 2.5 min (6 processes):
  31 of 36 arrays byte-identical to `harness/bank/MANIFEST.json`; q and states (both splits) and train qo differ
  (last-bit float output of WaterSim on Windows; qo rounded to 0.001 mm/day). See PREREG section 1.
- 22:52 (correction: the entry above says 23:20; the machine clock said about 22:52). Check (2) of PREREG section 1
  passed: the v1.0 `seq_lstm` arm scores sim-val 0.4255 and dev 0.4408 on the rebuilt p1 (its stored numbers).
- 22:57 b2a built (2.5 min, 5 processes): runs with reservoirs 16.6 % -> 19.9 % (p1 had 3.3 % of runs with a storage
  attribute and no reservoir; b2a has 0). Mean simulated flow 1.556 -> 1.562 mm/day.
- 22:58 b2c built (mosaic). calib_prior.py (dev only): p1's 4 coefficients reproduced (max |diff| 0.0005). LOO robust
  RMSE of log m*: M0 constant 0.153, M1 snow 0.177, M2 FlowCast 0.159 -> M0 chosen as pre-registered:
  log m = -0.026 + N(0, 0.153) (median m 0.97). Note: the plain LOO RMSE ranks them the other way (M2 0.206, M0
  0.231): the attributes help on the few outlying basins and not on the typical one. The pre-registered robust
  criterion stands. b2b building (1 process).
- 22:59 control queue started (5 processes): p1, b2a, b2c x seeds 0-2. About 0.24 s/step, ~24 min per run.
