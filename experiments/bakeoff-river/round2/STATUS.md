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
