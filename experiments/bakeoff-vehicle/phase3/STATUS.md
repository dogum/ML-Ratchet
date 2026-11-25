# Vehicle Bake-off Phase 3 (exam, page, release): STATUS

## Exclusion decision (coordinator, BEFORE the exam, compute only)
- `tab_tabpfn` and `tab_tabpfn_conformal` excluded via `harness/registry/excluded.json` (reason recorded verbatim:
  "compute: ~937 single-thread CPU-min on a shared 2-core box; TabPFN is a yardstick (lens, not a release); its sim/dev
  numbers stand; a dated supplementary exam can be run later on a larger machine."). Their registry records are kept unchanged.
- Nothing else excluded.
- Harness fix needed for the documented mechanism: `register.registered()` read every `registry/*.json` as an arm record,
  so `excluded.json` would have crashed it (KeyError 'arm_id') and could not drop an already-registered arm. Minimal fix:
  skip `excluded.json`, and leave out records whose `arm_dir` is listed in it (`phase3/logs/register_exclusion_fix.diff`;
  original in `phase3/logs/register.py.orig`). Harness hash changes, so `--check` notes "harness code changed since this
  arm registered (recorded)" for each arm; arm hashes are unaffected (no arm hashes harness files).
- Note for the record: agent A measured tab_tabpfn's real predict cost at ~10.6 ms/row (~20-30 min for the exam, and the
  same for its wrapper); the 480 / 457 min registry estimates divide a one-off ~23 s context fit over a 24-row smoke batch.
  The exclusion stands as decided; the supplementary exam is therefore cheap.

## Exam
- 20:42 `exam.py --check` after exclusion: PASSED, 21 arms, est. predict CPU 187 min (`logs/check_after_exclusion.log`).
- exam.py has no multi-worker mode (driver runs one worker per arm, sequentially): run as designed, one process.

## If the container restarted
- `ps -eo pid,args | grep exam.py`. If none and `harness/results/EXAM_RUNNING.json` exists and `EXAM_DONE.json` does not:
  `cd /home/claude/bakeoff/vehicle/harness; source env.sh; cd /tmp; setsid nohup nice -n 5 python3
  /home/claude/bakeoff/vehicle/harness/exam.py --resume >> /home/claude/bakeoff/vehicle/phase3/logs/exam_run.log 2>&1 < /dev/null &`
  and log it below. NEVER `--run` again.

## Log
- 20:42:51 `exam.py --run` launched ONCE: run id **20261123T204251** (PID 1025; `setsid nohup nice -n 5`, cwd /tmp,
  own NUMBA_CACHE_DIR, log `phase3/logs/exam_run.log`). 21 arms in the lock file.
- 20:53 exam records built in 631 s (VED 25,793 / EPA 76,037 / HD 2,732 records). refs 5 s, climatology 80 s,
  enet_v21 62 s, enet_v21_conformal 64 s; formula running. No errors so far.
- Prepared while the exam runs (no exam data read): `phase3/analyze.py` (tested on the Phase 1 dev rehearsal ->
  `results/rehearsal_analysis.json`), `page/template.html`, `repo_patch/{a,b}` (copies of 3 repo site files).
- 21:09 all table arms done (0 errors); trace arms running (~25 min each, 8 workers; ETA ~00:20).
- River's model-families patch no longer applies to the repo as pulled (hunk `symbolic`: the repo now lists LawFinder
  under `used`). `repo_patch/ar` = repo + River patch with that hunk rebased (`repo_patch/river-rebased.patch`,
  `git apply --check` OK on a repo copy); the vehicle patch is made against both `a` (repo) and `ar`.
- 23:57:52 exam process exited; `EXAM_DONE.json` written: run **20261123T204251**, all 21 workers OK in ONE pass, no
  failure, no `--resume`; every arm re-hashed unchanged after scoring; 0 error rows. Worker wall times: refs 5 s,
  climatology 80, enet_v21 62 (+conformal 64), formula 93, gbm ~140 (+145), gp 8 (+9), mlp 56 (+56), trace_cnn 1460
  (+1463), gru 598 (+612), s4d 1031 (+1017), transformer 757 (+766). Aggregation (bootstrap) 23:18-23:57; its RSS
  peaked at ~2.5 GB (above the 1.2 GB guideline; harness code, single process, nothing else running; cgroup <= 5.0 GB
  incl. page cache).
- EnergyNet v2.1 reproduces its published exam (training/exam3.json) on all 19 in-scope rows: same n, |diff MdAPE| 0.
  Classical references reproduce their published exam baselines too.
- 00:00-00:15 analyze.py: the first version (one bootstrap per comparison) timed out at 10 min; rewritten so that each
  row's bootstrap distribution is computed once per set / column on shared cluster resamples (pairs = differences).
  Two runs of my own analysis process were stopped by PID for using 1.6-2.2 GB (> 1.2 GB rule); fixed (Arrow load with
  categorical strings, lazily built cluster indices): ~0.72 GB.
- 00:24 analyze.py done -> results/analysis.json, leaderboards.md (~8 min, 0.72 GB). extra.py -> results/extra_comparisons.json.
- 00:26 LawFinder conversion: arms_phys/lawfinder_road/exam_road.py run once, unchanged, with `--results` pointing at a
  filtered copy of scores.parquet (table track, its 9 sets and 9 rows; the full file needs ~2 GB in pandas) and `--out`
  phase3/results/lawfinder/exam_road.json (+ exam_road_points.parquet). Road law VED ICE exam MdAPE 14.0 %
  (road-load 33.0, MOVES-like 10.3, label 13.0, EnergyNet 8.6).
- 00:35-00:50 page built (`page/build_page.py` -> `page/vehicle-bakeoff.html`, 0.37 MB, self-contained), checked
  headless at 1280 and 390 px, light and dark (no horizontal overflow, no console errors; `page/shots/`), published:
  https://claude.ai/artifact/CimHwrUrCZ2w5jBsLJGTTk
- Model families patches: `repo_patch/make_patch.py` -> `vehicle.patch` (vs repo 2dd401e), `vehicle.after-river.patch`
  (vs repo + `river-rebased.patch`); both `git apply --check` + `git apply` OK on copies; TS loads in node (10 vbake).
- Docs: `/home/claude/bakeoff/vehicle/vehicle-bakeoff-status.md`, `release/bakeoff-vehicle.md`,
  `release/README_experiment.md`, `release/MERGE_NOTES.md`, `release/README_zip.md`.
