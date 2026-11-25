# Phase 3 (coordinator): exam, page, release: STATUS

## Exam
- 2026-11-23 07:49 (container clock): `exam.py --check` PASSED (21 arms, est. 302 CPU-min + refs ~10 min).
- 07:49:07 `exam.py --run` launched ONCE: run id **20261123T074907**
  (`setsid nohup nice -n 5 python3 exam.py --run`, log `phase3/logs/exam_run.log`).
- exam.py has no multi-worker option (its driver runs one worker subprocess per arm, sequentially), so it runs as
  designed with one process. Not hacked.

## If the container restarted
- Check `ps -eo pid,args | grep exam.py`. If no exam process and `harness/results/EXAM_RUNNING.json` exists and
  `EXAM_DONE.json` does not: `cd harness; source env.sh; setsid nohup nice -n 5 python3 exam.py --resume >>
  ../phase3/logs/exam_run.log 2>&1 < /dev/null &` and log it below. NEVER `--run` again.

## Prepared while the exam runs (no exam data read)
- `phase3/analyze.py` (boards, per-class, wins map, compute; tested on the dev rehearsal), `phase3/showcase_data.py`
  (cuts showcase.npz to the 30 storm windows + forcing P for display; run only after EXAM_DONE),
  `phase3/page/template.html` + `build_page.py`, `phase3/repo_patch/{a,b}` (copies of 3 repo site files; b = patched).

## Log
- 07:49 exam started. 07:53 refs done (259 s), climatology (54 s). 08:13 enkf_watersim done (1268 s; registry est. 53 min).
- 10:44:40 (EXAM_DONE.json) exam finished: all 21 workers OK in one pass, no failures, no `--resume` needed; every arm
  re-hashed unchanged after scoring; 0 error rows. Results: harness/results/exam_20261123T074907.json (+ dir).
  Worker times: refs 259 s, climatology 54, enkf 1268 (+conformal 1269), flowcast 484 (+476), lstm 109 (+109),
  ssm 367 (+377), tcn 308 (+297), transformer 717 (+706), dpl 67 (+64), formula 64 (+64), gbm 1538 (+1547), gp 198 (+191).
- 10:56 analyze.py -> results/analysis.json, leaderboards.md; showcase_data.py -> results/showcase_page.json (read
  ERA5-Land P for the 30 storm windows of the 10 showcase basins, display only); results/extra_comparisons.txt.
- Page built (page/build_page.py), checked headless at 1280 and 390 px, light and dark (no overflow, no console
  errors), published: https://claude.ai/artifact/NrerY4M22XDVxfzH9VFFnW
- for_repo: docs + experiment folder + model-families.patch (git apply --check OK on copies; TS loads in node).

## DONE
Exam run once; analysis, page, release zip and river-bakeoff-status.md written. Nothing is running.
