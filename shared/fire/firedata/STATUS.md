# FireData (agent B) STATUS  (resume from "Next")
Scripts: shared/fire/data/scripts (resumable; run with `python3 -I`). Logs + pid files: shared/fire/data/logs. Temp: shared/fire/data/tmp.
Package: shared/fire/firedata (`sys.path.insert(0, 'shared/fire'); import firedata as fd`). Full docs: **shared/fire/data/DATA.md**.

## READY for agent A
- **splits.json — READY, FROZEN** (18 Nov 23:44Z, seed 20261118, sha256 d73e2d9f…). `shared/fire/data/proc/splits.json`,
  per-fire table `proc/fires.parquet` (`fd.fires('dev')`). Unit = MTBS fire. 40 dev / 4197 exam / 2859 aux_rx (Rx, never
  scored) / 643 nonCONUS excluded. Exam periods: early 2012-16 (1647), main 2017-21 (1859), recent 2022-24 (691).
  GOFER: 4 dev (Antelope 2021, CZU 2020, Creek 2020, W-5 Cold Springs 2020), 24 exam. WSTS: 4 dev, 568 exam, 35 mixed (excluded).
  FireSize (FPA-FOD 2012-20): 8913 dev, 655869 exam (`proc/firesize_split.parquet`). Moisture: `proc/moisture_split.json`.
- **Guard — READY.** `fd.guard.check_landscape(bbox)`, `fd.guard.check_weather(bbox, t0, t1)` raise ExamLeakError;
  `fd.guard.exam_mask(grid)` gives cells to blank. Loaders refuse exam fires unless `for_exam_scoring=True`.
  NOTE: many dev windows touch older/nearby exam footprints (reburns). Dev landscapes are free for calibration/replay;
  for TRAINING banks call check_landscape or blank exam_mask cells.
- **Dev-fire packages — READY (40/40)**: `fd.progression(id)` (PM-to-PM daily burned masks, 90 m EPSG:5070 window grid),
  `fd.perimeters(id)` (12-hourly FEDS, km2), `fd.firepix(id)` (VIIRS 375 m), `fd.landscape(id)` (LANDFIRE 90 m contract
  stack, proc/landscapes/<id>.npz), `fd.weather(id)` (hourly HRRR 'series' + 3 km 'grid', proc/weather/hrrr/<id>.npz;
  'daily' gridMET at centroid). `fd.gofer('Creek|2020')` hourly GOES perimeters.
  Gaps: 1-5.5 % hours NaN on 2017-18 fires (real archive gaps; repair pass confirmed); hourly precip NaN before
  mid-2018 (no hrrrzarr fcst; 14/40 dev fires) -> use gridMET daily pr. hrrrzarr is float32 from 2024 (handled).
- **Exam metric — READY**: `fd.metrics.nextday(now, nxt, prob, cell_km2, window=fd.metrics.score_window(now))`.
  Never score the whole window (it is built from the FINAL perimeter).
- **Baselines — READY**: proc/baselines/spread_summary.json (exam 3030 fires / 25105 growth days; best classical
  fl_prev AP 0.252, F1 0.184), firesize_exam.json, moisture_exam.json, fire_days_{dev,exam}.parquet, spread_params.json.
- **Exam replay subset (48 fires 2017-24, frozen, proc/baselines/replay_subset.json)**: landscapes + HRRR + precip DONE
  (for final exam scoring only).

## Running (19 Nov ~03:00Z)
- fetch_wsts.py full proc/wsts/order.txt (pid logs/wsts_full.pid; ~2 fires/min, 607 fires; dev done first). Resumable.

## Known issues / help_wanted
- NFMD dead fuel sticks: wfas.net/nfmd 404, moved into FEMS; not obtained. RAWS: Synoptic key needed (skipped).
- FIRMS not used (FEDS carries S-NPP pixels). No HRRR before 2016-08-25 (exam 'early' fires have only gridMET).

## Next (if resumed)
- Phase 1 done. Only the WSTS stream is still running (resumable: rerun the same command after a restart).
  Splits, replay subset, baselines are frozen: do not rerun make_splits / make_replay_subset / baselines on exam.
