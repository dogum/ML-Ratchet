# WaterData (agent B) STATUS  (resume from "Next")
Scripts: water/data/scripts (resumable). Logs: water/data/logs. Temp: water/data/tmp. Package: water/waterdata.
Docs: water/waterdata/DATA.md (sources, units, splits, baseline tables, gotchas).

## READY for agent A (since 18 Nov ~08:10)
- Splits READY: water/data/proc/splits.json (seed 20261118). CAMELS-US dev 52 / exam 619 (exam_531 487).
- Dev-basin forcing + streamflow + attributes READY (all 671 CAMELS-US): wd.forcing / wd.streamflow / wd.spec.
- NOTE (changed 18 Nov ~09:00): spec['area_km2'] is now the area CAMELS used for mm/day (= NWIS contributing area
  where published; derive_area.py). Only 36 basins changed by > 2%; forcing and Caravan Q_mm unchanged.

## Done
- Caravan CAMELS-US (671), CAMELS-CL (505), CAMELS-BR (870) timeseries + attributes for all regions; specs
- NWIS daily 1980-2026 (671) + annual peaks (48,769 at 669 gauges); implied-area fix
- NWM v3.0 retrospective daily (local + UTC day) at 650 gauges 1980-01..2023-01; hourly kept 2015-10..2023-01
- SNOTEL 880 stations (of 934) -> proc/snotel; split dev 39 / exam 772
- ResOpsUS v2 678 dams + GRanD attrs; CDEC 18 CA reservoirs 1990-2025; split dev 15 / exam 250 / aux 413
- Baselines: GR4J (671 basins, checked vs airGR 1e-7 mm), flow (clim, persistence, GR4J, NN U-dev/U-k12, NWM),
  detective signatures, SNOTEL (clim, degree-day), reservoirs (historical, SOP, rule, DP; + CDEC recent);
  proc/baselines/summary.json
- DATA.md written

## Phase 1 COMPLETE (18 Nov 2026) except CAMELS-AUS timeseries
- Flood-peak baselines added (proc/baselines/peak_summary.json). summary.json / checks.json final.
- Disk: water/data 2.9 GB (proc/caravan 1.5 GB, raw/resopsus zip 0.3 GB, raw/nwm hourly 2015-10..2023-01 0.2 GB).
  NWM hourly for other months was aggregated to daily and deleted (fetch_nwm.py would re-fetch them if re-run).

## Still running (no action needed unless it fails)
- fetch_caravan.py second pass (the first crashed on CAMELS-AUS netCDF4 files at byte 23.84 GB; converter fixed).
  Log logs/fetch_caravan2.log; ~2-2.5 h from 14:50 UTC. Watcher scripts/watch_aus.sh then runs
  `make_splits.py other` (pre-registered rule, seeded per region; adds camelsaus only) and appends a line below.
- If the container restarted: re-run `python3 -I scripts/fetch_caravan.py` (re-streams from byte 0, skips existing
  files) and then `python3 -I scripts/make_splits.py other`.
- 2026-11-18T16:54Z: CAMELS-AUS timeseries complete (561); split frozen by make_splits.py other (see logs/make_splits_aus.log)
