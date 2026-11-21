# FireSim (agent A) status — resume notes

Started 18 Nov 2026 ~17:50 UTC. Package `shared/fire/firesim/firesim`, checks `shared/fire/checks`.
Rules: one thread, nice long jobs, own PIDs in firesim/pids/, temp in firesim/tmp. Never write to shared/fire/data or shared/fire/firedata.
Refs (PDF + pdftotext) in firesim/refdata: Andrews 2018 GTR-371, Scott & Burgan 2005 GTR-153, Rothermel 1983 GTR-143.

## Plan
1. fuel models table, Rothermel/Albini surface (numba), WAF, slope, wind-slope vector, ellipse, crown, spotting
2. moisture: Nelson 2000 dead fuel, live/curing; FWI; diurnal weather helpers
3. spread: MTT (Dijkstra large stencil, time-varying), level-set, CA; spotting, barriers, suppression
4. observe(): VIIRS/MODIS/GOES emulator
5. generators: landscape (parametric + B's real tiles), weather
6. checks: published, pyretechnics (venv firesim/.refenv), analytic, crown, Nelson, speed; dev replays on B's data
7. README, demo.py

## Done (18 Nov, evening)
- package: fuels (53 models from GTR-371 T8; FM7/FM9 Anderson originals; GR6 heat 9000 per S&B T7), rothermel (numba
  bundle + BehavePlus combine), crown (VW77 I0, R91, Cruz05, VW93/S&R CFB), spotting (Albini surface, Schroeder Pig),
  moisture (Nelson-type stick model, Simard EMC, GSI live), weather (contract, daily->hourly, generator), fwi,
  landscape, kernel (MTT Dijkstra, time-dependent, 8/16/32 stencils + two-point simplex updates, crown, spotting,
  containment lines, initial attack), simulate, behave, observe (VIIRS/MODIS/GOES), generate (parametric), metrics
- pyretechnics built in firesim/.refenv (Cython 3.0.12, .c regenerated). checks/ref/pyre_run.py -> out/pyre_values.json
- checks: published_examples.json 436/436; pyretechnics_crosscheck.json (same table: 5e-5 median, flat/slope-only)
- MEMORY: Prior Wars filled the bash cgroup (6.27 GB limit) for ~1 h; messaged main. Check
  /sys/fs/cgroup/memory/process_api/*/claude-code-bash/memory.usage_in_bytes before big runs.
- L/W default = behaveplus (1+0.25UE); Anderson 1983 optional (U mph). Wind limit default none (Andrews 2013).

## Resumed 18 Nov 23:40 UTC after container restart (ls shows local time UTC-5; json 'generated' is UTC)
- found done: analytic.json (mtt16+simplex area ratio 0.98 mean / 0.94 min, L/W err <=5.5%, orientation range 3.4%;
  level_set(32) 0.99/0.97; strip + fuel break + refraction ~1e-3), crown.json, moisture.json (synthetic), fwi.json
- FIXED generator bug: canopy was zeroed everywhere (nb mask `fuel>=91` caught all SB40 codes 101-204); now 91-99.
- FIXED ledger growth check: now line integral of normal ROS on the mid-time isochrone (was area-weighted, ratio 2.4 -> 1.1)
- ADDED simulate_batch() and simulate(prepared=...) (prepare once, kernel per ignition)

- kernel speed-up (exact pruning of edges already slower than the best arrival + 8-slot float32 look-ahead state
  cache): worst case 90 m 2.9 -> 1.7 s. Re-ran analytic + crown (identical to 1e-8) and speed.
- ledger growth_check counts only active fronts (ratio ~1.05-1.15 uniform, 1.2-1.4 heterogeneous; documented)
- demo.py (demo_out/demo.png, ~2 s), README.md drafted (dev replay / VIIRS sections pending)
- B: splits.json frozen (40 dev fires, MTBS Event_IDs); firedata API has perimeters/firepix/progression/landscape/
  weather; landscapes + HRRR packages not yet READY. FEDS firepix t is a half-day label (no overpass time).

- realdata.py (B adapters: landscape, HRRR series/zones with grid-north wind rotation, random_landscape,
  random_weather guarded), generate_landscape/generate_weather kind='real'; replay.py (replay_day, ensemble, calibrated
  params); kernel heap grows dynamically, look-ahead slots 4 on >400k-cell grids; numba cache hash = njit modules only
- satellite.observe: overpass tuples may carry explicit pixel size; cell_frp post_scale

- observe_viirs.json DONE (40 dev fires, 1570 overpasses): uncal median ratio -0.14 log10, Spearman 0.79; calibrated
  (energy x2, night FRP50 0.25) +0.01, 0.80, 62% within x2. Adopted as satellite defaults (ENERGY_MULT, viirs night).
- moisture_real.json (vs gridMET fm100/fm1000; NFMD sticks unobtainable): 100-h 3.1 pts drier, r 0.80 (8 fires so
  far: rerun when B's gridMET points are complete)
- realdata.fill_precip_from_daily (gridMET pr/24 where HRRR precip missing: 14 dev fires 2017-18)
- replay sweep restarted 19:52 local (growth days only; precip fill); first run failed on a str column (fixed)

## Phase 1 DONE (18 Nov 23:20 local / 19 Nov ~04:20 UTC)
- replays: sweep 40 fires / 470 growth days; calibration fitted on 39 (Creek held out: its sweep was still running);
  calibrated ensemble on 39 fires (Creek calibrated pass stopped for compute; held-out deterministic via sweep).
  uncal F1 0.153, log area ratio +1.08; cal F1 0.192-0.194, AP 0.154 (ens), LOFO F1 0.184; persistence Sorensen 0.96.
  factors GR .5 GS .35 SH .35 TL 1.0 TU .35 default .5 first-day .7 (checks/replay_calibration.json)
- final reruns on final code: analytic, crown, speed (meets target: worst 2.0 s at 60 m), moisture_real (18 fires)
- README.md complete; demo.py + demo_out/{demo,replay_examples}.png

## If resumed
- nothing running. Optional: `cd checks && python3 replay_dev.py calibrated CA3720111927220200905` (~1.5 h) then
  `python3 replay_dev.py report` (do NOT re-run `calibrate`: it would refit with Creek and invalidate the calibrated
  pass). Exam replays are B's / Phase 2's job using firesim.replay + replay_calibration.json.
