# WaterSim (agent A) status — resume notes

Started 18 Nov 2026. Package `water/watersim/watersim`, checks `water/checks`.
Rules: one thread, nice long jobs, own PIDs in watersim/pids/, temp in watersim/tmp. Never write to water/data or water/waterdata.

## Plan
1. core: pet, snow (degree-day bands + energy balance), models (hbv, gr4j, sacsma, +vic_lite?), routing (uh, muskingum(-cunge), kinematic), reservoir, ledger, simulate, batch
2. generator (priors; CAMELS attrs on non-exam basins once B's splits exist), weather generator, forcing reuse
3. checks: analytic, references (airGR / HBV), speed, dev calibration (B's splits), SNOTEL
4. README.md

## Done (18 Nov ~08:15)
- package core: pet, snow (dd + eb), models hbv/gr4j/sacsma/vic_lite, engine (numba), routing, reservoir, simulate,
  simulate_batch/simulate_many, generator, weather generator, labels/signatures/metrics, calibrate (JADE DE)
- ledgers close to 1e-16 on all combos tried (incl. reservoirs cascade, irrigation)
- reference R env: watersim/.refenv (micromamba + conda-forge r-base 4.4 + gfortran; airGR 1.7.9 + TUWmodel 1.1-1
  compiled from CRAN into .refenv/rlib). checks/ref/*.R write CSVs; check_reference.py -> reference_models.json:
  GR4J == airGR to 1e-7 mm (single-precision output), NSE 0.79882 = published 0.7988; HBV snow+soil == TUWmodel
  to 1e-12, runoff NSE 0.9990
- B's data READY (splits.json; waterdata API). SRAD is NET shortwave (handled: srad_is_net from forcing_meta)
- dev calibration running in background: checks/calibrate_dev.py (pid in pids/calibrate_dev.pid), resumable,
  per basin x model JSON in checks/out/dev_calib/. Windows: cal 1999-10..2008-09; val_early 1980-10..1989-09,
  val_late 2008-10..2014-09 (exam years 1989-10..1999-09 and 2015-10..2023-09 never used; blanked in obs)

## Done (18 Nov, later)
- routing rewritten: Muskingum/MC use end-of-day samples + linear inflow; MC cascade advances all reaches per substep
  (chained daily reaches over-diffused: NSE 0.88 -> 0.9999998 vs Hayami). Substeps h <= K/10 for plain Muskingum.
- SAC-SMA: FLAND1 tiny-store zeroing now booked as ET (ledger exact). numba cache auto-cleared on source change
  (_nb.check_cache, sha1 stamp) -- numba does NOT track cross-module edits.
- sub-daily mode (hbv/sacsma/vic_lite; PET disaggregated by solar elevation; infiltration excess Hortonian with
  capacity from soil_conductivity); GR4J and reservoirs daily only.
- checks written & passing: reference_models.json, analytic.json, balance.json (120/120), speed.json,
  generator.json (+ v0/v1 variants), prior_skill.json, dev_calibration.json (KGE), dev_calibrated_params.json
- prior correction: tools/fit_prior.py SHRINK=0.5 -> watersim/data/prior_adjust.json (v2; chosen by mean KS on dev
  signatures: v0 0.25, v1 0.21, v2 0.17 camels prior)
- README.md written (numbers filled except NSE-objective medians + SNOTEL)

## Running / next
- NSE-objective dev calibration: OBJ=nse calibrate_dev.py (pid pids/calibrate_dev_nse.pid) -> out/dev_calib_nse;
  then OBJ=nse python3 summarize_dev.py -> dev_calibration_nse.json; put medians into README
- SNOTEL: B fetching (water/data/proc/snotel; splits pending). Then check_snotel.py on DEV stations only.
- maybe: simulate_network helper, observation-noise helper

## Done (later still)
- NSE-objective dev calibration done -> dev_calibration_nse.json / dev_calibrated_params_nse.json (README table)
- prior_skill.json re-run for the kept prior (v0 vs v2); v0/v1 kept in prior_skill_v0_v1.json
- check_reservoir.py -> reservoir_dev.json (B's 15 dev dams; shape_b prior now 2.5 exp(N(0,0.35)))
- simulate_network (tree of MC reaches, ledger in hm3), noise.py (gauge + forcing error), obs_noise= in simulate
- calibrate(target='SWE', model='none') for snow-only; check_snotel.py written (waits for B's splits['snotel'])
- demo.py -> demo_watershed.png
- rerunning reference/analytic/speed/generator after engine changes (tmp/rerun.sh, logs/rerun.log)

## PHASE 1 COMPLETE (18 Nov ~14:00 UTC)
- SNOTEL: check_snotel.py on 33/39 dev stations -> snotel.json (DD cal-val NSE 0.89, EB 0.93; defaults 0.70/0.82)
- weather generator fit_climate + check_weather.py -> weather_generator.json (stats ratios ~1.0)
- README.md final (checks table, benchmarks, speed, limits, Phase 2 notes); demo.py -> demo_watershed.png
- re-run everything: checks/run_all.sh (~1.5 h incl. both dev calibrations)
- disk: watersim ~1.4 GB (of which .refenv R toolchain ~1.3 GB, reference only; delete if space is needed)
