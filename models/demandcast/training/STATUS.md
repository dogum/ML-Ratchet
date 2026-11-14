# DemandCast status (resume notes)

Phase 2 GridSim project: day-ahead hourly demand, wind and solar per BA (8 focus BAs) with calibrated error bars,
trained only on simulated regions (building-stock thermal model + wind/solar fleet physics), graded on 2023-2025.

## Layout
- `src/` code. Pipeline order:
  1. `wxprep.py` -> data/wx/<BA>.npz (weather bundles + pvlib CFs), data/fleet/ (real farms)
  2. `renew.py` -> data/prior/<BA>_renew.npz (nominal physics wind/solar, real fleet)
  3. `calib_demand.py dev|dev2019|dev2021|train` -> data/calib/demand_*.json; train -> data/noise/demand_resid.npz
  4. `calib_renew.py dev|dev2019|dev2021` -> data/calib/renew_*.json (k scale, error stats, solar timestamp lag)
  5. `physics_check.py` -> checks/physics.json
  6. `sim.py <name> <seed> <nreg> <days>` (env DC_SIM=v1|v2) -> data/banks/<name>.npz
  7. `realbank.py dev|exam` -> data/banks/real_dev.npz / real_exam.npz (dev uses cross-year fits)
  8. `train.py <ckpt> <banks>`; `ratchet.py <ckpt> "<note>" --val val_vX`; `devreport.py <ckpt>`
  9. `exam.py <ckpt>` (final only)
- `fetch_openmeteo.py` (background, rate-limited; resumable cache data/openmeteo/), `make_tfc.py` (2022 T-forecast errors)

## Done
- Physics checks (checks/physics.json). Dev fits in-sample MAPE 2.2-3.5 %.
- r1_base (sim v1): dev gate demand 2.06 %, wind 21.4 %, solar 24.3 %; accepted (first rung).
- Found: sim v1 prior errors ~2x real dev -> sim v2 (calibrated). CISO/PJM EIA-930 solar leads physics by 1 h -> lag fix.

- r2 (sim v2) rejected: solar broke on real (night-time negative EIA output vs ~0-std sim features). r3 (clamp + scale
  floor) +0.009 objective -> rejected by the 0.01 rule. r4 (v1+v2 banks, 384-384) ACCEPTED (0.408).
- Twin test found a simulator bug: residual-noise offsets ran over all years, ~60 % of hours had no noise -> sim v3.
  Generating val_v3, train_v3a, train_v3b (running). Then: rescore ladder on val_v3, r5 on v3 banks.
- exam bank: forecast availability bits (1 T, 2 RH, 4 GHI, 8 wind); Open-Meteo heavily rate-limited (shared IP).

## Next
- val_v2/train_v2 banks (running), rebuild real_dev (running), rescore r1 on val_v2, train r2 on v2.
- When Open-Meteo 2022 pop done: make_tfc.py -> rung with real T-forecast errors. Exam bank after fetch completes.
- Page (web/), release/, project doc.
