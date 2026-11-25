# arms_tab (agent B: tables, physics-in-the-loop, formulas): STATUS

**DONE (2026-11-23 06:20 local).** All four arms registered. Exam NOT run (not ours to run).

## READY for C
**tab_dpl REGISTERED (hash 5a86b90b36cc9ef1): READY for C.** Path: `/home/claude/bakeoff/river/arms_tab/tab_dpl`.
`armlib.load_predict(path)` -> (predict, module); `module.params_of(batch)` = dict of the 17 learned parameters + fixed
CFR/CWH: HBV FC LP BETA PERC UZL K0 K1 K2 MAXBAS; degree-day snow TT TTI CFMAX SFCF TLAPSE PGRAD (CFR 0.05, CWH 0.1);
CH_VEL (gamma channel UH, routing='uh'); PMULT (multiplies forcing P before WaterSim). `module.wsim.run_batch(batch,
params)` runs the vendored WaterSim exactly as the arm does (model hbv, snow degree_day, 5 elevation bands from attrs
elev mean +- range/2, PET from the forcing, spin-up = the 365 warm-up rows, no reservoirs). List the folder in your
ratchet.json extra_files if you reuse it. It will not change again.

## Registered arms (rule: sim val median NSE > incumbent + 0.01 AND dev >= 0.95 x incumbent; 800 val runs, 52 dev basins)
| Arm | Hash | Released | Sim val NSE | Dev NSE | Dev KGE | cov50/90 | CPU h |
|---|---|---|---|---|---|---|---|
| tab_gbm | 86b5fa6bcbd682a8 | r2 (no monotone constraints) | 0.398 | 0.421 | 0.444 | 0.55/0.95 | 0.43 |
| tab_gp | 2806d12cceedb34f | r2 (20 features, M 500) | 0.313 | 0.369 | 0.364 | 0.72/0.96 | 0.94 |
| tab_dpl | 5a86b90b36cc9ef1 | r1 (loss MSE of q/p_mean) | 0.414 | 0.383 | 0.403 | 0.53/0.96 | 2.25 |
| tab_formula | 3f8047a78c60ebe0 | r0 (17 formulas from tab_dpl) | 0.415 | 0.386 | 0.412 | 0.53/0.95 | 0.25 |

Rungs (sim / dev):
- gbm: r0 0.367/0.418 base; r1 2x data + 127 leaves 0.371/0.403 rejected; r2 no monotone 0.398/0.421 KEPT/released;
  r3 + elevation-band snow features 0.407/0.418 rejected (+0.009).
- gp: r0 0.291/0.364 base; r1 log target 0.219/0.278 rejected; r2 20 features 0.313/0.369 released; r3 M 1000 0.308/0.352 rejected.
- dpl: pilots (NSE*, log(1+NSE*) losses) aborted, volume bias 0.75; r0 bounded NSE 0.354/0.365 base; r1 MSE q/p_mean
  0.414/0.383 released; r2 fine-tune batch 128 0.418/0.388 rejected (+0.004).
- formula: r0 0.415/0.386 released; r1 constants tuned through torch 0.408/0.376 rejected; r2 most accurate per front
  0.418/0.392 rejected (+0.003).

Checks: torch HBV == vendored WaterSim to 7.1e-14 mm/day (30 cases, `tab_dpl/work/verify_hbv.json`); numpy trees ==
LightGBM 1.2e-14; numpy MLP == torch 9e-16; GBM and GP registered folders reproduce their eval NSEs exactly.

## Compute log (single-thread CPU, process_time)
- gbm: tables 7+15+15 s, reorder ~20 s; failed launch ~125 s; r0 223 s; r1 609 s; r2 221 s; r3 289 s => 0.43 h
- gp: r0 652 s; r1 732 s; r2 724 s; r3 1266 s => 0.94 h
- dpl: verify 25 s; pilot_nse 336 s; pilot_lognse 558 s; r0 2682 s; r1 2948 s; r2 1530 s; exports ~25 s => 2.25 h
- formula: search 251 s; r1 tune 613 s; exports ~25 s => 0.25 h (+ teacher, counted in dpl)
- Harness evaluations (not training): gbm ~0.8 h, others a few minutes each.

## Layout
- `src/` training + helper scripts (not part of any artefact); `.venv/` lightgbm 4.6.0 (training only)
- `<arm>/work/rungs/<rung>/` training outputs; `<arm>/work/eval/<rung>_summary.json` harness evaluations
- Feature tables were deleted after training (rebuild: `python3 src/build_tab.py OUT.npz 100 [NWIN]`, ~10 s).
