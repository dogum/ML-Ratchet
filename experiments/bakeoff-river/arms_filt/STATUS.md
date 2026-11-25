# arms_filt (agent C): gauged EnKF + conformal layer — STATUS

Updated 2026-11-23 11:10 UTC. The exam is NOT ours to run.

## Registered by agent C (`register.py --list`)
```
enkf_watersim            bd85c4003560 OK /home/claude/bakeoff/river/arms_filt/enkf_watersim
enkf_watersim_conformal  c4f91252aa3b OK /home/claude/bakeoff/river/arms_filt/conformal/enkf_watersim_conformal
flowcast_v1_conformal    da549e656a0f OK /home/claude/bakeoff/river/arms_filt/conformal/flowcast_v1_conformal
tab_dpl_conformal        e73ea1c1e1d2 OK /home/claude/bakeoff/river/arms_filt/conformal/tab_dpl_conformal
tab_formula_conformal    dacb390ac9f0 OK /home/claude/bakeoff/river/arms_filt/conformal/tab_formula_conformal
tab_gbm_conformal        986e6c2ed612 OK /home/claude/bakeoff/river/arms_filt/conformal/tab_gbm_conformal
```

## EnKF rungs (tuned on 250 sim TRAIN runs; decisions on sim val 800 + dev 52; lead-3 rule)
| Rung | Sim val NSE L1/L3/L7 | Dev NSE L1/L3/L7 | Decision |
|---|---|---|---|
| r0 state-only | 0.777 / 0.538 / 0.450 | 0.640 / 0.500 / 0.460 | base |
| r1 + 8 augmented log params | 0.839 / 0.672 / 0.589 | 0.635 / 0.528 / 0.413 | released |
| r2 + UZL, c 0.2 x P | 0.852 / 0.705 / 0.645 | 0.595 / 0.488 / 0.459 | rejected (dev gate) |
| r3 r1 + tab_dpl params | 0.842 / 0.673 / 0.575 | 0.610 / 0.461 / 0.403 | rejected (sim +0.001, dev gate) |
| open loop (diagnostic) | | 0.381 / 0.383 / 0.379 | - |
All 3 rungs used. Compute 0.98 CPU-h (tuning 0.64 + rung evals 0.32 + smoke 0.02). Re-registered with --replace after
logging r3 (code unchanged; card + ratchet only).

## Conformal layer
- `conformal/lib/cf_mondrian.py` (method), `conformal/fit_wrapper.py <base_dir> [--register]` (one wrapper),
  `conformal/wrap_pending.py [--run]` (every registered, unwrapped arm; refits re-registered ones with --replace).
- `tools/watch_wrap.sh` (pid in tools/watch_wrap.pid) runs wrap_pending every 2 min while agent C is active;
  stop with `touch tools/STOP`.
- Dev coverage and CRPS (`python3 tools/conformal_table.py`):

| Base arm | Job | Raw cov50 / cov90 (median) | Conformal LOBO (median) | Mean over basins raw -> LOBO | CRPS/Q raw -> LOBO |
|---|---|---|---|---|---|
| enkf_watersim | gauged_L1 | 0.75 / 0.96 | 0.55 / 0.93 | 0.76 / 0.95 -> 0.57 / 0.91 | 0.192 -> 0.190 |
| enkf_watersim | gauged_L3 | 0.63 / 0.92 | 0.51 / 0.92 | 0.65 / 0.91 -> 0.54 / 0.90 | 0.252 -> 0.251 |
| enkf_watersim | gauged_L7 | 0.52 / 0.87 | 0.49 / 0.91 | 0.56 / 0.87 -> 0.52 / 0.90 | 0.302 -> 0.305 |
| flowcast_v1 | ungauged | 0.58 / 0.97 | 0.54 / 0.95 | 0.50 / 0.95 -> 0.49 / 0.88 | 0.380 -> 0.388 |
| flowcast_v1 | gauged_L1 | 0.65 / 0.96 | 0.55 / 0.93 | 0.58 / 0.87 -> 0.51 / 0.89 | 0.225 -> 0.219 |
| flowcast_v1 | gauged_L3 | 0.62 / 0.95 | 0.54 / 0.93 | 0.55 / 0.85 -> 0.51 / 0.89 | 0.246 -> 0.246 |
| flowcast_v1 | gauged_L7 | 0.57 / 0.93 | 0.52 / 0.93 | 0.50 / 0.84 -> 0.50 / 0.89 | 0.281 -> 0.283 |
| tab_dpl | ungauged | 0.53 / 0.96 | 0.49 / 0.93 | 0.50 / 0.92 -> 0.49 / 0.88 | 0.401 -> 0.406 |
| tab_formula | ungauged | 0.53 / 0.95 | 0.49 / 0.93 | 0.51 / 0.92 -> 0.49 / 0.89 | 0.401 -> 0.419 |
| tab_gbm | ungauged | 0.55 / 0.95 | 0.53 / 0.95 | 0.50 / 0.86 -> 0.51 / 0.88 | 0.386 -> 0.398 |

## Next
- Poll harness/registry/ and run `conformal/wrap_pending.py --run` as arms_seq / arms_tab arms register.

## If the container restarted
Nothing of ours needs resuming (all results on disk). Check `ps` for own jobs before starting new ones.
Logs: `arms_filt/logs/`. Tuning records: `arms_filt/work/tune/*.jsonl`.
