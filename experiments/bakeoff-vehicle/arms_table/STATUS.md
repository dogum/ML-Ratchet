# arms_table (agent A, table learners): STATUS

**DONE (2026-11-23).** All four arms registered and verified. Exam NOT run (Phase 3). Nothing of ours is running.
Rule (all arms): keep a rung if frozen sim val skill > incumbent + 0.01 AND dev D16 <= incumbent D16 + 0.01.

## Registered arms (do not touch these folders except work/)
| Arm | Family | Hash | Released | Sim val skill | Sim val MdAPE | Dev D16 | cov50/90 sim | Train CPU h |
|---|---|---|---|---|---|---|---|---|
| tab_mlp | mlp | ab35c4248d5c8283 | r3 (5 nets x 30 ep) | 0.668 | 0.0775 | 0.1203 | 0.46/0.87 | 0.62 |
| tab_gbm | gbm | 57349fe2297a02a8 | r2 (K 8 tier draws, 127 leaves) | 0.632 | 0.0860 | 0.1279 | 0.46/0.85 | 0.24 |
| tab_gp | gp | 131f5c73c54ed1a9 | r3 (32+14 inputs, M 500) | 0.596 | 0.0942 | 0.1260 | 0.68/0.94 | 0.81 |
| tab_tabpfn | tabfm (lens, not a release) | cb26e0c49f796b7d | r1 (ctx 2,048 sim trips, 40 inputs) | 0.563 | 0.1017 | 0.1249 | 0.47/0.88 | 0 (in-context) |
References: EnergyNet v2.1 0.680 / 0.0746 / 0.1133; E_phys 0.048 / 0.2104 / 0.2278.
Every registered folder reproduces its rung's sim val numbers exactly (work/eval_released/).

## All rungs (sim val skill / MdAPE / dev D16)
- mlp: r0 1 net 60 ep 0.646/0.0827/0.1205 base; r1 5 nets 0.657/0.0796/0.1188 kept; r2 5 x 180 ep 0.597/0.0943/0.1224
  rejected (memorises the one bank); r3 5 x 30 ep 0.668/0.0775/0.1203 RELEASED.
- gbm: r0 K 3 0.621/0.0877/0.1321 base; r1 monotone (7 signed inputs) 0.616/0.0895/0.1253 rejected (sim -0.005, dev
  -0.007 better); r2 K 8 + 127 leaves 0.632/0.0860/0.1279 RELEASED; r3 K 12 lr 0.03 0.637/0.0846/0.1221 rejected (+0.006).
- gp: r0 20+14 inputs M 500 0.557/0.1025/0.1271 base; r1 M 1000 0.559/0.1025/0.1352 rejected; r2 one GP per powertrain
  0.585/0.0969/0.1406 rejected by dev (+0.0135); r3 32+14 inputs 0.596/0.0942/0.1260 RELEASED.
- tabpfn: r0 ctx 1,024 0.527/0.1110/0.1348 base; r1 ctx 2,048 0.563/0.1017/0.1249 RELEASED; r2 balanced arch x class
  0.544/0.1067/0.1259 rejected; r3 3,072 x 30 inputs 0.561/0.1013/0.1298 rejected.

## Notes for Phase 3
- tab_tabpfn exam CPU: registry says 480 min, but that is the one-off context fit (~23 s, lazy, at the first predict call)
  divided over the 24-row VED smoke batch. Real cost ~10.6 ms CPU per row (dev gate: 128 s for ~12k rows) => ~20-30 min
  for ~121k exam rows, and the same again for arm C's tab_tabpfn_conformal wrapper. Peak memory ~1.1 GB per process.
- tab_tabpfn needs torch (system python3) + its pylib/ (copied tabpfn 10.2.0 + 4 deps; empty entry_points.txt added to
  each dist-info because importlib probes them). Register / audit it from a cwd OUTSIDE /home/claude/bakeoff: torch's
  fake modules torch.classes / torch.ops have __file__ '_classes.py' / '_ops.py', which register.py resolves against
  the cwd. Credit "Built with PriorLabs-TabPFN"; label "lens, not a release".
- Bands of tab_mlp / tab_gbm are raw (narrow on real data: VED ICE cov90 0.73 / 0.71); tab_gp too wide on easy rows
  (VED ICE cov50 0.75); tab_tabpfn closest to nominal (VED ICE 0.51 / 0.91). Calibration = arm C's conformal layer.

## Layout
- `env.sh` (source first), `src/` training + helpers, `<arm>/training/` copies of the scripts + rung configs (hashed, not
  read at predict time), `<arm>/work/rungs/<rung>/` training outputs, `<arm>/work/cand/<rung>/` candidate folders,
  `<arm>/work/eval/` harness evaluations; `python3 src/board.py` prints every evaluated rung. Disk ~200 MB.
- LightGBM / TabPFN were run with /home/claude/priorwars/venv/bin/python (read-only use).

## Compute log (single-thread CPU, process_time, training only)
- mlp: r0 90 s; r1 471 s (5 nets incl. r0's); r2 1,482 s; r3 263 s => 2,216 s = 0.62 h
- gbm: r0 108; r1 120; r2 244; r3 399 => 871 s = 0.24 h
- gp: r0 670; r1 894 (incl. a ~140 s attempt stopped at 1.4 GB); r2 980 (4 GPs); r3 359 => 2,903 s = 0.81 h
- tabpfn: 0 training; context fit at load 8-33 s per process; setup probes ~470 s (speed / memory, not training)
- harness evaluations (prediction CPU, not training): mlp/gbm/gp < 30 s each; tabpfn ~200 s each
