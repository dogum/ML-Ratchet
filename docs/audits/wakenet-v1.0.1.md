# WakeNet v1.0.1 | Demo display filter and graph-invariance verification

**Cross-checked minor demo reporting issue.** Reviewer A independently reran reviewer B's `dc30c02` probe against model base `61c70bf`, checked all 42 shipped parameter tensors for dead units, audited node/edge feature ranges across the six packaged snapshots, and tested permutation invariance when an offline turbine occupies index `0`. [Original report and probe](reviews/reviewer-b/wakenet-v1.0.1.md) · [Demo loop](../../models/wakenet/demo.py#L68) · [Exam mask](../../models/wakenet/training/exam.py#L227)

**What reproduced.** Across the six shipped real farm snapshots, [`demo.py:68`](../../models/wakenet/demo.py#L68) caps per-snapshot display at six online downstream turbines sorted by deepest predicted wake and accumulates summary MAE inside that loop (`24` of `31` eligible turbine readings). On those 24 displayed readings, WakeNet narrowly leads tuned v3 physics (`14.04` vs `14.29` percentage points, 68% coverage `66.7%`). Scoring all `31` eligible online downstream readings reverses the example ranking in favor of tuned physics (`14.90` vs `14.43`, coverage `64.5%`), matching the formal exam's all-waked mask.

**Independent second-pass checks.** All 42 weight arrays in `wakenet_weights.npz` are finite with zero dead rows. Across all six snapshots, graph features remain tightly bounded (`node_x ∈ [−0.74, +1.53]`, `edge_x ∈ [−2.50, +2.67]`), Torch↔NumPy parity reproduced within `2.38e-7`, coordinate/wind rotation and turbine permutation invariance held (`0` and `1.79e-7`), and all nine candidate rung decisions (`r2..r10`) matched the three-gate rule. Because [`wakenet.py:163`](../../models/wakenet/wakenet.py#L163) indexes `nx[0, 2]` and `nx[0, 6]` for farm-level scalars, swapping the parked offline turbine (`idx = 4` in `04_kelmarsh_parked_unit.npz`) into index `0` was tested explicitly and confirmed exact invariance (`farm_wake_sig` diff `0.00e+00`, `max |eta diff| = 0.00e+00`) because those two slots hold uniform farm-level meteorological inputs.

**What remains open.** The formal exam already scores all eligible waked turbines, so the display filter does not affect published exam scores. Fresh retraining, timestamp-level split reconstruction, and independent refitting of the wind-speed-binned widening factors (`by_u`) remain open.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/wakenet_v1_0_1_probe.py
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_dc30c02_probe.py wakenet
```

The second-pass probe uses NumPy and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_dc30c02_probe.py)
