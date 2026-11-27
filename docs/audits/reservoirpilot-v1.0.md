# ReservoirPilot v1.0 | Future observed storage can change today's release

**Confirmed evaluation-boundary defect with bounded evidence of impact.** At fixed capacity, the six packaged dam replays reproduced their archived costs and future-inflow changes left earlier releases identical. The public task builder, however, raises the supplied capacity to the highest storage observed anywhere in calibration or evaluation. The ResOpsUS loader takes its maximum over the full record, including days outside the selected scoring block. It also interpolates before splitting and includes the split date in both slices. [Public task construction](../../models/reservoirpilot/reservoirpilot.py#L43) · [ResOpsUS loader](../../models/reservoirpilot/training/task.py#L20)

In an independent synthetic case with 366 calibration days, 30 evaluation days and a known 100 hm³ capacity, changing only the **last** storage observation from 80 to 100 hm³ left the first release at 21.85347 m³/s. Changing it to 120 hm³ raised the inferred capacity to 120 hm³ and moved the first release to 21.79156 m³/s. The first-day inflow, storage and demand history were identical. This demonstrates a pathway from a later observation to an earlier action in the public API.

Two of the six archived dams reached their supplied capacity exactly. That equality alone cannot identify which date or metadata field set it. The lagged-inflow and 10% capacity stress tests preserved the ordering against the tuned rule on the five nontrivial packaged dams. Reconstructing actual calibration-date capacity for each dam and rescoring the full exam would establish the historical effect. Torch and NumPy agreed within 2.50e−16 on the six costs. The ten logged promotion decisions reproduced.

Model snapshot `61c70bf`, cross-check at `b53c7c7`. Both probes ran with NumPy, pandas and CPU PyTorch.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-a/reservoirpilot_v1_0_probe.py
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/peer_b53c7c7_probe.py
```

[First-pass evidence](reviews/reviewer-a/reservoirpilot-v1.0.md) · [Independent probe](reviews/reviewer-b/peer_b53c7c7_probe.py)

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| Capacity raised by storage observed in the scored window | Yes | **Opt-in fix, re-score queued.** `make_task(..., causal_capacity=True)` takes capacity from metadata and the calibration record only. The default stays as the exam defined it. | The exam's definition (`training/task.py`, mirroring WaterData's `run_baselines_reservoir.py`) applies to every arm, baselines included, so changing it is a re-score of the whole board, queued for the owner. |
| Loader interpolates before splitting; split date in both halves | Yes (minor) | Queued with the same re-score (it changes the shared exam definition). | — |
