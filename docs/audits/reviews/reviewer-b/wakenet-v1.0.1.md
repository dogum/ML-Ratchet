# WakeNet v1.0.1 | Which turbines count in the demo?

**First pass, reviewer B. Minor reporting issue in the demo.** The shipped archive identifies itself as v1.0.1, rung r6. Audited model base `61c70bf` from checkout `4820cf3`. WakeNet estimates downstream power relative to the operating farm's unwaked front row. The examples supply contemporaneous front-row power for absolute kW. The ratio comparison below uses that same observed reference for every method. [Operating inputs](../../../../models/wakenet/MODEL_CARD.md) · [Release readout](../../../../models/wakenet/wakenet.py#L195)

## A display limit changes the summary and the ranking

The demo selects at most six online downstream turbines per snapshot, sorted by deepest predicted wake. Its summary accumulates error and interval hits inside that display loop. Across all six shipped snapshots the denominator changes from 31 eligible turbine readings to 24. The formal exam explicitly defines the all-waked mask as every online turbine outside the free-stream group. [Demo accumulation](../../../../models/wakenet/demo.py#L68) · [Exam mask](../../../../models/wakenet/training/exam.py#L227)

| Same six snapshots | Readings | WakeNet MAE | Tuned v3 physics MAE | Textbook physics MAE | WakeNet 68% interval coverage |
|---|---:|---:|---:|---:|---:|
| Demo display selection | 24 | 14.04 | 14.29 | 14.83 | 66.7% |
| All eligible turbines | 31 | 14.90 | 14.43 | 16.66 | 64.5% |

MAE is in percentage points of the front-row power ratio, weighted equally per turbine reading. Tuned physics wins on all 31 readings, while WakeNet wins on the 24 displayed. The formal exam uses the broader mask. Six snapshots do not establish the population ranking or calibration.

## Why the learned correction trails tuned physics

- All shipped parameters were finite. Across the six real graphs, all five raw Torch and NumPy outputs agreed within `2.39e-7`. Rotating coordinates and meteorological wind direction together left the free-stream mask and predicted ratios unchanged. Reversing turbine order and undoing the permutation produced a maximum ratio difference of `1.20e-7`. Both implementations share the graph inputs in the parity test. The geometry controls separately exercise the public graph builder.
- The archived r6 twin error was 6.62 points, compared with 9.48 on real dev. For the separate r9 tuned-anchor candidate, the learned correction correlated −0.037 with the observed residual across 16,212 real waked readings. Its MAE was 9.44 points against 9.31 for its anchor. Poor transfer of the correction is a plausible explanation. These experiments cannot isolate simulator mismatch from loss design, model capacity or measurement error. [r6 twins](../../../../models/wakenet/training/twin_report.json) · [r9 residuals](../../../../models/wakenet/training/twin_residuals_r9.json)
- All nine candidate decisions plus the opening baseline matched the published rules using the three separate H1, H2 and farm error gates. R10 cleared the simulation margin by gaining 0.0124 over r6 on val4. Its worst dev ratio was 1.0053 or less. The log records it as kept and separately explains why r6 stayed the release. Earlier checkpoints unavailable for val4 keep their val3 comparisons. [Ladder](../../../../models/wakenet/ratchet.yaml) · [val4 reference](../../../../models/wakenet/training/v4_reference.json)
- The model stores dev widening factors by wind speed. Its low-speed bin reports 47.5% coverage before widening and 68.3% afterward across 2,260 readings. Its high-speed bin remains at 88.0% with factor 1. Training seeds NumPy and Torch. Fresh training, timestamp-level split reconstruction, and independent refitting of the dev calibration remain open.

Run from the repository root with NumPy, PyYAML and CPU PyTorch.

```sh
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/wakenet_v1_0_1_probe.py
```
