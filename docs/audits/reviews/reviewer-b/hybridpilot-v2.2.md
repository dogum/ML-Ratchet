# HybridPilot v2.2 | Missing road grade changes earlier no-preview control

**First pass. A later road-grade measurement can change an earlier engine decision in the RoadSim integration.** The released controller allocates power each second under the driver's current traction request. Its default has no route preview and no navigation distance. RoadSim first normalizes the entire route, linearly interpolating missing grade between surrounding finite observations. Both the controller and the simulated drivetrain then see that reconstructed grade. [RoadSim normalization](../../../../shared/vehicle/roadsim/roadsim/sim.py#L53) · [Policy features](../../../../models/hybridpilot/hybridpilot.py#L139)

On each of the three packaged VED trips, we selected a moving segment and withheld ten consecutive grade samples. We then raised grade by four percentage points starting at the **first sample after the gap**, leaving all earlier observations, speed, initial state and controller settings identical. We ran both versions closed loop and compared outputs strictly before that future sample. Controls with complete grade gave zero difference in all three fields on all trips.

| VED trip | Last compared second | Maximum earlier engine-power difference | Maximum earlier battery-current difference | Earlier engine state changed |
|---|---:|---:|---:|---|
| 125/1275 | 51 | 8.64 kW | 23.83 A | Yes |
| 492/2826 | 43 | 0 kW | 35.98 A | No |
| 428/465 | 38 | 11.51 kW | 15.53 A | No |

This isolates a shared input-preparation defect for live or incrementally supplied grade. It also changes physical demand, so the power deltas include drivetrain response. All three archived example grade series are complete. The example loader zero-fills missing grade, while the exam route loader passes VED grade onward. The prevalence of open gaps across the 432 exam cases and their effect on policy comparisons remain unmeasured. [Example loader](../../../../models/hybridpilot/pipeline.py#L49) · [Exam VED route loader](../../../../models/hybridpilot/training/hp_env.py#L244)

**The optional preview reads one speed/grade endpoint beyond its stated horizon.** For a preview ending at second `k+L`, the final road-power interval uses speed and grade at `k+L+1`. We held those two values at their advertised endpoint for that final interval and reran the three trips under pinned v3.1 physics. No-preview runs were identical. With 5 s and 30 s preview, two trips still had identical fuel and final SOC. On VED 428/465, the bounded 30 s run used **0.273 g less fuel** out of 398.405 g and ended **0.000735 lower in SOC**. All engine on/off samples matched. These paired changes do not establish a fuel saving at equal battery use. RoadSim's current one-second traction request already uses its ending speed. On complete-input controls, changes strictly beyond that interval left the sampled no-preview history features unchanged. [Road power and preview bins](../../../../models/hybridpilot/hybridpilot.py#L85) · [Driver request](../../../../shared/vehicle/roadsim/roadsim/sim.py#L314)

**Training and scoring checks.** This policy has 66 inputs scaled from simulated case-level training data. It has no EnergyNet work-fraction denominator. On these three no-navigation, 30 s preview trajectories, all 49 varying features had normalized standard deviation of at least 0.01. All weights were finite with zero low-norm output columns. Torch/NumPy maximum raw-head errors were 9.85e−6 for shipped v2 and 1.19e−5 for historical v1, slightly above the audit's 1e−5 reference for v1. The trainer seeds Torch and splits by case. All 23 entries on the v2.2 release ladder replayed under the rule in force at each row. The policy gives point decisions. The card reports vehicle-cluster intervals and its simulator-to-log comparison separately. We did not rebuild the DP bank, replay the entire exam or measure performance on physical vehicles. [Trainer](../../../../models/hybridpilot/training/train.py#L49) · [Promotion rules](../../../../models/hybridpilot/training/ratchet.py#L63) · [Corrected exam and simulator limits](../../../../models/hybridpilot/MODEL_CARD.md#L94)

Model snapshot `61c70bf`, audit checkout `b53c7c7`. Uses NumPy, SciPy, CPU PyTorch and the bundled RoadSim. [Reproduction probe](hybridpilot_v2_2_probe.py)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/hybridpilot_v2_2_probe.py
```
