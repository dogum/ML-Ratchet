# TripDetective v2.1 · One missing power trace shifts a whole bus day

**First pass · material public-inference defect under meter loss.** In the shipped 14-leg ZTBus mission, replacing one leg’s DC power by NaNs lowered its estimated mass by 1,327 kg and raised every other leg’s estimate by 239 kg. All 14 original traces have complete power. Model snapshot `61c70bf`, audit checkout `ea2bfde`. [Public bus reader](../../../../models/tripdetective/tripdetective.py#L376)

## Bus mission

The bus reader converts missing power to zero before checking sample validity, declares the battery channel present, then subtracts a window-weighted day mean from each leg. The intended reading uses a complete bus mission. Peer dependence across legs is therefore expected. The defect is giving an unmeasured leg a finite reading and weight in the shared reference.

| Experiment on the longest leg (index 8, 8,695 samples) | Selected leg | Effect on each of the 13 unchanged legs |
|---|---:|---:|
| Original 14-leg mission | 20,430.50 kg | Reference |
| All power samples missing | 19,103.65 kg | +238.83 kg |
| All power samples exactly zero | 19,103.65 kg | Identical to missing |
| Power missing and every sample explicitly invalid | 18,868.59 kg | +281.14 kg |

Compared with omitting the leg entirely, including it with NaN power raises each remaining estimate by 199.74 kg. Reversing leg order restores the same per-leg results after reversing the outputs. The released bus interval has a fixed 520 kg half-width. This example quantifies propagation through the mission. Its prevalence and effect on passenger-mass accuracy in the full ZTBus exam remain unmeasured. [Day calibration](../../../../models/tripdetective/tripdetective.py#L393) · [Mission and exam design](../../../../models/tripdetective/training/busexam.py)

## Cars and the shared road simulation

The public car reader takes completed trip logs and replaces missing grade by zero. It consumes no later grade observation to fill that gap. Across all four VED examples, all-NaN and flat grade returned identical curb masses. Relative to the measured grade, shifts ranged from −25.47 kg to +30.30 kg. The HybridPilot controller’s earlier-decision result cannot be transferred to this offline reader solely because training shares RoadSim. [Car feature construction](../../../../models/tripdetective/tripdetective.py#L41)

## Export and promotion controls

Both weight files were finite with zero near-zero kernel rows out of 1,343 each. Restoring the folded weights into the original PyTorch class reproduced the tested car and bus heads within `2.91e-7` on real and random zero-padded inputs (ledgers in network units). All 7, 16, and 18 logged car decisions across their respective scoring sets, both bus network decisions, and all six mission-calibration challengers matched the published rules. The v2.1 engine-stop constraint gained only 0.000158 on the simulation objective and raised dev per-trip type accuracy by 0.03257. It failed the original network rule and passed a separately documented output-constraint rule adopted after that result. This makes the rule change visible to the reader. The trainer explicitly seeds PyTorch. Historical seed variance, independent split reconstruction and interval coverage during power loss remain open. [Promotion rules](../../../../models/tripdetective/ratchet.yaml) · [Twin and coverage results](../../../../models/tripdetective/MODEL_CARD.md)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/tripdetective_v2_1_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/wave5_export_probe.py tripdetective
```
