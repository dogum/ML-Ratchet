# ReservoirPilot v1.0 | Clean policy math, evaluation-period `cap` boundary

**Reviewer A, first pass.** Model base `61c70bf`, audit checkout `d903cb6`, released rung `r8_hd_clim_g4_xl`. **Verdict: MINOR HYGIENE.** The shipped 15-feature MLP policy (`reservoirpilot_weights.npz`, 5,314 weights) is strictly causal with respect to future inflow, matches the differentiable PyTorch training rollout to float64 machine precision, and replays every logged promotion decision. The main operational boundary issue sits in task construction rather than the policy network: both `make_task` and the ResOpsUS exam loader define reservoir capacity `cap` using the maximum observed storage across the **entire** record, including the evaluation half.

## Evaluation-period storage enters `cap`

- **What failed?** Both [`reservoirpilot.py:49`](../../../../models/reservoirpilot/reservoirpilot.py#L49) (`make_task`) and [`training/task.py:39`](../../../../models/reservoirpilot/training/task.py#L39) (`resops_task`) raise catalogued GRanD capacity to `max(cap_hm3, max(storage_hm3))` across both calibration and evaluation periods (in contrast, [`cdec_task`](../../../../models/reservoirpilot/training/task.py#L63) uses calibration-period storage only). In addition, [`task.py:29-30`](../../../../models/reservoirpilot/training/task.py#L29-L30) linearly interpolates gaps up to 5 days (`interpolate(limit=5)`) across the full daily record before splitting `cal` and `ev`, so short gaps inside the scored window use post-gap observations. Because `cap` enters `S / cap`, `log(cap / qK)`, headroom, spill pressure, and the spill threshold itself, evaluation-period peak storage can leak into the task definition when a dam exceeded its catalogued capacity during the scored decade.
- **What changed when tested?** On the six shipped ResOpsUS exam dams in [`examples/exam_dams.json`](../../../../models/reservoirpilot/examples/exam_dams.json), **2 of 6** (`grand_148` and `grand_470`) have `max(S_obs) / cap == 1.0000`, consistent with `cap` being pinned to the evaluation-period maximum. As a counterfactual stress test, shrinking `cap` by 10% (`cap * 0.90`, clipping `S0`) moved `J` slightly upward for both `Pilot` and `tuned_rule` without reversing the win on any of the five non-zero-cost dams (`grand_1023`: `0.11344 vs 0.14108`, `grand_148`: `0.11902 vs 0.15543`, `grand_471`: `0.02512 vs 0.03163`, `grand_470`: `0.12762 vs 0.22436`, `grand_541`: `0.08962 vs 0.12855`; `grand_1112`: `0.00004 vs 0.00000`).
- **What consequence remains unmeasured?** Rebuilding all 244 ResOpsUS exam dams from raw ResOpsUS records (not stored in the repo) to count how many dams had `cap` raised strictly by `ev.storage_hm3`, and how many days inside `ev` were filled by `interpolate(limit=5)`, remains unmeasured without `mlratchet data water`.

## Causality, same-day inflow, and feature sensitivity

- **Future inflow:** Multiplying inflow by `5×` strictly after the midpoint day (`t > mid`) changed policy releases on `t <= mid` by **`0.000e+00`** across all six shipped dams.
- **Same-day inflow clairvoyance check:** By WaterData's protocol, all causal policies observe day `t`'s recorded inflow `I[t]` when choosing day `t`'s release. Replaying all six dams with a **1-day lagged inflow** (`seen[t] = I[t-1]`, initialized at `Icurve[doy[0]-1]`, while the true mass balance still evolves under `I[t]`) keeps `Pilot` ahead of `tuned_rule` on all five non-trivial dams (`grand_1023`: `0.0911 vs 0.0995`, `grand_148`: `0.1168 vs 0.1499`, `grand_471`: `0.0207 vs 0.0254`, `grand_470`: `0.1200 vs 0.2120`, `grand_541`: `0.0870 vs 0.1251`).
- **Calibration-half inflow climatology ablation:** Zeroing `Icurve` (`noclim`) degraded `J` sharply on `grand_470` (`0.11627 -> 0.20420`), `grand_471` (`0.01959 -> 0.03230`), and `grand_1023` (`0.08148 -> 0.09567`), confirming genuine reliance on the seasonal climatology features rather than a dead feature block.
- **Unpriced storage caveat (already documented in `model.yaml`):** On the six shipped dams, `Pilot` holds mean storage at `17.8%–54.8%` of capacity (consistent with the published 244-dam median `38%` vs SOP `59%`).

## Weights, export parity, seeding, and ratchet replay

- **Weights & normalization:** All arrays in `reservoirpilot_weights.npz` are finite with **0 dead rows or columns** and well-conditioned feature scales (`sd ∈ [0.1914, 2.6970]`).
- **Export parity:** Reconstructing the PyTorch float64 MLP via [`training/policy.py`](../../../../models/reservoirpilot/training/policy.py) (`torch_policy` + `rollout_torch`) matched [`reservoirpilot.py`](../../../../models/reservoirpilot/reservoirpilot.py) (`Pilot.run`) across all six shipped dams within **`2.50e-16`** in `J`, and every shipped dam reproduced its archived `exam_J["pilot"]` value to `< 1e-6`.
- **Seeding & resume:** [`training/train.py:86,127`](../../../../models/reservoirpilot/training/train.py#L86) seeds both NumPy (`default_rng(a.seed)`) and PyTorch (`torch.manual_seed(a.seed)`), and saves/restores `rng` in resume checkpoints (though not Torch's RNG state; training uses NumPy for batch sampling and `S0` draws).
- **Ratchet replay:** All **10 rows** in [`training/ratchet.json`](../../../../models/reservoirpilot/training/ratchet.json) (9 scored rows + 1 diverged `r6_rule_clim_g4`) match the published `g4` promotion rule (`S_J > inc + 0.01` and `dev J_median <= 1.05 * inc`), selecting `r8_hd_clim_g4_xl`.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/reservoirpilot_v1_0_probe.py
```

Requires NumPy, pandas, and CPU PyTorch. [Probe script](reservoirpilot_v1_0_probe.py)
