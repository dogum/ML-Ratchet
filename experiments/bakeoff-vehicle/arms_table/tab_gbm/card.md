# tab_gbm: gradient-boosted trees (Vehicle Bake-off, table track)

**What it is.** LightGBM trees on EnergyNet's 151 table inputs, predicting EnergyNet's correction
z = log(E / E_phys) to its generic physics estimate. Trained with LightGBM 4.7.0 (one thread), then every tree is exported
to flat numpy arrays (`gbm_trees.npz`; `treeexport.py`). Inference is numpy-only: the numpy trees equal LightGBM to
4e-15 in z.

**Inputs.** The table track's 151 features as given (route vector 90, physics vector 12, vehicle vector 49 with the
tier's masking). No extra features.

**Outputs.** pred = E_phys x exp(mean model). The 5 / 25 / 50 / 75 / 95 % quantiles come from five LightGBM quantile
models of z. They are sorted, and none crossed on the dev gate. No components.

**Training data.** Only the bake-off's simulated train bank (28,800 trips, EnergyNet v2.1's recipe, RoadSim physics v2).
Zero real labels. Each trip appears K times with independent tier draws (`bake.draw_tiers`, seeds 1000 + k). This is
the tree analogue of EnergyNet redrawing tiers every epoch. Rows with y <= 0.02 MJ are dropped. Early stopping uses
held-out train vehicles (vehicle index % 10 == 9), never the frozen val set.

**Released rung: r2.** K = 8 (229,800 rows; 206,848 train + 22,952 early-stopping rows). Mean model: L2 loss, 127 leaves,
min 50 rows per leaf, lr 0.05, feature / bagging fraction 0.8, lambda_l2 1, 472 trees. Quantile models: 31 leaves,
lr 0.1, 190-1,085 trees.

## Ratchet (rule: sim val skill > incumbent + 0.01 AND dev D16 <= incumbent + 0.01)

| Rung | Change | Sim val skill | Sim val MdAPE | Dev D16 | Decision |
|---|---|---|---|---|---|
| r0 | base: K = 3 tier draws, 63 leaves | 0.621 | 0.0877 | 0.1321 | base |
| r1 | monotone constraints where physics fixes the sign (below) | 0.616 | 0.0895 | 0.1253 | rejected (sim -0.005) |
| r2 | K = 8 tier draws, 127 leaves | **0.632** | **0.0860** | **0.1279** | **released** (+0.011) |
| r3 | K = 12 tier draws, lr 0.03 | 0.637 | 0.0846 | 0.1221 | rejected (+0.006 only) |

References: EnergyNet v2.1 skill 0.680 / D16 0.1133; E_phys alone 0.048 / 0.2278.

**Monotone rung (r1).** The target is a correction to a physics estimate, so the sign is fixed only where E_phys leaves the
effect out. The constrained inputs were:
- the window-sticker label energy relative to E_phys, fuel and electric (+): a car rated as thirstier than generic physics
  uses more;
- cold start (+) and warm start (-);
- log engine displacement (+), for engine friction;
- the share of trip time below 0 C (+), for heating and warm-up;
- the share of trip time above 25 C (+), for air conditioning.

Grade work is not constrained. E_phys already contains it, and the correction's sign versus climb is not fixed: engines
run more efficiently under load. LightGBM cannot constrain quantile objectives, so only the mean model was constrained.

Result: sim val dropped 0.005 in skill, while dev D16 improved by 0.007. The dev gains were EPA US06 0.148 -> 0.124,
hybrid HWY 0.128 -> 0.104, BEV HWY 0.117 -> 0.083 and ICCT. That gain is under the 0.01 seed noise, so by the rule the rung
is rejected. The pattern is still notable: the constraint costs a little on the simulator and helps a little on real data.

**Dev gate, r2 (MdAPE, bias, cov50 / cov90).**
- VED: ICE 0.071 (-0.012, 0.27 / 0.71); HEV 0.124; EV 0.151 (+0.113 bias); PHEV 0.182.
- EPA conventional: FTP 0.073 (0.56 / 0.90); HWY 0.061; US06 0.106; cold FTP 0.103; SC03 0.093.
- EPA hybrid FTP 0.157; EPA BEV city 0.057, highway 0.085.
- Heavy duty: Mersin diesel 0.141, CNG 0.243; ICCT 0.296.

**Compute.** Training 0.24 single-thread CPU-hours over all four rungs: r0 108 s, r1 120 s, r2 244 s, r3 399 s
(process_time). Prediction costs about 6 s CPU for the 3,985 sim val trips and 15 s for the full dev gate.

**Limits.**
- Bands are the raw quantile models. They are narrow on real data (VED ICE cov90 0.71), as for every sim-trained arm.
  Calibration is left to arm C's conformal layer.
- Trees extrapolate flat outside the simulated range, for example to very long heavy-duty trips (ICCT).
- No component split.
- Trained on one of v2.1's three banks (1/3 of its data).

Files: `predict.py`, `treeexport.py`, `gbm_trees.npz`; `training/` holds the training script and the rung configs. They
are not read at predict time.
