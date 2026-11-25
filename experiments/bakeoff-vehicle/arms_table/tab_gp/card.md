# tab_gp: sparse Gaussian process (Vehicle Bake-off, table track)

**What it is.** A sparse Gaussian process regression:
- SGPR, the collapsed variational bound of Titsias (2009);
- 500 inducing points and an ARD squared-exponential kernel;
- it predicts EnergyNet's correction z = log(E / E_phys) to its physics estimate.

It is adapted from the River Bake-off's tab_gp. Inference is numpy only (`gp.npz`).

**Inputs (46 dims, standardised).**
- The 32 top-ranked of the table track's 151 inputs by the GBM arm's r0 gain ranking, computed on the simulated train bank
  only. They include the label energy vs physics, log E_phys and the physics ratios, log mass, temperatures, speeds,
  climb, start SOC, cold / warm flags, displacement, power and model year.
- The 14 one-hot architecture / class columns.

A GP needs a scoped input set: the cost and the ARD fit grow with the dimension.

**Outputs.** The predictive distribution of z is Gaussian: latent variance plus one noise variance. Quantiles are
mu + sd x z_q; pred = E_phys x exp(mu), the median. The 5 quantiles never cross. No components.

**Training.**
- Data: the train bank only (28,800 trips) under K = 3 tier draws, minus 10 % held-out train vehicles (the check set).
  That leaves 77,568 rows. Zero real labels.
- Kernel hyperparameters and inducing inputs: Adam (lr 0.02, 300 steps) on the bound over 15,000 random rows.
- Then the exact SGPR posterior over all 77,568 rows at the fixed hyperparameters.
- Fitted noise variance 0.043 (sd 0.21 in log space). On the held-out train vehicles: median |z error| 0.096,
  cov50 / cov90 0.67 / 0.94.
- Script: `training/train_gp.py`. `predict.py` can also route rows to one GP per powertrain group (r2); the released
  rung is a single GP.

## Ratchet (rule: sim val skill > incumbent + 0.01 AND dev D16 <= incumbent + 0.01)

| Rung | Change | Sim val skill | Sim val MdAPE | Dev D16 | Decision |
|---|---|---|---|---|---|
| r0 | 20 ranked inputs + 14 one-hots, M 500 | 0.557 | 0.1025 | 0.1271 | base |
| r1 | M 1000 | 0.559 | 0.1025 | 0.1352 | rejected (+0.002) |
| r2 | one GP per powertrain group | 0.585 | 0.0969 | 0.1406 | rejected by dev (+0.0135) |
| r3 | 32 ranked inputs + 14 one-hots, M 500 | **0.596** | **0.0942** | **0.1260** | **released** (+0.039) |

References: EnergyNet v2.1 0.680 / 0.1133; MLP control 0.668 / 0.1203; GBM 0.632 / 0.1279.

**What the rungs say.**
- More inducing points do not help, as in the River Bake-off: the limit is the kernel, not the approximation.
- More inputs help the most.
- Separate GPs per powertrain fit the simulator better but transfer worse to real dynamometer tests. Pooling across
  powertrains regularises the correction, and the dev gate caught the difference.

**Dev gate, r3 (MdAPE, bias, cov50 / cov90).**
- VED: ICE 0.082 (-0.059, 0.75 / 1.00); HEV 0.119; EV 0.131; PHEV 0.174 (0.48 / 0.77).
- EPA conventional: FTP 0.071; HWY 0.105 (+0.100 bias); US06 0.197 (+0.197 bias); cold FTP 0.062; SC03 0.115.
- EPA hybrid FTP 0.149; EPA BEV city 0.065, highway 0.057.
- Heavy duty: Mersin diesel 0.140, CNG 0.174; ICCT 0.299.

**Compute.** Training 0.81 single-thread CPU-hours over all rungs (process_time):
- r0: 670 s;
- r1: 894 s, including a first attempt stopped at 1.4 GB memory;
- r2: 980 s (4 GPs);
- r3: 359 s.

Prediction costs under 1 s CPU for sim val or for the dev gate.

**Limits.**
- One noise variance for every trip, so the bands are too wide where the simulator is easy (VED ICE cov50 0.75) and too
  narrow where it is hard (PHEV).
- The stationary kernel extrapolates to the prior mean (z = mean correction) far from the training inputs.
- Long steady high-speed cycles are biased high (US06 +20 %).
- 46 of the 151 inputs.
