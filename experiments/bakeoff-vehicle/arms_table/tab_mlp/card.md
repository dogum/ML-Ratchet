# tab_mlp: MLP control (Vehicle Bake-off, table track)

**What it is.** The in-bake-off control. EnergyNet v2.1's own network and training recipe, retrained from scratch on the
bake-off's shared simulated train bank. It shows what v2.1's architecture does with the same data and budget as every
other arm. The released model is an **ensemble of 5 networks** (seeds 0-4).

**Network.** 151 inputs -> 256 -> 256 -> 256 -> 128 (GELU), with three outputs:
- the mean of z = log(E / E_phys);
- log sigma, clamped to [-5, 2];
- 5 log component shares: aero, rolling, brakes, aux / HVAC, powertrain losses.

About 207k parameters per member. The forward pass is numpy (`model_np.py`, a byte copy of EnergyNet's); the weights are
`weights/s*.npz` in EnergyNet's export format.

**Training (EnergyNet's train.py, version e, mode corr, era on).**
- Loss: Gaussian NLL of z, plus 0.2 x the MSE of the component shares.
- Optimiser: AdamW (lr 2e-3, weight decay 1e-4), OneCycle schedule, gradient clip 5, batch 256.
- Tiers are redrawn every epoch (`bake.draw_tiers`); input scaling comes from one tier draw.
- Data: the train bank only (28,800 trips). Zero real labels.
- v2.1 itself trained on three such banks (86,400 trips) for 60 epochs.
- Script: `training/train_mlp.py`, adapted to the bake API.

**Outputs.**
- pred = E_phys x exp(mean of the members' z): the median, as EnergyNet reports it.
- Band: log-normal with sigma^2 = mean member sigma^2 + the variance of the member means.
- The band is **not widened**: no simulation or dev factors. v2.1's release multiplies by per-architecture sim factors
  and per-group dev factors. Calibration is left to arm C's conformal layer.
- Components (5, MJ) from the component head.

## Ratchet (rule: sim val skill > incumbent + 0.01 AND dev D16 <= incumbent + 0.01)

| Rung | Change | Sim val skill | Sim val MdAPE | Dev D16 | Decision |
|---|---|---|---|---|---|
| r0 | v2.1 recipe, 1 network, 60 epochs | 0.646 | 0.0827 | 0.1205 | base |
| r1 | ensemble of 5 seeds | 0.657 | 0.0796 | 0.1188 | kept (+0.011) |
| r2 | 180 epochs per member (step-matched to v2.1's 3 banks) | 0.597 | 0.0943 | 0.1224 | rejected (-0.060) |
| r3 | 30 epochs per member | **0.668** | **0.0775** | **0.1203** | **released** (+0.011) |

Reference: EnergyNet v2.1 (5 networks, 3x the data, 60 epochs) has skill 0.680 and D16 0.1133.

**What the rungs say.**
- On one bank, extra epochs do not stand in for extra trips. At 180 epochs the networks memorise the bank: train
  |log error| 0.028 against 0.060 at 60 epochs, and sim val skill falls by 0.06.
- Halving to 30 epochs helps (+0.011). v2.1's 60-epoch schedule was set for 3x the data.
- The remaining gap to v2.1 (0.668 vs 0.680 skill; D16 0.1203 vs 0.1133) is about what EnergyNet's own data-size rungs
  measured: +0.019 skill from 1x to 3x data.

**Dev gate, r3 (MdAPE, bias, cov50 / cov90).**
- VED: ICE 0.069 (-0.020, 0.34 / 0.73); HEV 0.123; EV (Leaf) 0.116; PHEV 0.181 (-0.076 bias, cov90 0.54).
- EPA conventional: FTP 0.085; HWY 0.060; US06 0.085; cold FTP 0.101; SC03 0.105.
- EPA hybrid FTP 0.141, HWY 0.108; EPA BEV city 0.088, highway 0.074.
- Heavy duty: Mersin diesel 0.156, CNG 0.198 (-0.191 bias); ICCT 0.236.

**Compute.** Training 0.62 single-thread CPU-hours over all rungs (process_time):
- r0: 90 s (1 network);
- r1: 471 s (5 networks; the first is r0's);
- r2: 1,482 s;
- r3: 263 s.

Prediction costs about 2 s CPU for the 3,985 sim val trips and 6 s for the dev gate.

**Limits.**
- Raw bands are too narrow on real data (VED ICE cov90 0.73, PHEV 0.54).
- One third of v2.1's training data.
- PHEV trips remain the weakest real group, as for v2.1.
- The CNG bus is biased low (-19 %).
