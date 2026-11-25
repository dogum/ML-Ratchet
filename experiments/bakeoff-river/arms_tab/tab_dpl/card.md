# tab_dpl: "decides through physics" (River Bake-off, arm B3)

**What it is.** Differentiable parameter learning in the style of Tsai et al. (2021). A small neural network
(18 -> 64 -> 64 -> 17, SiLU) reads a basin's 18-value static vector and outputs the parameters of a conceptual
hydrological model. The flow itself always comes from that model: WaterSim's HBV runoff routine (9 parameters),
its degree-day snow routine over 5 elevation bands (6 parameters; CFR 0.05 and CWH 0.1 fixed), its gamma channel
unit hydrograph (channel velocity) and a precipitation multiplier PMULT applied to the forcing. Protocol: **ungauged**.

**Built-in assumption.** Every basin behaves like an HBV bucket model; only its parameters differ, and those
parameters are a smooth function of the basin's attributes and climate. The network never predicts flow directly,
so mass balance, recessions, snow storage and routing are guaranteed by the simulator.

**How it was trained.** Through the simulator. `arms_tab/src/hbv_torch.py` re-implements WaterSim's HBV +
degree-day snow + elevation bands + snow-covered-area PET reduction + triangular and gamma unit hydrographs + the
irrigation module in torch. Check against the harness's vendored WaterSim on 30 cases (6 sim val windows, 4 dev
basins, 3 random parameter sets each; float64): max |difference| **7.1e-14 mm/day**, median 1.4e-14
(`work/verify_hbv.json`). Gradients flow from the loss through 730 simulated days into the network weights.
Data: sim TRAIN bank only (runs 0-11,399; 11,400+ held out for monitoring and the error bands): random windows of
365 warm-up rows (no gradient) + 730 days, batch 64, Adam 2e-3 -> 1e-4 (cosine), 2,000 iterations. Target: qo.

**Inference.** `predict.py`: numpy MLP -> parameters -> `wsim.run_batch` = `watersim.simulate(model='hbv',
snow='degree_day', routing='uh')` from the harness's vendored WaterSim (spec from the batch attributes; the 365
warm-up rows are the spin-up; PET from the forcing; no reservoirs, since their specs are not inputs). Mean = the
simulated flow. Bands: the relative residual (qo - q) / (q + 0.05 p_mean) on 600 held-out sim-train runs, binned by
flow level (10 bins), its 5/25/50/75/95 % quantiles (`dpl_weights.npz`: edges, rq). numpy MLP = torch MLP to 9e-16.
`numpy_only: false` (WaterSim is numba-compiled; the arm IS a WaterSim run). ~0.2 CPU s per dev basin (14 years).

**For arm C.** `params_of(batch)` returns the 17 parameters (+ CFR, CWH) for any batch: the attribute -> WaterSim
parameter mapping, reusable as an EnKF prior (`armlib.load_predict(<this folder>)` returns the module).

**Ratchet** (rule: sim val median NSE > incumbent + 0.01 and dev NSE >= 0.95 x incumbent; all 800 val runs, 52 dev basins):

| Rung | Change | Sim val NSE | Dev NSE | Decision |
|---|---|---|---|---|
| r0 | base, bounded NSE loss x/(1+x), x = NSE* per run | 0.354 | 0.365 | base |
| r1 | **loss = MSE of flow / p_mean** (input scale, no per-run variance weighting) | **0.414** | **0.383** | **released** (+0.060) |
| r2 | warm start from r1, effective batch 128, lr 3e-4 -> 3e-5, 500 its | 0.418 | 0.388 | rejected (+0.004) |

Two pilot runs of the base were aborted and are logged in the compute: with the usual NSE* loss and with
log(1 + NSE*), the network cut PMULT to ~0.78 and the volume to ~0.75 of the observed, and its median NSE fell
below that of HBV with WaterSim's default parameters.

**Released r1 on the dev gate** (52 basins, WY2001-2014): NSE 0.383, KGE 0.403, FHV -27.8 %, 50/90 % coverage
0.53 / 0.96, CRPS / mean flow 0.401. Paired bootstrap of the NSE median difference: vs FlowCast v1.0 -0.042
(95 % CI -0.075..+0.007, a tie), vs U-dev regionalisation -0.084 (-0.188..-0.018, a loss), vs NWM -0.190 (loss),
vs GR4J calibrated per basin -0.203 (loss). By class: snow 0.54, arid 0.19, humid 0.56, regulated 0.27, large 0.24.

**What its behaviour suggests.**
- The simulator carries most of the skill: HBV with WaterSim's DEFAULT parameters already scores a median NSE of
  ~0.33-0.39 on held-out sim runs; the learned mapping adds a few hundredths, and the released arm reaches 0.414 on
  sim val, close to the LSTM (0.426), with 6k weights and a guaranteed water balance.
- Calibrating the same HBV separately for each of 48 held-out runs reaches a median NSE of 0.91 (in-sample, on
  the 730 fitted days): the structure is not the
  bottleneck; the attributes pin down the parameters only loosely (the bank's latent factors and parameter scatter
  are invisible to any ungauged method).
- The loss matters more than anything else here: a per-run variance-normalised loss (NSE*) rewards shrinking
  volumes when the runoff amplitude is uncertain, because dry runs dominate it. Scaling by an input (p_mean)
  instead removed the bias (+0.06 sim NSE).
- On real dev data it trails the LSTM and the gradient-boosted trees (0.383 vs 0.425 / 0.421) mostly in large and
  regulated basins (no reservoirs, a single lumped HBV per basin).

**Learned parameters** (5 / 50 / 95 % over the sim val basins): FC 133 / 155 / 273 mm, LP 0.43 / 0.62 / 0.82, BETA
~1.0-1.2, PERC 0.24 / 1.1 / 3.5 mm/d, K2 0.015 / 0.044 / 0.072 /d, MAXBAS 1.1-1.5 d, CFMAX 1.5 / 2.0 / 2.9 mm/C/d,
TLAPSE -8.6 / -7.6 / -5.9 C/km, PMULT 0.62 / 0.87 / 1.20, CH_VEL 0.77 / 0.95 / 1.2 m/s.

**Compute.** 2.25 single-thread CPU hours for all rungs and pilots (see ratchet.json). Exam estimate ~5 min.

**Limits.** One model structure (HBV + degree-day) for every basin; no reservoirs (17 % of sim runs and the
regulated real basins have them); the bands are a fixed residual model by flow level, not a parameter ensemble;
attributes explain only part of the parameter variation, so ungauged skill is capped (see the calibration ceiling).

**Files.** `predict.py`, `wsim.py` (spec + WaterSim call), `physarm.py` (bounds, numpy MLP, bands),
`dpl_weights.npz`, `ratchet.json`. Training code (not part of the artefact): `arms_tab/src/hbv_torch.py`,
`verify_hbv.py`, `train_dpl.py`, `train_dpl_net.py`, `export_phys.py`.
