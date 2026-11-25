# tab_gp: sparse Gaussian process (River Bake-off, arm B2)

**What it is.** A Gaussian process regression on daily rows: tomorrow's and today's rain, a snow proxy, wetness
memories and two climate indices go in; the GP returns a Gaussian mean and variance of the transformed flow
z = sqrt(qo / p_mean). Sparse (inducing-point) GP, because an exact GP on 200,000 rows is impossible on one core:
SGPR (Titsias 2009 collapsed variational bound) with 500 inducing inputs and an ARD squared-exponential kernel.
Inference is numpy only (`predict.py`, `gp.npz`). Protocol: **ungauged**.

**Built-in assumption.** Flow is a smooth function of the inputs: days with similar recent water, snow and climate
have similar flow, with similarity measured by one length scale per input (ARD), and the noise is Gaussian with one
variance (in sqrt space). Uncertainty comes for free from the posterior.

**What was scoped** (the brief allows it): 20 of tab_gbm's 47 daily features, no basin attributes beyond aridity
and snow fraction, M = 500 inducing points, hyperparameters and inducing inputs optimised on 20,000 random rows, the
posterior accumulated over 200,000 rows (of tab_gbm's 1.2 M-row sim-train table; runs 11,400+ held out). Features:
P(t+1), P(t), liquid input W(t), W(t-1), melt(t), EMAs of W over 3/10/30/90/365 days, the 50/300 mm bucket fill
fractions and spills, a 60-day slow store, EMA10 T, EMA30 PET, log1p SWE proxy, log aridity, snow fraction.

**Outputs.** Quantiles = the Gaussian 5/25/50/75/95 % points of z, mapped back by max(z, 0)^2 x p_mean; the mean is
the exact E[max(z, 0)^2] of that Gaussian (latent + noise variance) x p_mean.

**Ratchet** (rule: sim val median NSE > incumbent + 0.01 and dev NSE >= 0.95 x incumbent; all 800 val runs, 52 dev basins):

| Rung | Change | Sim val NSE | Dev NSE | Decision |
|---|---|---|---|---|
| r0 | base: 14 features, M 500, sqrt target | 0.291 | 0.364 | base |
| r1 | log target (log-normal back-transform) | 0.219 | 0.278 | rejected |
| r2 | **20 features** | **0.313** | **0.369** | **released** (+0.022) |
| r3 | M 1000 inducing inputs | 0.308 | 0.352 | rejected |

**Released r2 on the dev gate** (52 basins, WY2001-2014): NSE 0.369, KGE 0.364, FHV -42.2 %, 50/90 % coverage
0.72 / 0.96, CRPS / mean flow 0.419. Paired bootstrap of the NSE median difference: vs FlowCast v1.0 -0.055
(95 % CI -0.122..+0.023, a tie), vs U-dev -0.097 (-0.176..-0.049, a loss), vs NWM and GR4J (losses). By class:
snow 0.61 (the best of all arms here; FlowCast 0.45), arid 0.10, humid 0.50, regulated 0.26, large 0.21.

**What its behaviour suggests.**
- The smoothness assumption costs peaks: FHV -42 % (the trees: -31 %). The stationary kernel and one noise level
  average flood days with ordinary wet days; high flows are under-predicted.
- One Gaussian noise variance in sqrt space gives bands that are too wide in the middle (50 % band covers 72 %) but
  right at 90 % (0.96); the sqrt transform beat the log transform by a wide margin (r1), whose back-transform
  inflates means at low flows and damps them at peaks.
- Doubling the inducing points did not help (r3): with 20 inputs the posterior is limited by the kernel's
  smoothness and the noise in the targets, not by the number of basis functions.
- It does best where flow is a smooth function of a snowpack proxy (snow basins 0.61) and worst in arid basins,
  where flow is a rare, thresholded response.
- Learned length scales (standardised inputs): melt 15 and the 300 mm bucket spill 44 (nearly ignored); PET EMA30
  2.7, EMA10 T 3.1, the 50 mm bucket 3.2 and aridity 3.1 (most used).

**Compute.** 0.94 single-thread CPU hours for all rungs. Inference ~0.55 CPU s per dev basin (14 years); exam
estimate ~10 min.

**Limits.** A scoped model (subset of features, 200k of 1.2M rows, 500 inducing points); homoscedastic noise in
sqrt space; no basin attributes beyond two climate indices; peaks under-predicted.

**Files.** `predict.py` (numpy SGPR predictive), `feats.py` (shared feature code; the GP reads its 20 columns),
`gp.npz` (Z, length scales, variances, alpha, W = Kmm^-1 - Sigma^-1, standardisation), `ratchet.json`. Training
code: `arms_tab/src/train_gp.py`, `build_tab.py`.
