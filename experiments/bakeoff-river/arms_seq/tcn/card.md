# TCN (dilated causal 1-D CNN) (`seq_tcn`)

River Bake-off, group A (sequence learners). Family: **1-D convolutional network (temporal convolutional network)** (`cnn1d` on the Model families page). Protocol: **ungauged** (forcing + static attributes only; never reads streamflow). numpy-only inference (`predict.py` -> `../common/seqlib.py` + `weights.npz`).

## The built-in assumption
The flow today is a learned function of the weather at fixed lags. A stack of causal convolutions reads the last 1, 2, 4 ... 256 days at fixed offsets (receptive field 1,023 days, more than the 730-day window), the same way on every date, and builds longer features with depth. There is no explicit store carried forward: memory is a fixed-lag pattern detector, like a bank of learned unit hydrographs with nonlinear mixing between layers.

## Network
Linear 29 -> 26 channels, then 9 residual blocks (pre-LayerNorm, causal convolution with kernel 3 and dilation 1, 2, 4, ..., 256 days, GELU, 1x1 channel mix), final LayerNorm, dropout 0.1 before the head.
Parameters: 26,266 (the four sequence arms hold 24.7k-26.9k).

## Shared with every sequence arm (only the core differs)
- Inputs: FlowCast's 11 scaled daily features (P/10, log1p P, PET/5, T_mean/T_min/T_max /10, SRAD/150, vp/1000, wind/3, sin/cos day of year) + FlowCast's 18-value static vector (batch["static"], standardised with the train bank mean/sd) repeated every day; target in units of the basin mean precipitation exp(static[11]); 730-day windows (365 warm-up + 365 target), inference in 365-day blocks each with its own 365-day warm-up (FlowCast). Causal: day t uses forcing up to day t only.
- Head: Linear(hidden -> 6): mean = softplus; 5/25/50/75/95 % quantiles = cumulative softplus (monotone); units of p_mean, times p_mean -> mm/day. Bands are raw (no dev widening).
- Loss: FlowCast's: NSE* on the mean with an input-only scale (Fu runoff ratio of the forcing aridity, w = 2.6) + the same-scaled pinball loss on the quantiles.
- Training data: the harness bank (12,000 simulated runs x 4 years, physics p1, FlowCast's recipe), random 730-day windows, the same rng seed so every arm sees the same windows in the same order; Adam, lr 2e-3 cosine to 1e-4, gradient clip 1.0, batch 32, last step used (no early stopping). Zero real labels.

## Ratchet
Rule: keep if sim val median NSE (all 800 frozen runs) > incumbent + 0.01 AND dev median NSE (52 dev basins, WY2001-2014) >= 0.95 x incumbent's. Sim val = harness `evaluate.py`, all 800 frozen runs; dev = 52 dev basins.

| Rung | Change | Steps | Sim val NSE | Dev NSE | Dev KGE | Dev FHV % | Dev cov50/90 | CPU h | Decision | Why |
|---|---|---|---|---|---|---|---|---|---|---|
| r0 | base: FlowCast's training recipe (3000 steps) | 3000 | 0.401 | 0.412 | 0.438 | -35.8 | 0.57/0.96 | 0.26 | **base** | first rung; same recipe as the LSTM control's r0. |
| r1 | 2x training: 6000 steps (cosine over 6000) | 6000 | 0.416 | 0.420 | 0.433 | -33.3 | 0.59/0.97 | 0.55 | **released** | sim +0.015 over r0 (paired bootstrap +0.001..+0.032) and dev 0.420 >= 0.95 x 0.412: kept. Released (r2 did not beat it). |
| r2 | gated activations (WaveNet tanh x sigmoid) instead of GELU; channels 26 -> 20 to keep 26.9k parameters; 6000 steps | 6000 | 0.405 | 0.425 | 0.444 | -33.8 | 0.58/0.97 | 0.45 | **rejected** | sim 0.405 vs 0.416 (-0.012) although dev 0.425: multiplicative gating did not help the convolution. |

## Released rung r1: dev gate (52 basins, WY2001-2014)
NSE 0.420, KGE 0.433, FHV -33.3 %, 50/90 % coverage 0.59/0.97, CRPS/mean flow 0.391. By class (median NSE): snow 0.54 (n 10), arid 0.19 (n 14), humid 0.58 (n 11), regulated 0.28 (n 5), large 0.32 (n 6). FlowCast v1.0 as released: 0.425 (snow 0.45, arid 0.18, humid 0.56, regulated 0.28).

## Compute
Training 1.25 single-thread CPU-hours in all (3 rungs; one thread (torch.set_num_threads(1)), time.process_time() of the training process, shared 2-core machine). Budget: about 4.

## numpy release check
numpy float64 (`predict.py`) vs the torch float32 checkpoint on 19 cases (16 sim val windows, 3 dev basins x 14 years): max |diff| 4.1e-06 (mean) and 9.1e-06 (quantiles) of the mean flow; against torch in float64 6.3e-15 / 1.5e-14.

## Limits
- Ungauged only: it never reads streamflow.
- Memory is a fixed set of lags inside the 730-day window (each 365-day block gets its own 365-day warm-up): no multi-year carry-over, and the response to a storm is shaped by learned kernels rather than by a store that depends on how wet the basin already is (that dependence can only come from the nonlinear mixing between layers).
- Dev seed noise (measured on the LSTM control) is ~0.02 sd: dev differences under ~0.04 between arms are not meaningful.
- Bands are raw pinball quantiles (no dev widening). Its 90 % band is the widest of the four sequence arms on sim val (coverage 0.97).

## Notes
- Training cost per step is about 0.3 s on one core (LSTM 0.11 s, S4D 0.5-0.6 s, daily-token transformer 0.7 s); numpy inference ~0.015 s per 365-day block.
- The quick validation printed during training (first 200 val runs) is monitoring only; every decision uses harness evaluate.py on all 800 runs + the dev gate.
