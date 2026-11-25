# State-space model (S4D with a pointwise encoder) (`seq_ssm`)

River Bake-off, group A (sequence learners). Family: **State-space and recurrent models (S4D diagonal state space)** (`ssm` on the Model families page). Protocol: **ungauged** (forcing + static attributes only; never reads streamflow). numpy-only inference (`predict.py` -> `../common/seqlib.py` + `weights.npz`).

## The built-in assumption
Memory is linear and time-invariant: each of 48 channels passes through a fixed filter made of 16 decaying, oscillating exponentials (S4D), i.e. a bank of linear reservoirs with learned time constants from days to years, applied the same way whatever the weather. Nonlinearity happens only between the filters. That is the unit-hydrograph / linear-reservoir view of a catchment, stacked three deep. The released rung adds a pointwise nonlinear encoder before the first filter, so the day's inputs can be partitioned (rain vs snow, thresholds) before they enter the linear memory.

## Network
Linear 29 -> 48 channels, a residual pointwise MLP (48 -> 48 -> 48, GELU; added in r2), then 3 S4D layers (pre-LayerNorm; per channel 16 complex diagonal modes, S4D-Lin initialisation, zero-order hold with a learned step in 1e-3..1e-1, the 730-day kernel applied by FFT convolution, skip term D; then GELU and a GLU channel mix, residual), final LayerNorm, dropout 0.1 before the head. numpy builds the same kernel and convolves by FFT (length 2048, exact linear convolution).
Parameters: 30,438 (the four sequence arms hold 24.7k-26.9k).

## Shared with every sequence arm (only the core differs)
- Inputs: FlowCast's 11 scaled daily features (P/10, log1p P, PET/5, T_mean/T_min/T_max /10, SRAD/150, vp/1000, wind/3, sin/cos day of year) + FlowCast's 18-value static vector (batch["static"], standardised with the train bank mean/sd) repeated every day; target in units of the basin mean precipitation exp(static[11]); 730-day windows (365 warm-up + 365 target), inference in 365-day blocks each with its own 365-day warm-up (FlowCast). Causal: day t uses forcing up to day t only.
- Head: Linear(hidden -> 6): mean = softplus; 5/25/50/75/95 % quantiles = cumulative softplus (monotone); units of p_mean, times p_mean -> mm/day. Bands are raw (no dev widening).
- Loss: FlowCast's: NSE* on the mean with an input-only scale (Fu runoff ratio of the forcing aridity, w = 2.6) + the same-scaled pinball loss on the quantiles.
- Training data: the harness bank (12,000 simulated runs x 4 years, physics p1, FlowCast's recipe), random 730-day windows, the same rng seed so every arm sees the same windows in the same order; Adam, lr 2e-3 cosine to 1e-4, gradient clip 1.0, batch 32, last step used (no early stopping). Zero real labels.

## Ratchet
Rule: keep if sim val median NSE (all 800 frozen runs) > incumbent + 0.01 AND dev median NSE (52 dev basins, WY2001-2014) >= 0.95 x incumbent's. Sim val = harness `evaluate.py`, all 800 frozen runs; dev = 52 dev basins.

| Rung | Change | Steps | Sim val NSE | Dev NSE | Dev KGE | Dev FHV % | Dev cov50/90 | CPU h | Decision | Why |
|---|---|---|---|---|---|---|---|---|---|---|
| r0 | base: S4D (3 layers, 48 channels, 16 complex modes each), FlowCast's training recipe (3000 steps) | 3000 | 0.398 | 0.410 | 0.401 | -33.9 | 0.56/0.96 | 0.50 | **base** | first rung. |
| r1 | 2x training: 6000 steps | 6000 | 0.399 | 0.428 | 0.448 | -29.4 | 0.56/0.96 | 0.88 | **rejected** | sim +0.001 (0.399 vs 0.398): unlike the LSTM, TCN and transformer, the linear time-invariant SSM gains nothing from more training (dev 0.428). |
| r2 | pointwise nonlinear encoder before the SSM layers (residual MLP 48 -> 48 -> 48, GELU); 30.4k parameters; 6000 steps | 6000 | 0.422 | 0.435 | 0.430 | -30.1 | 0.58/0.97 | 0.93 | **released** | sim 0.422 vs incumbent r0 0.398 (+0.024; r1 shows the 6000 steps alone add nothing) and dev 0.435 >= 0.95 x 0.410: kept. Released (r3 did not beat it). A nonlinearity BEFORE the linear memory (e.g. rain vs snow, thresholds) is what the fixed filter lacked. |
| r3 | Mamba-lite: replace the S4D layers by selective state-space layers (input-dependent step dt and input/output maps B, C; scalar decay per head, 6 heads x 8, state 16; Mamba-2 style, gated output; no pre-MLP); 28.9k parameters; 6000 steps | 6000 | 0.415 | 0.429 | 0.436 | -29.5 | 0.61/0.97 | 1.39 | **rejected** | sim 0.415 vs incumbent r2 0.422 (-0.007; dev 0.429 vs 0.435). Selectivity beats the plain fixed filter (r1 0.399) but not the fixed filter with a nonlinear encoder in front. torch's chunked SSD equals numpy's day-by-day recurrence to 1.6e-15 (float64). |

## Released rung r2: dev gate (52 basins, WY2001-2014)
NSE 0.435, KGE 0.430, FHV -30.1 %, 50/90 % coverage 0.58/0.97, CRPS/mean flow 0.392. By class (median NSE): snow 0.56 (n 10), arid 0.19 (n 14), humid 0.59 (n 11), regulated 0.27 (n 5), large 0.27 (n 6). FlowCast v1.0 as released: 0.425 (snow 0.45, arid 0.18, humid 0.56, regulated 0.28).

## Compute
Training 3.69 single-thread CPU-hours in all (4 rungs; one thread (torch.set_num_threads(1)), time.process_time() of the training process, shared 2-core machine). Budget: about 4.

## numpy release check
numpy float64 (`predict.py`) vs the torch float32 checkpoint on 19 cases (16 sim val windows, 3 dev basins x 14 years): max |diff| 2.8e-05 (mean) and 7.2e-05 (quantiles) of the mean flow; against torch in float64 7.8e-15 / 1.9e-14.

## Limits
- Ungauged only: it never reads streamflow.
- Memory is limited to the 730-day window (each 365-day block gets its own 365-day warm-up).
- The S4D filter is the same for every day: how strongly a storm becomes flow cannot depend on how wet the basin already is except through the nonlinear layers between filters. The pointwise encoder (r2) fixes the input side only.
- Dev seed noise (measured on the LSTM control) is ~0.02 sd: dev differences under ~0.04 between arms are not meaningful.
- Bands are raw pinball quantiles (no dev widening).

## Notes
- Training cost per step: S4D about 0.5 s on one core (FFT convolutions), Mamba-lite 0.8 s; numpy inference ~0.017 s per 365-day block.
- numpy vs torch float32: ~3e-5 of the mean flow (torch's float32 complex exponentials over 730 lags); numpy equals torch float64 to 8e-15.
- The quick validation printed during training (first 200 val runs) is monitoring only; every decision uses harness evaluate.py on all 800 runs + the dev gate.
