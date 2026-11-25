# Transformer (causal encoder over daily tokens) (`seq_transformer`)

River Bake-off, group A (sequence learners). Family: **Attention and transformers** (`attn` on the Model families page). Protocol: **ungauged** (forcing + static attributes only; never reads streamflow). numpy-only inference (`predict.py` -> `../common/seqlib.py` + `weights.npz`).

## The built-in assumption
Any earlier day can be looked up directly. Each day's output is built from learned weighted averages (attention) over all earlier days of the 730-day window, the weights set by content and by a sinusoidal position encoding. There is no built-in store that fills and drains and no built-in fading of memory: memory is a lookup, not an integrator. Causal mask: day t attends to days <= t only.

## Network
Linear 29 -> 32, + sinusoidal position encoding (positions 0-729), 3 pre-LayerNorm blocks of causal multi-head self-attention (released r3: 2 heads of size 16; r0/r1: 4 heads of size 8) and a GELU feed-forward layer (64), final LayerNorm, dropout 0.1 before the head. r2 (rejected) used causal 7-day patches instead of daily tokens.
Parameters: 26,854 (the four sequence arms hold 24.7k-26.9k).

## Shared with every sequence arm (only the core differs)
- Inputs: FlowCast's 11 scaled daily features (P/10, log1p P, PET/5, T_mean/T_min/T_max /10, SRAD/150, vp/1000, wind/3, sin/cos day of year) + FlowCast's 18-value static vector (batch["static"], standardised with the train bank mean/sd) repeated every day; target in units of the basin mean precipitation exp(static[11]); 730-day windows (365 warm-up + 365 target), inference in 365-day blocks each with its own 365-day warm-up (FlowCast). Causal: day t uses forcing up to day t only.
- Head: Linear(hidden -> 6): mean = softplus; 5/25/50/75/95 % quantiles = cumulative softplus (monotone); units of p_mean, times p_mean -> mm/day. Bands are raw (no dev widening).
- Loss: FlowCast's: NSE* on the mean with an input-only scale (Fu runoff ratio of the forcing aridity, w = 2.6) + the same-scaled pinball loss on the quantiles.
- Training data: the harness bank (12,000 simulated runs x 4 years, physics p1, FlowCast's recipe), random 730-day windows, the same rng seed so every arm sees the same windows in the same order; Adam, lr 2e-3 cosine to 1e-4, gradient clip 1.0, batch 32, last step used (no early stopping). Zero real labels.

## Ratchet
Rule: keep if sim val median NSE (all 800 frozen runs) > incumbent + 0.01 AND dev median NSE (52 dev basins, WY2001-2014) >= 0.95 x incumbent's. Sim val = harness `evaluate.py`, all 800 frozen runs; dev = 52 dev basins.

| Rung | Change | Steps | Sim val NSE | Dev NSE | Dev KGE | Dev FHV % | Dev cov50/90 | CPU h | Decision | Why |
|---|---|---|---|---|---|---|---|---|---|---|
| r0 | base: daily tokens, FlowCast's training recipe (3000 steps) | 3000 | 0.364 | 0.362 | 0.365 | -40.4 | 0.62/0.96 | 0.59 | **base** | first rung; learns slowly (quick val 0.30 / 0.34 / 0.39 at 1k / 2k / 3k steps vs the LSTM's 0.38 / 0.40 / 0.45). |
| r1 | 2x training: 6000 steps | 6000 | 0.401 | 0.389 | 0.393 | -33.5 | 0.60/0.97 | 1.27 | **kept** | sim +0.037 over r0 (paired bootstrap +0.016..+0.053) and dev 0.389 >= 0.95 x 0.362: kept (superseded by r3). |
| r2 | causal 7-day patches (105 tokens, 2 blocks): each day reads the attention output of all earlier complete weeks + a depthwise causal conv over its own last 7 days; 25.8k parameters; 6000 steps | 6000 | 0.341 | 0.353 | 0.384 | -36.2 | 0.56/0.96 | 0.14 | **rejected** | sim 0.341 vs incumbent r1 0.401: weekly tokens lose more than they save (10x cheaper per step). |
| r3 | r1 with 2x training (12000 steps, cosine over 12000) and 2 attention heads of size 16 instead of 4 of size 8 (same 26.9k parameters; halves the attention cost so the run fits the 4 CPU-hour budget) | 12000 | 0.420 | 0.411 | 0.406 | -30.1 | 0.61/0.97 | 1.55 | **released** | sim 0.4205 vs incumbent r1 0.4008 (+0.020; paired bootstrap +0.002..+0.032) and dev 0.411 >= 0.95 x 0.389 (dev +0.022, CI -0.028..+0.047): kept. Released (last rung). Two changes in one rung (steps and heads), so the gain cannot be split between them; the LSTM's 12000-step rung gained only +0.003, so most of it is probably the transformer's slower learning catching up. |

## Released rung r3: dev gate (52 basins, WY2001-2014)
NSE 0.411, KGE 0.406, FHV -30.1 %, 50/90 % coverage 0.61/0.97, CRPS/mean flow 0.400. By class (median NSE): snow 0.51 (n 10), arid 0.20 (n 14), humid 0.59 (n 11), regulated 0.32 (n 5), large 0.30 (n 6). FlowCast v1.0 as released: 0.425 (snow 0.45, arid 0.18, humid 0.56, regulated 0.28).

## Compute
Training 3.55 single-thread CPU-hours in all (4 rungs; one thread (torch.set_num_threads(1)), time.process_time() of the training process, shared 2-core machine). Budget: about 4.

## numpy release check
numpy float64 (`predict.py`) vs the torch float32 checkpoint on 19 cases (16 sim val windows, 3 dev basins x 14 years): max |diff| 1.2e-05 (mean) and 3.3e-05 (quantiles) of the mean flow; against torch in float64 7.4e-07 / 1.9e-06.

## Limits
- Ungauged only: it never reads streamflow.
- Memory is the 730-day window (each 365-day block gets its own 365-day warm-up). Attention cost grows with the square of the window: numpy inference ~0.07 s per block, about 15x the LSTM (the exam estimate is in the registry).
- It is the slowest learner of the four: it needed 12000 steps (4x FlowCast's recipe) to reach the other arms' 6000-step level; quick validation (200 runs) was flat from 8k to 12k steps (0.449, 0.441, 0.444, 0.453, 0.446), so its standing here is mostly about attention under this budget rather than an unfinished run, but a longer schedule was not tried.
- Dev seed noise (measured on the LSTM control) is ~0.02 sd: dev differences under ~0.04 between arms are not meaningful.
- Bands are raw pinball quantiles (no dev widening).

## Notes
- Training cost per step about 0.7 s on one core with 4 heads of size 8 (r0, r1) and 0.47 s with 2 heads of size 16 (r3), mostly PyTorch's CPU flash attention; the patched r2 was 0.08 s.
- The quick validation printed during training (first 200 val runs) is monitoring only; every decision uses harness evaluate.py on all 800 runs + the dev gate.
