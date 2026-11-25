# LSTM (FlowCast's network, retrained on the bake-off bank) (`seq_lstm`)

River Bake-off, group A (sequence learners). Family: **State-space and recurrent models (LSTM)** (`ssm` on the Model families page). Protocol: **ungauged** (forcing + static attributes only; never reads streamflow). numpy-only inference (`predict.py` -> `../common/seqlib.py` + `weights.npz`).

## The built-in assumption
A small learned state is carried from day to day and updated through gates: each day the network decides how much of each memory cell to keep (forget gate), what to add (input gate) and what to release into the output. That is a learned version of a bucket model: snow, soil and groundwater stores that fill and drain. It is FlowCast v1.0's choice and the in-bake-off control: same network, same loss, same bank as the three other sequence arms.

## Network
One LSTM layer with 64 units over the 29 daily channels, dropout 0.1 before the head: exactly FlowCast v1.0's network (`models/flowcast/training/src/model.py`), retrained from scratch on the harness bank with FlowCast's recipe.
Parameters: 24,710 (the four sequence arms hold 24.7k-26.9k).

## Shared with every sequence arm (only the core differs)
- Inputs: FlowCast's 11 scaled daily features (P/10, log1p P, PET/5, T_mean/T_min/T_max /10, SRAD/150, vp/1000, wind/3, sin/cos day of year) + FlowCast's 18-value static vector (batch["static"], standardised with the train bank mean/sd) repeated every day; target in units of the basin mean precipitation exp(static[11]); 730-day windows (365 warm-up + 365 target), inference in 365-day blocks each with its own 365-day warm-up (FlowCast). Causal: day t uses forcing up to day t only.
- Head: Linear(hidden -> 6): mean = softplus; 5/25/50/75/95 % quantiles = cumulative softplus (monotone); units of p_mean, times p_mean -> mm/day. Bands are raw (no dev widening).
- Loss: FlowCast's: NSE* on the mean with an input-only scale (Fu runoff ratio of the forcing aridity, w = 2.6) + the same-scaled pinball loss on the quantiles.
- Training data: the harness bank (12,000 simulated runs x 4 years, physics p1, FlowCast's recipe), random 730-day windows, the same rng seed so every arm sees the same windows in the same order; Adam, lr 2e-3 cosine to 1e-4, gradient clip 1.0, batch 32, last step used (no early stopping). Zero real labels.

## Ratchet
Rule: keep if sim val median NSE (all 800 frozen runs) > incumbent + 0.01 AND dev median NSE (52 dev basins, WY2001-2014) >= 0.95 x incumbent's. Sim val = harness `evaluate.py`, all 800 frozen runs; dev = 52 dev basins.

| Rung | Change | Steps | Sim val NSE | Dev NSE | Dev KGE | Dev FHV % | Dev cov50/90 | CPU h | Decision | Why |
|---|---|---|---|---|---|---|---|---|---|---|
| r0 | FlowCast r1b recipe exactly (3000 steps x 32 windows) | 3000 | 0.410 | 0.391 | 0.420 | -34.1 | 0.56/0.95 | 0.10 | **base** | the control: FlowCast's network and recipe on the shared bank. Scores below FlowCast v1.0 as released (sim 0.426, dev 0.425) at the same recipe; 3 seeds agree on sim (0.409-0.410), so the gap is not seed noise (see notes). |
| r1 | 2x training: 6000 steps (cosine over 6000) | 6000 | 0.425 | 0.441 | 0.438 | -29.5 | 0.57/0.96 | 0.19 | **released** | sim +0.016 over r0 (> 0.01 margin; seed noise on sim is ~0.001) and dev 0.441 >= 0.95 x 0.391: kept. Released (r2 did not beat it). |
| r2 | 4x training: 12000 steps | 12000 | 0.428 | 0.440 | 0.442 | -26.4 | 0.58/0.96 | 0.40 | **rejected** | sim +0.003 over r1 (< 0.01 margin; paired bootstrap -0.011..+0.017), dev 0.440 vs 0.441: no gain from more training beyond 6000 steps. |

Seed re-runs of r0 (noise check, not rungs): seed1: sim 0.410, dev 0.431; seed2: sim 0.409, dev 0.404.

## Released rung r1: dev gate (52 basins, WY2001-2014)
NSE 0.441, KGE 0.438, FHV -29.5 %, 50/90 % coverage 0.57/0.96, CRPS/mean flow 0.389. By class (median NSE): snow 0.54 (n 10), arid 0.18 (n 14), humid 0.57 (n 11), regulated 0.28 (n 5), large 0.31 (n 6). FlowCast v1.0 as released: 0.425 (snow 0.45, arid 0.18, humid 0.56, regulated 0.28).

## Compute
Training 0.87 single-thread CPU-hours in all (3 rungs + 2 seed re-runs (noise check); one thread (torch.set_num_threads(1)), time.process_time() of the training process, shared 2-core machine). Budget: about 4.

## numpy release check
numpy float64 (`predict.py`) vs the torch float32 checkpoint on 19 cases (16 sim val windows, 3 dev basins x 14 years): max |diff| 3.2e-06 (mean) and 4.9e-06 (quantiles) of the mean flow; against torch in float64 3.2e-15 / 7.9e-15.

## Limits
- Ungauged only: it never reads streamflow, so it cannot correct a basin's own water balance (ERA5-Land undercatch, losses). The water-yield multiplier p1 is learned only as a prior from attributes.
- Memory is limited to the 365-day warm-up of each block (FlowCast's design): multi-year groundwater carry-over is not represented.
- Dev seed noise is large: three seeds of r0 gave dev 0.391, 0.431, 0.404 on the same sim score (0.409-0.410). Dev differences under ~0.04 between arms are not meaningful.
- Bands are raw pinball quantiles; FlowCast widened its 90 % band x1.08 on dev, this arm does not (arm C's conformal layer calibrates every arm).

## Notes
- Retraining FlowCast's exact recipe (3000 steps) on the harness bank gave sim 0.410 (3 seeds: 0.410, 0.410, 0.409) against 0.426 for the released FlowCast v1.0 weights on the same frozen set. Candidate causes, not resolved: the harness train bank was built with the repo's latest WaterSim (6 files differ from the copy FlowCast used; the val set reproduces FlowCast's), and torch initialisation/dropout streams. At 6000 steps the retrained network matches FlowCast v1.0 on sim (0.4255 vs 0.426) and is higher on dev (0.441 vs 0.425; within dev seed noise).
- The quick validation printed during training (first 200 val runs) is monitoring only; every decision uses harness evaluate.py on all 800 runs + the dev gate.
