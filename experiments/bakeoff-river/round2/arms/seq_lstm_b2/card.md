# seq_lstm_b2: LSTM (FlowCast's network) retrained on bank b2 (River Bake-off round 2)

**What it is.** `seq_lstm` from round 1 with its released recipe and hyperparameters unchanged (seq_lstm r1: H 64, dropout 0.1, 6,000 steps x 32 windows, lr 0.002 cosine, seed 0),
trained on bank **b2** instead of p1. Ungauged. numpy only at inference. Zero real labels in training.

**Bank b2** = FlowCast's p1 recipe with one dev-calibrated change (round2/PREREG.md, rung b2b): the ERA5-Land
water-yield multiplier m is drawn from a constant prior, log m = -0.026 + N(0, 0.153), instead of p1's regression on
elevation, snow fraction and wetness. The constant form won the pre-registered leave-one-basin-out test on the 52 dev
basins (robust RMSE 0.153 vs 0.159 for p1's form). Dry basins no longer get m well below 1, so the bank holds less
"lost" water there: mean simulated flow 1.556 -> 1.411 mm/day.

**Rung** (rule: sim val on b2's frozen val > v1.0 + 0.01 and dev >= 0.95 x v1.0):

| | Sim val (b2 frozen) | Dev NSE | Dev KGE | Dev by class (snow / arid / humid / regulated / large) |
|---|---|---|---|---|
| v1.0 (`seq_lstm`, bank p1) | 0.394 | 0.441 | 0.438 | 0.535 / 0.182 / 0.573 / 0.279 / 0.307 |
| b2 (this arm) | 0.445 | 0.439 | 0.450 | 0.589 / -0.1 / 0.569 / 0.311 / -0.1 |

Decision: **kept** (sim val on b2's frozen set 0.4449 vs v1.0 0.3935 (+0.0514 > 0.01) and dev 0.4386 >= 0.95 x 0.4408: kept). Sim val on p1's frozen set: 0.408.
The sim criterion mostly measures the bank shift itself: a net trained on b2 is scored on b2's own val set,
which the v1.0 net never saw. The dev criterion is the one that speaks to real basins, and it is a non-inferiority
check (>= 95 %), not a gain.

**Dev, other numbers.** FHV -27.9 %, 50 / 90 % coverage 0.61 / 0.96, CRPS / mean
flow 0.365. Paired against the dev references: udev -0.028 tie, uk12 -0.019 tie, gr4j -0.147 loss, nwm -0.134 loss, clim +0.347 win.

**Compute.** 0.317 single-thread CPU-hours (this rung).

**Limits.** One seed (the round-1 recipe's seed 0); the LSTM control's dev seed spread on this bank was 0.439-0.443.
Dev is 52 basins; class medians rest on 5-14 basins each.
