# seq_ssm_b2: S4D state space + MLP encoder retrained on bank b2 (River Bake-off round 2)

**What it is.** `seq_ssm` from round 1 with its released recipe and hyperparameters unchanged (seq_ssm r2: d 48, N 16, 3 layers, pointwise MLP encoder 48, 6,000 steps x 32, seed 0),
trained on bank **b2** instead of p1. Ungauged. numpy only at inference. Zero real labels in training.

**Bank b2** = FlowCast's p1 recipe with one dev-calibrated change (round2/PREREG.md, rung b2b): the ERA5-Land
water-yield multiplier m is drawn from a constant prior, log m = -0.026 + N(0, 0.153), instead of p1's regression on
elevation, snow fraction and wetness. The constant form won the pre-registered leave-one-basin-out test on the 52 dev
basins (robust RMSE 0.153 vs 0.159 for p1's form). Dry basins no longer get m well below 1, so the bank holds less
"lost" water there: mean simulated flow 1.556 -> 1.411 mm/day.

**Rung** (rule: sim val on b2's frozen val > v1.0 + 0.01 and dev >= 0.95 x v1.0):

| | Sim val (b2 frozen) | Dev NSE | Dev KGE | Dev by class (snow / arid / humid / regulated / large) |
|---|---|---|---|---|
| v1.0 (`seq_ssm`, bank p1) | 0.383 | 0.435 | 0.430 | 0.565 / 0.192 / 0.594 / 0.272 / 0.267 |
| b2 (this arm) | 0.424 | 0.432 | 0.437 | 0.584 / -0.133 / 0.591 / 0.347 / -0.19 |

Decision: **kept** (sim val on b2's frozen set 0.4238 vs v1.0 0.3828 (+0.0410 > 0.01) and dev 0.4316 >= 0.95 x 0.4349: kept). Sim val on p1's frozen set: 0.403.
The sim criterion mostly measures the bank shift itself: a net trained on b2 is scored on b2's own val set,
which the v1.0 net never saw. The dev criterion is the one that speaks to real basins, and it is a non-inferiority
check (>= 95 %), not a gain.

**Dev, other numbers.** FHV -28.4 %, 50 / 90 % coverage 0.56 / 0.97, CRPS / mean
flow 0.373. Paired against the dev references: udev -0.035 tie, uk12 -0.026 tie, gr4j -0.154 loss, nwm -0.141 loss, clim +0.340 win.

**Compute.** 1.245 single-thread CPU-hours (this rung).

**Limits.** One seed (the round-1 recipe's seed 0); the LSTM control's dev seed spread on this bank was 0.439-0.443.
Dev is 52 basins; class medians rest on 5-14 basins each.
