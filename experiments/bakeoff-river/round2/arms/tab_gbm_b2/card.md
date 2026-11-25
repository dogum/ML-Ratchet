# tab_gbm_b2: Gradient-boosted trees (LightGBM -> numpy trees) retrained on bank b2 (River Bake-off round 2)

**What it is.** `tab_gbm` from round 1 with its released recipe and hyperparameters unchanged (tab_gbm r2: 47 features (feats.py), 1 window x 100 days per train run (1.2 M rows), L2 mean booster (63 leaves, lr 0.05, early stop on train runs >= 11,400) + 5 quantile boosters (31 leaves, lr 0.1); no monotone constraints. LightGBM 4.7.0 here (round 1: 4.6.0); inference walks the exported numpy trees),
trained on bank **b2** instead of p1. Ungauged. numpy only at inference. Zero real labels in training.

**Bank b2** = FlowCast's p1 recipe with one dev-calibrated change (round2/PREREG.md, rung b2b): the ERA5-Land
water-yield multiplier m is drawn from a constant prior, log m = -0.026 + N(0, 0.153), instead of p1's regression on
elevation, snow fraction and wetness. The constant form won the pre-registered leave-one-basin-out test on the 52 dev
basins (robust RMSE 0.153 vs 0.159 for p1's form). Dry basins no longer get m well below 1, so the bank holds less
"lost" water there: mean simulated flow 1.556 -> 1.411 mm/day.

**Rung** (rule: sim val on b2's frozen val > v1.0 + 0.01 and dev >= 0.95 x v1.0):

| | Sim val (b2 frozen) | Dev NSE | Dev KGE | Dev by class (snow / arid / humid / regulated / large) |
|---|---|---|---|---|
| v1.0 (`tab_gbm`, bank p1) | 0.366 | 0.421 | 0.444 | 0.56 / 0.202 / 0.613 / 0.365 / 0.325 |
| b2 (this arm) | 0.415 | 0.430 | 0.439 | 0.58 / -0.152 / 0.565 / 0.39 / -0.208 |

Decision: **kept** (sim val on b2's frozen set 0.4152 vs v1.0 0.3659 (+0.0492 > 0.01) and dev 0.4303 >= 0.95 x 0.4209: kept). Sim val on p1's frozen set: None.
The sim criterion mostly measures the bank shift itself: a net trained on b2 is scored on b2's own val set,
which the v1.0 net never saw. The dev criterion is the one that speaks to real basins, and it is a non-inferiority
check (>= 95 %), not a gain.

**Dev, other numbers.** FHV -29.3 %, 50 / 90 % coverage 0.55 / 0.94, CRPS / mean
flow 0.388. Paired against the dev references: udev -0.036 tie, uk12 -0.027 tie, gr4j -0.156 loss, nwm -0.142 loss, clim +0.338 win.

**Compute.** 0.097 single-thread CPU-hours (this rung).

**Limits.** One seed (the round-1 recipe's seed 0); the LSTM control's dev seed spread on this bank was 0.439-0.443.
Dev is 52 basins; class medians rest on 5-14 basins each.

**Export check.** numpy trees == LightGBM 4.7.0 to 1.6e-14 (relative units) on 20,000 held-out training rows (`train_gbm.py`); 47 features (the released `feats.py`). Best iterations: mean 149, quantiles 91-219. A first training attempt used `arms_tab/src/feats.py` (round 1's rejected 54-feature r3 module) by mistake; it crashed at evaluation and was discarded unscored.
