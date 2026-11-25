# ens_e1: equal-weight ensemble of the five v1.0 tied arms (River Bake-off round 2)

**What it is.** The five arms that tied on the round-1 ungauged exam, run unchanged and averaged: gradient-boosted
trees (`tab_gbm`), FlowCast v1.0 as released (`flowcast_v1`, ungauged weights), the retrained LSTM (`seq_lstm`), the
dilated TCN (`seq_tcn`) and the S4D state space (`seq_ssm`). Protocol: **ungauged**. numpy only (every member is).

**Combination** (fixed in PREREG section 5 before any result). Per target day, weights 1/5 each: the point
prediction `mean` is the mean of the members' means; each quantile (5, 25, 50, 75, 95 %) is the mean of the
members' same quantile (quantile averaging), then sorted. No re-weighting, no widening, nothing fitted on dev.

**Built-in assumption.** The members' errors are partly independent, so averaging cancels some of them. They all
learned from the same simulated bank (p1) and read the same forcing, so errors the bank causes are shared and do
not cancel.

**Training.** None. Members are the registered v1.0 arms (hashes in round 1's registry), hashed again here through
`extra_files`. Zero real labels.

**Dev gate (52 dev basins, WY2001-2014).** Median NSE 0.445 (KGE 0.435, FHV -32.5 %, 50/90 % coverage 0.59 / 0.97, CRPS / mean flow 0.383). Best member seq_lstm 0.441: the gate (>= 0.95 x 0.441) passes. The gain over the best member (+0.004; paired bootstrap -0.026..+0.025) is under the pre-registered 0.01, so there is no claim that ensembling helps on dev. Against FlowCast v1.0 +0.020 (tie), U-dev -0.021 (tie), NWM -0.128 and GR4J -0.141 (losses). Sim val on p1's frozen set 0.435 (members 0.398-0.426). Per class: snow 0.53, arid 0.20, humid 0.59, regulated 0.29, large 0.35. Details: `round2/results/ensembles_dev.json`.

**Limits.** Exam cost is the sum of the five members (about 43 CPU-min for both periods in round 1). Equal
weights by design: dev has 52 basins, too few to fit weights without overfitting.
