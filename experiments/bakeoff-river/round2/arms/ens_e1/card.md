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

**Dev gate and numbers.** See `ratchet.json` (filled from `round2/results/ensembles_dev.json`).

**Limits.** Exam cost is the sum of the five members (about 43 CPU-min for both periods in round 1). Equal
weights by design: dev has 52 basins, too few to fit weights without overfitting.
