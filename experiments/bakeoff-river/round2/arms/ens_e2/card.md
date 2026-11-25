# ens_e2: equal-weight ensemble of the five round-2 entries (River Bake-off round 2)

**What it is.** The five tied families' round-2 entries, run unchanged and averaged: trees, LSTM, TCN and S4D each
retrained on bank b2 (`tab_gbm_b2`, `seq_lstm_b2`, `seq_tcn_b2`, `seq_ssm_b2`; all four b2 rungs were kept) and
FlowCast v1.0 as released (`flowcast_v1`, not retrained). Protocol: **ungauged**. numpy only.

**Combination** (PREREG section 5, fixed before any result): per target day, weights 1/5; `mean` = mean of the
members' means; each quantile = mean of the members' same quantile, then sorted. Nothing fitted on dev.

**Built-in assumption.** As E1: partly independent member errors cancel. Four members now share bank b2, FlowCast
keeps p1, so the bank-driven errors are only partly shared.

**Dev gate (52 dev basins, WY2001-2014).** Median NSE 0.454 (KGE 0.431, FHV -30.1 %, 50 / 90 % coverage
0.60 / 0.97, CRPS / mean flow 0.363). Best member seq_lstm_b2 0.439: the gate passes, and the gain (+0.015) clears
the pre-registered 0.01 for "ensembling helps on dev", but the paired bootstrap against that member is a tie
(-0.013..+0.037). Against E1 +0.009 (tie), FlowCast v1.0 +0.029 (tie), U-dev -0.013 (tie), NWM -0.119 and GR4J
-0.132 (losses). Per class: snow 0.58, arid 0.08, humid 0.60, regulated 0.33, large 0.08 (E1: 0.53, 0.20, 0.59,
0.29, 0.35). Sim val on b2's frozen set 0.446. Details: `round2/results/ensembles_dev.json`.

**Disagreement.** The members' relative spread on dev correlates with E2's error at Spearman 0.21 (95 % CI
-0.10..0.48): not a usable flag by the pre-registered bar. It does track the volume error (0.57, 0.33..0.74).

**Limits.** Exam cost is the sum of the five members (about 45 CPU-min for both periods). The b2 members trade
snowy-basin skill for dry-basin skill (see their cards); the ensemble inherits that.
