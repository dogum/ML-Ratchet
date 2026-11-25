# tab_gbm: gradient-boosted trees (River Bake-off, arm B1)

**What it is.** A table learner. Each target day becomes one row of 47 hand-built numbers (recent rain, a crude
snowpack, slow and fast "wetness" memories, temperature, season, basin attributes). LightGBM grows decision trees on
those rows. The trees are exported to plain numpy arrays (`gbm_trees.npz`) and `predict.py` walks them with numpy
only; numpy equals LightGBM to 1.2e-14 (relative units) on 20,000 held-out rows. Protocol: **ungauged**.

**Built-in assumption.** Flow on a day is a function of a fixed, hand-picked summary of the past. There is no
learned memory: whatever the features do not carry (e.g. a deep groundwater store with a 2-year time scale), the
trees cannot see.

**Inputs and features** (`feats.py`, only from the batch's forcing and 18-value static vector):
- P(t+1), P(t), P(t-1), P(t-2), sum P(t-6..t-3) (P(t+1) is legitimate: the batch carries one extra forcing day, and
  forcing days are UTC while flow days are local);
- a one-band degree-day snow proxy (threshold 0 C, 2 C rain/snow interval, 3 mm/C/day): liquid input W = rain + melt
  at t, t-1, t-2, melt(t), SWE and log1p(SWE);
- exponential moving averages of W over 3, 10, 30, 90, 365 days (initialised at the warm-up mean);
- two buckets (50 and 300 mm: fill fraction, spill EMA, drain) and a 60-day linear store fed by the larger bucket;
- T(t), EMA10 T, PET(t), EMA30 PET; sin/cos of day of year; the static vector.
Water features and the target are divided by p_mean (the forcing's own mean precipitation), so wet and dry basins
share one scale. Target = qo / p_mean (the gauge-like simulated flow).

**Training.** Sim TRAIN bank only: one random 731-day window per run, 100 random target days per window,
1.2 M rows; runs 11,400-11,999 held out for early stopping (never the frozen val set). Six boosters: the mean (L2)
and the 5/25/50/75/95 % quantiles (pinball loss). The output is x p_mean, clipped at 0; quantiles are sorted.

**Ratchet** (rule: sim val median NSE > incumbent + 0.01 and dev NSE >= 0.95 x incumbent; all 800 val runs, 52 dev basins):

| Rung | Change | Sim val NSE | Dev NSE | Decision |
|---|---|---|---|---|
| r0 | base, monotone +1 constraints on the 21 water-input features (mean model) | 0.367 | 0.418 | base |
| r1 | 2x data (2.4 M rows), 127 leaves, lr 0.03 | 0.371 | 0.403 | rejected (+0.004) |
| r2 | **no monotone constraints** | **0.398** | **0.421** | **released** (+0.031) |
| r3 | + 7 elevation-band snow features | 0.407 | 0.418 | rejected (+0.009; bootstrap CI -0.012..+0.020) |

**Released r2 on the dev gate** (52 basins, WY2001-2014): NSE 0.421, KGE 0.444, FHV -30.8 %, 50/90 % coverage
0.55 / 0.95, CRPS / mean flow 0.386. Paired bootstrap vs FlowCast v1.0 (LSTM, 0.425): NSE -0.004 (95 % CI
-0.031..+0.058), a tie. By class (NSE): snow 0.56 (FlowCast 0.45), arid 0.20 (0.18), humid 0.61 (0.56), regulated
0.36 (0.28), large 0.33 (0.35).

**What its behaviour suggests.**
- With good hand-built memory features, trees tie the LSTM on real dev data while scoring lower on the simulator
  (0.398 vs 0.426): the tree learns less of the simulator's fine structure but transfers about as well.
- The physics sign constraints cost 0.031 sim val NSE (and nothing on dev). "More recent water -> more flow" holds
  one feature at a time, but not jointly for 21 strongly correlated water features: forcing all of them to be
  monotone at once removed useful interactions. The constraint was dropped by the ratchet; the quantile boosters were
  never constrained (LightGBM does not allow it).
- More data and bigger trees did not help (r1): the limit is the noise in the targets (gauge-noise model, unknown
  model structure, latent parameters), not the sample size.

**Compute.** 0.43 single-thread CPU hours for all rungs (see ratchet.json). Inference: ~2.3 CPU s per dev basin
(14 years); exam estimate ~35 min for both periods.

**Limits.** No learned memory beyond the fixed features (multi-year storage is invisible); quantiles come from
separate boosters (sorted when they cross); the snow proxy is crude (one band, fixed parameters); flows below zero
are clipped. Arid basins remain poor (dev arid NSE 0.20), as for every ungauged learner here.

**Files.** `predict.py` (numpy inference), `feats.py` (features), `treeexport.py` (tree arrays and the numpy
predictor), `gbm_trees.npz` (six ensembles), `ratchet.json`. Training code (not part of the artefact):
`arms_tab/src/build_tab.py`, `train_gbm.py`.
