# seq_lstm_conformal: LSTM (FlowCast's network, retrained on the bake-off bank) + Mondrian conformal bands

**What.** The registered arm `seq_lstm` (hash `93421517ac8df0b7`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C). The mean and the median (q50) are the base arm's, so NSE, KGE and
FHV are identical to `seq_lstm`; only coverage, CRPS and band width change. Protocols: ungauged.

**Method** (`../lib/cf_mondrian.py`). Strata fixed before looking at any residual: the first of snow / arid / humid
in the basin's class list, else "other" (regulated-only, large-only, unclassed). Scores in log space with offset
c = 0.05 x mean forcing precipitation: upper endpoint log(y+c) - log(q_hi+c), lower endpoint
log(q_lo+c) - log(y+c). Per stratum, protocol/lead and endpoint the shift is the weighted (1 - a/2)
quantile (a = 0.10 or 0.50) of the scores over the stratum's dev basin-days, each basin weighted equally. Strata with fewer
than 6 dev basins fall back to the pooled shift. Applied as q_hi' = (q_hi+c) e^d_hi - c,
q_lo' = (q_lo+c) e^-d_lo - c, then made monotone. Shifts can be negative (bands narrow): conformal, not widen-only.

**Data.** Fitted on the base arm's daily dev predictions only (52 dev basins, WY2001-2014, Caravan Q;
`evaluate.py --save-preds`). Zero real labels in the base arm's training; no exam data. Nothing else is learned.

**Dev coverage (median over basins; 50 % / 90 % band) and CRPS / mean flow.** "In-sample" is what `evaluate.py`
reports for this wrapper on dev (the shifts were fitted on the same basins, so it is optimistic). "LOBO" refits the
shifts without each basin and scores it: the honest estimate. The fit targets the MEAN coverage over basins
(every basin weighted equally); per-basin coverage is skewed (a few basins are badly covered, most are over-covered),
so the median over basins sits above the nominal 0.50 / 0.90 while the mean is close to it (last column).

| Job | Raw cov50 / cov90 | Conformal in-sample | Conformal LOBO | Raw CRPS/Q | In-sample | LOBO | MEAN over basins: raw -> LOBO |
|---|---|---|---|---|---|---|---|
| ungauged | 0.57 / 0.96 | 0.55 / 0.95 | 0.54 / 0.95 | 0.389 | 0.386 | 0.388 | 0.52 / 0.87 -> 0.49 / 0.88 |

**Shifts** (log space; lower d_lo / upper d_hi; positive = widen).

| Job | Stratum (dev basins) | 90 % band lo / hi | 50 % band lo / hi |
|---|---|---|---|
| ungauged | snow (10) | -0.440 / +0.059 | -0.508 / +0.274 |
| ungauged | arid (12) | +0.048 / -0.025 | +0.109 / -0.341 |
| ungauged | humid (8) | -0.437 / -0.059 | -0.220 / -0.021 |
| ungauged | other (22) | -0.115 / -0.191 | -0.151 / -0.078 |

**Compute.** Fit 0.003 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`seq_lstm`'s card (0.867 CPU-h). Inference = the base arm + microseconds.

**Limits.** Calibrated on 52 dev basins (8-22 per stratum): coverage holds on average over basins like these, not
for every basin; basin-to-basin variation in coverage stays large (see fit_report.json). Days within a basin are
not exchangeable (persistence of errors), so the guarantee is at the basin level, approximately. Exam periods
(WY1990-99 main, WY2016-23 recent) differ in climate and gauge source (NWIS for recent) from dev.
