# flowcast_v1_conformal: FlowCast v1.0 as released + Mondrian conformal bands

**What.** The registered arm `flowcast_v1` (hash `6400ea99ca6cfc34`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C). The mean and the median (q50) are the base arm's, so NSE, KGE and
FHV are identical to `flowcast_v1`; only coverage, CRPS and band width change. Protocols: ungauged, gauged.

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
| ungauged | 0.58 / 0.97 | 0.54 / 0.95 | 0.54 / 0.95 | 0.380 | 0.386 | 0.388 | 0.50 / 0.95 -> 0.49 / 0.88 |
| gauged_L1 | 0.65 / 0.96 | 0.55 / 0.93 | 0.55 / 0.93 | 0.225 | 0.219 | 0.219 | 0.58 / 0.87 -> 0.51 / 0.89 |
| gauged_L3 | 0.62 / 0.95 | 0.54 / 0.93 | 0.54 / 0.93 | 0.246 | 0.245 | 0.246 | 0.55 / 0.85 -> 0.51 / 0.89 |
| gauged_L7 | 0.57 / 0.93 | 0.52 / 0.93 | 0.52 / 0.93 | 0.281 | 0.282 | 0.283 | 0.50 / 0.84 -> 0.50 / 0.89 |

**Shifts** (log space; lower d_lo / upper d_hi; positive = widen).

| Job | Stratum (dev basins) | 90 % band lo / hi | 50 % band lo / hi |
|---|---|---|---|
| ungauged | snow (10) | -0.631 / +0.022 | -0.528 / +0.323 |
| ungauged | arid (12) | +0.000 / -0.137 | +0.172 / -0.395 |
| ungauged | humid (8) | -0.604 / -0.118 | -0.249 / -0.018 |
| ungauged | other (22) | -0.224 / -0.194 | -0.194 / -0.002 |
| gauged_L1 | snow (10) | -0.344 / +0.015 | -0.359 / +0.120 |
| gauged_L1 | arid (12) | +0.085 / -0.311 | +0.124 / -0.229 |
| gauged_L1 | humid (8) | -0.372 / -0.165 | -0.233 / -0.045 |
| gauged_L1 | other (22) | -0.034 / -0.183 | -0.141 / -0.061 |
| gauged_L3 | snow (10) | -0.339 / +0.051 | -0.341 / +0.143 |
| gauged_L3 | arid (12) | +0.086 / -0.253 | +0.122 / -0.218 |
| gauged_L3 | humid (8) | -0.345 / -0.143 | -0.192 / -0.043 |
| gauged_L3 | other (22) | -0.015 / -0.117 | -0.109 / -0.053 |
| gauged_L7 | snow (10) | -0.298 / +0.088 | -0.294 / +0.182 |
| gauged_L7 | arid (12) | +0.087 / -0.151 | +0.118 / -0.198 |
| gauged_L7 | humid (8) | -0.276 / -0.093 | -0.118 / -0.025 |
| gauged_L7 | other (22) | +0.020 / -0.053 | -0.066 / -0.033 |

**Compute.** Fit 0.005 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`flowcast_v1`'s card (0.0 CPU-h). Inference = the base arm + microseconds.

**Limits.** Calibrated on 52 dev basins (8-22 per stratum): coverage holds on average over basins like these, not
for every basin; basin-to-basin variation in coverage stays large (see fit_report.json). Days within a basin are
not exchangeable (persistence of errors), so the guarantee is at the basin level, approximately. Exam periods
(WY1990-99 main, WY2016-23 recent) differ in climate and gauge source (NWIS for recent) from dev.
