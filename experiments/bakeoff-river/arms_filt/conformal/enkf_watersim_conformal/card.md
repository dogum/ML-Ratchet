# enkf_watersim_conformal: Ensemble Kalman filter on WaterSim (HBV + snow), joint state-parameter + Mondrian conformal bands

**What.** The registered arm `enkf_watersim` (hash `bd85c40035602083`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C). The mean and the median (q50) are the base arm's, so NSE, KGE and
FHV are identical to `enkf_watersim`; only coverage, CRPS and band width change. Protocols: gauged.

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
| gauged_L1 | 0.75 / 0.96 | 0.55 / 0.93 | 0.55 / 0.93 | 0.192 | 0.190 | 0.190 | 0.76 / 0.95 -> 0.57 / 0.91 |
| gauged_L3 | 0.63 / 0.92 | 0.51 / 0.92 | 0.51 / 0.92 | 0.252 | 0.251 | 0.251 | 0.65 / 0.91 -> 0.54 / 0.90 |
| gauged_L7 | 0.52 / 0.87 | 0.49 / 0.91 | 0.49 / 0.91 | 0.302 | 0.305 | 0.305 | 0.56 / 0.87 -> 0.52 / 0.90 |

**Shifts** (log space; lower d_lo / upper d_hi; positive = widen).

| Job | Stratum (dev basins) | 90 % band lo / hi | 50 % band lo / hi |
|---|---|---|---|
| gauged_L1 | snow (10) | -0.186 / -0.141 | -0.124 / -0.081 |
| gauged_L1 | arid (12) | +0.000 / -0.002 | +0.000 / -0.009 |
| gauged_L1 | humid (8) | -0.155 / -0.077 | -0.096 / -0.058 |
| gauged_L1 | other (22) | -0.063 / -0.056 | -0.078 / -0.057 |
| gauged_L3 | snow (10) | -0.177 / -0.039 | -0.127 / -0.022 |
| gauged_L3 | arid (12) | +0.000 / +0.047 | +0.000 / -0.006 |
| gauged_L3 | humid (8) | -0.118 / +0.069 | -0.087 / +0.017 |
| gauged_L3 | other (22) | -0.017 / +0.069 | -0.068 / -0.020 |
| gauged_L7 | snow (10) | -0.117 / +0.128 | -0.108 / +0.017 |
| gauged_L7 | arid (12) | +0.009 / +0.126 | +0.000 / -0.005 |
| gauged_L7 | humid (8) | -0.029 / +0.201 | -0.050 / +0.077 |
| gauged_L7 | other (22) | +0.000 / +0.190 | -0.050 / +0.025 |

**Compute.** Fit 0.026 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`enkf_watersim`'s card (0.98 CPU-h). Inference = the base arm + microseconds.

**Limits.** Calibrated on 52 dev basins (8-22 per stratum): coverage holds on average over basins like these, not
for every basin; basin-to-basin variation in coverage stays large (see fit_report.json). Days within a basin are
not exchangeable (persistence of errors), so the guarantee is at the basin level, approximately. Exam periods
(WY1990-99 main, WY2016-23 recent) differ in climate and gauge source (NWIS for recent) from dev.
