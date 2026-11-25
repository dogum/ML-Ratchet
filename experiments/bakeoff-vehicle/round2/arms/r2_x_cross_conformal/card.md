# r2_x_cross_conformal: Cross-family ensemble: transformer + GRU + MLP + GP, equal weights (round 2, X) + Mondrian conformal bands

**What.** The registered arm `r2_x_cross` (hash `d1f140fda3455a4f`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `r2_x_cross`; only coverage, CRPS and band width change. Tracks: trace.

**Method.** Strata fixed from the inputs before looking at any residual: powertrain (conventional / hybrid / plug-in /
battery-electric, from `arch`) x regime (heavy duty if class is bus or truck; dynamometer test if the input's
dyno flag is set; otherwise road). Strata with fewer than 20 dev units fall back to the regime pooled over
powertrains, then the powertrain pooled over regimes, then all units. Scores in log space per band endpoint
(asymmetric CQR): s_hi = log y - log q_hi, s_lo = log q_lo - log y; the shift is the ceil((n+1)(1-a/2))/n quantile of
the stratum's dev units (a = 0.10 for the 90 % band, 0.50 for the 50 % band), every unit weighted equally. Applied as
q_hi' = q_hi e^d_hi, q_lo' = q_lo e^-d_lo, then made monotone. Negative shifts narrow a band: conformal, not widen-only.

**Data.** Fitted on the base arm's dev predictions only (`evaluate.py --sets dev`: VED 16 dev cars, EPA dev families,
Mersin dev week, ICCT dev truck, 5 ZTBus legs; EnergyNet's scoring units). Zero real labels in the base arm's
training; no exam data. Nothing else is learned.

**Dev coverage** (50 % / 90 % band) and mean CRPS (log). "In-sample" = the shifts fitted on all dev units and scored
on the same units (optimistic). "Cross-fit" = grouped 10-fold: shifts refitted without each fold of clusters (VED cars,
EPA vehicles, Mersin trips, ICCT tests, ZTBus missions) and scored on that fold: the honest estimate.

**Track trace** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.48 / 0.84
-> in-sample 0.48 / 0.87 -> cross-fit 0.47 / 0.85; CRPS
0.0922 -> 0.0881 / 0.0899.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.53 / 0.93 | 0.50 / 0.90 | 0.46 / 0.89 | 0.052 | 0.051 | 0.052 |
| VED_HEV | 275 | 0.49 / 0.89 | 0.51 / 0.91 | 0.49 / 0.91 | 0.091 | 0.090 | 0.094 |
| VED_EV | 258 | 0.45 / 0.82 | 0.51 / 0.91 | 0.43 / 0.79 | 0.110 | 0.108 | 0.114 |
| VED_PHEV | 318 | 0.40 / 0.77 | 0.51 / 0.91 | 0.52 / 0.90 | 0.184 | 0.160 | 0.161 |
| EPA_conv_FTP | 2877 | 0.55 / 0.92 | 0.47 / 0.87 | 0.47 / 0.87 | 0.064 | 0.066 | 0.066 |
| EPA_conv_HWY | 2835 | 0.63 / 0.96 | 0.58 / 0.95 | 0.58 / 0.95 | 0.051 | 0.048 | 0.048 |
| EPA_conv_US06 | 371 | 0.41 / 0.88 | 0.41 / 0.88 | 0.41 / 0.88 | 0.078 | 0.074 | 0.074 |
| EPA_conv_FTPcold | 280 | 0.53 / 0.86 | 0.36 / 0.84 | 0.36 / 0.84 | 0.084 | 0.086 | 0.086 |
| EPA_conv_SC03 | 302 | 0.47 / 0.81 | 0.35 / 0.76 | 0.35 / 0.76 | 0.095 | 0.098 | 0.098 |
| EPA_hev_FTP | 70 | 0.37 / 0.76 | 0.54 / 0.90 | 0.53 / 0.90 | 0.098 | 0.087 | 0.087 |
| EPA_hev_HWY | 72 | 0.82 / 1.00 | 0.62 / 1.00 | 0.64 / 1.00 | 0.047 | 0.051 | 0.051 |
| EPA_bev_FTP | 98 | 0.66 / 0.88 | 0.47 / 0.88 | 0.50 / 0.87 | 0.075 | 0.063 | 0.064 |
| EPA_bev_HWY | 97 | 0.54 / 0.84 | 0.53 / 0.96 | 0.52 / 0.94 | 0.063 | 0.053 | 0.053 |
| Mersin_diesel | 56 | 0.34 / 0.79 | 0.50 / 0.93 | 0.50 / 0.91 | 0.108 | 0.096 | 0.097 |
| Mersin_cng | 56 | 0.14 / 0.64 | 0.50 / 0.95 | 0.52 / 0.93 | 0.149 | 0.112 | 0.113 |
| ICCT | 3 | 0.33 / 0.67 | 0.33 / 0.33 | 0.33 / 0.33 | 0.125 | 0.167 | 0.179 |
| ZTBus_legs | 5 | 0.40 / 1.00 | 0.80 / 1.00 | 0.80 / 1.00 | 0.080 | 0.082 | 0.085 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.60 / 0.86 | 0.50 / 0.92 | 0.51 / 0.90 | 0.069 | 0.059 |
| bev/hd | 5 | 0.40 / 1.00 | 0.80 / 1.00 | 0.80 / 1.00 | 0.080 | 0.085 |
| bev/road | 258 | 0.45 / 0.82 | 0.51 / 0.91 | 0.43 / 0.79 | 0.110 | 0.114 |
| hev/dyno | 229 | 0.50 / 0.90 | 0.51 / 0.91 | 0.51 / 0.90 | 0.075 | 0.072 |
| hev/road | 275 | 0.49 / 0.89 | 0.51 / 0.91 | 0.49 / 0.91 | 0.091 | 0.094 |
| ice/dyno | 6665 | 0.57 / 0.93 | 0.50 / 0.90 | 0.50 / 0.90 | 0.062 | 0.061 |
| ice/hd | 115 | 0.24 / 0.71 | 0.50 / 0.92 | 0.50 / 0.90 | 0.128 | 0.107 |
| ice/road | 645 | 0.53 / 0.93 | 0.50 / 0.90 | 0.46 / 0.89 | 0.052 | 0.052 |
| phev/road | 318 | 0.40 / 0.77 | 0.51 / 0.91 | 0.52 / 0.90 | 0.184 | 0.161 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | -0.050 / +0.008 | -0.041 / +0.029 |
| ice/dyno | 6665 (ice/dyno) | +0.000 / -0.044 | +0.001 / -0.032 |
| ice/hd | 115 (ice/hd) | -0.075 / +0.216 | -0.026 / +0.165 |
| hev/road | 275 (hev/road) | +0.030 / -0.005 | +0.017 / -0.016 |
| hev/dyno | 229 (hev/dyno) | -0.088 / +0.091 | -0.018 / +0.035 |
| hev/hd | 120 (*/hd) | -0.098 / +0.201 | -0.026 / +0.156 |
| phev/road | 318 (phev/road) | -0.121 / +0.382 | -0.088 / +0.203 |
| phev/dyno | 7089 (*/dyno) | -0.004 / -0.038 | -0.002 / -0.028 |
| phev/hd | 120 (*/hd) | -0.098 / +0.201 | -0.026 / +0.156 |
| bev/road | 258 (bev/road) | +0.178 / +0.054 | +0.035 / -0.001 |
| bev/dyno | 195 (bev/dyno) | -0.157 / +0.178 | -0.083 / +0.036 |
| bev/hd | 120 (*/hd) | -0.098 / +0.201 | -0.026 / +0.156 |

**Compute.** Fit 0.002 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`r2_x_cross`'s card (0.0 CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
