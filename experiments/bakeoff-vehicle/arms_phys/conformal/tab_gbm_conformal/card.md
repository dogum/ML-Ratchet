# tab_gbm_conformal: Gradient-boosted trees (LightGBM, numpy export) + Mondrian conformal bands

**What.** The registered arm `tab_gbm` (hash `57349fe2297a02a8`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `tab_gbm`; only coverage, CRPS and band width change. Tracks: table.

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

**Track table** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.40 / 0.78
-> in-sample 0.49 / 0.86 -> cross-fit 0.46 / 0.84; CRPS
0.0994 -> 0.0937 / 0.0962.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.27 / 0.71 | 0.50 / 0.90 | 0.45 / 0.88 | 0.058 | 0.053 | 0.055 |
| VED_HEV | 275 | 0.38 / 0.87 | 0.51 / 0.91 | 0.47 / 0.89 | 0.097 | 0.095 | 0.099 |
| VED_EV | 258 | 0.24 / 0.59 | 0.54 / 0.91 | 0.33 / 0.65 | 0.138 | 0.115 | 0.135 |
| VED_PHEV | 318 | 0.27 / 0.85 | 0.51 / 0.91 | 0.44 / 0.92 | 0.155 | 0.149 | 0.153 |
| EPA_conv_FTP | 2877 | 0.56 / 0.90 | 0.52 / 0.90 | 0.52 / 0.89 | 0.066 | 0.067 | 0.067 |
| EPA_conv_HWY | 2835 | 0.53 / 0.90 | 0.57 / 0.95 | 0.57 / 0.95 | 0.059 | 0.056 | 0.056 |
| EPA_conv_US06 | 371 | 0.26 / 0.80 | 0.29 / 0.87 | 0.29 / 0.87 | 0.094 | 0.087 | 0.087 |
| EPA_conv_FTPcold | 280 | 0.35 / 0.75 | 0.23 / 0.74 | 0.24 / 0.74 | 0.097 | 0.099 | 0.100 |
| EPA_conv_SC03 | 302 | 0.36 / 0.70 | 0.32 / 0.69 | 0.32 / 0.69 | 0.110 | 0.111 | 0.111 |
| EPA_hev_FTP | 70 | 0.71 / 0.93 | 0.67 / 0.93 | 0.64 / 0.93 | 0.086 | 0.087 | 0.088 |
| EPA_hev_HWY | 72 | 0.39 / 1.00 | 0.38 / 1.00 | 0.33 / 1.00 | 0.064 | 0.061 | 0.061 |
| EPA_bev_FTP | 98 | 0.56 / 0.90 | 0.72 / 0.91 | 0.72 / 0.90 | 0.083 | 0.068 | 0.068 |
| EPA_bev_HWY | 97 | 0.45 / 0.78 | 0.71 / 0.97 | 0.74 / 0.97 | 0.077 | 0.061 | 0.061 |
| Mersin_diesel | 56 | 0.48 / 0.80 | 0.50 / 0.91 | 0.50 / 0.91 | 0.101 | 0.090 | 0.091 |
| Mersin_cng | 56 | 0.23 / 0.71 | 0.54 / 0.96 | 0.52 / 0.95 | 0.133 | 0.100 | 0.100 |
| ICCT | 3 | 0.33 / 0.33 | 0.33 / 0.33 | 0.33 / 0.33 | 0.174 | 0.200 | 0.206 |
| ZTBus_legs | 5 | 0.20 / 0.80 | 1.00 / 1.00 | 1.00 / 1.00 | 0.087 | 0.082 | 0.082 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.51 / 0.84 | 0.72 / 0.94 | 0.73 / 0.93 | 0.080 | 0.065 |
| bev/hd | 5 | 0.20 / 0.80 | 1.00 / 1.00 | 1.00 / 1.00 | 0.087 | 0.082 |
| bev/road | 258 | 0.24 / 0.59 | 0.54 / 0.91 | 0.33 / 0.65 | 0.138 | 0.135 |
| hev/dyno | 229 | 0.55 / 0.93 | 0.52 / 0.91 | 0.49 / 0.91 | 0.075 | 0.075 |
| hev/road | 275 | 0.38 / 0.87 | 0.51 / 0.91 | 0.47 / 0.89 | 0.097 | 0.099 |
| ice/dyno | 6665 | 0.52 / 0.88 | 0.51 / 0.90 | 0.51 / 0.90 | 0.068 | 0.067 |
| ice/hd | 115 | 0.36 / 0.75 | 0.51 / 0.92 | 0.50 / 0.91 | 0.118 | 0.099 |
| ice/road | 645 | 0.27 / 0.71 | 0.50 / 0.90 | 0.45 / 0.88 | 0.058 | 0.055 |
| phev/road | 318 | 0.27 / 0.85 | 0.51 / 0.91 | 0.44 / 0.92 | 0.155 | 0.153 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | +0.007 / +0.098 | +0.015 / +0.047 |
| ice/dyno | 6665 (ice/dyno) | +0.041 / -0.007 | +0.013 / -0.025 |
| ice/hd | 115 (ice/hd) | -0.028 / +0.166 | -0.076 / +0.139 |
| hev/road | 275 (hev/road) | +0.006 / +0.073 | +0.044 / -0.002 |
| hev/dyno | 229 (hev/dyno) | -0.018 / +0.021 | +0.017 / -0.015 |
| hev/hd | 120 (*/hd) | -0.047 / +0.156 | -0.079 / +0.137 |
| phev/road | 318 (phev/road) | +0.050 / +0.071 | +0.015 / +0.152 |
| phev/dyno | 7089 (*/dyno) | +0.034 / -0.002 | +0.011 / -0.021 |
| phev/hd | 120 (*/hd) | -0.047 / +0.156 | -0.079 / +0.137 |
| bev/road | 258 (bev/road) | +0.251 / +0.052 | +0.170 / -0.056 |
| bev/dyno | 195 (bev/dyno) | -0.119 / +0.140 | -0.120 / +0.066 |
| bev/hd | 120 (*/hd) | -0.047 / +0.156 | -0.079 / +0.137 |

**Compute.** Fit 0.006 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`tab_gbm`'s card (0.24 CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
