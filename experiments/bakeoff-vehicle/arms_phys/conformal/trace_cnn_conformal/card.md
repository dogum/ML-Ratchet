# trace_cnn_conformal: 1-D dilated CNN on the trip trace + Mondrian conformal bands

**What.** The registered arm `trace_cnn` (hash `d430085a60921451`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `trace_cnn`; only coverage, CRPS and band width change. Tracks: trace.

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

**Track trace** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.41 / 0.77
-> in-sample 0.49 / 0.86 -> cross-fit 0.49 / 0.85; CRPS
0.0972 -> 0.0912 / 0.0926.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.42 / 0.86 | 0.50 / 0.90 | 0.48 / 0.88 | 0.052 | 0.050 | 0.052 |
| VED_HEV | 275 | 0.43 / 0.84 | 0.50 / 0.91 | 0.49 / 0.89 | 0.096 | 0.094 | 0.097 |
| VED_EV | 258 | 0.40 / 0.77 | 0.51 / 0.91 | 0.49 / 0.85 | 0.116 | 0.111 | 0.112 |
| VED_PHEV | 318 | 0.38 / 0.75 | 0.51 / 0.91 | 0.51 / 0.88 | 0.198 | 0.172 | 0.178 |
| EPA_conv_FTP | 2877 | 0.42 / 0.89 | 0.44 / 0.87 | 0.43 / 0.87 | 0.065 | 0.067 | 0.067 |
| EPA_conv_HWY | 2835 | 0.49 / 0.87 | 0.60 / 0.95 | 0.60 / 0.95 | 0.056 | 0.051 | 0.051 |
| EPA_conv_US06 | 371 | 0.26 / 0.76 | 0.39 / 0.86 | 0.39 / 0.86 | 0.082 | 0.071 | 0.071 |
| EPA_conv_FTPcold | 280 | 0.37 / 0.84 | 0.29 / 0.81 | 0.29 / 0.81 | 0.086 | 0.088 | 0.088 |
| EPA_conv_SC03 | 302 | 0.49 / 0.81 | 0.50 / 0.80 | 0.49 / 0.80 | 0.094 | 0.094 | 0.094 |
| EPA_hev_FTP | 70 | 0.57 / 0.77 | 0.57 / 0.87 | 0.57 / 0.87 | 0.087 | 0.083 | 0.084 |
| EPA_hev_HWY | 72 | 0.51 / 1.00 | 0.58 / 0.93 | 0.57 / 0.93 | 0.047 | 0.049 | 0.049 |
| EPA_bev_FTP | 98 | 0.62 / 0.88 | 0.57 / 0.86 | 0.58 / 0.86 | 0.079 | 0.067 | 0.068 |
| EPA_bev_HWY | 97 | 0.40 / 0.78 | 0.62 / 0.97 | 0.62 / 0.96 | 0.079 | 0.059 | 0.059 |
| Mersin_diesel | 56 | 0.29 / 0.70 | 0.50 / 0.96 | 0.50 / 0.95 | 0.102 | 0.088 | 0.089 |
| Mersin_cng | 56 | 0.11 / 0.48 | 0.50 / 0.91 | 0.52 / 0.91 | 0.152 | 0.109 | 0.110 |
| ICCT | 3 | 0.33 / 0.33 | 0.33 / 0.33 | 0.33 / 0.33 | 0.165 | 0.206 | 0.212 |
| ZTBus_legs | 5 | 0.00 / 0.80 | 0.60 / 1.00 | 0.60 / 1.00 | 0.095 | 0.087 | 0.090 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.51 / 0.83 | 0.59 / 0.91 | 0.60 / 0.91 | 0.079 | 0.064 |
| bev/hd | 5 | 0.00 / 0.80 | 0.60 / 1.00 | 0.60 / 1.00 | 0.095 | 0.090 |
| bev/road | 258 | 0.40 / 0.77 | 0.51 / 0.91 | 0.49 / 0.85 | 0.116 | 0.112 |
| hev/dyno | 229 | 0.47 / 0.86 | 0.51 / 0.91 | 0.49 / 0.91 | 0.071 | 0.070 |
| hev/road | 275 | 0.43 / 0.84 | 0.50 / 0.91 | 0.49 / 0.89 | 0.096 | 0.097 |
| ice/dyno | 6665 | 0.44 / 0.87 | 0.50 / 0.90 | 0.50 / 0.90 | 0.064 | 0.062 |
| ice/hd | 115 | 0.20 / 0.58 | 0.50 / 0.92 | 0.50 / 0.91 | 0.128 | 0.102 |
| ice/road | 645 | 0.42 / 0.86 | 0.50 / 0.90 | 0.48 / 0.88 | 0.052 | 0.052 |
| phev/road | 318 | 0.38 / 0.75 | 0.51 / 0.91 | 0.51 / 0.88 | 0.198 | 0.178 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | -0.007 / +0.042 | -0.021 / +0.040 |
| ice/dyno | 6665 (ice/dyno) | +0.060 / -0.020 | +0.032 / -0.024 |
| ice/hd | 115 (ice/hd) | -0.062 / +0.279 | -0.031 / +0.152 |
| hev/road | 275 (hev/road) | +0.060 / +0.020 | +0.042 / -0.018 |
| hev/dyno | 229 (hev/dyno) | -0.046 / +0.051 | +0.010 / +0.025 |
| hev/hd | 120 (*/hd) | -0.073 / +0.278 | -0.031 / +0.152 |
| phev/road | 318 (phev/road) | -0.168 / +0.379 | -0.132 / +0.222 |
| phev/dyno | 7089 (*/dyno) | +0.055 / -0.012 | +0.029 / -0.020 |
| phev/hd | 120 (*/hd) | -0.073 / +0.278 | -0.031 / +0.152 |
| bev/road | 258 (bev/road) | +0.128 / +0.165 | -0.023 / +0.061 |
| bev/dyno | 195 (bev/dyno) | -0.142 / +0.231 | -0.095 / +0.070 |
| bev/hd | 120 (*/hd) | -0.073 / +0.278 | -0.031 / +0.152 |

**Compute.** Fit 0.007 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`trace_cnn`'s card (1.572 CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
