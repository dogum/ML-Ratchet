# trace_gru_conformal: Bidirectional GRU on the trip trace + Mondrian conformal bands

**What.** The registered arm `trace_gru` (hash `ef837cb12962b17b`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `trace_gru`; only coverage, CRPS and band width change. Tracks: trace.

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

**Track trace** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.38 / 0.79
-> in-sample 0.50 / 0.89 -> cross-fit 0.48 / 0.86; CRPS
0.0939 -> 0.0882 / 0.0899.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.44 / 0.85 | 0.50 / 0.90 | 0.46 / 0.89 | 0.055 | 0.053 | 0.054 |
| VED_HEV | 275 | 0.44 / 0.88 | 0.51 / 0.91 | 0.49 / 0.89 | 0.093 | 0.092 | 0.095 |
| VED_EV | 258 | 0.39 / 0.78 | 0.51 / 0.91 | 0.45 / 0.82 | 0.113 | 0.110 | 0.113 |
| VED_PHEV | 318 | 0.39 / 0.86 | 0.55 / 0.91 | 0.51 / 0.87 | 0.203 | 0.176 | 0.181 |
| EPA_conv_FTP | 2877 | 0.44 / 0.82 | 0.48 / 0.87 | 0.48 / 0.87 | 0.069 | 0.068 | 0.068 |
| EPA_conv_HWY | 2835 | 0.44 / 0.90 | 0.54 / 0.95 | 0.54 / 0.95 | 0.055 | 0.054 | 0.054 |
| EPA_conv_US06 | 371 | 0.43 / 0.82 | 0.52 / 0.87 | 0.52 / 0.87 | 0.066 | 0.065 | 0.065 |
| EPA_conv_FTPcold | 280 | 0.28 / 0.79 | 0.33 / 0.84 | 0.33 / 0.84 | 0.090 | 0.088 | 0.088 |
| EPA_conv_SC03 | 302 | 0.34 / 0.71 | 0.43 / 0.80 | 0.42 / 0.80 | 0.103 | 0.101 | 0.101 |
| EPA_hev_FTP | 70 | 0.57 / 0.77 | 0.57 / 0.91 | 0.57 / 0.91 | 0.087 | 0.083 | 0.085 |
| EPA_hev_HWY | 72 | 0.42 / 1.00 | 0.60 / 1.00 | 0.56 / 1.00 | 0.053 | 0.056 | 0.056 |
| EPA_bev_FTP | 98 | 0.56 / 0.87 | 0.59 / 0.91 | 0.59 / 0.90 | 0.078 | 0.072 | 0.075 |
| EPA_bev_HWY | 97 | 0.45 / 0.87 | 0.42 / 0.93 | 0.42 / 0.92 | 0.057 | 0.058 | 0.058 |
| Mersin_diesel | 56 | 0.34 / 0.64 | 0.52 / 0.93 | 0.52 / 0.89 | 0.118 | 0.096 | 0.097 |
| Mersin_cng | 56 | 0.16 / 0.38 | 0.52 / 0.95 | 0.50 / 0.95 | 0.174 | 0.115 | 0.116 |
| ICCT | 3 | 0.00 / 0.67 | 0.33 / 0.67 | 0.33 / 0.33 | 0.089 | 0.124 | 0.134 |
| ZTBus_legs | 5 | 0.20 / 1.00 | 0.80 / 1.00 | 0.80 / 1.00 | 0.075 | 0.084 | 0.088 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.51 / 0.87 | 0.51 / 0.92 | 0.51 / 0.91 | 0.068 | 0.067 |
| bev/hd | 5 | 0.20 / 1.00 | 0.80 / 1.00 | 0.80 / 1.00 | 0.075 | 0.088 |
| bev/road | 258 | 0.39 / 0.78 | 0.51 / 0.91 | 0.45 / 0.82 | 0.113 | 0.113 |
| hev/dyno | 229 | 0.38 / 0.86 | 0.51 / 0.91 | 0.48 / 0.91 | 0.073 | 0.073 |
| hev/road | 275 | 0.44 / 0.88 | 0.51 / 0.91 | 0.49 / 0.89 | 0.093 | 0.095 |
| ice/dyno | 6665 | 0.43 / 0.85 | 0.50 / 0.90 | 0.50 / 0.90 | 0.065 | 0.065 |
| ice/hd | 115 | 0.24 / 0.51 | 0.51 / 0.93 | 0.50 / 0.90 | 0.145 | 0.107 |
| ice/road | 645 | 0.44 / 0.85 | 0.50 / 0.90 | 0.46 / 0.89 | 0.055 | 0.054 |
| phev/road | 318 | 0.39 / 0.86 | 0.55 / 0.91 | 0.51 / 0.87 | 0.203 | 0.181 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | -0.001 / +0.039 | -0.022 / +0.042 |
| ice/dyno | 6665 (ice/dyno) | +0.039 / +0.040 | +0.017 / +0.007 |
| ice/hd | 115 (ice/hd) | -0.033 / +0.325 | -0.030 / +0.202 |
| hev/road | 275 (hev/road) | +0.038 / +0.005 | +0.025 / +0.005 |
| hev/dyno | 229 (hev/dyno) | -0.001 / +0.095 | +0.021 / +0.030 |
| hev/hd | 120 (*/hd) | -0.050 / +0.303 | -0.030 / +0.198 |
| phev/road | 318 (phev/road) | -0.241 / +0.233 | -0.180 / +0.233 |
| phev/dyno | 7089 (*/dyno) | +0.032 / +0.042 | +0.016 / +0.007 |
| phev/hd | 120 (*/hd) | -0.050 / +0.303 | -0.030 / +0.198 |
| bev/road | 258 (bev/road) | +0.175 / +0.140 | +0.031 / +0.023 |
| bev/dyno | 195 (bev/dyno) | -0.054 / +0.226 | -0.011 / +0.013 |
| bev/hd | 120 (*/hd) | -0.050 / +0.303 | -0.030 / +0.198 |

**Compute.** Fit 0.007 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`trace_gru`'s card (1.648 CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
