# trace_transformer_conformal: Transformer on the trip trace (20-s patch tokens) + Mondrian conformal bands

**What.** The registered arm `trace_transformer` (hash `d42add9c9877dd38`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `trace_transformer`; only coverage, CRPS and band width change. Tracks: trace.

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

**Track trace** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.38 / 0.74
-> in-sample 0.49 / 0.87 -> cross-fit 0.48 / 0.85; CRPS
0.0982 -> 0.0893 / 0.0915.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.43 / 0.83 | 0.50 / 0.90 | 0.47 / 0.88 | 0.054 | 0.052 | 0.054 |
| VED_HEV | 275 | 0.40 / 0.87 | 0.50 / 0.91 | 0.51 / 0.91 | 0.098 | 0.096 | 0.099 |
| VED_EV | 258 | 0.37 / 0.69 | 0.51 / 0.91 | 0.45 / 0.77 | 0.117 | 0.109 | 0.119 |
| VED_PHEV | 318 | 0.33 / 0.77 | 0.55 / 0.91 | 0.50 / 0.87 | 0.224 | 0.187 | 0.194 |
| EPA_conv_FTP | 2877 | 0.39 / 0.80 | 0.44 / 0.87 | 0.44 / 0.87 | 0.075 | 0.072 | 0.073 |
| EPA_conv_HWY | 2835 | 0.54 / 0.93 | 0.60 / 0.96 | 0.60 / 0.96 | 0.050 | 0.051 | 0.051 |
| EPA_conv_US06 | 371 | 0.32 / 0.81 | 0.36 / 0.85 | 0.36 / 0.84 | 0.078 | 0.079 | 0.079 |
| EPA_conv_FTPcold | 280 | 0.31 / 0.78 | 0.41 / 0.84 | 0.40 / 0.83 | 0.099 | 0.096 | 0.096 |
| EPA_conv_SC03 | 302 | 0.29 / 0.67 | 0.36 / 0.76 | 0.36 / 0.76 | 0.104 | 0.100 | 0.100 |
| EPA_hev_FTP | 70 | 0.49 / 0.76 | 0.56 / 0.91 | 0.56 / 0.91 | 0.102 | 0.088 | 0.089 |
| EPA_hev_HWY | 72 | 0.78 / 1.00 | 0.51 / 1.00 | 0.49 / 1.00 | 0.046 | 0.047 | 0.047 |
| EPA_bev_FTP | 98 | 0.48 / 0.84 | 0.58 / 0.88 | 0.58 / 0.87 | 0.085 | 0.068 | 0.070 |
| EPA_bev_HWY | 97 | 0.40 / 0.76 | 0.63 / 0.96 | 0.62 / 0.95 | 0.069 | 0.054 | 0.054 |
| Mersin_diesel | 56 | 0.23 / 0.62 | 0.52 / 0.95 | 0.52 / 0.93 | 0.109 | 0.092 | 0.093 |
| Mersin_cng | 56 | 0.05 / 0.39 | 0.52 / 0.95 | 0.52 / 0.93 | 0.155 | 0.109 | 0.110 |
| ICCT | 3 | 0.33 / 0.33 | 0.33 / 0.33 | 0.33 / 0.33 | 0.106 | 0.127 | 0.137 |
| ZTBus_legs | 5 | 0.40 / 0.80 | 0.80 / 1.00 | 0.80 / 1.00 | 0.076 | 0.080 | 0.083 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.44 / 0.80 | 0.61 / 0.92 | 0.60 / 0.91 | 0.077 | 0.062 |
| bev/hd | 5 | 0.40 / 0.80 | 0.80 / 1.00 | 0.80 / 1.00 | 0.076 | 0.083 |
| bev/road | 258 | 0.37 / 0.69 | 0.51 / 0.91 | 0.45 / 0.77 | 0.117 | 0.119 |
| hev/dyno | 229 | 0.53 / 0.90 | 0.51 / 0.91 | 0.50 / 0.90 | 0.074 | 0.071 |
| hev/road | 275 | 0.40 / 0.87 | 0.50 / 0.91 | 0.51 / 0.91 | 0.098 | 0.099 |
| ice/dyno | 6665 | 0.44 / 0.85 | 0.50 / 0.90 | 0.50 / 0.90 | 0.067 | 0.066 |
| ice/hd | 115 | 0.15 / 0.50 | 0.51 / 0.93 | 0.51 / 0.91 | 0.131 | 0.102 |
| ice/road | 645 | 0.43 / 0.83 | 0.50 / 0.90 | 0.47 / 0.88 | 0.054 | 0.054 |
| phev/road | 318 | 0.33 / 0.77 | 0.55 / 0.91 | 0.50 / 0.87 | 0.224 | 0.194 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | +0.009 / +0.043 | -0.015 / +0.037 |
| ice/dyno | 6665 (ice/dyno) | +0.030 / +0.031 | +0.001 / +0.022 |
| ice/hd | 115 (ice/hd) | -0.008 / +0.273 | +0.017 / +0.166 |
| hev/road | 275 (hev/road) | +0.075 / -0.003 | +0.063 / -0.023 |
| hev/dyno | 229 (hev/dyno) | -0.065 / +0.095 | -0.049 / +0.021 |
| hev/hd | 120 (*/hd) | -0.026 / +0.272 | +0.017 / +0.161 |
| phev/road | 318 (phev/road) | -0.205 / +0.421 | -0.151 / +0.306 |
| phev/dyno | 7089 (*/dyno) | +0.024 / +0.037 | -0.002 / +0.025 |
| phev/hd | 120 (*/hd) | -0.026 / +0.272 | +0.017 / +0.161 |
| bev/road | 258 (bev/road) | +0.230 / +0.104 | +0.089 / -0.008 |
| bev/dyno | 195 (bev/dyno) | -0.088 / +0.269 | -0.066 / +0.072 |
| bev/hd | 120 (*/hd) | -0.026 / +0.272 | +0.017 / +0.161 |

**Compute.** Fit 0.007 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`trace_transformer`'s card (1.635 CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
