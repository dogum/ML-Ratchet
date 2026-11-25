# r2_d2_transformer_conformal: Transformer over 20-s patches, 5-member ensemble on 3 banks (round 2, D2) + Mondrian conformal bands

**What.** The registered arm `r2_d2_transformer` (hash `a83d255c2a2424da`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `r2_d2_transformer`; only coverage, CRPS and band width change. Tracks: trace.

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

**Track trace** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.44 / 0.75
-> in-sample 0.51 / 0.87 -> cross-fit 0.50 / 0.86; CRPS
0.0940 -> 0.0877 / 0.0894.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.38 / 0.81 | 0.50 / 0.90 | 0.48 / 0.89 | 0.052 | 0.050 | 0.051 |
| VED_HEV | 275 | 0.44 / 0.85 | 0.51 / 0.91 | 0.48 / 0.91 | 0.097 | 0.096 | 0.099 |
| VED_EV | 258 | 0.40 / 0.66 | 0.51 / 0.91 | 0.46 / 0.81 | 0.117 | 0.111 | 0.117 |
| VED_PHEV | 318 | 0.36 / 0.74 | 0.51 / 0.91 | 0.50 / 0.89 | 0.194 | 0.171 | 0.172 |
| EPA_conv_FTP | 2877 | 0.43 / 0.85 | 0.43 / 0.86 | 0.43 / 0.86 | 0.067 | 0.068 | 0.068 |
| EPA_conv_HWY | 2835 | 0.56 / 0.91 | 0.59 / 0.96 | 0.59 / 0.95 | 0.047 | 0.046 | 0.046 |
| EPA_conv_US06 | 371 | 0.49 / 0.87 | 0.52 / 0.91 | 0.51 / 0.91 | 0.065 | 0.063 | 0.063 |
| EPA_conv_FTPcold | 280 | 0.42 / 0.83 | 0.38 / 0.83 | 0.38 / 0.83 | 0.086 | 0.086 | 0.086 |
| EPA_conv_SC03 | 302 | 0.42 / 0.76 | 0.41 / 0.78 | 0.41 / 0.78 | 0.091 | 0.090 | 0.090 |
| EPA_hev_FTP | 70 | 0.53 / 0.77 | 0.57 / 0.93 | 0.57 / 0.90 | 0.092 | 0.082 | 0.082 |
| EPA_hev_HWY | 72 | 0.79 / 1.00 | 0.61 / 1.00 | 0.60 / 1.00 | 0.044 | 0.046 | 0.046 |
| EPA_bev_FTP | 98 | 0.57 / 0.87 | 0.62 / 0.89 | 0.63 / 0.89 | 0.078 | 0.063 | 0.063 |
| EPA_bev_HWY | 97 | 0.46 / 0.79 | 0.62 / 0.93 | 0.62 / 0.93 | 0.070 | 0.055 | 0.055 |
| Mersin_diesel | 56 | 0.32 / 0.66 | 0.50 / 0.93 | 0.50 / 0.91 | 0.117 | 0.095 | 0.097 |
| Mersin_cng | 56 | 0.14 / 0.38 | 0.52 / 0.96 | 0.50 / 0.95 | 0.168 | 0.114 | 0.115 |
| ICCT | 3 | 0.33 / 0.33 | 0.33 / 0.33 | 0.33 / 0.33 | 0.119 | 0.166 | 0.178 |
| ZTBus_legs | 5 | 0.00 / 0.80 | 0.80 / 1.00 | 0.80 / 0.80 | 0.081 | 0.083 | 0.088 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.52 / 0.83 | 0.62 / 0.91 | 0.63 / 0.91 | 0.074 | 0.059 |
| bev/hd | 5 | 0.00 / 0.80 | 0.80 / 1.00 | 0.80 / 0.80 | 0.081 | 0.088 |
| bev/road | 258 | 0.40 / 0.66 | 0.51 / 0.91 | 0.46 / 0.81 | 0.117 | 0.117 |
| hev/dyno | 229 | 0.54 / 0.89 | 0.51 / 0.91 | 0.50 / 0.90 | 0.070 | 0.067 |
| hev/road | 275 | 0.44 / 0.85 | 0.51 / 0.91 | 0.48 / 0.91 | 0.097 | 0.099 |
| ice/dyno | 6665 | 0.48 / 0.87 | 0.50 / 0.90 | 0.50 / 0.90 | 0.060 | 0.060 |
| ice/hd | 115 | 0.23 / 0.51 | 0.50 / 0.93 | 0.50 / 0.91 | 0.142 | 0.108 |
| ice/road | 645 | 0.38 / 0.81 | 0.50 / 0.90 | 0.48 / 0.89 | 0.052 | 0.051 |
| phev/road | 318 | 0.36 / 0.74 | 0.51 / 0.91 | 0.50 / 0.89 | 0.194 | 0.172 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | +0.020 / +0.051 | -0.010 / +0.044 |
| ice/dyno | 6665 (ice/dyno) | +0.036 / +0.003 | +0.009 / -0.008 |
| ice/hd | 115 (ice/hd) | -0.075 / +0.280 | -0.029 / +0.185 |
| hev/road | 275 (hev/road) | +0.071 / +0.048 | +0.021 / +0.012 |
| hev/dyno | 229 (hev/dyno) | -0.069 / +0.112 | -0.016 / +0.023 |
| hev/hd | 120 (*/hd) | -0.090 / +0.280 | -0.029 / +0.184 |
| phev/road | 318 (phev/road) | -0.050 / +0.436 | -0.065 / +0.226 |
| phev/dyno | 7089 (*/dyno) | +0.031 / +0.010 | +0.007 / -0.004 |
| phev/hd | 120 (*/hd) | -0.090 / +0.280 | -0.029 / +0.184 |
| bev/road | 258 (bev/road) | +0.237 / +0.106 | +0.064 / -0.000 |
| bev/dyno | 195 (bev/dyno) | -0.136 / +0.194 | -0.095 / +0.059 |
| bev/hd | 120 (*/hd) | -0.090 / +0.280 | -0.029 / +0.184 |

**Compute.** Fit 0.002 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`r2_d2_transformer`'s card (4.883 CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
