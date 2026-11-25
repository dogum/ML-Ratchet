# tab_gp_conformal: Sparse Gaussian process (SGPR, 500 inducing points) + Mondrian conformal bands

**What.** The registered arm `tab_gp` (hash `131f5c73c54ed1a9`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `tab_gp`; only coverage, CRPS and band width change. Tracks: table.

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

**Track table** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.61 / 0.93
-> in-sample 0.47 / 0.86 -> cross-fit 0.46 / 0.85; CRPS
0.0986 -> 0.0943 / 0.0960.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.75 / 1.00 | 0.50 / 0.90 | 0.48 / 0.89 | 0.067 | 0.055 | 0.057 |
| VED_HEV | 275 | 0.62 / 0.95 | 0.50 / 0.91 | 0.48 / 0.89 | 0.090 | 0.088 | 0.093 |
| VED_EV | 258 | 0.55 / 0.88 | 0.51 / 0.91 | 0.37 / 0.81 | 0.116 | 0.115 | 0.123 |
| VED_PHEV | 318 | 0.48 / 0.77 | 0.51 / 0.91 | 0.51 / 0.91 | 0.187 | 0.168 | 0.169 |
| EPA_conv_FTP | 2877 | 0.82 / 0.98 | 0.52 / 0.89 | 0.52 / 0.89 | 0.069 | 0.064 | 0.064 |
| EPA_conv_HWY | 2835 | 0.70 / 1.00 | 0.55 / 0.95 | 0.55 / 0.95 | 0.071 | 0.058 | 0.058 |
| EPA_conv_US06 | 371 | 0.31 / 0.92 | 0.25 / 0.84 | 0.25 / 0.84 | 0.113 | 0.104 | 0.104 |
| EPA_conv_FTPcold | 280 | 0.79 / 0.89 | 0.45 / 0.83 | 0.45 / 0.83 | 0.093 | 0.092 | 0.092 |
| EPA_conv_SC03 | 302 | 0.60 / 0.90 | 0.20 / 0.66 | 0.20 / 0.66 | 0.104 | 0.117 | 0.118 |
| EPA_hev_FTP | 70 | 0.37 / 0.97 | 0.30 / 0.87 | 0.31 / 0.89 | 0.097 | 0.089 | 0.090 |
| EPA_hev_HWY | 72 | 0.94 / 1.00 | 0.89 / 1.00 | 0.89 / 1.00 | 0.058 | 0.053 | 0.053 |
| EPA_bev_FTP | 98 | 0.85 / 0.90 | 0.44 / 0.85 | 0.44 / 0.83 | 0.076 | 0.067 | 0.068 |
| EPA_bev_HWY | 97 | 0.80 / 0.99 | 0.58 / 0.99 | 0.56 / 0.99 | 0.069 | 0.055 | 0.056 |
| Mersin_diesel | 56 | 0.52 / 0.89 | 0.50 / 0.95 | 0.52 / 0.93 | 0.100 | 0.094 | 0.095 |
| Mersin_cng | 56 | 0.29 / 0.82 | 0.50 / 0.93 | 0.52 / 0.91 | 0.128 | 0.109 | 0.109 |
| ICCT | 3 | 0.33 / 1.00 | 0.33 / 0.33 | 0.33 / 0.33 | 0.138 | 0.180 | 0.189 |
| ZTBus_legs | 5 | 0.60 / 1.00 | 0.80 / 1.00 | 0.80 / 1.00 | 0.076 | 0.076 | 0.079 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.83 / 0.94 | 0.51 / 0.92 | 0.50 / 0.91 | 0.073 | 0.062 |
| bev/hd | 5 | 0.60 / 1.00 | 0.80 / 1.00 | 0.80 / 1.00 | 0.076 | 0.079 |
| bev/road | 258 | 0.55 / 0.88 | 0.51 / 0.91 | 0.37 / 0.81 | 0.116 | 0.123 |
| hev/dyno | 229 | 0.54 / 0.99 | 0.50 / 0.91 | 0.50 / 0.90 | 0.083 | 0.079 |
| hev/road | 275 | 0.62 / 0.95 | 0.50 / 0.91 | 0.48 / 0.89 | 0.090 | 0.093 |
| ice/dyno | 6665 | 0.73 / 0.98 | 0.50 / 0.90 | 0.50 / 0.90 | 0.075 | 0.068 |
| ice/hd | 115 | 0.40 / 0.86 | 0.50 / 0.92 | 0.51 / 0.90 | 0.115 | 0.105 |
| ice/road | 645 | 0.75 / 1.00 | 0.50 / 0.90 | 0.48 / 0.89 | 0.067 | 0.057 |
| phev/road | 318 | 0.48 / 0.77 | 0.51 / 0.91 | 0.51 / 0.91 | 0.187 | 0.169 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | -0.232 / -0.116 | -0.135 / -0.012 |
| ice/dyno | 6665 (ice/dyno) | -0.093 / -0.191 | -0.017 / -0.105 |
| ice/hd | 115 (ice/hd) | -0.165 / +0.076 | -0.064 / +0.096 |
| hev/road | 275 (hev/road) | -0.038 / -0.075 | -0.006 / -0.077 |
| hev/dyno | 229 (hev/dyno) | -0.157 / -0.047 | -0.040 / +0.015 |
| hev/hd | 120 (*/hd) | -0.187 / +0.075 | -0.064 / +0.092 |
| phev/road | 318 (phev/road) | -0.163 / +0.401 | -0.105 / +0.185 |
| phev/dyno | 7089 (*/dyno) | -0.098 / -0.178 | -0.020 / -0.101 |
| phev/hd | 120 (*/hd) | -0.187 / +0.075 | -0.064 / +0.092 |
| bev/road | 258 (bev/road) | +0.136 / +0.049 | +0.009 / -0.031 |
| bev/dyno | 195 (bev/dyno) | -0.244 / +0.013 | -0.107 / -0.039 |
| bev/hd | 120 (*/hd) | -0.187 / +0.075 | -0.064 / +0.092 |

**Compute.** Fit 0.007 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`tab_gp`'s card (0.81 CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
