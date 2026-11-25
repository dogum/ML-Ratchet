# formula_conformal: Trip-energy law (LawFinder-style symbolic regression, one formula per powertrain) + Mondrian conformal bands

**What.** The registered arm `formula` (hash `a5691cba86f7ee5c`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `formula`; only coverage, CRPS and band width change. Tracks: table, trace.

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

**Track table** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.50 / 0.85
-> in-sample 0.54 / 0.85 -> cross-fit 0.53 / 0.82; CRPS
0.1280 -> 0.1176 / 0.1211.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.60 / 0.99 | 0.58 / 0.90 | 0.54 / 0.86 | 0.081 | 0.066 | 0.069 |
| VED_HEV | 275 | 0.45 / 0.87 | 0.54 / 0.91 | 0.53 / 0.85 | 0.134 | 0.117 | 0.122 |
| VED_EV | 258 | 0.55 / 0.91 | 0.51 / 0.91 | 0.41 / 0.73 | 0.118 | 0.117 | 0.130 |
| VED_PHEV | 318 | 0.62 / 0.92 | 0.56 / 0.91 | 0.48 / 0.76 | 0.239 | 0.209 | 0.233 |
| EPA_conv_FTP | 2877 | 0.42 / 0.84 | 0.64 / 0.91 | 0.64 / 0.91 | 0.125 | 0.099 | 0.099 |
| EPA_conv_HWY | 2835 | 0.24 / 0.78 | 0.64 / 0.96 | 0.64 / 0.96 | 0.133 | 0.095 | 0.095 |
| EPA_conv_US06 | 371 | 0.47 / 0.91 | 0.79 / 0.90 | 0.79 / 0.90 | 0.109 | 0.091 | 0.091 |
| EPA_conv_FTPcold | 280 | 0.51 / 0.86 | 0.55 / 0.76 | 0.55 / 0.76 | 0.137 | 0.133 | 0.133 |
| EPA_conv_SC03 | 302 | 0.44 / 0.82 | 0.18 / 0.42 | 0.18 / 0.42 | 0.138 | 0.177 | 0.177 |
| EPA_hev_FTP | 70 | 0.76 / 0.80 | 0.69 / 0.89 | 0.69 / 0.86 | 0.108 | 0.090 | 0.090 |
| EPA_hev_HWY | 72 | 0.15 / 0.96 | 0.61 / 1.00 | 0.65 / 1.00 | 0.127 | 0.096 | 0.096 |
| EPA_bev_FTP | 98 | 0.70 / 0.99 | 0.42 / 0.86 | 0.41 / 0.85 | 0.080 | 0.066 | 0.067 |
| EPA_bev_HWY | 97 | 0.96 / 1.00 | 0.60 / 0.99 | 0.59 / 0.99 | 0.066 | 0.049 | 0.049 |
| Mersin_diesel | 56 | 0.52 / 0.82 | 0.54 / 0.93 | 0.54 / 0.91 | 0.111 | 0.092 | 0.092 |
| Mersin_cng | 56 | 0.23 / 0.71 | 0.52 / 0.95 | 0.54 / 0.95 | 0.162 | 0.115 | 0.115 |
| ICCT | 3 | 0.33 / 0.33 | 0.33 / 0.33 | 0.33 / 0.33 | 0.180 | 0.269 | 0.276 |
| ZTBus_legs | 5 | 0.60 / 1.00 | 0.80 / 1.00 | 0.80 / 1.00 | 0.094 | 0.102 | 0.104 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.83 / 0.99 | 0.51 / 0.92 | 0.50 / 0.92 | 0.073 | 0.058 |
| bev/hd | 5 | 0.60 / 1.00 | 0.80 / 1.00 | 0.80 / 1.00 | 0.094 | 0.104 |
| bev/road | 258 | 0.55 / 0.91 | 0.51 / 0.91 | 0.41 / 0.73 | 0.118 | 0.130 |
| hev/dyno | 229 | 0.45 / 0.92 | 0.59 / 0.93 | 0.60 / 0.92 | 0.111 | 0.096 |
| hev/road | 275 | 0.45 / 0.87 | 0.54 / 0.91 | 0.53 / 0.85 | 0.134 | 0.122 |
| ice/dyno | 6665 | 0.35 / 0.82 | 0.62 / 0.90 | 0.62 / 0.90 | 0.129 | 0.102 |
| ice/hd | 115 | 0.37 / 0.76 | 0.52 / 0.92 | 0.53 / 0.91 | 0.138 | 0.108 |
| ice/road | 645 | 0.60 / 0.99 | 0.58 / 0.90 | 0.54 / 0.86 | 0.081 | 0.069 |
| phev/road | 318 | 0.62 / 0.92 | 0.56 / 0.91 | 0.48 / 0.76 | 0.239 | 0.233 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | -0.248 / -0.075 | -0.164 / +0.047 |
| ice/dyno | 6665 (ice/dyno) | +0.099 / -0.284 | +0.154 / -0.215 |
| ice/hd | 115 (ice/hd) | -0.208 / +0.187 | -0.111 / +0.170 |
| hev/road | 275 (hev/road) | -0.196 / +0.177 | -0.184 / +0.135 |
| hev/dyno | 229 (hev/dyno) | +0.060 / -0.194 | +0.099 / -0.212 |
| hev/hd | 120 (*/hd) | -0.231 / +0.176 | -0.111 / +0.165 |
| phev/road | 318 (phev/road) | -0.528 / +0.082 | -0.377 / +0.088 |
| phev/dyno | 7089 (*/dyno) | +0.095 / -0.284 | +0.146 / -0.213 |
| phev/hd | 120 (*/hd) | -0.231 / +0.176 | -0.111 / +0.165 |
| bev/road | 258 (bev/road) | +0.004 / -0.039 | -0.018 / -0.014 |
| bev/dyno | 195 (bev/dyno) | -0.201 / -0.232 | -0.033 / -0.166 |
| bev/hd | 120 (*/hd) | -0.231 / +0.176 | -0.111 / +0.165 |
**Track trace** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.50 / 0.85
-> in-sample 0.55 / 0.85 -> cross-fit 0.53 / 0.82; CRPS
0.1280 -> 0.1176 / 0.1211.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.60 / 0.99 | 0.58 / 0.90 | 0.54 / 0.86 | 0.081 | 0.066 | 0.069 |
| VED_HEV | 275 | 0.45 / 0.87 | 0.54 / 0.91 | 0.53 / 0.85 | 0.134 | 0.117 | 0.122 |
| VED_EV | 258 | 0.55 / 0.91 | 0.51 / 0.91 | 0.41 / 0.73 | 0.118 | 0.117 | 0.130 |
| VED_PHEV | 318 | 0.62 / 0.92 | 0.56 / 0.91 | 0.48 / 0.76 | 0.239 | 0.209 | 0.233 |
| EPA_conv_FTP | 2877 | 0.42 / 0.84 | 0.64 / 0.91 | 0.64 / 0.91 | 0.125 | 0.099 | 0.099 |
| EPA_conv_HWY | 2835 | 0.25 / 0.78 | 0.64 / 0.96 | 0.64 / 0.96 | 0.133 | 0.095 | 0.095 |
| EPA_conv_US06 | 371 | 0.47 / 0.91 | 0.79 / 0.90 | 0.79 / 0.90 | 0.109 | 0.091 | 0.091 |
| EPA_conv_FTPcold | 280 | 0.51 / 0.86 | 0.55 / 0.76 | 0.55 / 0.76 | 0.137 | 0.133 | 0.133 |
| EPA_conv_SC03 | 302 | 0.44 / 0.82 | 0.18 / 0.42 | 0.18 / 0.42 | 0.138 | 0.177 | 0.177 |
| EPA_hev_FTP | 70 | 0.76 / 0.80 | 0.69 / 0.89 | 0.69 / 0.86 | 0.108 | 0.090 | 0.090 |
| EPA_hev_HWY | 72 | 0.15 / 0.96 | 0.61 / 1.00 | 0.65 / 1.00 | 0.127 | 0.096 | 0.096 |
| EPA_bev_FTP | 98 | 0.70 / 0.99 | 0.44 / 0.86 | 0.42 / 0.85 | 0.080 | 0.066 | 0.067 |
| EPA_bev_HWY | 97 | 0.96 / 1.00 | 0.60 / 0.99 | 0.59 / 0.99 | 0.066 | 0.049 | 0.049 |
| Mersin_diesel | 56 | 0.52 / 0.82 | 0.54 / 0.93 | 0.54 / 0.91 | 0.111 | 0.092 | 0.092 |
| Mersin_cng | 56 | 0.23 / 0.71 | 0.54 / 0.95 | 0.54 / 0.95 | 0.162 | 0.115 | 0.115 |
| ICCT | 3 | 0.33 / 0.33 | 0.33 / 0.33 | 0.33 / 0.33 | 0.180 | 0.270 | 0.276 |
| ZTBus_legs | 5 | 0.60 / 1.00 | 0.80 / 1.00 | 0.80 / 1.00 | 0.094 | 0.102 | 0.104 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.83 / 0.99 | 0.52 / 0.92 | 0.50 / 0.92 | 0.073 | 0.058 |
| bev/hd | 5 | 0.60 / 1.00 | 0.80 / 1.00 | 0.80 / 1.00 | 0.094 | 0.104 |
| bev/road | 258 | 0.55 / 0.91 | 0.51 / 0.91 | 0.41 / 0.73 | 0.118 | 0.130 |
| hev/dyno | 229 | 0.45 / 0.92 | 0.59 / 0.93 | 0.60 / 0.92 | 0.111 | 0.096 |
| hev/road | 275 | 0.45 / 0.87 | 0.54 / 0.91 | 0.53 / 0.85 | 0.134 | 0.122 |
| ice/dyno | 6665 | 0.36 / 0.82 | 0.62 / 0.90 | 0.62 / 0.90 | 0.128 | 0.102 |
| ice/hd | 115 | 0.37 / 0.76 | 0.53 / 0.92 | 0.53 / 0.91 | 0.138 | 0.108 |
| ice/road | 645 | 0.60 / 0.99 | 0.58 / 0.90 | 0.54 / 0.86 | 0.081 | 0.069 |
| phev/road | 318 | 0.62 / 0.92 | 0.56 / 0.91 | 0.48 / 0.76 | 0.239 | 0.233 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | -0.248 / -0.075 | -0.164 / +0.047 |
| ice/dyno | 6665 (ice/dyno) | +0.099 / -0.284 | +0.154 / -0.215 |
| ice/hd | 115 (ice/hd) | -0.208 / +0.187 | -0.111 / +0.170 |
| hev/road | 275 (hev/road) | -0.196 / +0.177 | -0.184 / +0.135 |
| hev/dyno | 229 (hev/dyno) | +0.060 / -0.194 | +0.099 / -0.212 |
| hev/hd | 120 (*/hd) | -0.231 / +0.176 | -0.111 / +0.165 |
| phev/road | 318 (phev/road) | -0.528 / +0.082 | -0.377 / +0.088 |
| phev/dyno | 7089 (*/dyno) | +0.095 / -0.284 | +0.146 / -0.213 |
| phev/hd | 120 (*/hd) | -0.231 / +0.176 | -0.111 / +0.165 |
| bev/road | 258 (bev/road) | +0.004 / -0.039 | -0.018 / -0.014 |
| bev/dyno | 195 (bev/dyno) | -0.201 / -0.232 | -0.033 / -0.165 |
| bev/hd | 120 (*/hd) | -0.231 / +0.176 | -0.111 / +0.165 |

**Compute.** Fit 0.013 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`formula`'s card (0.87 CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
