# phys_anchor_conformal: Physics anchor: road-load estimate E_phys alone (+ simulation bands) + Mondrian conformal bands

**What.** The registered arm `phys_anchor` (hash `c646d0fb9e424d1b`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `phys_anchor`; only coverage, CRPS and band width change. Tracks: table, trace.

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

**Track table** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.47 / 0.85
-> in-sample 0.47 / 0.83 -> cross-fit 0.45 / 0.81; CRPS
0.1534 -> 0.1456 / 0.1495.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.63 / 0.99 | 0.57 / 0.90 | 0.54 / 0.89 | 0.094 | 0.079 | 0.082 |
| VED_HEV | 275 | 0.63 / 0.94 | 0.50 / 0.91 | 0.49 / 0.87 | 0.116 | 0.111 | 0.115 |
| VED_EV | 258 | 0.55 / 0.91 | 0.51 / 0.91 | 0.29 / 0.72 | 0.141 | 0.140 | 0.161 |
| VED_PHEV | 318 | 0.64 / 0.91 | 0.51 / 0.91 | 0.44 / 0.83 | 0.221 | 0.206 | 0.226 |
| EPA_conv_FTP | 2877 | 0.68 / 0.95 | 0.56 / 0.92 | 0.56 / 0.92 | 0.095 | 0.095 | 0.095 |
| EPA_conv_HWY | 2835 | 0.13 / 0.68 | 0.54 / 0.94 | 0.54 / 0.94 | 0.172 | 0.118 | 0.118 |
| EPA_conv_US06 | 371 | 0.09 / 0.70 | 0.49 / 0.90 | 0.49 / 0.90 | 0.200 | 0.141 | 0.141 |
| EPA_conv_FTPcold | 280 | 0.67 / 0.87 | 0.34 / 0.76 | 0.34 / 0.75 | 0.125 | 0.135 | 0.135 |
| EPA_conv_SC03 | 302 | 0.52 / 0.87 | 0.19 / 0.54 | 0.19 / 0.54 | 0.136 | 0.167 | 0.167 |
| EPA_hev_FTP | 70 | 0.79 / 1.00 | 0.51 / 0.93 | 0.51 / 0.93 | 0.092 | 0.076 | 0.077 |
| EPA_hev_HWY | 72 | 0.24 / 1.00 | 0.62 / 0.96 | 0.68 / 0.96 | 0.120 | 0.089 | 0.090 |
| EPA_bev_FTP | 98 | 0.27 / 1.00 | 0.42 / 0.89 | 0.42 / 0.89 | 0.122 | 0.090 | 0.090 |
| EPA_bev_HWY | 97 | 0.96 / 1.00 | 0.74 / 0.95 | 0.74 / 0.93 | 0.076 | 0.059 | 0.059 |
| Mersin_diesel | 56 | 0.54 / 0.75 | 0.46 / 0.91 | 0.46 / 0.89 | 0.171 | 0.144 | 0.147 |
| Mersin_cng | 56 | 0.23 / 0.77 | 0.57 / 0.98 | 0.57 / 0.98 | 0.205 | 0.153 | 0.155 |
| ICCT | 3 | 0.00 / 0.33 | 0.00 / 0.00 | 0.00 / 0.00 | 0.368 | 0.527 | 0.534 |
| ZTBus_legs | 5 | 0.20 / 1.00 | 0.80 / 0.80 | 0.80 / 0.80 | 0.139 | 0.130 | 0.133 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.61 / 1.00 | 0.58 / 0.92 | 0.58 / 0.91 | 0.099 | 0.075 |
| bev/hd | 5 | 0.20 / 1.00 | 0.80 / 0.80 | 0.80 / 0.80 | 0.139 | 0.133 |
| bev/road | 258 | 0.55 / 0.91 | 0.51 / 0.91 | 0.29 / 0.72 | 0.141 | 0.161 |
| hev/dyno | 229 | 0.45 / 1.00 | 0.54 / 0.91 | 0.55 / 0.91 | 0.109 | 0.096 |
| hev/road | 275 | 0.63 / 0.94 | 0.50 / 0.91 | 0.49 / 0.87 | 0.116 | 0.115 |
| ice/dyno | 6665 | 0.41 / 0.82 | 0.52 / 0.90 | 0.52 / 0.90 | 0.137 | 0.112 |
| ice/hd | 115 | 0.37 / 0.75 | 0.50 / 0.92 | 0.50 / 0.91 | 0.193 | 0.161 |
| ice/road | 645 | 0.63 / 0.99 | 0.57 / 0.90 | 0.54 / 0.89 | 0.094 | 0.082 |
| phev/road | 318 | 0.64 / 0.91 | 0.51 / 0.91 | 0.44 / 0.83 | 0.221 | 0.226 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | -0.248 / -0.126 | -0.183 / +0.042 |
| ice/dyno | 6665 (ice/dyno) | +0.111 / -0.329 | +0.156 / -0.179 |
| ice/hd | 115 (ice/hd) | -0.287 / +0.245 | -0.122 / +0.249 |
| hev/road | 275 (hev/road) | -0.142 / -0.007 | -0.113 / +0.041 |
| hev/dyno | 229 (hev/dyno) | -0.026 / -0.306 | +0.059 / -0.183 |
| hev/hd | 120 (*/hd) | -0.287 / +0.234 | -0.122 / +0.247 |
| phev/road | 318 (phev/road) | -0.420 / +0.156 | -0.251 / +0.058 |
| phev/dyno | 7089 (*/dyno) | +0.109 / -0.329 | +0.151 / -0.182 |
| phev/hd | 120 (*/hd) | -0.287 / +0.234 | -0.122 / +0.247 |
| bev/road | 258 (bev/road) | -0.034 / +0.099 | +0.011 / -0.060 |
| bev/dyno | 195 (bev/dyno) | -0.116 / -0.440 | +0.052 / -0.230 |
| bev/hd | 120 (*/hd) | -0.287 / +0.234 | -0.122 / +0.247 |
**Track trace** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.47 / 0.85
-> in-sample 0.47 / 0.83 -> cross-fit 0.45 / 0.81; CRPS
0.1534 -> 0.1456 / 0.1495.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.63 / 0.99 | 0.57 / 0.90 | 0.54 / 0.89 | 0.094 | 0.079 | 0.082 |
| VED_HEV | 275 | 0.63 / 0.94 | 0.51 / 0.91 | 0.49 / 0.87 | 0.116 | 0.111 | 0.115 |
| VED_EV | 258 | 0.55 / 0.91 | 0.51 / 0.91 | 0.29 / 0.72 | 0.141 | 0.140 | 0.161 |
| VED_PHEV | 318 | 0.64 / 0.91 | 0.51 / 0.91 | 0.44 / 0.83 | 0.221 | 0.206 | 0.226 |
| EPA_conv_FTP | 2877 | 0.68 / 0.95 | 0.56 / 0.92 | 0.56 / 0.92 | 0.095 | 0.095 | 0.095 |
| EPA_conv_HWY | 2835 | 0.13 / 0.68 | 0.54 / 0.94 | 0.54 / 0.94 | 0.172 | 0.118 | 0.118 |
| EPA_conv_US06 | 371 | 0.09 / 0.70 | 0.49 / 0.90 | 0.49 / 0.90 | 0.200 | 0.141 | 0.141 |
| EPA_conv_FTPcold | 280 | 0.67 / 0.87 | 0.34 / 0.76 | 0.34 / 0.76 | 0.125 | 0.135 | 0.135 |
| EPA_conv_SC03 | 302 | 0.52 / 0.87 | 0.19 / 0.54 | 0.19 / 0.54 | 0.136 | 0.167 | 0.167 |
| EPA_hev_FTP | 70 | 0.79 / 1.00 | 0.51 / 0.89 | 0.51 / 0.91 | 0.092 | 0.076 | 0.077 |
| EPA_hev_HWY | 72 | 0.24 / 1.00 | 0.62 / 0.96 | 0.68 / 0.96 | 0.120 | 0.089 | 0.090 |
| EPA_bev_FTP | 98 | 0.27 / 1.00 | 0.42 / 0.89 | 0.42 / 0.89 | 0.122 | 0.090 | 0.090 |
| EPA_bev_HWY | 97 | 0.96 / 1.00 | 0.74 / 0.95 | 0.74 / 0.93 | 0.076 | 0.059 | 0.059 |
| Mersin_diesel | 56 | 0.54 / 0.75 | 0.46 / 0.91 | 0.46 / 0.89 | 0.171 | 0.144 | 0.147 |
| Mersin_cng | 56 | 0.23 / 0.77 | 0.57 / 0.98 | 0.57 / 0.98 | 0.205 | 0.153 | 0.155 |
| ICCT | 3 | 0.00 / 0.33 | 0.00 / 0.00 | 0.00 / 0.00 | 0.368 | 0.527 | 0.534 |
| ZTBus_legs | 5 | 0.20 / 1.00 | 0.80 / 0.80 | 0.80 / 0.80 | 0.139 | 0.130 | 0.133 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.61 / 1.00 | 0.58 / 0.92 | 0.58 / 0.91 | 0.099 | 0.075 |
| bev/hd | 5 | 0.20 / 1.00 | 0.80 / 0.80 | 0.80 / 0.80 | 0.139 | 0.133 |
| bev/road | 258 | 0.55 / 0.91 | 0.51 / 0.91 | 0.29 / 0.72 | 0.141 | 0.161 |
| hev/dyno | 229 | 0.45 / 1.00 | 0.54 / 0.90 | 0.55 / 0.91 | 0.109 | 0.096 |
| hev/road | 275 | 0.63 / 0.94 | 0.51 / 0.91 | 0.49 / 0.87 | 0.116 | 0.115 |
| ice/dyno | 6665 | 0.41 / 0.82 | 0.52 / 0.90 | 0.52 / 0.90 | 0.137 | 0.112 |
| ice/hd | 115 | 0.37 / 0.75 | 0.50 / 0.92 | 0.50 / 0.91 | 0.193 | 0.161 |
| ice/road | 645 | 0.63 / 0.99 | 0.57 / 0.90 | 0.54 / 0.89 | 0.094 | 0.082 |
| phev/road | 318 | 0.64 / 0.91 | 0.51 / 0.91 | 0.44 / 0.83 | 0.221 | 0.226 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | -0.248 / -0.126 | -0.183 / +0.042 |
| ice/dyno | 6665 (ice/dyno) | +0.111 / -0.329 | +0.156 / -0.179 |
| ice/hd | 115 (ice/hd) | -0.287 / +0.245 | -0.122 / +0.249 |
| hev/road | 275 (hev/road) | -0.142 / -0.007 | -0.113 / +0.041 |
| hev/dyno | 229 (hev/dyno) | -0.026 / -0.306 | +0.059 / -0.183 |
| hev/hd | 120 (*/hd) | -0.287 / +0.234 | -0.122 / +0.247 |
| phev/road | 318 (phev/road) | -0.420 / +0.156 | -0.251 / +0.058 |
| phev/dyno | 7089 (*/dyno) | +0.109 / -0.329 | +0.150 / -0.182 |
| phev/hd | 120 (*/hd) | -0.287 / +0.234 | -0.122 / +0.247 |
| bev/road | 258 (bev/road) | -0.034 / +0.099 | +0.011 / -0.060 |
| bev/dyno | 195 (bev/dyno) | -0.116 / -0.440 | +0.052 / -0.230 |
| bev/hd | 120 (*/hd) | -0.287 / +0.234 | -0.122 / +0.247 |

**Compute.** Fit 0.013 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`phys_anchor`'s card (0.002 CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
