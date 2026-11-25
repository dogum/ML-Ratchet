# tab_mlp_conformal: MLP control (EnergyNet v2.1's network, retrained on the shared bank) + Mondrian conformal bands

**What.** The registered arm `tab_mlp` (hash `ab35c4248d5c8283`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `tab_mlp`; only coverage, CRPS and band width change. Tracks: table.

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

**Track table** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.33 / 0.73
-> in-sample 0.48 / 0.86 -> cross-fit 0.47 / 0.85; CRPS
0.0993 -> 0.0918 / 0.0936.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.34 / 0.73 | 0.50 / 0.90 | 0.46 / 0.88 | 0.055 | 0.052 | 0.054 |
| VED_HEV | 275 | 0.43 / 0.84 | 0.51 / 0.91 | 0.50 / 0.91 | 0.098 | 0.097 | 0.099 |
| VED_EV | 258 | 0.34 / 0.69 | 0.51 / 0.91 | 0.46 / 0.90 | 0.111 | 0.106 | 0.109 |
| VED_PHEV | 318 | 0.24 / 0.54 | 0.51 / 0.91 | 0.49 / 0.87 | 0.193 | 0.162 | 0.166 |
| EPA_conv_FTP | 2877 | 0.43 / 0.81 | 0.47 / 0.88 | 0.47 / 0.88 | 0.069 | 0.068 | 0.068 |
| EPA_conv_HWY | 2835 | 0.52 / 0.87 | 0.57 / 0.95 | 0.57 / 0.95 | 0.048 | 0.048 | 0.048 |
| EPA_conv_US06 | 371 | 0.38 / 0.82 | 0.42 / 0.90 | 0.42 / 0.90 | 0.074 | 0.074 | 0.074 |
| EPA_conv_FTPcold | 280 | 0.31 / 0.73 | 0.38 / 0.83 | 0.38 / 0.82 | 0.089 | 0.086 | 0.086 |
| EPA_conv_SC03 | 302 | 0.32 / 0.66 | 0.35 / 0.76 | 0.35 / 0.76 | 0.106 | 0.102 | 0.102 |
| EPA_hev_FTP | 70 | 0.21 / 0.76 | 0.47 / 0.90 | 0.49 / 0.90 | 0.112 | 0.098 | 0.099 |
| EPA_hev_HWY | 72 | 0.24 / 0.93 | 0.54 / 0.86 | 0.53 / 0.83 | 0.057 | 0.059 | 0.059 |
| EPA_bev_FTP | 98 | 0.46 / 0.82 | 0.62 / 0.93 | 0.62 / 0.92 | 0.088 | 0.070 | 0.070 |
| EPA_bev_HWY | 97 | 0.45 / 0.75 | 0.53 / 0.91 | 0.52 / 0.90 | 0.069 | 0.054 | 0.055 |
| Mersin_diesel | 56 | 0.25 / 0.73 | 0.52 / 0.91 | 0.54 / 0.89 | 0.113 | 0.102 | 0.103 |
| Mersin_cng | 56 | 0.05 / 0.61 | 0.50 / 0.95 | 0.46 / 0.93 | 0.147 | 0.114 | 0.115 |
| ICCT | 3 | 0.33 / 0.33 | 0.33 / 0.33 | 0.33 / 0.33 | 0.159 | 0.178 | 0.190 |
| ZTBus_legs | 5 | 0.40 / 1.00 | 0.80 / 1.00 | 0.80 / 1.00 | 0.080 | 0.084 | 0.087 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.46 / 0.78 | 0.57 / 0.92 | 0.57 / 0.91 | 0.079 | 0.062 |
| bev/hd | 5 | 0.40 / 1.00 | 0.80 / 1.00 | 0.80 / 1.00 | 0.080 | 0.087 |
| bev/road | 258 | 0.34 / 0.69 | 0.51 / 0.91 | 0.46 / 0.90 | 0.111 | 0.109 |
| hev/dyno | 229 | 0.27 / 0.88 | 0.51 / 0.91 | 0.51 / 0.90 | 0.081 | 0.076 |
| hev/road | 275 | 0.43 / 0.84 | 0.51 / 0.91 | 0.50 / 0.91 | 0.098 | 0.099 |
| ice/dyno | 6665 | 0.45 / 0.83 | 0.50 / 0.90 | 0.50 / 0.90 | 0.063 | 0.062 |
| ice/hd | 115 | 0.16 / 0.66 | 0.50 / 0.91 | 0.50 / 0.90 | 0.131 | 0.111 |
| ice/road | 645 | 0.34 / 0.73 | 0.50 / 0.90 | 0.46 / 0.88 | 0.055 | 0.054 |
| phev/road | 318 | 0.24 / 0.54 | 0.51 / 0.91 | 0.49 / 0.87 | 0.193 | 0.166 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | +0.027 / +0.076 | -0.003 / +0.043 |
| ice/dyno | 6665 (ice/dyno) | +0.048 / +0.033 | +0.004 / +0.014 |
| ice/hd | 115 (ice/hd) | +0.003 / +0.207 | +0.003 / +0.163 |
| hev/road | 275 (hev/road) | +0.107 / +0.049 | +0.039 / +0.005 |
| hev/dyno | 229 (hev/dyno) | -0.013 / +0.141 | +0.034 / +0.050 |
| hev/hd | 120 (*/hd) | -0.007 / +0.187 | +0.003 / +0.161 |
| phev/road | 318 (phev/road) | +0.103 / +0.471 | +0.000 / +0.240 |
| phev/dyno | 7089 (*/dyno) | +0.044 / +0.038 | +0.003 / +0.017 |
| phev/hd | 120 (*/hd) | -0.007 / +0.187 | +0.003 / +0.161 |
| bev/road | 258 (bev/road) | +0.169 / +0.114 | +0.065 / +0.027 |
| bev/dyno | 195 (bev/dyno) | -0.088 / +0.245 | -0.069 / +0.079 |
| bev/hd | 120 (*/hd) | -0.007 / +0.187 | +0.003 / +0.161 |

**Compute.** Fit 0.007 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`tab_mlp`'s card (0.62 CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
