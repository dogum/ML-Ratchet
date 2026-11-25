# tab_tabpfn_conformal: TabPFN v2 in context on simulated rows (a lens, not a release; Built with PriorLabs-TabPFN) + Mondrian conformal bands

**What.** The registered arm `tab_tabpfn` (hash `cb26e0c49f796b7d`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `tab_tabpfn`; only coverage, CRPS and band width change. Tracks: table.

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

**Track table** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.42 / 0.86
-> in-sample 0.45 / 0.85 -> cross-fit 0.45 / 0.84; CRPS
0.0953 -> 0.0903 / 0.0914.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.51 / 0.91 | 0.50 / 0.90 | 0.49 / 0.85 | 0.054 | 0.054 | 0.056 |
| VED_HEV | 275 | 0.44 / 0.85 | 0.51 / 0.91 | 0.47 / 0.89 | 0.103 | 0.097 | 0.101 |
| VED_EV | 258 | 0.48 / 0.88 | 0.51 / 0.91 | 0.49 / 0.88 | 0.122 | 0.121 | 0.122 |
| VED_PHEV | 318 | 0.53 / 0.94 | 0.51 / 0.91 | 0.53 / 0.90 | 0.164 | 0.152 | 0.154 |
| EPA_conv_FTP | 2877 | 0.53 / 0.91 | 0.52 / 0.89 | 0.52 / 0.89 | 0.068 | 0.065 | 0.065 |
| EPA_conv_HWY | 2835 | 0.40 / 0.92 | 0.58 / 0.96 | 0.58 / 0.96 | 0.071 | 0.059 | 0.059 |
| EPA_conv_US06 | 371 | 0.16 / 0.80 | 0.33 / 0.89 | 0.33 / 0.89 | 0.106 | 0.083 | 0.083 |
| EPA_conv_FTPcold | 280 | 0.46 / 0.84 | 0.17 / 0.71 | 0.16 / 0.71 | 0.092 | 0.107 | 0.108 |
| EPA_conv_SC03 | 302 | 0.36 / 0.76 | 0.15 / 0.58 | 0.15 / 0.58 | 0.114 | 0.131 | 0.131 |
| EPA_hev_FTP | 70 | 0.39 / 1.00 | 0.59 / 0.94 | 0.59 / 0.94 | 0.068 | 0.065 | 0.065 |
| EPA_hev_HWY | 72 | 0.39 / 1.00 | 0.47 / 0.96 | 0.47 / 0.96 | 0.072 | 0.065 | 0.065 |
| EPA_bev_FTP | 98 | 0.65 / 0.88 | 0.42 / 0.89 | 0.42 / 0.86 | 0.078 | 0.079 | 0.079 |
| EPA_bev_HWY | 97 | 0.29 / 0.59 | 0.61 / 0.94 | 0.63 / 0.93 | 0.127 | 0.087 | 0.087 |
| Mersin_diesel | 56 | 0.46 / 0.82 | 0.50 / 0.95 | 0.50 / 0.91 | 0.087 | 0.079 | 0.080 |
| Mersin_cng | 56 | 0.27 / 0.71 | 0.52 / 0.91 | 0.52 / 0.91 | 0.130 | 0.100 | 0.102 |
| ICCT | 3 | 0.33 / 1.00 | 0.33 / 0.33 | 0.33 / 0.33 | 0.070 | 0.100 | 0.106 |
| ZTBus_legs | 5 | 0.20 / 1.00 | 0.60 / 1.00 | 0.80 / 0.80 | 0.102 | 0.100 | 0.103 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.47 / 0.73 | 0.51 / 0.91 | 0.52 / 0.89 | 0.102 | 0.083 |
| bev/hd | 5 | 0.20 / 1.00 | 0.60 / 1.00 | 0.80 / 0.80 | 0.102 | 0.103 |
| bev/road | 258 | 0.48 / 0.88 | 0.51 / 0.91 | 0.49 / 0.88 | 0.122 | 0.122 |
| hev/dyno | 229 | 0.46 / 0.98 | 0.51 / 0.91 | 0.50 / 0.90 | 0.071 | 0.068 |
| hev/road | 275 | 0.44 / 0.85 | 0.51 / 0.91 | 0.47 / 0.89 | 0.103 | 0.101 |
| ice/dyno | 6665 | 0.44 / 0.90 | 0.51 / 0.90 | 0.50 / 0.90 | 0.074 | 0.068 |
| ice/hd | 115 | 0.37 / 0.77 | 0.50 / 0.91 | 0.50 / 0.90 | 0.107 | 0.091 |
| ice/road | 645 | 0.51 / 0.91 | 0.50 / 0.90 | 0.49 / 0.85 | 0.054 | 0.056 |
| phev/road | 318 | 0.53 / 0.94 | 0.51 / 0.91 | 0.53 / 0.90 | 0.164 | 0.154 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | -0.018 / +0.008 | -0.022 / +0.015 |
| ice/dyno | 6665 (ice/dyno) | +0.055 / -0.082 | +0.057 / -0.086 |
| ice/hd | 115 (ice/hd) | -0.103 / +0.234 | -0.056 / +0.119 |
| hev/road | 275 (hev/road) | +0.071 / -0.074 | +0.084 / -0.063 |
| hev/dyno | 229 (hev/dyno) | -0.039 / -0.084 | +0.054 / -0.023 |
| hev/hd | 120 (*/hd) | -0.105 / +0.230 | -0.056 / +0.119 |
| phev/road | 318 (phev/road) | -0.211 / +0.111 | -0.135 / +0.121 |
| phev/dyno | 7089 (*/dyno) | +0.049 / -0.068 | +0.055 / -0.080 |
| phev/hd | 120 (*/hd) | -0.105 / +0.230 | -0.056 / +0.119 |
| bev/road | 258 (bev/road) | +0.053 / +0.076 | +0.029 / -0.014 |
| bev/dyno | 195 (bev/dyno) | -0.176 / +0.233 | -0.123 / +0.137 |
| bev/hd | 120 (*/hd) | -0.105 / +0.230 | -0.056 / +0.119 |

**Compute.** Fit 0.058 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`tab_tabpfn`'s card (0.0 CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
