# trace_s4d_conformal: S4D state-space model on the trip trace + Mondrian conformal bands

**What.** The registered arm `trace_s4d` (hash `280230f3ffa5dcee`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `trace_s4d`; only coverage, CRPS and band width change. Tracks: trace.

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

**Track trace** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.36 / 0.72
-> in-sample 0.52 / 0.86 -> cross-fit 0.51 / 0.85; CRPS
0.1035 -> 0.0936 / 0.0950.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.40 / 0.83 | 0.50 / 0.90 | 0.46 / 0.88 | 0.056 | 0.054 | 0.056 |
| VED_HEV | 275 | 0.41 / 0.85 | 0.51 / 0.91 | 0.46 / 0.89 | 0.098 | 0.097 | 0.099 |
| VED_EV | 258 | 0.44 / 0.77 | 0.51 / 0.91 | 0.52 / 0.86 | 0.113 | 0.109 | 0.112 |
| VED_PHEV | 318 | 0.35 / 0.74 | 0.57 / 0.91 | 0.55 / 0.88 | 0.222 | 0.184 | 0.188 |
| EPA_conv_FTP | 2877 | 0.36 / 0.79 | 0.42 / 0.86 | 0.42 / 0.86 | 0.079 | 0.078 | 0.078 |
| EPA_conv_HWY | 2835 | 0.54 / 0.90 | 0.61 / 0.96 | 0.61 / 0.96 | 0.053 | 0.054 | 0.054 |
| EPA_conv_US06 | 371 | 0.40 / 0.78 | 0.47 / 0.86 | 0.47 / 0.86 | 0.079 | 0.076 | 0.076 |
| EPA_conv_FTPcold | 280 | 0.28 / 0.78 | 0.32 / 0.81 | 0.32 / 0.81 | 0.104 | 0.102 | 0.102 |
| EPA_conv_SC03 | 302 | 0.32 / 0.76 | 0.38 / 0.81 | 0.39 / 0.82 | 0.105 | 0.103 | 0.103 |
| EPA_hev_FTP | 70 | 0.47 / 0.76 | 0.60 / 0.93 | 0.57 / 0.93 | 0.088 | 0.078 | 0.080 |
| EPA_hev_HWY | 72 | 0.56 / 1.00 | 0.61 / 0.86 | 0.58 / 0.86 | 0.048 | 0.057 | 0.056 |
| EPA_bev_FTP | 98 | 0.43 / 0.84 | 0.76 / 0.91 | 0.76 / 0.90 | 0.099 | 0.075 | 0.076 |
| EPA_bev_HWY | 97 | 0.35 / 0.67 | 0.66 / 0.96 | 0.67 / 0.96 | 0.090 | 0.063 | 0.063 |
| Mersin_diesel | 56 | 0.32 / 0.64 | 0.50 / 0.96 | 0.52 / 0.96 | 0.109 | 0.090 | 0.092 |
| Mersin_cng | 56 | 0.11 / 0.38 | 0.52 / 0.91 | 0.50 / 0.91 | 0.166 | 0.112 | 0.113 |
| ICCT | 3 | 0.00 / 0.00 | 0.33 / 0.33 | 0.33 / 0.33 | 0.145 | 0.168 | 0.173 |
| ZTBus_legs | 5 | 0.60 / 0.80 | 1.00 / 1.00 | 1.00 / 1.00 | 0.084 | 0.080 | 0.082 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.39 / 0.75 | 0.71 / 0.93 | 0.71 / 0.93 | 0.095 | 0.070 |
| bev/hd | 5 | 0.60 / 0.80 | 1.00 / 1.00 | 1.00 / 1.00 | 0.084 | 0.082 |
| bev/road | 258 | 0.44 / 0.77 | 0.51 / 0.91 | 0.52 / 0.86 | 0.113 | 0.112 |
| hev/dyno | 229 | 0.45 / 0.89 | 0.51 / 0.91 | 0.49 / 0.91 | 0.071 | 0.070 |
| hev/road | 275 | 0.41 / 0.85 | 0.51 / 0.91 | 0.46 / 0.89 | 0.098 | 0.099 |
| ice/dyno | 6665 | 0.43 / 0.83 | 0.50 / 0.90 | 0.50 / 0.90 | 0.070 | 0.070 |
| ice/hd | 115 | 0.21 / 0.50 | 0.50 / 0.92 | 0.50 / 0.92 | 0.138 | 0.104 |
| ice/road | 645 | 0.40 / 0.83 | 0.50 / 0.90 | 0.46 / 0.88 | 0.056 | 0.056 |
| phev/road | 318 | 0.35 / 0.74 | 0.57 / 0.91 | 0.55 / 0.88 | 0.222 | 0.188 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | -0.011 / +0.056 | -0.025 / +0.052 |
| ice/dyno | 6665 (ice/dyno) | +0.052 / +0.033 | +0.018 / +0.010 |
| ice/hd | 115 (ice/hd) | -0.047 / +0.296 | -0.040 / +0.185 |
| hev/road | 275 (hev/road) | +0.056 / +0.035 | +0.028 / +0.015 |
| hev/dyno | 229 (hev/dyno) | -0.035 / +0.122 | +0.006 / +0.066 |
| hev/hd | 120 (*/hd) | -0.054 / +0.296 | -0.050 / +0.177 |
| phev/road | 318 (phev/road) | -0.179 / +0.384 | -0.194 / +0.291 |
| phev/dyno | 7089 (*/dyno) | +0.043 / +0.041 | +0.015 / +0.015 |
| phev/hd | 120 (*/hd) | -0.054 / +0.296 | -0.050 / +0.177 |
| bev/road | 258 (bev/road) | +0.135 / +0.127 | -0.006 / +0.044 |
| bev/dyno | 195 (bev/dyno) | -0.179 / +0.278 | -0.121 / +0.098 |
| bev/hd | 120 (*/hd) | -0.054 / +0.296 | -0.050 / +0.177 |

**Compute.** Fit 0.008 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`trace_s4d`'s card (2.75 CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
