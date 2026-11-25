# enet_v21_conformal: EnergyNet v2.1 as released + Mondrian conformal bands

**What.** The registered arm `enet_v21` (hash `7f29a28638ca3e91`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `enet_v21`; only coverage, CRPS and band width change. Tracks: table.

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

**Track table** (8705 dev units). Mean over the 16 D16 sets: cov50 / cov90 raw 0.47 / 0.86
-> in-sample 0.48 / 0.86 -> cross-fit 0.48 / 0.86; CRPS
0.0926 -> 0.0916 / 0.0927.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
| VED_ICE | 645 | 0.49 / 0.89 | 0.50 / 0.90 | 0.46 / 0.88 | 0.051 | 0.051 | 0.052 |
| VED_HEV | 275 | 0.52 / 0.88 | 0.51 / 0.91 | 0.50 / 0.90 | 0.103 | 0.102 | 0.105 |
| VED_EV | 258 | 0.49 / 0.88 | 0.51 / 0.91 | 0.48 / 0.86 | 0.115 | 0.114 | 0.116 |
| VED_PHEV | 318 | 0.53 / 0.84 | 0.51 / 0.91 | 0.49 / 0.90 | 0.175 | 0.163 | 0.166 |
| EPA_conv_FTP | 2877 | 0.46 / 0.82 | 0.45 / 0.87 | 0.45 / 0.87 | 0.078 | 0.075 | 0.075 |
| EPA_conv_HWY | 2835 | 0.56 / 0.91 | 0.55 / 0.94 | 0.56 / 0.94 | 0.053 | 0.053 | 0.053 |
| EPA_conv_US06 | 371 | 0.64 / 0.91 | 0.58 / 0.91 | 0.58 / 0.91 | 0.060 | 0.064 | 0.064 |
| EPA_conv_FTPcold | 280 | 0.34 / 0.67 | 0.41 / 0.81 | 0.41 / 0.81 | 0.101 | 0.096 | 0.096 |
| EPA_conv_SC03 | 302 | 0.40 / 0.70 | 0.43 / 0.83 | 0.43 / 0.84 | 0.102 | 0.098 | 0.098 |
| EPA_hev_FTP | 70 | 0.24 / 0.77 | 0.40 / 0.91 | 0.41 / 0.91 | 0.110 | 0.099 | 0.099 |
| EPA_hev_HWY | 72 | 0.35 / 1.00 | 0.44 / 0.86 | 0.46 / 0.86 | 0.062 | 0.066 | 0.067 |
| EPA_bev_FTP | 98 | 0.71 / 0.90 | 0.78 / 0.99 | 0.77 / 0.99 | 0.083 | 0.075 | 0.075 |
| EPA_bev_HWY | 97 | 0.41 / 0.77 | 0.29 / 0.82 | 0.30 / 0.82 | 0.061 | 0.052 | 0.052 |
| Mersin_diesel | 56 | 0.62 / 0.91 | 0.46 / 0.89 | 0.48 / 0.88 | 0.101 | 0.091 | 0.092 |
| Mersin_cng | 56 | 0.34 / 0.86 | 0.57 / 0.98 | 0.54 / 0.98 | 0.125 | 0.101 | 0.102 |
| ICCT | 3 | 0.33 / 1.00 | 0.33 / 0.33 | 0.33 / 0.33 | 0.103 | 0.165 | 0.169 |
| ZTBus_legs | 5 | 0.80 / 1.00 | 0.80 / 0.80 | 0.80 / 0.80 | 0.071 | 0.071 | 0.074 |

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
| bev/dyno | 195 | 0.56 / 0.84 | 0.53 / 0.91 | 0.53 / 0.91 | 0.072 | 0.064 |
| bev/hd | 5 | 0.80 / 1.00 | 0.80 / 0.80 | 0.80 / 0.80 | 0.071 | 0.074 |
| bev/road | 258 | 0.49 / 0.88 | 0.51 / 0.91 | 0.48 / 0.86 | 0.115 | 0.116 |
| hev/dyno | 229 | 0.35 / 0.91 | 0.50 / 0.91 | 0.50 / 0.91 | 0.080 | 0.078 |
| hev/road | 275 | 0.52 / 0.88 | 0.51 / 0.91 | 0.50 / 0.90 | 0.103 | 0.105 |
| ice/dyno | 6665 | 0.50 / 0.85 | 0.50 / 0.90 | 0.50 / 0.90 | 0.068 | 0.067 |
| ice/hd | 115 | 0.48 / 0.89 | 0.51 / 0.92 | 0.50 / 0.91 | 0.113 | 0.099 |
| ice/road | 645 | 0.49 / 0.89 | 0.50 / 0.90 | 0.46 / 0.88 | 0.051 | 0.052 |
| phev/road | 318 | 0.53 / 0.84 | 0.51 / 0.91 | 0.49 / 0.90 | 0.175 | 0.166 |

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
| ice/road | 645 (ice/road) | -0.009 / +0.021 | -0.009 / +0.013 |
| ice/dyno | 6665 (ice/dyno) | -0.002 / +0.076 | -0.021 / +0.023 |
| ice/hd | 115 (ice/hd) | -0.254 / +0.084 | -0.121 / +0.110 |
| hev/road | 275 (hev/road) | +0.055 / -0.008 | +0.012 / -0.013 |
| hev/dyno | 229 (hev/dyno) | -0.057 / +0.073 | +0.007 / +0.038 |
| hev/hd | 120 (*/hd) | -0.254 / +0.082 | -0.121 / +0.101 |
| phev/road | 318 (phev/road) | -0.149 / +0.201 | -0.086 / +0.149 |
| phev/dyno | 7089 (*/dyno) | -0.011 / +0.079 | -0.021 / +0.025 |
| phev/hd | 120 (*/hd) | -0.254 / +0.082 | -0.121 / +0.101 |
| bev/road | 258 (bev/road) | +0.032 / -0.010 | +0.040 / -0.042 |
| bev/dyno | 195 (bev/dyno) | -0.059 / +0.129 | -0.057 / +0.046 |
| bev/hd | 120 (*/hd) | -0.254 / +0.082 | -0.121 / +0.101 |

**Compute.** Fit 0.007 CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`enet_v21`'s card (0 CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
