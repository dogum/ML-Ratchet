# River Bake-off exam leaderboards (run 20261123T074907)

Medians over basins. Paired bootstrap over basins of the difference in medians (10,000 resamples); win/loss only when the 95 % interval excludes 0. Conformal rows share their base arm's mean (same NSE/KGE/FHV).

### main: ungauged

| Row | n | NSE | KGE | \|FHV\| % | cov50 / cov90 (mean over basins) | CRPS / mean Q | vs FlowCast v1.0, NSE (95 % CI) |
|---|---|---|---|---|---|---|---|
| FlowCast v1.0 (LSTM, as released) | 619 | 0.341 | 0.367 | 39.1 | 0.52 / 0.95 | 0.392 |  |
| LSTM (FlowCast net, retrained) | 619 | 0.340 | 0.381 | 38.0 | 0.53 / 0.88 | 0.390 | -0.001 (-0.009..+0.013) tie |
| State space (S4D + encoder) | 619 | 0.336 | 0.373 | 38.2 | 0.51 / 0.90 | 0.397 | -0.005 (-0.015..+0.009) tie |
| TCN (dilated 1-D CNN) | 619 | 0.338 | 0.379 | 37.8 | 0.52 / 0.90 | 0.392 | -0.003 (-0.013..+0.010) tie |
| Transformer | 619 | 0.323 | 0.362 | 36.5 | 0.54 / 0.88 | 0.400 | -0.018 (-0.029..-0.002) loss |
| Gradient-boosted trees | 619 | 0.345 | 0.383 | 36.0 | 0.51 / 0.87 | 0.398 | +0.004 (-0.007..+0.018) tie |
| Sparse Gaussian process | 619 | 0.297 | 0.310 | 44.4 | 0.64 / 0.94 | 0.433 | -0.044 (-0.061..-0.023) loss |
| Physics in the loop (HBV) | 619 | 0.313 | 0.361 | 36.9 | 0.50 / 0.91 | 0.413 | -0.029 (-0.043..-0.013) loss |
| 17 formulas in HBV | 619 | 0.306 | 0.367 | 36.4 | 0.50 / 0.91 | 0.420 | -0.035 (-0.050..-0.016) loss |
| FlowCast v1.0 (LSTM, as released) + conformal | 619 | 0.341 | 0.367 | 39.1 | 0.47 / 0.88 | 0.414 | +0.000 (+0.000..+0.000) tie |
| LSTM (FlowCast net, retrained) + conformal | 619 | 0.340 | 0.381 | 38.0 | 0.46 / 0.88 | 0.412 | -0.001 (-0.009..+0.013) tie |
| State space (S4D + encoder) + conformal | 619 | 0.336 | 0.373 | 38.2 | 0.47 / 0.88 | 0.410 | -0.005 (-0.015..+0.009) tie |
| TCN (dilated 1-D CNN) + conformal | 619 | 0.338 | 0.379 | 37.8 | 0.46 / 0.88 | 0.413 | -0.003 (-0.012..+0.010) tie |
| Transformer + conformal | 619 | 0.323 | 0.362 | 36.5 | 0.46 / 0.87 | 0.417 | -0.018 (-0.028..-0.002) loss |
| Gradient-boosted trees + conformal | 619 | 0.345 | 0.383 | 36.0 | 0.48 / 0.88 | 0.405 | +0.004 (-0.007..+0.018) tie |
| Sparse Gaussian process + conformal | 619 | 0.297 | 0.310 | 44.4 | 0.47 / 0.88 | 0.440 | -0.044 (-0.061..-0.023) loss |
| Physics in the loop (HBV) + conformal | 619 | 0.313 | 0.361 | 36.9 | 0.47 / 0.87 | 0.442 | -0.029 (-0.042..-0.013) loss |
| 17 formulas in HBV + conformal | 619 | 0.306 | 0.367 | 36.4 | 0.46 / 0.87 | 0.442 | -0.035 (-0.049..-0.015) loss |
| Climatology (Budyko-Fu) | 619 | -0.029 | -0.452 | 88.5 | 0.53 / 0.89 | 0.601 | -0.370 (-0.388..-0.343) loss |
| U-dev regionalisation | 619 | 0.372 | 0.433 | 34.3 |  | 0.613 | +0.031 (+0.014..+0.056) win |
| U-k12 regionalisation | 619 | 0.385 | 0.462 | 31.3 |  | 0.612 | +0.044 (+0.028..+0.067) win |
| GR4J calibrated per basin | 619 | 0.473 | 0.602 | 20.3 |  | 0.532 | +0.132 (+0.109..+0.158) win |
| NOAA NWM v3.0 | 600 | 0.550 | 0.596 | 23.8 |  | 0.511 | +0.205 (+0.181..+0.235) win |
| Own day-of-year climatology | 619 | 0.065 | -0.054 | 80.7 |  | 0.738 | -0.276 (-0.293..-0.250) loss |

### main: gauged, lead 1 d

| Row | n | NSE | KGE | \|FHV\| % | cov50 / cov90 (mean over basins) | CRPS / mean Q | vs FlowCast v1.0, NSE (95 % CI) |
|---|---|---|---|---|---|---|---|
| Ensemble Kalman filter (HBV) | 619 | 0.599 | 0.680 | 19.9 | 0.75 / 0.94 | 0.207 | +0.030 (+0.004..+0.069) win |
| Ensemble Kalman filter (HBV) + conformal | 619 | 0.599 | 0.680 | 19.9 | 0.56 / 0.89 | 0.207 | +0.030 (+0.004..+0.069) win |
| FlowCast v1.0 (LSTM, as released) | 619 | 0.569 | 0.635 | 24.5 | 0.57 / 0.89 | 0.248 |  |
| FlowCast v1.0 (LSTM, as released) + conformal | 619 | 0.569 | 0.635 | 24.5 | 0.46 / 0.87 | 0.248 | +0.000 (+0.000..+0.000) tie |
| Persistence | 619 | 0.475 | 0.738 | 0.0 |  | 0.326 | -0.093 (-0.132..-0.060) loss |
| GR4J calibrated per basin | 619 | 0.473 | 0.602 | 20.3 |  | 0.532 | -0.096 (-0.122..-0.072) loss |
| NOAA NWM v3.0 | 600 | 0.550 | 0.596 | 23.8 |  | 0.511 | -0.025 (-0.057..+0.011) tie |

### main: gauged, lead 3 d

| Row | n | NSE | KGE | \|FHV\| % | cov50 / cov90 (mean over basins) | CRPS / mean Q | vs FlowCast v1.0, NSE (95 % CI) |
|---|---|---|---|---|---|---|---|
| Ensemble Kalman filter (HBV) | 619 | 0.465 | 0.524 | 33.5 | 0.63 / 0.90 | 0.277 | -0.009 (-0.035..+0.009) tie |
| Ensemble Kalman filter (HBV) + conformal | 619 | 0.465 | 0.524 | 33.5 | 0.53 / 0.89 | 0.275 | -0.009 (-0.034..+0.009) tie |
| FlowCast v1.0 (LSTM, as released) | 619 | 0.474 | 0.576 | 28.4 | 0.54 / 0.87 | 0.277 |  |
| FlowCast v1.0 (LSTM, as released) + conformal | 619 | 0.474 | 0.576 | 28.4 | 0.45 / 0.88 | 0.283 | +0.000 (+0.000..+0.000) tie |
| Persistence | 619 | -0.190 | 0.405 | 0.0 |  | 0.600 | -0.665 (-0.706..-0.630) loss |
| GR4J calibrated per basin | 619 | 0.473 | 0.602 | 20.3 |  | 0.532 | -0.001 (-0.027..+0.017) tie |
| NOAA NWM v3.0 | 600 | 0.550 | 0.596 | 23.8 |  | 0.511 | +0.072 (+0.036..+0.101) win |

### main: gauged, lead 7 d

| Row | n | NSE | KGE | \|FHV\| % | cov50 / cov90 (mean over basins) | CRPS / mean Q | vs FlowCast v1.0, NSE (95 % CI) |
|---|---|---|---|---|---|---|---|
| Ensemble Kalman filter (HBV) | 619 | 0.405 | 0.454 | 38.5 | 0.54 / 0.85 | 0.320 | -0.021 (-0.044..+0.007) tie |
| Ensemble Kalman filter (HBV) + conformal | 619 | 0.405 | 0.454 | 38.5 | 0.50 / 0.89 | 0.314 | -0.021 (-0.043..+0.007) tie |
| FlowCast v1.0 (LSTM, as released) | 619 | 0.426 | 0.548 | 29.1 | 0.50 / 0.85 | 0.313 |  |
| FlowCast v1.0 (LSTM, as released) + conformal | 619 | 0.426 | 0.548 | 29.1 | 0.45 / 0.88 | 0.320 | +0.000 (+0.000..+0.000) tie |
| Persistence | 619 | -0.465 | 0.265 | 0.0 |  | 0.756 | -0.892 (-0.924..-0.860) loss |
| GR4J calibrated per basin | 619 | 0.473 | 0.602 | 20.3 |  | 0.532 | +0.047 (+0.025..+0.065) win |
| NOAA NWM v3.0 | 600 | 0.550 | 0.596 | 23.8 |  | 0.511 | +0.121 (+0.091..+0.147) win |

### recent: ungauged

| Row | n | NSE | KGE | \|FHV\| % | cov50 / cov90 (mean over basins) | CRPS / mean Q | vs FlowCast v1.0, NSE (95 % CI) |
|---|---|---|---|---|---|---|---|
| FlowCast v1.0 (LSTM, as released) | 603 | 0.378 | 0.397 | 40.1 | 0.51 / 0.95 | 0.383 |  |
| LSTM (FlowCast net, retrained) | 603 | 0.391 | 0.401 | 37.8 | 0.53 / 0.88 | 0.383 | +0.013 (+0.005..+0.021) win |
| State space (S4D + encoder) | 603 | 0.386 | 0.402 | 36.4 | 0.51 / 0.90 | 0.388 | +0.008 (-0.005..+0.023) tie |
| TCN (dilated 1-D CNN) | 603 | 0.382 | 0.415 | 38.1 | 0.52 / 0.90 | 0.380 | +0.004 (-0.004..+0.020) tie |
| Transformer | 603 | 0.360 | 0.396 | 35.7 | 0.54 / 0.88 | 0.387 | -0.018 (-0.029..-0.003) loss |
| Gradient-boosted trees | 603 | 0.388 | 0.413 | 36.5 | 0.51 / 0.89 | 0.385 | +0.010 (+0.000..+0.024) win |
| Sparse Gaussian process | 603 | 0.344 | 0.332 | 46.3 | 0.67 / 0.94 | 0.412 | -0.034 (-0.053..-0.017) loss |
| Physics in the loop (HBV) | 603 | 0.366 | 0.386 | 37.1 | 0.51 / 0.91 | 0.398 | -0.012 (-0.032..-0.003) loss |
| 17 formulas in HBV | 603 | 0.363 | 0.393 | 36.1 | 0.51 / 0.92 | 0.399 | -0.015 (-0.028..-0.005) loss |
| FlowCast v1.0 (LSTM, as released) + conformal | 603 | 0.378 | 0.397 | 40.1 | 0.48 / 0.89 | 0.397 | +0.000 (+0.000..+0.000) tie |
| LSTM (FlowCast net, retrained) + conformal | 603 | 0.391 | 0.401 | 37.8 | 0.47 / 0.88 | 0.392 | +0.013 (+0.005..+0.021) win |
| State space (S4D + encoder) + conformal | 603 | 0.386 | 0.402 | 36.4 | 0.49 / 0.89 | 0.393 | +0.008 (-0.005..+0.023) tie |
| TCN (dilated 1-D CNN) + conformal | 603 | 0.382 | 0.415 | 38.1 | 0.47 / 0.89 | 0.393 | +0.004 (-0.004..+0.020) tie |
| Transformer + conformal | 603 | 0.360 | 0.396 | 35.7 | 0.48 / 0.88 | 0.391 | -0.018 (-0.029..-0.003) loss |
| Gradient-boosted trees + conformal | 603 | 0.388 | 0.413 | 36.5 | 0.49 / 0.89 | 0.394 | +0.010 (+0.000..+0.023) win |
| Sparse Gaussian process + conformal | 603 | 0.344 | 0.332 | 46.3 | 0.48 / 0.89 | 0.422 | -0.034 (-0.052..-0.018) loss |
| Physics in the loop (HBV) + conformal | 603 | 0.366 | 0.386 | 37.1 | 0.49 / 0.88 | 0.422 | -0.012 (-0.032..-0.004) loss |
| 17 formulas in HBV + conformal | 603 | 0.363 | 0.393 | 36.1 | 0.48 / 0.88 | 0.419 | -0.015 (-0.028..-0.004) loss |
| Climatology (Budyko-Fu) | 603 | -0.030 | -0.451 | 88.4 | 0.53 / 0.90 | 0.595 | -0.408 (-0.424..-0.392) loss |
| U-dev regionalisation | 603 | 0.433 | 0.469 | 35.3 |  | 0.600 | +0.054 (+0.034..+0.068) win |
| U-k12 regionalisation | 603 | 0.443 | 0.469 | 31.8 |  | 0.589 | +0.064 (+0.046..+0.095) win |
| GR4J calibrated per basin | 603 | 0.517 | 0.604 | 22.2 |  | 0.505 | +0.139 (+0.120..+0.159) win |
| NOAA NWM v3.0 | 600 | 0.687 | 0.802 | 7.7 |  | 0.461 | +0.309 (+0.287..+0.328) win |
| Own day-of-year climatology | 603 | 0.053 | -0.065 | 79.5 |  | 0.717 | -0.325 (-0.342..-0.308) loss |

### recent: gauged, lead 1 d

| Row | n | NSE | KGE | \|FHV\| % | cov50 / cov90 (mean over basins) | CRPS / mean Q | vs FlowCast v1.0, NSE (95 % CI) |
|---|---|---|---|---|---|---|---|
| Ensemble Kalman filter (HBV) | 603 | 0.615 | 0.702 | 21.2 | 0.76 / 0.94 | 0.200 | +0.023 (-0.003..+0.047) tie |
| Ensemble Kalman filter (HBV) + conformal | 603 | 0.615 | 0.702 | 21.2 | 0.57 / 0.90 | 0.197 | +0.023 (-0.002..+0.047) tie |
| FlowCast v1.0 (LSTM, as released) | 603 | 0.591 | 0.652 | 23.7 | 0.57 / 0.89 | 0.241 |  |
| FlowCast v1.0 (LSTM, as released) + conformal | 603 | 0.591 | 0.652 | 23.7 | 0.47 / 0.88 | 0.243 | +0.000 (+0.000..+0.000) tie |
| Persistence | 603 | 0.480 | 0.740 | 0.0 |  | 0.328 | -0.112 (-0.151..-0.077) loss |
| GR4J calibrated per basin | 603 | 0.517 | 0.604 | 22.2 |  | 0.505 | -0.074 (-0.098..-0.056) loss |
| NOAA NWM v3.0 | 600 | 0.687 | 0.802 | 7.7 |  | 0.461 | +0.096 (+0.065..+0.118) win |

### recent: gauged, lead 3 d

| Row | n | NSE | KGE | \|FHV\| % | cov50 / cov90 (mean over basins) | CRPS / mean Q | vs FlowCast v1.0, NSE (95 % CI) |
|---|---|---|---|---|---|---|---|
| Ensemble Kalman filter (HBV) | 603 | 0.487 | 0.539 | 33.2 | 0.64 / 0.90 | 0.264 | -0.024 (-0.047..-0.002) loss |
| Ensemble Kalman filter (HBV) + conformal | 603 | 0.487 | 0.539 | 33.2 | 0.54 / 0.90 | 0.263 | -0.024 (-0.046..-0.002) loss |
| FlowCast v1.0 (LSTM, as released) | 603 | 0.511 | 0.587 | 28.8 | 0.54 / 0.87 | 0.274 |  |
| FlowCast v1.0 (LSTM, as released) + conformal | 603 | 0.511 | 0.587 | 28.8 | 0.46 / 0.88 | 0.278 | +0.000 (+0.000..+0.000) tie |
| Persistence | 603 | -0.210 | 0.395 | 0.0 |  | 0.582 | -0.721 (-0.775..-0.654) loss |
| GR4J calibrated per basin | 603 | 0.517 | 0.604 | 22.2 |  | 0.505 | +0.006 (-0.010..+0.028) tie |
| NOAA NWM v3.0 | 600 | 0.687 | 0.802 | 7.7 |  | 0.461 | +0.176 (+0.153..+0.200) win |

### recent: gauged, lead 7 d

| Row | n | NSE | KGE | \|FHV\| % | cov50 / cov90 (mean over basins) | CRPS / mean Q | vs FlowCast v1.0, NSE (95 % CI) |
|---|---|---|---|---|---|---|---|
| Ensemble Kalman filter (HBV) | 603 | 0.422 | 0.467 | 38.5 | 0.55 / 0.86 | 0.312 | -0.044 (-0.061..-0.020) loss |
| Ensemble Kalman filter (HBV) + conformal | 603 | 0.422 | 0.467 | 38.5 | 0.52 / 0.90 | 0.306 | -0.044 (-0.061..-0.019) loss |
| FlowCast v1.0 (LSTM, as released) | 603 | 0.466 | 0.562 | 29.4 | 0.49 / 0.86 | 0.304 |  |
| FlowCast v1.0 (LSTM, as released) + conformal | 603 | 0.466 | 0.562 | 29.4 | 0.46 / 0.89 | 0.310 | +0.000 (+0.000..+0.000) tie |
| Persistence | 603 | -0.459 | 0.271 | 0.0 |  | 0.735 | -0.924 (-0.960..-0.870) loss |
| GR4J calibrated per basin | 603 | 0.517 | 0.604 | 22.2 |  | 0.505 | +0.052 (+0.035..+0.072) win |
| NOAA NWM v3.0 | 600 | 0.687 | 0.802 | 7.7 |  | 0.461 | +0.222 (+0.201..+0.244) win |

### Where each assumption wins (median NSE; best learner, and who is statistically tied with it)

| Board | Class | n | Best learner (NSE) | Tied with it | Best CRPS row | Best reference (NSE) | best learner vs best reference |
|---|---|---|---|---|---|---|---|
| main/ungauged | all | 619 | Gradient-boosted trees (0.345) | FlowCast v1.0 (LSTM, as released), LSTM (FlowCast net, retrained), State space (S4D + encoder), TCN (dilated 1-D CNN) | LSTM (FlowCast net, retrained) (0.3904) | NOAA NWM v3.0 (0.5502) | -0.203 loss |
| main/ungauged | snow | 118 | 17 formulas in HBV (0.4616) | FlowCast v1.0 (LSTM, as released), LSTM (FlowCast net, retrained), State space (S4D + encoder), TCN (dilated 1-D CNN), Transformer, Gradient-boosted trees, Sparse Gaussian process, Physics in the loop (HBV) | Gradient-boosted trees (0.3276) | GR4J calibrated per basin (0.6749) | -0.213 loss |
| main/ungauged | arid | 174 | LSTM (FlowCast net, retrained) (0.2167) | FlowCast v1.0 (LSTM, as released), Transformer, Gradient-boosted trees, Physics in the loop (HBV), 17 formulas in HBV | FlowCast v1.0 (LSTM, as released) + conformal (0.5959) | NOAA NWM v3.0 (0.3155) | -0.099 loss |
| main/ungauged | humid | 128 | State space (S4D + encoder) (0.5285) | LSTM (FlowCast net, retrained), TCN (dilated 1-D CNN), Gradient-boosted trees, Sparse Gaussian process | LSTM (FlowCast net, retrained) (0.3021) | GR4J calibrated per basin (0.6642) | -0.136 loss |
| main/ungauged | regulated | 37 | Gradient-boosted trees (0.3527) | FlowCast v1.0 (LSTM, as released), LSTM (FlowCast net, retrained), State space (S4D + encoder), TCN (dilated 1-D CNN), Transformer, Sparse Gaussian process, Physics in the loop (HBV), 17 formulas in HBV | Physics in the loop (HBV) (0.3683) | GR4J calibrated per basin (0.5623) | -0.210 loss |
| main/ungauged | large | 48 | FlowCast v1.0 (LSTM, as released) (0.2257) | LSTM (FlowCast net, retrained), Transformer, Gradient-boosted trees, Sparse Gaussian process, Physics in the loop (HBV) | Transformer + conformal (0.4509) | NOAA NWM v3.0 (0.4225) | -0.197 tie |
| main/gauged_L1 | all | 619 | Ensemble Kalman filter (HBV) (0.5989) | none | Ensemble Kalman filter (HBV) (0.2071) | NOAA NWM v3.0 (0.5502) | +0.049 win |
| main/gauged_L1 | snow | 118 | Persistence (0.9275) | Ensemble Kalman filter (HBV) | Ensemble Kalman filter (HBV) + conformal (0.1171) | GR4J calibrated per basin (0.6749) | +0.253 win |
| main/gauged_L1 | arid | 174 | Ensemble Kalman filter (HBV) (0.5361) | FlowCast v1.0 (LSTM, as released), Persistence | Ensemble Kalman filter (HBV) + conformal (0.2731) | NOAA NWM v3.0 (0.3155) | +0.216 win |
| main/gauged_L1 | humid | 128 | Ensemble Kalman filter (HBV) (0.7365) | FlowCast v1.0 (LSTM, as released) | Ensemble Kalman filter (HBV) + conformal (0.1629) | GR4J calibrated per basin (0.6642) | +0.072 win |
| main/gauged_L1 | regulated | 37 | Ensemble Kalman filter (HBV) (0.7172) | FlowCast v1.0 (LSTM, as released), Persistence | Ensemble Kalman filter (HBV) + conformal (0.1712) | GR4J calibrated per basin (0.5623) | +0.155 win |
| main/gauged_L1 | large | 48 | Persistence (0.8742) | none | Ensemble Kalman filter (HBV) + conformal (0.1649) | NOAA NWM v3.0 (0.4225) | +0.452 win |
| main/gauged_L3 | all | 619 | FlowCast v1.0 (LSTM, as released) (0.4743) | Ensemble Kalman filter (HBV) | Ensemble Kalman filter (HBV) + conformal (0.2754) | NOAA NWM v3.0 (0.5502) | -0.072 loss |
| main/gauged_L3 | snow | 118 | Persistence (0.7634) | Ensemble Kalman filter (HBV), FlowCast v1.0 (LSTM, as released) | Ensemble Kalman filter (HBV) + conformal (0.1764) | GR4J calibrated per basin (0.6749) | +0.088 tie |
| main/gauged_L3 | arid | 174 | FlowCast v1.0 (LSTM, as released) (0.3915) | Ensemble Kalman filter (HBV) | Ensemble Kalman filter (HBV) + conformal (0.3504) | NOAA NWM v3.0 (0.3155) | +0.069 tie |
| main/gauged_L3 | humid | 128 | FlowCast v1.0 (LSTM, as released) (0.6451) | Ensemble Kalman filter (HBV) | Ensemble Kalman filter (HBV) (0.2078) | GR4J calibrated per basin (0.6642) | -0.019 tie |
| main/gauged_L3 | regulated | 37 | FlowCast v1.0 (LSTM, as released) (0.6082) | Ensemble Kalman filter (HBV) | Ensemble Kalman filter (HBV) (0.2483) | GR4J calibrated per basin (0.5623) | +0.046 tie |
| main/gauged_L3 | large | 48 | FlowCast v1.0 (LSTM, as released) (0.5295) | Ensemble Kalman filter (HBV), Persistence | Ensemble Kalman filter (HBV) (0.2456) | NOAA NWM v3.0 (0.4225) | +0.107 tie |
| main/gauged_L7 | all | 619 | FlowCast v1.0 (LSTM, as released) (0.4264) | Ensemble Kalman filter (HBV) | FlowCast v1.0 (LSTM, as released) (0.3134) | NOAA NWM v3.0 (0.5502) | -0.121 loss |
| main/gauged_L7 | snow | 118 | Ensemble Kalman filter (HBV) (0.6758) | FlowCast v1.0 (LSTM, as released) | FlowCast v1.0 (LSTM, as released) (0.221) | GR4J calibrated per basin (0.6749) | +0.001 tie |
| main/gauged_L7 | arid | 174 | FlowCast v1.0 (LSTM, as released) (0.3346) | Ensemble Kalman filter (HBV) | Ensemble Kalman filter (HBV) + conformal (0.402) | NOAA NWM v3.0 (0.3155) | +0.019 tie |
| main/gauged_L7 | humid | 128 | FlowCast v1.0 (LSTM, as released) (0.6074) | Ensemble Kalman filter (HBV) | FlowCast v1.0 (LSTM, as released) (0.2339) | GR4J calibrated per basin (0.6642) | -0.057 loss |
| main/gauged_L7 | regulated | 37 | FlowCast v1.0 (LSTM, as released) (0.5106) | Ensemble Kalman filter (HBV) | Ensemble Kalman filter (HBV) + conformal (0.291) | GR4J calibrated per basin (0.5623) | -0.052 tie |
| main/gauged_L7 | large | 48 | FlowCast v1.0 (LSTM, as released) (0.3759) | Ensemble Kalman filter (HBV) | Ensemble Kalman filter (HBV) + conformal (0.3159) | NOAA NWM v3.0 (0.4225) | -0.047 tie |
| recent/ungauged | all | 603 | LSTM (FlowCast net, retrained) (0.3914) | State space (S4D + encoder), TCN (dilated 1-D CNN), Gradient-boosted trees | TCN (dilated 1-D CNN) (0.3804) | NOAA NWM v3.0 (0.6874) | -0.296 loss |
| recent/ungauged | snow | 115 | Gradient-boosted trees (0.4929) | LSTM (FlowCast net, retrained), State space (S4D + encoder), TCN (dilated 1-D CNN), Transformer, Sparse Gaussian process, 17 formulas in HBV | Sparse Gaussian process (0.3241) | GR4J calibrated per basin (0.6971) | -0.204 loss |
| recent/ungauged | arid | 171 | LSTM (FlowCast net, retrained) (0.2316) | State space (S4D + encoder), Physics in the loop (HBV) | Transformer + conformal (0.5875) | NOAA NWM v3.0 (0.4806) | -0.250 loss |
| recent/ungauged | humid | 127 | State space (S4D + encoder) (0.5329) | LSTM (FlowCast net, retrained), TCN (dilated 1-D CNN), Gradient-boosted trees, Sparse Gaussian process | Gradient-boosted trees (0.2918) | NOAA NWM v3.0 (0.7186) | -0.186 loss |
| recent/ungauged | regulated | 35 | TCN (dilated 1-D CNN) (0.4371) | FlowCast v1.0 (LSTM, as released), LSTM (FlowCast net, retrained), State space (S4D + encoder), Transformer, Gradient-boosted trees, Physics in the loop (HBV) | LSTM (FlowCast net, retrained) (0.3626) | NOAA NWM v3.0 (0.7022) | -0.265 loss |
| recent/ungauged | large | 48 | LSTM (FlowCast net, retrained) (0.3494) | FlowCast v1.0 (LSTM, as released), State space (S4D + encoder), TCN (dilated 1-D CNN), Transformer, Gradient-boosted trees, Sparse Gaussian process, Physics in the loop (HBV), 17 formulas in HBV | FlowCast v1.0 (LSTM, as released) (0.3981) | NOAA NWM v3.0 (0.5165) | -0.167 loss |
| recent/gauged_L1 | all | 603 | Ensemble Kalman filter (HBV) (0.6147) | FlowCast v1.0 (LSTM, as released) | Ensemble Kalman filter (HBV) + conformal (0.1973) | NOAA NWM v3.0 (0.6874) | -0.072 loss |
| recent/gauged_L1 | snow | 115 | Persistence (0.94) | Ensemble Kalman filter (HBV) | Ensemble Kalman filter (HBV) + conformal (0.116) | GR4J calibrated per basin (0.6971) | +0.243 win |
| recent/gauged_L1 | arid | 171 | Ensemble Kalman filter (HBV) (0.5345) | FlowCast v1.0 (LSTM, as released), Persistence | Ensemble Kalman filter (HBV) + conformal (0.2663) | NOAA NWM v3.0 (0.4806) | +0.058 tie |
| recent/gauged_L1 | humid | 127 | Ensemble Kalman filter (HBV) (0.7378) | FlowCast v1.0 (LSTM, as released) | Ensemble Kalman filter (HBV) + conformal (0.1596) | NOAA NWM v3.0 (0.7186) | +0.019 tie |
| recent/gauged_L1 | regulated | 35 | Ensemble Kalman filter (HBV) (0.6395) | FlowCast v1.0 (LSTM, as released), Persistence | Ensemble Kalman filter (HBV) (0.1721) | NOAA NWM v3.0 (0.7022) | -0.063 tie |
| recent/gauged_L1 | large | 48 | Persistence (0.87) | none | Ensemble Kalman filter (HBV) + conformal (0.1532) | NOAA NWM v3.0 (0.5165) | +0.353 win |
| recent/gauged_L3 | all | 603 | FlowCast v1.0 (LSTM, as released) (0.511) | none | Ensemble Kalman filter (HBV) + conformal (0.2627) | NOAA NWM v3.0 (0.6874) | -0.176 loss |
| recent/gauged_L3 | snow | 115 | Persistence (0.7921) | Ensemble Kalman filter (HBV), FlowCast v1.0 (LSTM, as released) | Ensemble Kalman filter (HBV) + conformal (0.1793) | GR4J calibrated per basin (0.6971) | +0.095 tie |
| recent/gauged_L3 | arid | 171 | FlowCast v1.0 (LSTM, as released) (0.3944) | Ensemble Kalman filter (HBV) | Ensemble Kalman filter (HBV) + conformal (0.3553) | NOAA NWM v3.0 (0.4806) | -0.089 tie |
| recent/gauged_L3 | humid | 127 | Ensemble Kalman filter (HBV) (0.6823) | FlowCast v1.0 (LSTM, as released) | Ensemble Kalman filter (HBV) + conformal (0.2009) | NOAA NWM v3.0 (0.7186) | -0.036 tie |
| recent/gauged_L3 | regulated | 35 | Ensemble Kalman filter (HBV) (0.5739) | FlowCast v1.0 (LSTM, as released), Persistence | Ensemble Kalman filter (HBV) (0.2627) | NOAA NWM v3.0 (0.7022) | -0.128 tie |
| recent/gauged_L3 | large | 48 | FlowCast v1.0 (LSTM, as released) (0.5912) | Ensemble Kalman filter (HBV), Persistence | Ensemble Kalman filter (HBV) (0.2351) | NOAA NWM v3.0 (0.5165) | +0.075 tie |
| recent/gauged_L7 | all | 603 | FlowCast v1.0 (LSTM, as released) (0.4656) | none | FlowCast v1.0 (LSTM, as released) (0.304) | NOAA NWM v3.0 (0.6874) | -0.222 loss |
| recent/gauged_L7 | snow | 115 | Ensemble Kalman filter (HBV) (0.6879) | FlowCast v1.0 (LSTM, as released) | FlowCast v1.0 (LSTM, as released) (0.2207) | GR4J calibrated per basin (0.6971) | -0.009 tie |
| recent/gauged_L7 | arid | 171 | FlowCast v1.0 (LSTM, as released) (0.3077) | none | Ensemble Kalman filter (HBV) + conformal (0.4035) | NOAA NWM v3.0 (0.4806) | -0.173 loss |
| recent/gauged_L7 | humid | 127 | Ensemble Kalman filter (HBV) (0.6175) | FlowCast v1.0 (LSTM, as released) | Ensemble Kalman filter (HBV) + conformal (0.2278) | NOAA NWM v3.0 (0.7186) | -0.101 loss |
| recent/gauged_L7 | regulated | 35 | FlowCast v1.0 (LSTM, as released) (0.4958) | Ensemble Kalman filter (HBV) | FlowCast v1.0 (LSTM, as released) + conformal (0.3095) | NOAA NWM v3.0 (0.7022) | -0.206 loss |
| recent/gauged_L7 | large | 48 | FlowCast v1.0 (LSTM, as released) (0.4543) | Ensemble Kalman filter (HBV) | Ensemble Kalman filter (HBV) (0.2833) | NOAA NWM v3.0 (0.5165) | -0.062 tie |

### Compute

| Arm | Train CPU-h | Exam predict CPU-s | Stored numbers |
|---|---|---|---|
| Climatology (Budyko-Fu) | 0.001 | 0.2 |   |
| Ensemble Kalman filter (HBV) | 0.98 | 1071.2 | 0 no learned weights; 8 tuned filter settings |
| Ensemble Kalman filter (HBV) + conformal | 0.0261 | 1070.1 |   |
| FlowCast v1.0 (LSTM, as released) | 0.0 | 246.3 | 24710  |
| FlowCast v1.0 (LSTM, as released) + conformal | 0.005 | 243.3 |   |
| LSTM (FlowCast net, retrained) | 0.867 | 57.4 | 24710  |
| LSTM (FlowCast net, retrained) + conformal | 0.0034 | 57.7 |   |
| State space (S4D + encoder) | 3.693 | 311.7 | 30438  |
| State space (S4D + encoder) + conformal | 0.0077 | 320.0 |   |
| TCN (dilated 1-D CNN) | 1.251 | 252.6 | 26266  |
| TCN (dilated 1-D CNN) + conformal | 0.0071 | 243.6 |   |
| Transformer | 3.553 | 661.0 | 26854  |
| Transformer + conformal | 0.0146 | 648.2 |   |
| Physics in the loop (HBV) | 2.25 | 16.3 | 6481 MLP 18-64-64-17 |
| Physics in the loop (HBV) + conformal | 0.003 | 15.7 |   |
| 17 formulas in HBV | 0.25 | 16.2 |  17 formulas |
| 17 formulas in HBV + conformal | 0.0031 | 16.2 |   |
| Gradient-boosted trees | 0.43 | 1471.0 | 87876 1,172 trees, 87,876 nodes |
| Gradient-boosted trees + conformal | 0.037 | 1479.7 |   |
| Sparse Gaussian process | 0.94 | 145.9 | 260585 500 inducing points x 20 inputs + a 500 x 500 matrix |
| Sparse Gaussian process + conformal | 0.0059 | 141.2 |   |
