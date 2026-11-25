# Vehicle Bake-off exam leaderboards (run 20261123T204251)

E16 = mean over the 16 D16 sets of the exam MdAPE (lower is better). Intervals: paired cluster bootstrap with shared resamples per set (2,000), averaged over the 16 sets; win / loss only when the 95 % interval excludes 0. cov50 / cov90 and CRPS(log) are means over the 16 sets. Control: MLP (table), GRU (trace).

### table track: E16 (mean exam MdAPE over the 16 D16 sets) and selected sets

| Row | E16 | vs EnergyNet v2.1 (95 % CI) | vs control | VED ICE / HEV / EV / PHEV | EPA conv FTP / HWY | EPA hev FTP / HWY | EPA BEV city / HWY | Mersin d / CNG | ICCT | ZTBus legs | cov50 / cov90 | CRPS(log) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EnergyNet v2.1 (as released) | 0.110 |  | -0.012 win | 0.086 / 0.134 / 0.232 / 0.168 | 0.086 / 0.068 | 0.087 / 0.056 | 0.095 / 0.047 | 0.172 / 0.192 | 0.107 | 0.071 | 0.52 / 0.85 | 0.094 |
| MLP (EnergyNet net, retrained) | 0.121 | +0.012 (+0.009..+0.013) loss |  | 0.088 / 0.146 / 0.253 / 0.176 | 0.080 / 0.063 | 0.081 / 0.064 | 0.114 / 0.057 | 0.205 / 0.218 | 0.143 | 0.057 | 0.37 / 0.71 | 0.104 |
| MLP (EnergyNet net, retrained) + conformal | 0.121 | +0.012 (+0.009..+0.013) loss | +0.000 tie | 0.088 / 0.146 / 0.253 / 0.176 | 0.080 / 0.063 | 0.081 / 0.064 | 0.114 / 0.057 | 0.205 / 0.218 | 0.143 | 0.057 | 0.49 / 0.86 | 0.094 |
| Sparse Gaussian process | 0.126 | +0.016 (+0.012..+0.020) loss | +0.004 loss | 0.089 / 0.151 / 0.215 / 0.201 | 0.068 / 0.099 | 0.086 / 0.050 | 0.070 / 0.051 | 0.186 / 0.201 | 0.197 | 0.098 | 0.61 / 0.92 | 0.102 |
| Sparse Gaussian process + conformal | 0.126 | +0.016 (+0.012..+0.020) loss | +0.004 loss | 0.089 / 0.151 / 0.215 / 0.201 | 0.068 / 0.099 | 0.086 / 0.050 | 0.070 / 0.051 | 0.186 / 0.201 | 0.197 | 0.098 | 0.47 / 0.87 | 0.096 |
| Gradient-boosted trees | 0.126 | +0.017 (+0.014..+0.021) loss | +0.005 loss | 0.092 / 0.145 / 0.195 / 0.181 | 0.069 / 0.062 | 0.070 / 0.056 | 0.096 / 0.074 | 0.251 / 0.292 | 0.172 | 0.080 | 0.39 / 0.75 | 0.099 |
| Gradient-boosted trees + conformal | 0.126 | +0.017 (+0.014..+0.021) loss | +0.005 loss | 0.092 / 0.145 / 0.195 / 0.181 | 0.069 / 0.062 | 0.070 / 0.056 | 0.096 / 0.074 | 0.251 / 0.292 | 0.172 | 0.080 | 0.45 / 0.83 | 0.094 |
| Trip-energy law (symbolic) | 0.210 | +0.101 (+0.092..+0.108) loss | +0.089 loss | 0.140 / 0.167 / 0.243 / 0.214 | 0.240 / 0.305 | 0.383 / 0.260 | 0.123 / 0.107 | 0.237 / 0.259 | 0.275 | 0.086 | 0.42 / 0.83 | 0.141 |
| Trip-energy law (symbolic) + conformal | 0.210 | +0.101 (+0.092..+0.108) loss | +0.089 loss | 0.140 / 0.167 / 0.243 / 0.214 | 0.240 / 0.305 | 0.383 / 0.260 | 0.123 / 0.107 | 0.237 / 0.259 | 0.275 | 0.086 | 0.42 / 0.81 | 0.128 |
| Physics anchor (E_phys + sim bands) | 0.229 | +0.119 (+0.111..+0.128) loss | +0.107 loss | 0.330 / 0.192 / 0.234 / 0.204 | 0.156 / 0.119 | 0.156 / 0.113 | 0.170 / 0.078 | 0.301 / 0.332 | 0.656 | 0.202 | 0.41 / 0.83 | 0.164 |
| Physics anchor (E_phys + sim bands) + conformal | 0.229 | +0.119 (+0.111..+0.128) loss | +0.107 loss | 0.330 / 0.192 / 0.234 / 0.204 | 0.156 / 0.119 | 0.156 / 0.113 | 0.170 / 0.078 | 0.301 / 0.332 | 0.656 | 0.202 | 0.43 / 0.82 | 0.154 |
| EnergyNet v2.1 (as released) + conformal | 0.110 | +0.000 (+0.000..+0.000) tie | -0.012 win | 0.086 / 0.134 / 0.232 / 0.168 | 0.086 / 0.068 | 0.087 / 0.056 | 0.095 / 0.047 | 0.172 / 0.192 | 0.107 | 0.071 | 0.51 / 0.87 | 0.091 |
| Classical composite (MOVES-like / EPA regression / dev median) | 0.125 | +0.016 (+0.007..+0.024) loss | +0.004 tie | 0.103 / 0.153 / 0.190 / 0.413 | 0.071 / 0.050 | 0.163 / 0.215 | 0.098 / 0.065 | 0.086 / 0.071 | 0.070 | 0.143 |  | 0.160 |
| E_phys alone (road-load) | 0.229 | +0.119 (+0.111..+0.128) loss | +0.107 loss | 0.330 / 0.192 / 0.234 / 0.204 | 0.156 / 0.119 | 0.156 / 0.113 | 0.170 / 0.078 | 0.301 / 0.332 | 0.656 | 0.202 |  | 0.261 |
| Climatology (sim median MJ/km x distance) | 0.310 | +0.201 (+0.192..+0.213) loss | +0.189 loss | 0.188 / 0.232 / 0.228 / 0.390 | 0.221 / 0.779 | 0.488 / 0.500 | 0.388 / 0.322 | 0.114 / 0.133 | 0.360 | 0.164 | 0.43 / 0.92 | 0.189 |
| Window label x distance (dev-fitted) |  |  |  | 0.131 / 0.162 / 0.215 / 0.461 |  /  |  /  |  /  |  /  |  |  |  |  |
| MOVES-like VSP model (dev-fitted) |  |  |  | 0.103 / 0.153 / 0.190 / 0.413 |  /  |  /  |  /  |  /  |  |  |  |  |
| Class median per km (dev) |  |  |  | 0.228 / 0.172 / 0.271 / 0.427 |  /  |  /  |  /  |  /  |  |  |  |  |
| EPA regression (dev-fitted) |  |  |  |  /  /  /  | 0.071 / 0.050 | 0.163 / 0.215 | 0.098 / 0.065 |  /  |  |  |  |  |
| Dev median (EPA / HD) |  |  |  |  /  /  /  | 0.187 / 0.168 | 0.490 / 0.201 | 0.190 / 0.113 | 0.086 / 0.071 | 0.070 | 0.143 |  |  |

### trace track: E16 (mean exam MdAPE over the 16 D16 sets) and selected sets

| Row | E16 | vs EnergyNet v2.1 (95 % CI) | vs control | VED ICE / HEV / EV / PHEV | EPA conv FTP / HWY | EPA hev FTP / HWY | EPA BEV city / HWY | Mersin d / CNG | ICCT | ZTBus legs | cov50 / cov90 | CRPS(log) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EnergyNet v2.1 (as released) | 0.110 |  | -0.011 win | 0.086 / 0.134 / 0.232 / 0.168 | 0.086 / 0.068 | 0.087 / 0.056 | 0.095 / 0.047 | 0.172 / 0.192 | 0.107 | 0.071 | 0.52 / 0.85 | 0.094 |
| Transformer (20-s patches) | 0.117 | +0.007 (+0.004..+0.011) loss | -0.004 win | 0.087 / 0.148 / 0.224 / 0.201 | 0.085 / 0.067 | 0.088 / 0.053 | 0.116 / 0.073 | 0.189 / 0.202 | 0.080 | 0.069 | 0.38 / 0.75 | 0.101 |
| Transformer (20-s patches) + conformal | 0.117 | +0.007 (+0.004..+0.011) loss | -0.004 win | 0.087 / 0.148 / 0.224 / 0.201 | 0.085 / 0.067 | 0.088 / 0.053 | 0.116 / 0.073 | 0.189 / 0.202 | 0.080 | 0.069 | 0.49 / 0.89 | 0.090 |
| Bidirectional GRU | 0.120 | +0.011 (+0.007..+0.016) loss |  | 0.088 / 0.136 / 0.240 / 0.211 | 0.076 / 0.091 | 0.096 / 0.124 | 0.089 / 0.052 | 0.194 / 0.219 | 0.059 | 0.074 | 0.39 / 0.77 | 0.102 |
| Bidirectional GRU + conformal | 0.120 | +0.011 (+0.007..+0.016) loss | +0.000 tie | 0.088 / 0.136 / 0.240 / 0.211 | 0.076 / 0.091 | 0.096 / 0.124 | 0.089 / 0.052 | 0.194 / 0.219 | 0.059 | 0.074 | 0.48 / 0.88 | 0.092 |
| 1-D dilated CNN | 0.127 | +0.017 (+0.014..+0.020) loss | +0.007 loss | 0.086 / 0.142 / 0.281 / 0.201 | 0.074 / 0.083 | 0.108 / 0.075 | 0.103 / 0.092 | 0.189 / 0.204 | 0.154 | 0.060 | 0.37 / 0.72 | 0.108 |
| 1-D dilated CNN + conformal | 0.127 | +0.017 (+0.014..+0.020) loss | +0.007 loss | 0.086 / 0.142 / 0.281 / 0.201 | 0.074 / 0.083 | 0.108 / 0.075 | 0.103 / 0.092 | 0.189 / 0.204 | 0.154 | 0.060 | 0.44 / 0.82 | 0.098 |
| S4D state space | 0.130 | +0.020 (+0.016..+0.026) loss | +0.009 loss | 0.090 / 0.143 / 0.266 / 0.225 | 0.092 / 0.077 | 0.079 / 0.066 | 0.147 / 0.108 | 0.186 / 0.203 | 0.125 | 0.072 | 0.35 / 0.71 | 0.110 |
| S4D state space + conformal | 0.130 | +0.020 (+0.016..+0.026) loss | +0.009 loss | 0.090 / 0.143 / 0.266 / 0.225 | 0.092 / 0.077 | 0.079 / 0.066 | 0.147 / 0.108 | 0.186 / 0.203 | 0.125 | 0.072 | 0.50 / 0.86 | 0.097 |
| Trip-energy law (symbolic) | 0.210 | +0.101 (+0.092..+0.108) loss | +0.090 loss | 0.140 / 0.167 / 0.243 / 0.214 | 0.240 / 0.305 | 0.383 / 0.260 | 0.123 / 0.106 | 0.237 / 0.259 | 0.275 | 0.086 | 0.42 / 0.83 | 0.141 |
| Trip-energy law (symbolic) + conformal | 0.210 | +0.101 (+0.092..+0.108) loss | +0.090 loss | 0.140 / 0.167 / 0.243 / 0.214 | 0.240 / 0.305 | 0.383 / 0.260 | 0.123 / 0.106 | 0.237 / 0.259 | 0.275 | 0.086 | 0.42 / 0.81 | 0.128 |
| Physics anchor (E_phys + sim bands) | 0.229 | +0.119 (+0.111..+0.128) loss | +0.108 loss | 0.330 / 0.192 / 0.234 / 0.204 | 0.156 / 0.119 | 0.156 / 0.113 | 0.170 / 0.078 | 0.301 / 0.332 | 0.657 | 0.202 | 0.41 / 0.83 | 0.164 |
| Physics anchor (E_phys + sim bands) + conformal | 0.229 | +0.119 (+0.111..+0.128) loss | +0.108 loss | 0.330 / 0.192 / 0.234 / 0.204 | 0.156 / 0.119 | 0.156 / 0.113 | 0.170 / 0.078 | 0.301 / 0.332 | 0.657 | 0.202 | 0.43 / 0.82 | 0.154 |
| Classical composite (MOVES-like / EPA regression / dev median) | 0.125 | +0.016 (+0.007..+0.024) loss | +0.005 tie | 0.103 / 0.153 / 0.190 / 0.413 | 0.071 / 0.050 | 0.163 / 0.215 | 0.098 / 0.065 | 0.086 / 0.071 | 0.070 | 0.143 |  | 0.160 |
| E_phys alone (road-load) | 0.229 | +0.119 (+0.111..+0.128) loss | +0.108 loss | 0.330 / 0.192 / 0.234 / 0.204 | 0.156 / 0.119 | 0.156 / 0.113 | 0.170 / 0.078 | 0.301 / 0.332 | 0.656 | 0.202 |  | 0.261 |
| Climatology (sim median MJ/km x distance) | 0.310 | +0.201 (+0.192..+0.213) loss | +0.190 loss | 0.188 / 0.232 / 0.228 / 0.390 | 0.221 / 0.779 | 0.488 / 0.500 | 0.388 / 0.322 | 0.114 / 0.133 | 0.360 | 0.164 | 0.43 / 0.92 | 0.189 |
| Window label x distance (dev-fitted) |  |  |  | 0.131 / 0.162 / 0.215 / 0.461 |  /  |  /  |  /  |  /  |  |  |  |  |
| MOVES-like VSP model (dev-fitted) |  |  |  | 0.103 / 0.153 / 0.190 / 0.413 |  /  |  /  |  /  |  /  |  |  |  |  |
| Class median per km (dev) |  |  |  | 0.228 / 0.172 / 0.271 / 0.427 |  /  |  /  |  /  |  /  |  |  |  |  |
| EPA regression (dev-fitted) |  |  |  |  /  /  /  | 0.071 / 0.050 | 0.163 / 0.215 | 0.098 / 0.065 |  /  |  |  |  |  |
| Dev median (EPA / HD) |  |  |  |  /  /  /  | 0.187 / 0.168 | 0.490 / 0.201 | 0.190 / 0.113 | 0.086 / 0.071 | 0.070 | 0.143 |  |  |

### Where each assumption wins (lowest MdAPE among the raw learners of both tracks + EnergyNet v2.1)

| Column | n | Best | MdAPE | Tied with it | Best reference (MdAPE) | best vs best reference |
|---|---|---|---|---|---|---|
| VED_ICE | 14469 | trace_cnn|trace | 0.086 | tab_mlp|table, tab_gp|table, enet_v21|table, trace_gru|trace, trace_transformer|trace | moves_like|table (0.103) | -0.016 win |
| VED_HEV | 8241 | enet_v21|table | 0.134 | trace_gru|trace | moves_like|table (0.153) | -0.018 win |
| VED_EV | 167 | tab_gbm|table | 0.195 | none | moves_like|table (0.190) | 0.005 tie |
| VED_PHEV | 2916 | enet_v21|table | 0.168 | tab_mlp|table, tab_gbm|table | ephys|table (0.204) | -0.036 win |
| EPA_conv_FTP | 20065 | tab_gp|table | 0.068 | tab_gbm|table | regression|table (0.071) | -0.003 win |
| EPA_conv_HWY | 20426 | tab_gbm|table | 0.062 | tab_mlp|table | regression|table (0.050) | 0.012 loss |
| EPA_conv_US06 | 2832 | enet_v21|table | 0.052 | none | regression|table (0.069) | -0.017 win |
| EPA_conv_FTPcold | 1734 | tab_gp|table | 0.072 | none | regression|table (0.097) | -0.025 win |
| EPA_conv_SC03 | 1819 | trace_cnn|trace | 0.079 | none | regression|table (0.092) | -0.013 win |
| EPA_hev_FTP | 1098 | tab_gbm|table | 0.070 | none | ephys|table (0.156) | -0.087 win |
| EPA_hev_HWY | 1156 | tab_gp|table | 0.050 | trace_transformer|trace | ephys|table (0.113) | -0.063 win |
| EPA_bev_FTP | 746 | tab_gp|table | 0.070 | none | regression|table (0.098) | -0.028 win |
| EPA_bev_HWY | 746 | enet_v21|table | 0.047 | none | regression|table (0.065) | -0.018 win |
| Mersin_diesel | 652 | enet_v21|table | 0.172 | none | dev_median|table (0.086) | 0.086 loss |
| Mersin_cng | 672 | enet_v21|table | 0.192 | none | dev_median|table (0.071) | 0.120 loss |
| ICCT | 6 | trace_gru|trace | 0.059 | trace_transformer|trace | dev_median|table (0.070) | -0.011 tie |
| ZTBus_legs | 78 | tab_mlp|table | 0.057 | tab_gbm|table, enet_v21|table, trace_gru|trace, trace_cnn|trace, trace_s4d|trace, trace_transformer|trace | dev_median|table (0.143) | -0.087 win |
| city | 15833 | enet_v21|table | 0.109 | trace_gru|trace, trace_cnn|trace | moves_like|table (0.136) | -0.027 win |
| highway | 2884 | enet_v21|table | 0.094 | tab_gp|table, trace_gru|trace, trace_cnn|trace, trace_s4d|trace, trace_transformer|trace | moves_like|table (0.108) | -0.013 win |
| cold | 5932 | trace_gru|trace | 0.117 | tab_mlp|table, tab_gp|table, enet_v21|table, trace_transformer|trace | moves_like|table (0.139) | -0.023 win |
| hot | 3416 | enet_v21|table | 0.100 | trace_gru|trace, trace_cnn|trace, trace_s4d|trace, trace_transformer|trace | moves_like|table (0.130) | -0.030 win |
| hilly | 1056 | enet_v21|table | 0.089 | tab_mlp|table, trace_gru|trace | moves_like|table (0.121) | -0.032 win |
| short | 3565 | enet_v21|table | 0.118 | trace_gru|trace, trace_cnn|trace, trace_s4d|trace, trace_transformer|trace | moves_like|table (0.139) | -0.021 win |
| long | 934 | tab_gp|table | 0.094 | tab_mlp|table, tab_gbm|table, formula|table, enet_v21|table, trace_gru|trace, trace_cnn|trace, trace_s4d|trace, trace_transformer|trace | moves_like|table (0.113) | -0.018 tie |

### Compute

| Arm | Train CPU-h | Exam predict CPU-s | Stored numbers | E16 |
|---|---|---|---|---|
| Climatology (sim median MJ/km x distance) | 0.001 | {'table': 0.1, 'trace': 0.6} |  | {'table': 0.3103, 'trace': 0.3103} |
| EnergyNet v2.1 (as released) | 0 | {'table': 54.8} | 1022990 | {'table': 0.1096} |
| EnergyNet v2.1 (as released) + conformal | 0.0066 | {'table': 57.2} |  | {'table': 0.1096} |
| Trip-energy law (symbolic) | 0.87 | {'table': 0.1, 'trace': 16.0} | 66 | {'table': 0.2105, 'trace': 0.2105} |
| Trip-energy law (symbolic) + conformal | 0.0131 | {'table': 0.3, 'trace': 15.4} |  | {'table': 0.2105, 'trace': 0.2105} |
| Physics anchor (E_phys + sim bands) | 0.002 | {'table': 0.0, 'trace': 14.9} |  | {'table': 0.2285, 'trace': 0.2285} |
| Physics anchor (E_phys + sim bands) + conformal | 0.0135 | {'table': 0.2, 'trace': 14.8} |  | {'table': 0.2285, 'trace': 0.2285} |
| Gradient-boosted trees | 0.24 | {'table': 134.8} | 281493 | {'table': 0.1262} |
| Gradient-boosted trees + conformal | 0.0065 | {'table': 137.3} |  | {'table': 0.1262} |
| Sparse Gaussian process | 0.81 | {'table': 1.3} | 273687 | {'table': 0.1255} |
| Sparse Gaussian process + conformal | 0.0073 | {'table': 1.5} |  | {'table': 0.1255} |
| MLP (EnergyNet net, retrained) | 0.62 | {'table': 48.5} | 1022990 | {'table': 0.1212} |
| MLP (EnergyNet net, retrained) + conformal | 0.0065 | {'table': 49.7} |  | {'table': 0.1212} |
| 1-D dilated CNN | 1.572 | {'trace': 1376.4} | 86263 | {'trace': 0.1268} |
| 1-D dilated CNN + conformal | 0.0075 | {'trace': 1381.4} |  | {'trace': 0.1268} |
| Bidirectional GRU | 1.648 | {'trace': 521.5} | 53287 | {'trace': 0.1202} |
| Bidirectional GRU + conformal | 0.0071 | {'trace': 534.0} |  | {'trace': 0.1202} |
| S4D state space | 2.75 | {'trace': 950.7} | 48919 | {'trace': 0.1297} |
| S4D state space + conformal | 0.008 | {'trace': 936.3} |  | {'trace': 0.1297} |
| Transformer (20-s patches) | 1.635 | {'trace': 678.8} | 86551 | {'trace': 0.1166} |
| Transformer (20-s patches) + conformal | 0.007 | {'trace': 684.4} |  | {'trace': 0.1166} |
