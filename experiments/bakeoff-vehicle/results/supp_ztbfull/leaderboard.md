# Full-ZTBus supplementary exam (run 20261125T013250), beside the main exam 20261123T204251

EnergyNet's full ZTBus exam (handoff item 21): 5,847 labelled legs of the 1,115 exam missions of the Zurich trolleybus set, which the cloud machine did not have. MdAPE in %; bias = median (pred / y - 1). Missions: the bake-off's rule (sum of the labelled legs) and EnergyNet's rule (every leg + the standstill layover against the metered mission kWh). Pairs: paired bootstrap over missions (2,000 resamples) against EnergyNet v2.1; win / loss only when the 95 % interval excludes 0.

Reproduction: EnergyNet v2.1 legs 7.04 % (published 7.04 %), missions (EnergyNet rule) 5.59 % (published 5.59 %); dev-median baseline legs 14.01 % (published 14.01 %), missions 10.66 % (published 10.66 %).

| Row | Legs MdAPE | bias | cov90 | vs EnergyNet (legs) | Missions (bake-off rule) | vs EnergyNet | Missions (EnergyNet rule) |
|---|---|---|---|---|---|---|---|
| enet_v21 (table) | 7.04 | +0.3 | 0.96 |  | 5.22 |  | 5.59 |
| enet_v21_conformal (table) | 7.04 | +0.3 | 0.62 | +0.0 (+0.0..+0.0) tie | 5.22 | +0.0 (+0.0..+0.0) tie | 5.59 |
| trace_gru (trace) | 7.15 | -1.5 | 0.96 | +0.1 (-0.2..+0.3) tie | 5.79 | +0.6 (+0.2..+0.9) loss | 6.24 |
| trace_gru_conformal (trace) | 7.15 | -1.5 | 0.96 | +0.1 (-0.2..+0.3) tie | 5.79 | +0.6 (+0.2..+0.9) loss | 6.24 |
| tab_mlp (table) | 7.23 | +0.3 | 0.96 | +0.2 (-0.0..+0.4) tie | 5.74 | +0.5 (+0.1..+1.0) loss | 6.10 |
| tab_mlp_conformal (table) | 7.23 | +0.3 | 0.98 | +0.2 (-0.0..+0.4) tie | 5.74 | +0.5 (+0.1..+1.0) loss | 6.10 |
| trace_transformer (trace) | 7.29 | +1.2 | 0.92 | +0.3 (-0.0..+0.5) tie | 5.74 | +0.5 (+0.1..+0.8) loss | 5.89 |
| trace_transformer_conformal (trace) | 7.29 | +1.2 | 0.92 | +0.3 (-0.0..+0.4) tie | 5.74 | +0.5 (+0.1..+0.9) loss | 5.89 |
| trace_cnn (trace) | 7.52 | -1.6 | 0.96 | +0.5 (+0.2..+0.7) loss | 6.53 | +1.3 (+0.8..+1.9) loss | 6.80 |
| trace_cnn_conformal (trace) | 7.52 | -1.6 | 0.92 | +0.5 (+0.2..+0.7) loss | 6.53 | +1.3 (+0.9..+1.8) loss | 6.80 |
| tab_gbm (table) | 8.25 | -1.0 | 0.98 | +1.2 (+0.9..+1.5) loss | 5.99 | +0.8 (+0.2..+1.3) loss | 6.61 |
| tab_gbm_conformal (table) | 8.25 | -1.0 | 0.97 | +1.2 (+0.9..+1.5) loss | 5.99 | +0.8 (+0.2..+1.3) loss | 6.61 |
| trace_s4d (trace) | 8.62 | -6.1 | 0.94 | +1.6 (+1.2..+1.9) loss | 8.77 | +3.5 (+3.0..+4.2) loss | 9.69 |
| trace_s4d_conformal (trace) | 8.62 | -6.1 | 0.97 | +1.6 (+1.2..+1.9) loss | 8.77 | +3.5 (+3.0..+4.2) loss | 9.69 |
| formula (trace) | 9.29 | -6.9 | 0.99 | +2.3 (+1.8..+2.7) loss | 8.09 | +2.9 (+2.4..+3.3) loss | 8.15 |
| formula_conformal (trace) | 9.29 | -6.9 | 0.99 | +2.3 (+1.8..+2.7) loss | 8.09 | +2.9 (+2.4..+3.3) loss | 8.15 |
| formula_conformal (table) | 9.29 | -6.9 | 0.99 | +2.3 (+1.8..+2.6) loss | 8.09 | +2.9 (+2.4..+3.3) loss | 8.15 |
| tab_tabpfn (table) | 9.30 | +2.8 | 0.98 | +2.3 (+1.8..+2.7) loss | 8.20 | +3.0 (+2.4..+3.6) loss | 8.38 |
| tab_tabpfn_conformal (table) | 9.30 | +2.8 | 0.90 | +2.3 (+1.9..+2.7) loss | 8.20 | +3.0 (+2.4..+3.6) loss | 8.38 |
| tab_gp (table) | 9.52 | +4.8 | 0.98 | +2.5 (+2.0..+2.9) loss | 6.84 | +1.6 (+1.0..+2.2) loss | 6.51 |
| tab_gp_conformal (table) | 9.52 | +4.8 | 0.82 | +2.5 (+2.1..+3.0) loss | 6.84 | +1.6 (+0.9..+2.2) loss | 6.51 |
| dev_median (ref) | 14.01 | +2.4 |  |  | 10.65 |  | 10.66 |
| climatology (trace) | 15.16 | +8.7 | 0.98 | +8.1 (+7.4..+8.8) loss | 13.05 | +7.8 (+6.9..+8.4) loss | 12.64 |
| phys_anchor (trace) | 21.47 | -21.2 | 0.99 | +14.4 (+13.4..+15.5) loss | 21.27 | +16.0 (+15.2..+16.9) loss | 22.31 |
| phys_anchor_conformal (trace) | 21.47 | -21.2 | 0.85 | +14.4 (+13.4..+15.5) loss | 21.27 | +16.0 (+15.2..+16.9) loss | 22.31 |
| phys_anchor_conformal (table) | 21.48 | -21.2 | 0.85 | +14.4 (+13.4..+15.5) loss | 21.27 | +16.0 (+15.2..+16.9) loss | 22.31 |

The dev-median row predicts labelled legs only (its 1,173 layover and unlabelled records have no prediction); on missions it uses EnergyNet's baseline (pooled dev kWh/km x driven km).
