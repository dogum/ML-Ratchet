# TabPFN supplementary exam (run 20261124T230626), table track, beside the main exam 20261123T204251

E16 = mean exam MdAPE over the 16 D16 sets (lower is better); paired cluster bootstrap with shared resamples, as phase3/analyze.py. The main board is unchanged; these rows are a supplement.

| Row | E16 | E16 CRPS(log) |
|---|---|---|
| enet_v21 | 0.1096 | 0.0942 |
| tab_mlp | 0.1212 | 0.1038 |
| tab_mlp_conformal | 0.1212 | 0.0943 |
| classical | 0.1253 | 0.1599 |
| tab_gp | 0.1255 | 0.1025 |
| tab_gp_conformal | 0.1255 | 0.0957 |
| tab_gbm | 0.1262 | 0.0988 |
| tab_gbm_conformal | 0.1262 | 0.094 |
| tab_tabpfn | 0.1328 | 0.1052 |
| tab_tabpfn_conformal | 0.1328 | 0.0972 |
| formula | 0.2105 | 0.1406 |
| phys_anchor | 0.2285 | 0.1636 |
| ephys | 0.2285 | 0.2605 |
| climatology | 0.3103 | 0.189 |

| Pair | E16 diff (95 % CI) | verdict |
|---|---|---|
| tab_tabpfn vs enet_v21 | 0.0232 (0.0179..0.0285) | loss |
| tab_tabpfn vs tab_mlp | 0.0116 (0.0054..0.0188) | loss |
| tab_tabpfn vs tab_gbm | 0.0066 (0.002..0.0102) | loss |
| tab_tabpfn vs tab_gp | 0.0073 (0.0006..0.0144) | loss |
| tab_tabpfn vs classical | 0.0075 (0.001..0.0143) | loss |
| tab_tabpfn_conformal vs enet_v21 | 0.0232 (0.0179..0.0285) | loss |
| tab_tabpfn_conformal vs tab_mlp | 0.0116 (0.0054..0.0188) | loss |
| tab_tabpfn_conformal vs tab_gbm | 0.0066 (0.002..0.0102) | loss |
| tab_tabpfn_conformal vs tab_gp | 0.0073 (0.0006..0.0144) | loss |
| tab_tabpfn_conformal vs classical | 0.0075 (0.001..0.0143) | loss |

Same records: {'rows_checked': ['class_median', 'climatology', 'dev_median', 'enet_v21', 'enet_v21_conformal', 'ephys', 'formula', 'formula_conformal', 'label', 'moves_like', 'phys_anchor', 'phys_anchor_conformal', 'regression', 'tab_gbm', 'tab_gbm_conformal', 'tab_gp', 'tab_gp_conformal', 'tab_mlp', 'tab_mlp_conformal'], 'units_both': 1306017, 'only_in_supplement': 0, 'only_in_main': 0, 'max_rel_diff_y': 1.6047384721051827e-07, 'max_rel_diff_pred': 1.156053031081683e-11}
