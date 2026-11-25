# tab_gbm_b2_conformal: Gradient-boosted trees (LightGBM -> numpy trees) retrained on bank b2 + Mondrian conformal bands (River Bake-off round 2)

The registered round-2 arm `tab_gbm_b2` (hash `0326ced564c223b0`) run unchanged; its 5-95 % and 25-75 % bands are recalibrated by round 1's Mondrian split-conformal layer (`arms_filt/conformal/lib/cf_mondrian.py`, unchanged; strata snow / arid / humid / other; shifts in log space), fitted on `tab_gbm_b2`'s daily dev predictions (52 dev basins, WY2001-2014). Mean and q50 are the base arm's, so NSE, KGE and FHV are identical; only coverage, CRPS and band width change. No exam data.

Dev, mean over basins: cov50 0.48 -> 0.52 (leave-one-basin-out), cov90 0.84 -> 0.88; CRPS/Q 0.654 -> 0.584.

Compute: fit 0.0029 CPU-h. Limits: as round 1's wrappers (52 dev basins, 8-22 per stratum; days within a basin are not exchangeable).
