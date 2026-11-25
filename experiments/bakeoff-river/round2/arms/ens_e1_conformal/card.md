# ens_e1_conformal: E1: equal-weight ensemble of the five v1.0 tied arms + Mondrian conformal bands (River Bake-off round 2)

The registered round-2 arm `ens_e1` (hash `42733b915cc4d4ed`) run unchanged; its 5-95 % and 25-75 % bands are recalibrated by round 1's Mondrian split-conformal layer (`arms_filt/conformal/lib/cf_mondrian.py`, unchanged; strata snow / arid / humid / other; shifts in log space), fitted on `ens_e1`'s daily dev predictions (52 dev basins, WY2001-2014). Mean and q50 are the base arm's, so NSE, KGE and FHV are identical; only coverage, CRPS and band width change. No exam data.

Dev, mean over basins: cov50 0.52 -> 0.50 (leave-one-basin-out), cov90 0.88 -> 0.88; CRPS/Q 0.520 -> 0.497.

Compute: fit 0.0033 CPU-h. Limits: as round 1's wrappers (52 dev basins, 8-22 per stratum; days within a basin are not exchangeable).
