# seq_ssm_b2_conformal: S4D state space + MLP encoder retrained on bank b2 + Mondrian conformal bands (River Bake-off round 2)

The registered round-2 arm `seq_ssm_b2` (hash `c009c95f5618c7eb`) run unchanged; its 5-95 % and 25-75 % bands are recalibrated by round 1's Mondrian split-conformal layer (`arms_filt/conformal/lib/cf_mondrian.py`, unchanged; strata snow / arid / humid / other; shifts in log space), fitted on `seq_ssm_b2`'s daily dev predictions (52 dev basins, WY2001-2014). Mean and q50 are the base arm's, so NSE, KGE and FHV are identical; only coverage, CRPS and band width change. No exam data.

Dev, mean over basins: cov50 0.49 -> 0.51 (leave-one-basin-out), cov90 0.89 -> 0.88; CRPS/Q 0.653 -> 0.558.

Compute: fit 0.0018 CPU-h. Limits: as round 1's wrappers (52 dev basins, 8-22 per stratum; days within a basin are not exchangeable).
