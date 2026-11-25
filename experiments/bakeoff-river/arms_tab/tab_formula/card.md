# tab_formula: attribute -> parameter formulas inside WaterSim (River Bake-off, arm B4)

**What it is.** Seventeen short closed-form formulas turn a basin's 18 static values (attributes and climate
indices of the forcing) into WaterSim parameters: HBV (9), degree-day snow (6), channel velocity and a precipitation
multiplier. WaterSim then simulates the flow, exactly as in tab_dpl. Protocol: **ungauged**.

**Built-in assumption.** As tab_dpl (every basin is an HBV bucket model), plus: the attribute -> parameter link is
simple enough to write down. No network at inference; a hydrologist can read, check and argue with every law.

**How the formulas were found** (LawFinder-style, `lawsearch.py` vendored unchanged from LawFinder). The teacher is
tab_dpl r1, the network trained through the simulator. On 11,400 sim-train basins it gives each parameter's
unbounded value z. For each parameter, lawsearch evolves expression trees over the 18 named variables (operators
+ - * / ^ exp log sqrt inv sq; constants fitted by Levenberg-Marquardt; islands, regularised evolution, a hall of
fame per complexity) for 50 s and keeps the Pareto front of complexity vs error. The released rung takes the
simplest formula within 0.02 R^2 of the best one (complexity <= 15). The parameter is then
`lo + (hi - lo) sigmoid(z)` (log-scale for FC, K0, K1, K2, CH_VEL, PMULT) within WaterSim's bounds; a non-finite z
falls back to the teacher's median (3 of 11,400 train basins for MAXBAS).

**The released laws** (z, R^2 against the teacher; variables are the static vector: log10 area, elevation km,
log10 slope, ln p_mean, ln aridity, lp_freq = share of days with P < 1 mm, t_mean_c = T / 10 C, ...):

| Param | Size | R^2 | z = |
|---|---|---|---|
| FC | 13 | 0.73 | `-0.2686 - (reservoir_frac + 2 log_aridity) * (-1.346) * (lp_freq - 0.5026)` |
| LP | 9 | 0.89 | `0.4799 elev_mean_km^0.788 - lp_freq - log_aridity` |
| BETA | 10 | 0.54 | `-3.181 - forest_frac - (1.151 / log_area^elev_range_km)^2` |
| PERC | 7 | 0.91 | `frac_snow + log_p_mean - log_aridity - 2.714` |
| UZL | 10 | 0.88 | `-0.6785 (log_aridity - log_p_mean + 5.414 - exp(frac_snow))` |
| K0 | 11 | 0.65 | `-1.775 / (t_mean_c + (log_slope_mkm^elev_range_km)^(4.57^t_mean_c))` |
| K1 | 12 | 0.80 | `0.7461^log_area (exp(log_slope_mkm - 1.535) + log_p_mean + t_mean_c)` |
| K2 | 8 | 0.65 | `1.36 ln(2.332 + frac_snow - log_aridity)` |
| MAXBAS | 10 | 0.58 | `sqrt(-2.352 (forest_frac - log_area) + t_mean_c) - 5.479` |
| TT | 12 | 0.66 | `-0.2724 + t_mean_c / (t_mean_c + (lp_freq / soil_porosity)^2)` |
| TTI | 10 | 0.75 | `forest_frac + 2.684 log_p_mean - ln(log_area) + 0.622` |
| CFMAX | 10 | 0.60 | `-1.406 (exp(forest_frac / (1.47 + elev_mean_km)) - frac_snow)` |
| SFCF | 14 | 0.55 | `lp_freq^exp(2 forest_frac - (elev_range_km + log_p_mean) / 0.6471) - 1.911` |
| TLAPSE | 15 | 0.81 | `elev_range_km^frac_snow - 3.291 - forest_frac + log_p_mean + (log_slope_mkm elev_mean_km)^0.4774` |
| PGRAD | 10 | 0.74 | `elev_range_km^-0.2633 + (frac_snow - 1.508) sqrt(pet_mean)` |
| CH_VEL | 9 | 0.80 | `ln(exp(irrigation_frac) + log_area) - 1.407 + frac_snow` |
| PMULT | 9 | 0.92 | `elev_mean_km / 6.271 - lp_freq - log_aridity - 0.0501` |

Exact trees and constants: `formulas.json` (with every Pareto front). Some laws read like textbook hydrology:
the precipitation multiplier grows with elevation and wetness (the bank's own undercatch model has the same signs),
percolation grows with wetness and snow, baseflow recession K2 slows in arid basins, the hillslope routing base
MAXBAS lengthens with basin area while channel velocity rises with area (bigger rivers flow faster).

**Inference.** `predict.py`: formulas (numpy, `formula_eval.py` + `lawsearch.py`) -> parameters -> `wsim.run_batch`
(the harness's vendored WaterSim, as tab_dpl). Mean = simulated flow; bands = residual quantiles by flow level fitted
on 600 held-out sim-train runs (`bands.npz`). `numpy_only: false` (WaterSim is numba-compiled). ~0.2 CPU s per dev
basin.

**Ratchet** (rule: sim val median NSE > incumbent + 0.01 and dev NSE >= 0.95 x incumbent; all 800 val runs, 52 dev basins):

| Rung | Change | Sim val NSE | Dev NSE | Decision |
|---|---|---|---|---|
| r0 | **simplest formula within 0.02 R^2 of each front** | **0.415** | **0.386** | **released** |
| r1 | same structures, constants tuned through the torch simulator (MSE of q/p_mean) | 0.408 | 0.376 | rejected |
| r2 | most accurate formula on each front (complexity ~14 instead of ~10) | 0.418 | 0.392 | rejected (+0.003) |

**Released r0 on the dev gate** (52 basins, WY2001-2014): NSE 0.386, KGE 0.412, FHV -27.7 %, 50/90 % coverage
0.53 / 0.95, CRPS / mean flow 0.401. Paired bootstrap of the NSE median difference: vs FlowCast v1.0 -0.039
(95 % CI -0.084..+0.022, a tie), vs U-dev -0.080 (-0.176..-0.019, a loss), vs NWM -0.198 and GR4J -0.200 (losses).
By class: snow 0.51, arid 0.20, humid 0.55, regulated 0.30, large 0.26.

**What its behaviour suggests.** Distilling the 6k-weight network into 17 formulas of 7-15 symbols lost nothing
(0.415 vs 0.414 sim, 0.386 vs 0.383 dev), even though the formulas explain only 54-92 % of the teacher's parameter
variance: what the network learned about parameters is low-dimensional, and the simulator is forgiving of the rest.
Tuning the formula constants end-to-end through the simulator did not help (the MSE loss is not the NSE metric);
longer formulas did not help either. The ceiling is the teacher's, i.e. how much attributes say about parameters.

**Compute.** 0.25 single-thread CPU hours (search 251 s, constant tuning 613 s, exports); the teacher's 2.25 h are
in tab_dpl. Exam estimate ~5 min.

**Limits.** Inherits tab_dpl's limits (one structure, no reservoirs, fixed residual bands). The formulas were
searched on simulated basins; some (K0, SFCF, TLAPSE) contain nested powers that could behave oddly far outside the
bank's attribute ranges (z is clipped to +-12 and then bounded by a sigmoid, so parameters always stay within
WaterSim's bounds).

**Files.** `predict.py`, `formula_eval.py`, `lawsearch.py` (vendored, unchanged), `physarm.py`, `wsim.py`,
`formulas.json`, `bands.npz`, `ratchet.json`. Training code: `arms_tab/src/search_formula.py`, `formula_torch.py`,
`train_dpl.py` (r1), `export_phys.py`.
