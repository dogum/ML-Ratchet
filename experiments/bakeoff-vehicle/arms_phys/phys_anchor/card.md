# phys_anchor: the road-load physics estimate E_phys alone

**What it is.** The "no learning" row: EnergyNet's generic physics estimate E_phys, the number every EnergyNet
network corrects. Road-load wheel power from the trip trace (aero with the wind along the heading, rolling resistance
with the ISO 28580 tyre-temperature factor and surface factors, grade, inertia), integrated into positive and
negative wheel work, then turned into trip energy with three textbook forms (EnergyNet `features.e_phys`):
- conventional: E = W+ / 0.2845 + 4,642 W x (m / 1,500 kg)^0.7 x duration (engine-to-wheel efficiency + idle);
- hybrid: E = max(W+ - 0.3757 W-, 0.2 W+) / 0.3023 + 1,662 W x (m / 1,500)^0.7 x duration (regen);
- battery-electric: E = max(W+ / 0.8928 - 0.6305 W-, 0.3 W+) + 566 W x (m / 1,500)^0.7 x duration;
- plug-in: f x E_bev + (1 - f) x E_hev, f = the charge-depleting share from start SOC and battery size (0.5 unknown).
Road load at the tier: exact mass and coefficients where known, else reported mass or class priors.
Constants: EnergyNet's own (PHYS_K, fitted by EnergyNet on its simulated bank). **Dev-free: no real label touches
this arm, and no dev-fitted scale was released** (`dev_fitted_scale: false`; the scale rung was tried and rejected).

**Inputs.** Table track: the input log E_phys (X[:, 90]). Trace track: the same number recomputed from the 1 Hz trace,
spec and rl (`lawvars.from_trace` with a byte copy of EnergyNet's `features.py` = `bake.e_phys_from_trace`); dev
D16 0.2277 vs 0.2278 on the table (float16 traces).

**Bands.** Simulation only: q_k = E_phys x exp(r_k), r_k = the 5 / 25 / 50 / 75 / 95 % quantiles of log(y / E_phys) on
the simulated train bank (two tier draws) per architecture x light / heavy duty (`bands.json`). `pred` = E_phys
exactly (so MdAPE and bias equal the harness's E_phys reference row); `q50` = E_phys x the simulated median ratio
(conventional light duty x1.28), i.e. the band says "the truth is usually above road-load", which is true on the
simulator and on VED, but not on highway / dynamometer highway or the ICCT truck.

**Ratchet** (rule: sim val skill > incumbent + 0.01 AND dev D16 not worse by > 0.01):

| Rung | Change | Sim val skill | MdAPE | Dev D16 | Decision |
|---|---|---|---|---|---|
| r0 | E_phys, EnergyNet's defaults, sim bands | 0.048 | 0.210 | **0.2278** | **released** |
| r1 | x per-powertrain scale fitted on simulations (conv x1.22, hybrid x1.17, plug-in x1.09, BEV x1.07) | 0.199 | 0.189 | 0.2521 | rejected (D16 +0.024) |
| r2 | x per-powertrain scale fitted on DEV (calibration; conv x1.22, hybrid x1.18, plug-in x1.16, BEV x0.98) | 0.203 | 0.186 | 0.2416 (in-sample) | rejected (D16 +0.014) |

r1 / r2 numbers come from the harness's score.py functions in process (`../src/fscore.py`, which reproduces
evaluate.py's E_phys row exactly); r0 from evaluate.py on both tracks. Why the scales fail: road-load under-predicts
stop-and-go and cold trips (idle, warm-up, HVAC) and is close on steady highway driving; a single factor per powertrain
moves both, so VED ICE improves (0.299 -> 0.153) while EPA highway (0.093 -> 0.276), US06 (0.078 -> 0.295) and the ICCT
truck (0.865 -> 1.27) get worse. That is the job of a law with a time term (`../formula`).

**Released r0 on dev** (MdAPE / cov50 / cov90): VED ICE 0.299 / 0.63 / 0.99, HEV 0.202 / 0.63 / 0.94, EV 0.153 / 0.55 /
0.91, PHEV 0.225 / 0.64 / 0.91; EPA conv FTP 0.190 / 0.68 / 0.95, HWY 0.093 / 0.13 / 0.68; Mersin diesel 0.171 / 0.54 /
0.75; ICCT 0.865 / 0.00 / 0.33; ZTBus legs 0.378 / 0.20 / 1.00. Mean over the D16 sets: cov50 0.47, cov90 0.85, CRPS
(log) 0.153. Sim val: cov50 0.48, cov90 0.89, CRPS 0.155. The Mondrian conformal wrapper `phys_anchor_conformal`
recalibrates these bands on dev.

**Compute.** No training: 0.002 CPU-h (band quantiles + the two scale rungs). Inference ~1 microsecond per row (table),
~0.2 ms per row (trace, physics integrals).

**Limits.** E_phys's constants were fitted by EnergyNet on an earlier simulated bank; on this bank it is 28 % low for
conventional light duty (median), and on real heavy-duty trucks (ICCT) it is 2.4x too high. No cold-start, HVAC or
engine-map physics beyond the idle term.
