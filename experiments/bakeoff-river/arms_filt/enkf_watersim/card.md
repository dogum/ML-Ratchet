# enkf_watersim: ensemble Kalman filter on WaterSim (gauged track)

**Family:** data assimilation (`assim`). **Protocol:** gauged (observed flow up to day t − lead is an input).
**Released rung:** r1. Agent C, River Bake-off.

## What it is
A physical model that is corrected by the gauge every day. 32 copies ("members") of one WaterSim basin model
run side by side. Each day, each member is stepped forward with slightly different weather; then the observed flow
of that day pulls every member's stores towards values that would have produced it. To forecast day t + lead, the
members start from the corrected state of day t and run `lead` days with the known weather of those days
(a hindcast, as FlowCast's nowcast: the weather of the target days is known).

There is no neural network and no training in the usual sense. Nothing is learned from real flow before the window:
the filter learns about each basin only from that basin's own past flow inside the batch.

## The model inside (WaterSim, vendored, byte-identical to the repo)
- Snow: `watersim.snow.dd_step` (HBV-light degree-day) on up to 5 elevation bands built from the elevation mean and
  range, with WaterSim's lapse rate and normalised precipitation gradient; PET reduced by snow cover, as
  `watersim.engine.run_core`.
- Runoff: `watersim.models.hbv.step` (soil SM, upper box SUZ, lower box SLZ, triangular hillslope unit hydrograph).
- Channel: WaterSim's `routing='uh'` (gamma UH, shape 3, mean travel = Hack length / 1 m/s).
- Prior parameters (released r1): `watersim.generator.params_from_attributes` / `resolve_params` (WaterSim's attribute transfer
  functions with its dev-fitted prior adjustments) from a spec built from the batch's 11 attributes; members
  scattered around that median with WaterSim's own scatter model (scale 1.0). Precipitation multiplier prior =
  WaterSim p1's water-yield formula, log m = −0.088 + 0.058 elev_km + 0.153 frac_snow + 0.397 log(P/3 mm).
- Spin-up: the warm-up year's forcing is run once (no flow used), then the filter runs over warm-up + target rows.

## The filter (`enkf_core.py`, numba)
- State per member: snow ice + liquid per band, SM, SUZ, SLZ, the hillslope and channel UH buffers, and (r1)
  8 log parameters: precipitation multiplier, FC, BETA, K1, K2, PERC, K0, LP (joint state-parameter estimation,
  Moradkhani et al. 2005 style, random walk 0.003-0.005 per day).
- Observation: y = log(Q_obs + c), c = 0.02 × mean forcing P; predicted h = log(Q + c) of each member's flow that
  day. Serial square-root update (EnSRF, Whitaker & Hamill 2002), no perturbed observations, deterministic.
- States enter the update as log(x + 1 mm) (UH buffers log(x + c)): multiplicative, always-positive corrections.
  Snow is not updated by flow (it made no difference on sim train).
- Spread: daily multiplicative rain noise (sd 0.2), temperature noise (1 °C), multiplicative store noise (0.1),
  observation error 0.3 (log), relaxation of the posterior spread to the prior spread (RTPS 0.9), innovations
  clipped at 3 sd. These were tuned on 250 simulated TRAIN runs only (`tools/tune.py`, ~70 variants).
- Output: mean = ensemble mean; quantiles from the 32 members × 8 fixed draws of log-normal gauge noise (sd 0.1).
- Causality: the forecast for row t uses the analysis of row t − lead, i.e. qobs[: t − lead + 1] only
  (register.py's perturbation test). Missing observations: no update that day.

## Rungs (rule: lead-3 sim val NSE > incumbent + 0.01 AND lead-3 dev NSE ≥ 0.95 × incumbent's)

| Rung | Change | Sim val NSE L1 / L3 / L7 | Dev NSE L1 / L3 / L7 | Decision |
|---|---|---|---|---|
| r0 | state-only EnKF, attribute-prior parameters | 0.777 / 0.538 / 0.450 | 0.640 / 0.500 / 0.460 | base |
| r1 | + 8 augmented log parameters, wider parameter prior | 0.839 / 0.672 / 0.589 | 0.635 / 0.528 / 0.413 | **released** (+0.134 sim; dev 0.528 ≥ 0.475) |
| r2 | + UZL augmented, offset c 0.02 → 0.2 × P | 0.852 / 0.705 / 0.645 | 0.595 / 0.488 / 0.459 | rejected (dev 0.488 < 0.502) |
| r3 | r1 with agent B's physics-in-the-loop parameters (tab_dpl) instead of WaterSim's attribute priors | 0.842 / 0.673 / 0.575 | 0.610 / 0.461 / 0.403 | rejected (+0.001 sim; dev 0.461 < 0.502) |
| (diagnostic) | open loop: r0 without assimilation | | 0.381 / 0.383 / 0.379 | not a rung |

Sim val = the frozen 800 runs × 2 windows at each lead with qobs = gauge-like simulated flow; dev = 52 dev basins
WY2001-2014, Caravan Q.

## Released r1 on the dev gate (52 basins; medians)

| Lead | NSE | KGE | FHV % | cov50 / cov90 | CRPS / mean Q | vs FlowCast nowcast (paired bootstrap) | vs persistence | vs NWM |
|---|---|---|---|---|---|---|---|---|
| 1 day | 0.635 | 0.697 | −18.9 | 0.75 / 0.96 | 0.192 | +0.002, tie | +0.086, tie | +0.043, tie |
| 3 days | 0.528 | 0.556 | −31.8 | 0.63 / 0.92 | 0.252 | −0.040, tie | +0.667, win | −0.050, tie |
| 7 days | 0.413 | 0.456 | −35.9 | 0.52 / 0.87 | 0.302 | −0.107, loss | +0.928, win | −0.148, loss |

FlowCast nowcast dev: 0.633 / 0.567 / 0.520 (CRPS 0.225 / 0.246 / 0.281); persistence 0.549 / −0.139 / −0.515;
NWM 0.573; GR4J calibrated on 1999-2008 (overlaps dev) 0.586. Dev NSE by class at lead 3: snow 0.75, arid 0.36,
humid 0.67, regulated 0.57, large 0.14.

## Reading
- Assimilation is worth a lot: the same model without the gauge scores 0.38 on dev; the filter gives 0.64 at lead 1.
- The filter matches FlowCast's nowcast at lead 1 (best CRPS of the gauged rows on dev, 0.192) and fades with lead:
  its corrections sit in the stores and the forecast drifts back to what the (imperfect) model does with the rain.
- Peaks are too low on real basins (FHV −32 % at lead 3 vs −12 % on sim): the parameters learned in log space fit
  the low and middle flows; storms that ERA5-Land under-catches cannot be corrected before they are observed.
- The sim-to-real gap is larger than for the learners: r2 helped on sim and hurt on dev.
- Agent B's learned parameter mapping (r3) did not help the filter: equal on sim val, −0.067 at lead 3 on dev, though
  its peaks were less low (dev FHV −23 % vs −32 % at lead 3; KGE 0.58 vs 0.56). Once the gauge is assimilated, the
  starting parameters matter less than how the filter moves them.

## Compute
0.98 single-thread CPU-hours (0.64 tuning on sim train, 0.32 rung evaluations, 0.02 smoke tests). Inference ~0.26 / 0.37 / 0.53 CPU-s
per dev basin-period (14 years) at leads 1 / 3 / 7; first call compiles numba (~3 s).

## Limits
- One model structure (HBV) for every basin; WaterSim's bank uses four. No reservoirs or irrigation in the filter.
- Lumped forcing at the mean elevation; no forecast-weather uncertainty (the weather of the target days is known).
- Parameters drift with the random walk over a decade; no smoother, no restart from earlier analyses.
- Gauge noise is autocorrelated (WaterSim's AR(1) rating error, real ice/backwater effects); the filter assumes
  independent daily errors.
- numba is required (numpy_only false).
