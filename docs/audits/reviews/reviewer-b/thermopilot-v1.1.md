# ThermoPilot v1.1 | The building card carries information from later weather

**Reviewer B, first pass.** Model base `61c70bf`, audit checkout `fb19db6`. ThermoPilot v1.1 ships a classical model-predictive controller behind a comfort shield. The v1.0 imitation network remains an optional mode. The default reproduced all **2,232 packaged hourly setpoint pairs**. Its exam preparation has a separate information path through the building card.

## Future wind can change an earlier command

At issue time the controller gets a nominal building card, indoor state, a weather forecast, tariff information and the comfort schedule. The exam constructs that card by fitting a 12-parameter house model to the nominal twin's hourly coefficients over the evaluation episode. Those coefficients depend on the episode's realized wind through infiltration. The card summarizes that wind even though the controller receives no hourly wind channel. [Card fit](../../../../models/thermopilot/training/tp/features.py#L50) · [Wind-dependent coefficient](../../../../models/thermopilot/training/tp/mpc.py#L129) · [Exam episode](../../../../models/thermopilot/training/tp/episode.py#L49)

I built the same 150 m² heat-pump house twice. The weather records, including wind, agreed through hour 47. From hour 48 onward one had 2 m/s wind, the other 8 m/s. I rebuilt the nominal card for each and gave the public controllers identical forecasts, tariff, base load and starting state. Only the fitted house map changed. The coefficient multiplying indoor thermal-mass temperature in the air-temperature equation moved from **0.89788 to 0.87349**.

Across 24 issue hours and three indoor starting temperatures, the issued setpoints differed in **19 of 72 matched fresh starts**. At hour zero with the zone at 21.25 °C, the heating command moved from **22 to 20 °C**. Every issue time plus its 24-hour forecast ended before the modified wind began. This establishes a causal path in the evaluation setup. Its effect on monthly bills, comfort and incumbent rankings is unmeasured. The 72 cases are controlled starts in one synthetic house. A real deployment with a card fixed before the operating period would have a different information boundary.

## A shared weather cache can hide this effect

BuildingSim's prepared-weather cache hashes the outdoor-temperature series, timestamps and preparation settings. The key omits the wind series. Running the two fixtures in order, in one process with the cache warm, gave exactly the same hourly wind term for both. Rebuilding the high-wind case with a cleared cache changed its mean wind term after hour 48 from **0.000924 to 0.014784**, a factor of 16 as expected for a fourfold wind-speed change. The card experiment above uses the fresh calculation. This is a reproducible cache defect for weather variants with matching temperature and settings. Historical cache collisions during training or evaluation remain unverified. [Cache implementation](../../../../shared/building/buildingsim/buildingsim/weather.py#L219)

## Tariff and cold-start controls

- With a card fixed, changing tariffs and base loads strictly **beyond** the hour-zero 24-hour horizon left all 36 first-action costs and the issued setpoints exactly equal on each packaged example. Changing the previous-hour HVAC heat and electricity supplied to the uninitialized observer also had zero effect. Prior load and measured zone temperature become relevant as state accumulates. [Observer and planner](../../../../models/thermopilot/thermopilot_core.py#L276)
- Published time-of-use schedules are knowable. The ERCOT real-time tariff path instead constructs the price vector from realized hourly settlement prices and the planner reads the next 24 entries. To test sensitivity, I repriced each recorded trajectory with the first 768 hours of the repository's Houston 2023 data. Both controllers received identical current RT prices and states. Only future prices differed, using RT versus day-ahead prices with the same clipping and retail adder. Commands changed in **12, 32 and 45 of 744 hours** respectively. These are fixed-trajectory stress tests. They estimate neither the original exam's bill impact nor what day-ahead prices would have been published for every lead. [Historical price construction](../../../../models/thermopilot/training/tp/env.py#L177) · [Houston price series](../../../../models/batterypilot/examples/ERCOT_HB_HOUSTON_2023.csv)

## Model and ratchet checks

All optional-network weights were finite. Stored training scales ranged from 0.001 to 1.646. The largest absolute standardized input was 9.96 across the three packaged episodes. A float32 Torch reconstruction and the release's float64 MLP had maximum absolute logit differences of `1.60e-5` to `3.25e-5` across the three trajectories. This exceeds the audit's generic `1e-5` tolerance, but all **2,232 action argmaxes** agreed. The trainer seeds both Torch and NumPy. All seven rows in each archived ladder replayed, including reference rows and the extra requirement that a residual policy beat the shielded card-only controller. The published bootstrap resamples buildings with their episodes. This pass reran neither the closed-loop exam nor its intervals. [Trainer](../../../../models/thermopilot/training/scripts/train.py) · [Residual ladder](../../../../models/thermopilot/training/ratchet_r8.json) · [Interval calculation](../../../../models/thermopilot/training/scripts/intervals.py)

Run from the repository root with NumPy, pandas, SciPy, pvlib, Numba and CPU PyTorch. The synthetic fixture uses the repository's BuildingSim and needs no external survey or weather archive.

```sh
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/thermopilot_v1_1_probe.py
```
