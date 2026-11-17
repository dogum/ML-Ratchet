<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# ThermoPilot <sub>v1.0</sub>

> **A learned thermostat that pre-heats and pre-cools against real tariffs.**

> [!WARNING]
> Graded inside the simulator. There are no real thermostat logs (ecobee's Donate Your Data needs an application), so every exam number is BuildingSim with real weather, real tariffs and twin buildings, not a measured saving. On 37 real WPuQ heat-pump homes the simulator overstates the heating slope 2.7× and puts heat-pump load at night, and the ThermoPilot-vs-setback order flips in 18 of 37 homes between generic and calibrated twins.

*Every hour, which heating and cooling setpoints keep a home or small building comfortable for the lowest bill?*

ThermoPilot is a learned, causal thermostat / heat-pump controller for homes and small commercial buildings. Every hour it picks a heating and a cooling setpoint from a menu of 36 pairs (off, the comfort-band edge, or pre-heat / pre-cool by up to 3 K) to minimise the bill plus a penalty of $3 per kelvin-hour outside the occupant's comfort band (per 150 m² of floor). It was trained only in BuildingSim, by imitating a perfect-foresight dynamic-programming teacher (behaviour cloning, then three DAgger rounds), with a 6 h comfort shield. It was graded once on simulated twins of real exam units (48 RECS 2020 homes, 16 BDG2 small commercial buildings, 17 HUE homes, 38 WPuQ heat-pump homes) under their real weather and real tariffs. It always beats a fixed thermostat. The classical MPC with the same house and forecast beats it on every home set; on small commercial buildings with demand charges the two tie (95 % intervals, re-scored as seen on 17 Nov 2026).

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/thermopilot/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/thermopilot/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Each hour: the indoor air temperature now and over the last 3 hours (°C), the HVAC's own heat output (W) and the meter's electricity (kW) last hour, a 24 h forecast of outdoor temperature (°C) and solar (W/m²), 24 h of energy prices ($/kWh), demand-charge windows, rates and the running monthly peak, the occupant's comfort band for the next 24 h, and a building card: UA (W/K), thermal capacitance (J/K), floor area (m²), heating type, and heat-pump / furnace / AC capacity and COP tables vs outdoor temperature. |
| **Returns** | A heating and a cooling setpoint (°C; ±100 means that side is off) for the hour, after a comfort shield that never lets the zone fall behind the band of the next 6 h (recovery 1 K/h heating, 0.5 K/h cooling)<br>No uncertainty: the policy is deterministic; exam intervals are 95 % bootstrap intervals over units |
| **Learned from** | BuildingSim v1 (shared/building, BuildingData + BuildingSim): ISO 13790 5R1C zone with Perez solar, LBL infiltration and EnergyPlus-derived heat-pump (capacity and COP vs outdoor temperature, defrost, backup strips), DX, furnace and boiler curves with part load, run at 1 h steps through the step API with exact state rollback. Bills from real URDB tariffs (TOU, demand charges) and ERCOT real-time prices, gas at 2025 EIA averages. The teacher is a dynamic program over the mass temperature (0.25 K grid, 24 h horizon, hourly replanning) with the true house, the true weather and the true state; on demand tariffs it re-simulates the month with a monthly peak target (two-pass). The classical MPC and the teacher use BuildingSim's own one-step 5R1C maps. |
| **Noise** | Forecast error on outdoor temperature that grows with lead time; a perturbed 'true' house (envelope ±12 %, air leakage ±25 %, mass class, internal loads) that the controller never sees exactly; real weather years and real tariffs |
| **Network** | MLP 298 → 384 → 384 → 36 (276,516 parameters, 277k): 34 state / tariff / building-card features plus 11 × 24 h forecast rows; trained on soft targets softmax(−Q/τ) from the teacher's action values plus expected regret, behaviour cloning then 3 DAgger rounds; argmax over the 36 setpoint pairs, then the 6 h comfort shield |
| **Physics also enters** | inputs: A building card from the physics model: UA, thermal capacitance, heat-pump / furnace / AC capacity and COP tables evaluated on the 24 h temperature forecast<br>output: A comfort shield built on recovery rates (1 K/h heating, 0.5 K/h cooling) clips the network's setpoints so the zone can always reach the next 6 h of the band; the equipment model meets the setpoints within capacity |
| **Real data's jobs** | backdrop, gate, exam (never training labels) |
| **Dev set** | 30 episodes: 11 RECS 2020 dev respondents and 4 BDG2 dev buildings, January and July 2017 (frozen splits, seed 20261116); frozen simulated val set: 60 episodes |
| **Exam** | RECS 2020 exam respondents: 48 homes × January + July 2017, 12 real tariffs in 9 states (4 of 6 drawn per tariff slot, fixed before the exam); BDG2: 16 small offices / schools / shops × January + July 2017, 4 commercial tariffs; HUE: 17 BC homes, 21 episodes; WPuQ: 38 heat-pump homes, January 2019 |

## Run it

```bash
cd models/thermopilot
pip install numpy
python demo.py        # about 2 s
```

or from anywhere in the repo: `mlratchet demo thermopilot`.

```python
import numpy as np, thermopilot_core as tp
W = tp.load('thermopilot_v1.npz')
ctx = tp.make_ctx(local_hours, price, lo, hi, pref_lo, pref_hi, state, dperiod, drate, frate)   # pandas DatetimeIndex
pilot = tp.ThermoPilot(W, card, ctx, lam=3.0 * max(1, area_m2 / 150), base_W=expected_other_load_W)
for k in range(len(price)):
    sp = pilot.step(k, T_zone, q_hvac_prev_W, elec_prev_kW, forecast_T_out_24h, forecast_GHI_24h)
    # -> {'heat_sp': .., 'cool_sp': ..}  (°C; ±100 = that side off)
```

## The exam

Taken once with R7 (v1.0) after the release choice, never used for any choice; re-scored on 17 Nov 2026 with the released weights (seen). Twins of real exam units (perturbed true house, noisy forecasts) under their real weather and real tariffs, against a fixed thermostat, the occupant's own setback schedule, standard setback, two pre-cool rules, the in-band pre-cool rule tuned on dev, a classical MPC with the same nominal house and the same noisy forecast, and a perfect-foresight bound. Bill = energy + demand + gas for the month (fixed charges left out), summed over episodes. Skill = share of the perfect-foresight saving on cost + comfort (0 = fixed thermostat, 1 = perfect foresight). Every row gives ThermoPilot minus the best baseline with a 95 % bootstrap interval over units (homes / buildings; 2,000 resamples, seed 20261117); win or loss only when the interval excludes zero. On bill rows a baseline counts only if it keeps comfort within the ratchet's test (K h ≤ fixed + 1), so standard setback and the phase-1 pre-cool rule are listed but not counted. The re-score reproduced WPuQ exactly and BDG2 except the MPC; the RECS twins differ from the v1.0 build's, so the RECS rows moved (v1.0 values in each note). HUE was not re-run. Every number here is on real data that never chose anything.

| Metric | ThermoPilot | Baselines | |
|---|---|---|---|
| RECS exam twins (48 homes, 96 home-months, 12 tariffs), bill saving vs fixed thermostat<br><sub>Seen re-score. −1.55 points against the MPC, 95% CI [−2.28, −0.93] (bootstrap by home, n = 48). ThermoPilot vs the occupant's setback: −0.46 [−1.73, +0.56], tie. ThermoPilot 2.56 % [1.67, 3.44]; perfect foresight 4.25 %. v1.0 (90 % CI, RECS twins as built then): ThermoPilot 2.18 %, setback 2.24 %, MPC 3.50 %. ThermoPilot beats the best rule in 27 % of home-months (MPC 52 %).</sub> | **2.56%** | Fixed thermostat (dumb baseline): 0%<br>Setback (occupant's own schedule; 1.8 K h, fails the comfort test) (dumb baseline): 3.01%<br>Standard setback −4 / +4 K (835 K h, fails the comfort test) (dumb baseline): 9.38%<br>In-band pre-cool rule (classical): 3.1%<br>Phase-1 pre-cool rule (69 K h, fails the comfort test) (classical): 3.38%<br>Pre-cool rule tuned on dev (4 h, 3 K) (classical): 3.07%<br>MPC with forecast (same nominal house) (classical): 4.1% | ❌ loses |
| RECS exam twins, skill on cost + comfort (0 = fixed, 1 = perfect foresight)<br><sub>Seen re-score. −0.60 against the MPC, 95% CI [−0.85, −0.39] (bootstrap by home, n = 48). The dev-tuned pre-cool rule (0.35) also beats ThermoPilot's 0.24 here. v1.0: ThermoPilot 0.33, MPC 0.84.</sub> | **0.24** | Fixed thermostat (dumb baseline): 0<br>Setback (occupant's own schedule) (dumb baseline): -0.18<br>Standard setback −4 / +4 K (dumb baseline): -362<br>In-band pre-cool rule (classical): 0.28<br>Phase-1 pre-cool rule (classical): -32.2<br>Pre-cool rule tuned on dev (4 h, 3 K) (classical): 0.35<br>MPC with forecast (same nominal house) (classical): 0.84 | ❌ loses |
| RECS exam twins, kelvin-hours outside the comfort band per episode<br><sub>Seen re-score. +0.91 K h against the fixed thermostat, 95% CI [+0.20, +2.00] (bootstrap by home, n = 48). Better than setback (1.77), worse than both pre-cool rules in the band and the MPC. Mean PPD while home 23.5 % (fixed 23.0 %, MPC 24.0 %). v1.0: ThermoPilot 1.43, fixed 0.71, setback 2.52.</sub> | **1.28 K h** | Fixed thermostat (dumb baseline): 0.36 K h<br>Setback (occupant's own schedule) (dumb baseline): 1.77 K h<br>Standard setback −4 / +4 K (dumb baseline): 835 K h<br>In-band pre-cool rule (classical): 1 K h<br>Phase-1 pre-cool rule (classical): 69.5 K h<br>Pre-cool rule tuned on dev (4 h, 3 K) (classical): 0.88 K h<br>MPC with forecast (same nominal house) (classical): 0.49 K h | ❌ loses |
| BDG2 small commercial twins (16 buildings, 32 building-months, 4 demand tariffs), bill saving vs fixed thermostat<br><sub>Seen re-score. +0.08 points against the MPC, 95% CI [−0.85, +1.09] (bootstrap by building, n = 16). ThermoPilot vs setback: +1.58 [+0.30, +2.81], better. ThermoPilot, the rules and perfect foresight (7.23 %) reproduce v1.0 exactly; the MPC does not (v1.0 3.05 %, now 2.69 %; a one-hour difference can move a monthly demand peak). v1.0 called this row a loss on the point values. Tariffs PG&E B-19, OUC GSD, Xcel A15, Pepco GT.</sub> | **2.77%** | Fixed thermostat (dumb baseline): 0%<br>Setback (operating schedule) (dumb baseline): 1.18%<br>Standard setback −4 / +4 K (dumb baseline): 1.18%<br>In-band pre-cool rule (classical): -0.28%<br>Phase-1 pre-cool rule (127 K h, fails the comfort test) (classical): 0.2%<br>Pre-cool rule tuned on dev (4 h, 3 K) (classical): -0.03%<br>MPC with forecast (same nominal house) (classical): 2.69% | ➖ tie |
| BDG2 small commercial twins, skill on cost + comfort<br><sub>Seen re-score; new row (computed by exam_report.py in v1.0, not shown). −0.00 against the MPC, 95% CI [−0.11, +0.10] (bootstrap by building, n = 16). v1.0: ThermoPilot 0.451, MPC 0.471.</sub> | **0.451** | Fixed thermostat (dumb baseline): 0<br>Setback (operating schedule) (dumb baseline): 0.227<br>In-band pre-cool rule (classical): 0.128<br>Phase-1 pre-cool rule (classical): -12<br>Pre-cool rule tuned on dev (4 h, 3 K) (classical): 0.161<br>MPC with forecast (same nominal house) (classical): 0.452 | ➖ tie |
| BDG2 small commercial twins, kelvin-hours outside the comfort band per episode<br><sub>Seen re-score; new row. +0.05 K h against the fixed thermostat, 95% CI [0.00, +0.09] (bootstrap by building, n = 16; the interval touches zero). Phase-1 pre-cool rule 127 K h.</sub> | **0.05 K h** | Fixed thermostat (dumb baseline): 0 K h<br>Setback (operating schedule) (dumb baseline): 0 K h<br>In-band pre-cool rule (classical): 0 K h<br>Pre-cool rule tuned on dev (4 h, 3 K) (classical): 0 K h<br>MPC with forecast (same nominal house) (classical): 0.21 K h | ➖ tie |
| HUE BC home twins (17 homes, 21 episodes), bill saving vs fixed thermostat<br><sub>v1.0 values, not re-run on 17 Nov 2026 (the NASA POWER irradiance for Vancouver is not in the repository), so there is no 95 % interval; verdict on the point values. ThermoPilot 4.29 % [3.92, 4.66] (90 %); MPC 5.75 %, 0.17 K h; setback 5.89 % at 3.07 K h; ThermoPilot vs setback −1.70 [−1.89, −1.53] (90 %); perfect foresight 5.78 %. Stand-in tariff PG&E E-TOU-C; January (+ July for the 4 with AC), Vancouver 2017.</sub> | **4.3%** | Fixed thermostat (dumb baseline): 0%<br>Setback (occupant's schedule; 3.1 K h) (dumb baseline): 5.9%<br>Standard setback −4 / +4 K (675 K h) (dumb baseline): 11%<br>In-band pre-cool rule (classical): 5.7%<br>MPC with forecast (same nominal house) (classical): 5.7% | ❌ loses |
| HUE BC home twins, skill on cost + comfort<br><sub>v1.0 values (exam_r7.json), not re-run; no interval; verdict on the point values. New row.</sub> | **0.54** | Fixed thermostat (dumb baseline): 0<br>Setback (occupant's schedule) (dumb baseline): -0.54<br>In-band pre-cool rule (classical): 0.31<br>MPC with forecast (same nominal house) (classical): 0.95 | ❌ loses |
| HUE BC home twins, kelvin-hours outside the comfort band per episode<br><sub>v1.0 values (exam_r7.json), not re-run; no interval; verdict on the point values. New row.</sub> | **0.36 K h** | Fixed thermostat (dumb baseline): 0.14 K h<br>Setback (occupant's schedule) (dumb baseline): 3.07 K h<br>In-band pre-cool rule (classical): 1.34 K h<br>MPC with forecast (same nominal house) (classical): 0.17 K h | ❌ loses |
| WPuQ heat-pump home twins (38 homes, January 2019), bill saving vs fixed thermostat<br><sub>Seen re-score; reproduces v1.0 exactly. −4.03 points against the MPC, 95% CI [−4.66, −3.47] (bootstrap by home, n = 38). ThermoPilot vs setback: −2.45 [−3.16, −1.54], worse. Perfect foresight 7.50 %. Generic twins (no building metadata), stand-in tariff Xcel MN TOD.</sub> | **3.13%** | Fixed thermostat (dumb baseline): 0%<br>Setback (dumb baseline): 5.58%<br>Standard setback −4 / +4 K (869 K h, fails the comfort test) (dumb baseline): 7.7%<br>In-band pre-cool rule (= setback here) (classical): 5.58%<br>Pre-cool rule tuned on dev (= setback here) (classical): 5.58%<br>MPC with forecast (same nominal house) (classical): 7.16% | ❌ loses |
| WPuQ heat-pump home twins, skill on cost + comfort<br><sub>Seen re-score; new row. −0.60 against the MPC, 95% CI [−0.68, −0.53] (bootstrap by home, n = 38).</sub> | **0.35** | Fixed thermostat (dumb baseline): 0<br>Setback (dumb baseline): 0.65<br>Standard setback −4 / +4 K (dumb baseline): -118<br>In-band pre-cool rule (= setback here) (classical): 0.65<br>MPC with forecast (same nominal house) (classical): 0.95 | ❌ loses |
| WPuQ heat-pump home twins, kelvin-hours outside the comfort band per episode<br><sub>Seen re-score; new row. 0.00 against the fixed thermostat, 95% CI [0.00, 0.00] (n = 38): the wide 20.5-26 °C band is never left except by standard setback (869 K h).</sub> | **0 K h** | Fixed thermostat (dumb baseline): 0 K h<br>Setback (dumb baseline): 0.01 K h<br>MPC with forecast (same nominal house) (classical): 0.01 K h | ➖ tie |
| WPuQ reality check (report only): controller order on generic vs envelope-scaled twins, median Kendall tau<br><sub>Simulator vs simulator on 37 homes (v1.0; not re-run). 'MPC > ThermoPilot > fixed' holds; ThermoPilot vs setback flips in 18 of 37 homes. Saving on cost + comfort vs fixed, generic → scaled twins: perfect foresight 7.0 → 5.3 %, MPC 6.7 → 4.7 %, ThermoPilot 2.5 → 1.0 %, setback 4.5 → −3.2 %.</sub> | **0.7** | — | ℹ️ |

### Where it wins

- Always beats the fixed thermostat on bill: 2.6 % (RECS homes), 2.8 % (BDG2 commercial), 4.3 % (HUE), 3.1 % (WPuQ heat pumps).
- On commercial demand tariffs (BDG2) it ties the classical MPC (bill +0.08 points [−0.85, +1.09], skill 0.451 vs 0.452) and beats the operating-schedule setback by 1.6 points [0.3, 2.8], with 0.05 K h outside the band.
- Better comfort than the occupant's setback schedule: 1.3 vs 1.8 K h per home-month on RECS twins, 0.4 vs 3.1 K h on HUE.

### Where it loses

- The classical MPC with the same nominal house and the same noisy forecast beats it on every home set: RECS −1.55 points [−2.28, −0.93], WPuQ −4.03 [−4.66, −3.47], HUE 5.7 % vs 4.3 % (point); skill 0.24-0.54 against the MPC's 0.84-0.95.
- On homes it saves no more than the setback schedule (RECS −0.46 points [−1.73, +0.56], tie) or less (WPuQ −2.45 [−3.16, −1.54]; HUE −1.7, v1.0).
- The in-band pre-cool rule tuned on dev beats it on RECS skill (0.35 vs 0.24).
- Worse comfort than a fixed thermostat on RECS twins (1.28 vs 0.36 K h per episode, +0.91 [+0.20, +2.00]).

## Limits

- The 17 Nov 2026 re-score did not reproduce v1.0 on RECS-derived twins: the twin builder here gives different houses (e.g. RECS_116893: UA 287 → 434 W/K, light → medium mass), so the RECS rows and the dev / val ratchet numbers moved (R7 dev skill 0.360 → 0.482, val 0.562 → 0.556). WPuQ reproduced exactly; BDG2 exactly except the MPC (3.05 → 2.69 %). Cause: the twin builder picks among the 5 best ResStock matches with np.argsort, whose order of tied scores (68-way ties) depends on the numpy build and CPU; the BuildingSim edit of 17 Nov (1.0.1) is not the cause (twins identical before and after).
- The exam MPC reads the true future wind for infiltration (ThermoPilot sees none); with the wind at its monthly mean its dev skill drops 0.815 → 0.787.
- Graded in simulation: there are no real thermostat logs (ecobee Donate Your Data needs an application; Pecan Street is paid). Every exam number is BuildingSim with real weather, real tariffs and twin buildings.
- WPuQ reality check (37 real heat-pump homes, report only): generic twins overstate the heating slope 2.7× (median; IQR 1.4–5.4) and peak at night while real heat pumps run flat (hourly-profile r = 0.21). The controller order MPC > ThermoPilot > fixed holds (Kendall tau 0.70), but ThermoPilot vs setback flips in 18 of 37 homes, so that comparison cannot be trusted from the simulator. Time-shifting savings are the least certain part.
- Hourly ideal thermostat: one zone, no short-cycling or minimum run times, no humidity control; on/off counts are hourly.
- Twins come from survey characteristics plus a ResStock match, not audits. HUE and WPuQ use stand-in US tariffs.
- Perfect foresight is a practical bound (a DP on a 0.25 K grid with the 36-pair menu, plus rules), not a proven optimum.
- Heat pumps are under-represented in training (most generated and RECS homes heat with gas furnaces).
- The comfort weight (λ = $3 per K h per 150 m²) was fixed in training; the network sees it only through log(λ / (mean price × size)).

## The ratchet

**Objective:** Skill on the frozen simulated val set (60 episodes): sum over episodes of the relative saving of J vs the fixed thermostat, divided by the same sum for perfect foresight (0 = fixed thermostat, 1 = perfect foresight); J = control-dependent bill (energy + demand + gas) + λ × kelvin-hours outside the occupant's band  
**Rule:** Keep if val skill > incumbent + 0.01 AND the dev gate passes: dev skill > 0, dev skill ≥ incumbent dev skill − 0.03, and mean kelvin-hours per dev episode ≤ fixed thermostat + 1 K h.

| Rung | Change | Frozen-set score | Decision | Why | Dev skill | Dev K h / episode | Dev bill saving | Val beats fixed |
|---|---|---|---|---|---|---|---|---|
| r1 | Behaviour cloning of the perfect-foresight DP teacher (173 episodes, 127k hours), MLP 298-256-256-36, no shield | 0.084 | ○ rejected | val skill 0.084 vs incumbent 0.000; dev skill −3.02, dev K h 5.0 vs fixed 0.1: dev gate failed (comfort) | -3.022 | 4.98 | 1.4% | 73% |
| r2 | R1 weights + anticipatory comfort shield (stay within 1 K/h of every band edge in the next 4 h) | 0.377 | ○ rejected | val skill 0.377 vs 0.000; dev skill −0.81, dev K h 1.5 vs fixed 0.1: dev gate failed | -0.81 | 1.51 | 2.0% | 75% |
| r3 | R1 weights + 6 h comfort shield (heating 1 K/h, cooling 0.5 K/h recovery allowance) | 0.419 | ○ rejected | val skill 0.419 vs 0.000; dev skill −0.19, dev K h 0.6 vs fixed 0.1: dev gate failed (dev skill ≤ 0) | -0.191 | 0.6 | 2.0% | 78% |
| r4 | Retrain on the BC bank + DAgger round 1 (286 episodes, 210k hours; student = R1 + shield, beta 0.3), shield 4 | 0.517 | ● kept | val skill 0.517 vs 0.000; dev skill 0.043, dev K h 0.3 vs fixed 0.1: kept | 0.043 | 0.28 | 2.0% | 78% |
| r5 | Retrain on BC + DAgger rounds 1-2 (411 episodes; round 2 student = R4, beta 0.2), shield 4 | 0.499 | ○ rejected | val skill 0.499 vs incumbent 0.517 (needs +0.01); dev skill −0.068 vs 0.043: rejected | -0.068 | 0.51 | 2.4% | 77% |
| r6 | Wider network 298-384-384-36 (275k params), 24 epochs, same 440 episodes, shield 4 | 0.550 | ● kept | val skill 0.550 vs 0.517; dev skill 0.200 vs 0.043, dev K h 0.4: kept | 0.2 | 0.37 | 1.3% | 78% |
| r7 | R6 architecture + DAgger round 3 (533 episodes, 391k hours; round 3 student = R6, beta 0.1), shield 4 | 0.562 | **◆ released** | val skill 0.562 vs 0.550; dev skill 0.360 vs 0.200, dev K h 0.3: kept, released as v1.0 (exam run once afterwards) | 0.36 | 0.29 | 2.5% | 73% |

<sub>Columns after *Why*: training/ratchet.json: dev gate numbers (30 episodes), then the val set's fraction of episodes cheaper than fixed.</sub>

The starting incumbent is the fixed thermostat (skill 0). Before R1 was scored, the oracle was upgraded to the two-pass demand-peak version (an oracle that only amortised demand charges was worse than a fixed thermostat on commercial demand tariffs) and skill was changed from a ratio of sums (dominated by large commercial bills) to relative savings per episode; every rung below is scored on the same frozen set. Reference: the classical MPC with forecast scores 0.83 on val and 0.83 on dev. Rule notes, said plainly: the RECS exam uses 4 of the 6 respondents drawn per tariff slot, a cut made for compute before any exam result existed; and during the build the agent once ran `pkill -f` with its own script name to stop a job, which could have killed other processes — it matched only its own process. No exam result was used to choose anything. The R5 change text says 411 episodes while the model card and R6 say 440 (the logged text is kept). The R6 text says 275k parameters; the shipped 298-384-384-36 network has 276,516. Re-scored on 17 Nov 2026 in this repository with the released R7 weights (NOTEBOOK.md): val skill 0.556 (logged 0.562; MPC 0.859) and dev skill 0.482 (logged 0.360; MPC 0.815). The RECS respondent twins built here differ from the sandbox's (dev has 11 of them), so the logged rungs below cannot be re-scored like for like; R1-R6 weights were not kept. The next rung (a residual on the cheap MPC, planned in NOTEBOOK.md) compares with R7 as re-scored here.

Raw per-rung logs are in [`training/`](training/).

## Physics

BuildingSim v1 (shared/building, BuildingData + BuildingSim): ISO 13790 5R1C zone with Perez solar, LBL infiltration and EnergyPlus-derived heat-pump (capacity and COP vs outdoor temperature, defrost, backup strips), DX, furnace and boiler curves with part load, run at 1 h steps through the step API with exact state rollback. Bills from real URDB tariffs (TOU, demand charges) and ERCOT real-time prices, gas at 2025 EIA averages. The teacher is a dynamic program over the mass temperature (0.25 K grid, 24 h horizon, hourly replanning) with the true house, the true weather and the true state; on demand tariffs it re-simulates the month with a monthly peak target (two-pass). The classical MPC and the teacher use BuildingSim's own one-step 5R1C maps.

**Simulator**

- Generated homes and small commercial buildings (ResStock / ComStock / CBECS priors) and twins of RECS 2020 prior-split respondents (nearest ResStock dwelling by type, climate zone, heating system and fuel, cooling, area and vintage, with the respondent's own area, occupants and thermostat settings)
- The simulated 'true' house is a perturbed copy of the card the controller is told: envelope U × LN(0, 0.12), air leakage × LN(0, 0.25), thermal-mass class, plug loads, sizing
- Real 2016 BDG2 site weather (11 US sites) and TMYx typical years for 17 US cities; 13 residential and 5 commercial URDB tariffs, synthetic TOU / demand tariffs, ERCOT 2018 real-time prices + 10 c/kWh
- Forecast error on outdoor temperature grows with lead time; comfort bands narrower when home, wider asleep / away
- 533 episode-months (391k hours) labelled with the teacher's value of all 36 actions

**Checked against independent references**

- BuildingSim (platform checks, checks/*.json): steady conduction to 1.8e-9, RC step within 0.0035 K, ISO 13786 wall amplitude −1.6 %; BESTEST 29 of 34 values in the published ranges; ResStock twins (126 homes) site energy median 1.10×, daily r 0.92
- Numba DP sweep == numpy DP loop exactly (max difference 0)
- numpy inference == torch to 0 (float32); the JS port == Python on 2,232 / 2,232 demo hours; the numpy demo replays the closed-loop setpoints in 744 / 744 hours on each of 3 exam episodes
- Reality check (report only): on 37 WPuQ heat-pump homes, generic twins' heating slope is a median 2.7× the real one (IQR 1.4–5.4); real January profiles are flat while twins peak at night (profile r = 0.21)

## Help wanted

- Real thermostat logs: ecobee Donate Your Data (research application) would give a real exam of setpoints, run times and time constants.
- UK heat-pump field trials (RHPP, Electrification of Heat): UK Data Service registration; they would test the heat-pump load shape the WPuQ check found wrong.
- Pecan Street Dataport (paid): US homes with circuit-level HVAC data.
- EnergyPlus as a reference engine: no PyPI wheel with an engine and GitHub was blocked in the build sandbox; install it outside and compare BuildingSim's zone response.

## Next rungs

- More DAgger rounds plus a heat-pump-heavy bank (val skill still rose with every round).
- Make the student a residual on a cheap MPC (planned in NOTEBOOK.md, 17 Nov 2026): on dev the teacher departs from the MPC in only 4.1 % of hours, while ThermoPilot's corrections correlate only 0.10 / 0.14 (heating / cooling) with the teacher's.
- Calibrate residential twins on RECS dev annual energy, then build a new frozen val set and re-score every rung.
- A real exam once ecobee access exists.

## Files

| Path | What |
|---|---|
| `thermopilot_core.py`, `pipeline.py` | Inference — numpy/scipy only |
| `thermopilot_v1.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [Building Data Genome 2 (BDG2) — Miller et al., Scientific Data 7, 368 (2020)](https://github.com/buds-lab/building-data-genome-project-2) | noise (2016 site weather for training), dev (4 BDG2 dev buildings), exam (16 small commercial buildings, 2017 weather for the RECS twins) | CC BY-SA 4.0 (derived data share-alike; examples/bdg2/, lab/ and the instrument data are BDG2-derived) |
| [EIA Residential Energy Consumption Survey (RECS) 2020 microdata](https://www.eia.gov/consumption/residential/data/2020/) | truth (twins and comfort bands of prior-split respondents), dev (11 respondents), exam (48 respondents) | Public domain (US government) |
| [NREL ResStock 2025 R1 / ComStock 2025 R3 (OEDI)](https://resstock.nrel.gov/) | physics (generator priors, nearest-dwelling twin matching) | CC BY 4.0 |
| [HUE, the Hourly Usage of Energy dataset for buildings in British Columbia — Makonin, Data in Brief (2019)](https://doi.org/10.7910/DVN/N3HGRN) | exam (17 homes, Vancouver 2017 weather) | CC BY 4.0 |
| [WPuQ heat-pump and household data, Lower Saxony — Schlemminger et al., Scientific Data (2022)](https://doi.org/10.5281/zenodo.5642902) | exam (38 heat-pump homes, January 2019 weather), reality check (37 homes' real heat-pump electricity) | CC BY 4.0 |
| [Utility Rate Database (URDB), OpenEI / NREL](https://openei.org/wiki/Utility_Rate_Database) | physics (18 real tariffs for training and exam bills) | Open data, free use with credit |
| [ERCOT real-time settlement point prices 2018 / 2019 (via GridData)](https://www.ercot.com/mp/data-products) | physics (real-time tariff, + 10 c/kWh) | Public market data, free use with credit |
| [TMYx typical weather years, climate.onebuilding.org](https://climate.onebuilding.org/) | noise (17 US cities for training) | Cite Lawrie and Crawley, TMYx; the site says All Rights Reserved and states no reuse licence: used for simulation only, not redistributed |
| [NASA POWER hourly irradiance; EIA natural gas prices (2025 averages)](https://power.larc.nasa.gov/) | physics (solar for the HUE homes; gas price $15.34 residential, $10.98 commercial per Mcf) | Free use with credit / public domain (US government) |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
