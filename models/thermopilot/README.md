<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# ThermoPilot <sub>v1.1</sub>

> **A thermostat MPC that pre-heats and pre-cools against real tariffs.**

> [!WARNING]
> Graded in simulation. There are no real thermostat logs (ecobee's Donate Your Data needs an application), so every exam number is BuildingSim with real weather, real tariffs and twin buildings, not a measured saving. v1.1's house map is fitted to the twin's own nominal BuildingSim model, so it plans on a near-copy of the house the simulator starts from (the simulated house is a perturbed copy); a real thermostat would have to identify those 12 numbers from its own data, which was not tested. On 37 real WPuQ heat-pump homes the simulator overstates the heating slope 2.7× and puts heat-pump load at night.

*Every hour, which heating and cooling setpoints keep a home or small building comfortable for the lowest bill, planning 24 h ahead on a small model of the house?*

ThermoPilot is a causal thermostat / heat-pump controller for homes and small commercial buildings. Every hour it picks a heating and a cooling setpoint from a menu of 36 pairs (off, the comfort-band edge, or pre-heat / pre-cool by up to 3 K) to minimise the bill plus a penalty of $3 per kelvin-hour outside the occupant's comfort band (per 150 m² of floor). Since v1.1 the default controller is classical, with no learned parameters: a 24 h dynamic programme ("card MPC") on a 12-number linear map of the house carried by its building card, behind a rule-based 6 h comfort shield. It replaced the v1.0 network (an MLP that imitated a perfect-foresight teacher in BuildingSim), which stays in the package as mode='network': on the rebuilt simulated val set the card MPC scores 0.700 skill against the network's 0.561, and a residual network on top of it (rung R8) did not beat it. On simulated twins of real exam units (seen re-score, 17 Nov 2026) v1.1 beats v1.0 on every set, beats the classical MPC on small commercial buildings with demand charges, and still loses to the classical MPC on homes.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/thermopilot/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/thermopilot/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Each hour: the indoor air temperature (°C), the HVAC's own heat output (W) and the meter's electricity (kW) last hour, a 24 h forecast of outdoor temperature (°C) and solar (W/m²), 24 h of energy prices ($/kWh), demand-charge windows, rates and the running monthly peak, the occupant's comfort band for the next 24 h, the expected other electricity load, and a building card: heating type, heat-pump / furnace / AC capacity and COP tables vs outdoor temperature, fan power, and (v1.1) a 12-number linear house map (air and mass temperature one hour ahead as linear functions of the mass temperature, HVAC heat, outdoor temperature, solar and the other load). The v1.0 network also reads the last 3 hours of indoor temperature and the card's UA (W/K), thermal capacitance (J/K) and floor area. |
| **Returns** | A heating and a cooling setpoint (°C; ±100 means that side is off) for the hour, after a comfort shield that never lets the zone fall behind the band of the next 6 h (recovery 1 K/h heating, 0.5 K/h cooling)<br>v1.1 also returns the card MPC's expected cost ($) of each of the 36 setpoint pairs over the next 24 h<br>No uncertainty: the controller is deterministic; exam intervals are 95 % bootstrap intervals over units |
| **Learned from** | BuildingSim v1 (shared/building, BuildingData + BuildingSim): ISO 13790 5R1C zone with Perez solar, LBL infiltration and EnergyPlus-derived heat-pump (capacity and COP vs outdoor temperature, defrost, backup strips), DX, furnace and boiler curves with part load, run at 1 h steps through the step API with exact state rollback. Bills from real URDB tariffs (TOU, demand charges) and ERCOT real-time prices, gas at 2025 EIA averages. The teacher is a dynamic program over the mass temperature (0.25 K grid, 24 h horizon, hourly replanning) with the true house, the true weather and the true state; on demand tariffs it re-simulates the month with a monthly peak target (two-pass). The classical MPC and the teacher use BuildingSim's own one-step 5R1C maps. v1.1's card MPC uses the same dynamic programme on the card's 12-number linear house map instead (no wind, no indoor humidity), fitted to the nominal twin's hourly maps; on three dev episodes the map reproduces them to 0.08-0.19 K RMS. |
| **Noise** | Forecast error on outdoor temperature that grows with lead time; a perturbed 'true' house (envelope ±12 %, air leakage ±25 %, mass class, internal loads) that the controller never sees exactly; real weather years and real tariffs |
| **Network** | v1.1 (default) has no network: a backward dynamic programme over the house's mass temperature (0.25 K grid, 24 h horizon, replanned every hour, 36 setpoint pairs; cost = energy + gas + demand-charge increments over the running monthly peak + λ × K h outside the band) on the card's 12-number linear house map, with a dead-beat observer of the mass temperature from the measured air temperature, then the 6 h comfort shield. v1.0 (mode='network'): MLP 298 → 384 → 384 → 36 (276,516 parameters) on 34 state / tariff / building-card features plus 11 × 24 h forecast rows, trained on soft targets from a perfect-foresight teacher's action values plus expected regret (behaviour cloning, then 3 DAgger rounds); argmax over the 36 pairs, then the same shield |
| **Physics also enters** | inputs: A building card from the physics model: a 12-number linear house map least-squares fitted to the nominal twin's hourly ISO 13790 5R1C maps, and heat-pump / furnace / AC capacity and COP tables on the 24 h temperature forecast<br>output: v1.1 is the physics: a dynamic programme on the card's house map, behind a comfort shield built on recovery rates (1 K/h heating, 0.5 K/h cooling); the v1.0 network, kept as an option, was trained to imitate a dynamic programme with the true house and weather |
| **Real data's jobs** | backdrop, gate, exam (never training labels) |
| **Dev set** | 30 episodes: 11 RECS 2020 dev respondents and 4 BDG2 dev buildings, January and July 2017 (frozen splits, seed 20261116); frozen simulated val set: 60 episodes. Both frozen again on 17 Nov 2026 with stable RECS twins (tied ResStock matches ranked by bldg_id) from the same seeds; v1.1 was chosen there |
| **Exam** | RECS 2020 exam respondents: 48 homes × January + July 2017, 12 real tariffs in 9 states (4 of 6 drawn per tariff slot, fixed before the exam; the list was frozen again with stable twins on 17 Nov 2026, same respondents, 39 of 48 homes got a different twin); BDG2: 16 small offices / schools / shops × January + July 2017, 4 commercial tariffs; HUE: 17 BC homes, 21 episodes; WPuQ: 38 heat-pump homes, January 2019 |

## Run it

```bash
cd models/thermopilot
pip install numpy
python demo.py        # about 12 s
```

or from anywhere in the repo: `mlratchet demo thermopilot`.

```python
import numpy as np, thermopilot_core as tp
ctx = tp.make_ctx(local_hours, price, lo, hi, pref_lo, pref_hi, state, dperiod, drate, frate)   # pandas DatetimeIndex
# card: heating kind, capacity / COP tables, fan power, and card['map'] (tp.fit_map from the house model's hourly maps)
pilot = tp.pilot(card, ctx, lam=3.0 * max(1, area_m2 / 150), base_W=expected_other_load_W)    # v1.1 card MPC
# pilot = tp.pilot(card, ctx, lam, base_W, mode='network', W=tp.load('thermopilot_v1.npz'))     # v1.0 network
for k in range(len(price)):
    sp = pilot.step(k, T_zone, q_hvac_prev_W, elec_prev_kW, forecast_T_out_24h, forecast_GHI_24h)
    # -> {'heat_sp': .., 'cool_sp': ..}  (°C; ±100 = that side off)
```

## The exam

All seen: v1.0 took these sets once at release, and on 17 Nov 2026 v1.1 and v1.0 were scored again after v1.1 was chosen on val and dev; nothing was chosen from them. The RECS list was frozen again with stable twins (same respondents, 39 of 48 homes got a different twin), so both versions were re-scored there; BDG2 and WPuQ are as frozen. Twins of real exam units (perturbed true house, noisy forecasts) under their real weather and real tariffs, against a fixed thermostat, the occupant's own setback schedule, standard setback, two pre-cool rules, the in-band pre-cool rule tuned on dev (4 h, 3 K), a classical MPC with the same nominal house and the same noisy forecast (it also reads the true wind and indoor humidity), and a perfect-foresight bound. Bill = energy + demand + gas for the month (fixed charges left out), summed over episodes. Skill = share of the perfect-foresight saving on cost + comfort (0 = fixed thermostat, 1 = perfect foresight). Every row gives v1.1 minus the best baseline with a 95 % bootstrap interval over units (homes / buildings; 2,000 resamples, seed 20261117); win or loss only when the interval excludes zero. On bill rows a baseline counts only if it keeps comfort within the ratchet's test (K h ≤ fixed + 1). Each note gives v1.0 (the network, mode='network') on the same episodes and v1.1 minus v1.0 with its interval. HUE was not run (NASA POWER irradiance not fetched): v1.0's point values are kept in one info row. Every number here is on real data that never chose anything.

| Metric | ThermoPilot | Baselines | |
|---|---|---|---|
| RECS exam twins (48 homes, 96 home-months, 12 tariffs), bill saving vs fixed thermostat<br><sub>Seen, new RECS list. −0.93 points against the MPC, 95% CI [−1.30, −0.58] (bootstrap by home, n = 48). v1.0 network 2.34 % (−1.36 [−1.80, −0.95] against the MPC); v1.1 − v1.0 +0.43 [+0.24, +0.68]. v1.1 vs the occupant's setback: +0.49 [−0.28, +1.36], tie. v1.1 2.77 % [2.10, 3.49]; perfect foresight 3.91 %.</sub> | **2.77%** | Fixed thermostat (dumb baseline): 0%<br>Setback (occupant's own schedule; 2.5 K h, fails the comfort test) (dumb baseline): 2.28%<br>Standard setback −4 / +4 K (841 K h, fails the comfort test) (dumb baseline): 8.74%<br>In-band pre-cool rule (classical): 2.54%<br>Phase-1 pre-cool rule (69 K h, fails the comfort test) (classical): 3.39%<br>Pre-cool rule tuned on dev (4 h, 3 K) (classical): 2.56%<br>MPC with forecast (same nominal house) (classical): 3.7% | ❌ loses |
| RECS exam twins, skill on cost + comfort (0 = fixed, 1 = perfect foresight)<br><sub>Seen, new RECS list. −0.12 against the MPC, 95% CI [−0.20, −0.01] (bootstrap by home, n = 48). v1.0 network 0.28 (−0.54 [−0.75, −0.37]); v1.1 − v1.0 +0.42 [+0.25, +0.64].</sub> | **0.7** | Fixed thermostat (dumb baseline): 0<br>Setback (occupant's own schedule) (dumb baseline): -0.49<br>Standard setback −4 / +4 K (dumb baseline): -333<br>In-band pre-cool rule (classical): 0.2<br>Phase-1 pre-cool rule (classical): -28.5<br>Pre-cool rule tuned on dev (4 h, 3 K) (classical): 0.32<br>MPC with forecast (same nominal house) (classical): 0.82 | ❌ loses |
| RECS exam twins, kelvin-hours outside the comfort band per episode<br><sub>Seen, new RECS list. −0.14 K h against the MPC, 95% CI [−0.32, +0.03] (bootstrap by home, n = 48). v1.0 network 1.55 K h; v1.1 − v1.0 −1.13 [−2.19, −0.33]. Perfect foresight 0.31 K h.</sub> | **0.42 K h** | Fixed thermostat (dumb baseline): 0.74 K h<br>Setback (occupant's own schedule) (dumb baseline): 2.52 K h<br>Standard setback −4 / +4 K (dumb baseline): 841 K h<br>In-band pre-cool rule (classical): 1.3 K h<br>Phase-1 pre-cool rule (classical): 69.4 K h<br>Pre-cool rule tuned on dev (4 h, 3 K) (classical): 1.08 K h<br>MPC with forecast (same nominal house) (classical): 0.56 K h | ➖ tie |
| BDG2 small commercial twins (16 buildings, 32 building-months, 4 demand tariffs), bill saving vs fixed thermostat<br><sub>Seen. +2.28 points against the MPC, 95% CI [+1.46, +3.08] (bootstrap by building, n = 16). v1.0 network 2.77 % (+0.08 [−0.85, +1.09], tie); v1.1 − v1.0 +2.20 [+1.53, +2.92]. v1.1 vs setback +3.79 [+2.85, +4.71]. Perfect foresight 7.23 %. Why the card MPC beats the classical MPC here was not investigated; the classical MPC's BDG2 result is the one that did not reproduce v1.0's (3.05 → 2.69 %). Tariffs PG&E B-19, OUC GSD, Xcel A15, Pepco GT.</sub> | **4.97%** | Fixed thermostat (dumb baseline): 0%<br>Setback (operating schedule) (dumb baseline): 1.18%<br>Standard setback −4 / +4 K (dumb baseline): 1.18%<br>In-band pre-cool rule (classical): -0.28%<br>Phase-1 pre-cool rule (127 K h, fails the comfort test) (classical): 0.2%<br>Pre-cool rule tuned on dev (4 h, 3 K) (classical): -0.03%<br>MPC with forecast (same nominal house) (classical): 2.69% | ✅ wins |
| BDG2 small commercial twins, skill on cost + comfort<br><sub>Seen. +0.30 against the MPC, 95% CI [+0.18, +0.46] (bootstrap by building, n = 16). v1.0 network 0.451 (−0.00 [−0.11, +0.10], tie); v1.1 − v1.0 +0.31 [+0.19, +0.46].</sub> | **0.757** | Fixed thermostat (dumb baseline): 0<br>Setback (operating schedule) (dumb baseline): 0.227<br>In-band pre-cool rule (classical): 0.128<br>Phase-1 pre-cool rule (classical): -12<br>Pre-cool rule tuned on dev (4 h, 3 K) (classical): 0.161<br>MPC with forecast (same nominal house) (classical): 0.452 | ✅ wins |
| BDG2 small commercial twins, kelvin-hours outside the comfort band per episode<br><sub>Seen. 0.00 against the fixed thermostat, 95% CI [0.00, 0.00] (n = 16). v1.0 network 0.05 K h. Phase-1 pre-cool rule 127 K h.</sub> | **0 K h** | Fixed thermostat (dumb baseline): 0 K h<br>Setback (operating schedule) (dumb baseline): 0 K h<br>In-band pre-cool rule (classical): 0 K h<br>Pre-cool rule tuned on dev (4 h, 3 K) (classical): 0 K h<br>MPC with forecast (same nominal house) (classical): 0.21 K h | ➖ tie |
| WPuQ heat-pump home twins (38 homes, January 2019), bill saving vs fixed thermostat<br><sub>Seen. −1.22 points against the MPC, 95% CI [−1.32, −1.11] (bootstrap by home, n = 38). v1.0 network 3.13 % (−4.03 [−4.66, −3.47]); v1.1 − v1.0 +2.81 [+2.27, +3.42]. v1.1 vs setback +0.37 [−0.36, +1.30], tie. Perfect foresight 7.50 %. Generic twins (no building metadata), stand-in tariff Xcel MN TOD.</sub> | **5.95%** | Fixed thermostat (dumb baseline): 0%<br>Setback (dumb baseline): 5.58%<br>Standard setback −4 / +4 K (869 K h, fails the comfort test) (dumb baseline): 7.7%<br>In-band pre-cool rule (= setback here) (classical): 5.58%<br>Pre-cool rule tuned on dev (= setback here) (classical): 5.58%<br>MPC with forecast (same nominal house) (classical): 7.16% | ❌ loses |
| WPuQ heat-pump home twins, skill on cost + comfort<br><sub>Seen. −0.16 against the MPC, 95% CI [−0.18, −0.15] (bootstrap by home, n = 38). v1.0 network 0.35 (−0.60 [−0.68, −0.53]); v1.1 − v1.0 +0.44 [+0.37, +0.51].</sub> | **0.79** | Fixed thermostat (dumb baseline): 0<br>Setback (dumb baseline): 0.65<br>Standard setback −4 / +4 K (dumb baseline): -118<br>In-band pre-cool rule (= setback here) (classical): 0.65<br>MPC with forecast (same nominal house) (classical): 0.95 | ❌ loses |
| WPuQ heat-pump home twins, kelvin-hours outside the comfort band per episode<br><sub>Seen. 0.00 against the fixed thermostat, 95% CI [0.00, 0.00] (n = 38): the wide 20.5-26 °C band is never left except by standard setback (869 K h). v1.0 0.00.</sub> | **0 K h** | Fixed thermostat (dumb baseline): 0 K h<br>Setback (dumb baseline): 0.01 K h<br>MPC with forecast (same nominal house) (classical): 0.01 K h | ➖ tie |
| HUE BC home twins (17 homes, 21 episodes), bill saving vs fixed thermostat<br><sub>v1.1 was not run on HUE: the NASA POWER irradiance for Vancouver is not in the repository and was not fetched. v1.0 values (exam_r7.json, point values, no 95 % interval): v1.0 network 4.29 %, skill 0.54, 0.36 K h; MPC 5.75 %, skill 0.95, 0.17 K h; setback 5.89 % at 3.07 K h; perfect foresight 5.78 %. Stand-in tariff PG&E E-TOU-C.</sub> | **not scored** | Fixed thermostat (dumb baseline): 0<br>Setback (occupant's schedule; 3.1 K h) (dumb baseline): 5.9<br>In-band pre-cool rule (classical): 5.7<br>MPC with forecast (same nominal house) (classical): 5.7 | ℹ️ |
| WPuQ reality check (report only): controller order on generic vs envelope-scaled twins, median Kendall tau<br><sub>Simulator vs simulator on 37 homes (v1.0; not re-run). 'MPC > ThermoPilot > fixed' holds; ThermoPilot vs setback flips in 18 of 37 homes. Saving on cost + comfort vs fixed, generic → scaled twins: perfect foresight 7.0 → 5.3 %, MPC 6.7 → 4.7 %, ThermoPilot 2.5 → 1.0 %, setback 4.5 → −3.2 %.</sub> | **0.7** | — | ℹ️ |

### Where it wins

- v1.1 beats v1.0 on every exam set, bill and skill, with intervals above zero: bill +0.43 points (RECS), +2.20 (BDG2), +2.81 (WPuQ); skill +0.42, +0.31, +0.44 (seen).
- On small commercial buildings with demand charges (BDG2) it beats the classical MPC: bill 4.97 % vs 2.69 %, +2.28 points [+1.46, +3.08]; skill 0.757 vs 0.452, with no time outside the band.
- Comfort: 0.42 K h per RECS home-month (fixed thermostat 0.74, setback 2.52, classical MPC 0.56; tie with the MPC [−0.32, +0.03]); 0.00 on BDG2 and WPuQ.
- No learned parameters and no weights: it runs from the building card alone, in numpy, in about 4 s per simulated month.

### Where it loses

- The classical MPC (BuildingSim's own hourly 5R1C maps of the nominal house, the true wind and indoor humidity) still beats it on homes: RECS bill −0.93 points [−1.30, −0.58], skill −0.12 [−0.20, −0.01]; WPuQ bill −1.22 [−1.32, −1.11], skill −0.16 [−0.18, −0.15].
- On homes its bill saving ties the occupant's own setback schedule (RECS +0.49 points [−0.28, +1.36]; WPuQ +0.37 [−0.36, +1.30]); it keeps much better comfort than setback (0.42 vs 2.52 K h on RECS).
- Without the comfort shield the card MPC leaves the band often (0.91 K h per dev episode; dev skill 0.572 vs 0.720 with the shield): it does not anticipate the band tightening after a setback.
- A learned residual on top of it (R8) made it worse on val (0.667 vs 0.700, 3 seeds): the network moved the setpoints in 97 % of hours where the teacher departs from the card MPC in 6 %.

## Limits

- Graded in simulation: there are no real thermostat logs (ecobee Donate Your Data needs an application; Pecan Street is paid). Every exam number is BuildingSim with real weather, real tariffs and twin buildings, and every exam set is seen.
- The card's 12-number house map is fitted to the twin's own nominal BuildingSim maps over the month being controlled. In the simulator the true house is a perturbed copy, so there is model error, but a real thermostat would have to identify the map from its own logs; that was not built or tested.
- v1.1 was chosen on numbers seen as a reference row in R8's run (the same frozen val and dev sets); the rule was applied without changes, but the candidate was proposed after seeing them.
- WPuQ reality check (37 real heat-pump homes, report only, v1.0): generic twins overstate the heating slope 2.7× (median; IQR 1.4–5.4) and peak at night while real heat pumps run flat (hourly-profile r = 0.21). Time-shifting savings are the least certain part.
- The exam's classical MPC reads the true future wind for infiltration (with the wind at its monthly mean its dev skill dropped 0.815 → 0.787 on the old dev set) and the indoor humidity; the card MPC reads neither.
- Hourly ideal thermostat: one zone, no short-cycling or minimum run times, no humidity control; on/off counts are hourly.
- Twins come from survey characteristics plus a ResStock match, not audits. HUE and WPuQ use stand-in US tariffs. HUE was not scored for v1.1.
- Perfect foresight is a practical bound (a DP on a 0.25 K grid with the 36-pair menu, plus rules), not a proven optimum.
- The comfort weight (λ = $3 per K h per 150 m²) is an input; the v1.0 network was trained at one λ and sees it only through log(λ / (mean price × size)).
- Since 17 Nov 2026 the twin builder ranks tied ResStock matches by bldg_id (training/tp/twins.py), so RECS twins rebuild on any machine; v1.0's own RECS twins (sandbox tie order) cannot be rebuilt, so v1.0's release-day RECS numbers are not comparable with these.

## The ratchet

### ThermoPilot v1.0 (BuildingSim v1, 16-17 Nov 2026)

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

### Round 2: residual on the card MPC (R8, rejected) and the card MPC alone (R9 = v1.1); BuildingSim 1.0, 17 Nov 2026

**Objective:** The same skill on the frozen simulated val set (60 episodes), frozen again on 17 Nov 2026 with stable RECS twins (tied ResStock matches ranked by bldg_id): 19 of its 28 RECS-twin episodes changed house  
**Rule:** Keep if val skill > incumbent + 0.01 AND the dev gate passes (dev skill > 0, ≥ incumbent dev skill − 0.03, dev K h per episode ≤ fixed + 1). R8 (residual) also had to beat the anchor alone (card MPC behind the 6 h shield) on val by 0.01, with three seeds decided on the 3-seed mean. R9 is that anchor, so the anchor condition does not apply to it.

| Rung | Change | Frozen-set score | Decision | Why | Dev skill | Dev K h / episode | Dev bill saving | Val beats fixed |
|---|---|---|---|---|---|---|---|---|
| r7 | R7 = the v1.0 weights, re-scored on val and dev frozen again with stable RECS twins (the starting incumbent) | 0.561 | ● kept | Starting incumbent: val 0.561 (v1.0 build 0.562), dev 0.471 (v1.0 build 0.360), dev K h 0.09 | 0.471 | 0.09 | 2.5% | 78% |
| r8-s0 | Residual MLP 334-384-384-36 on the card MPC's 36 action values (last layer zero-initialised; logits = −Q/(S·0.1) + MLP), 6 h shield; 507 new episode-months (card MPC acting, then DAgger β 0.3 / 0.2 / 0.1), 371,663 hours; seed 0 | 0.670 | ○ rejected | val 0.670 vs R7 0.561 and dev 0.718 pass the incumbent rule, but the anchor alone scores 0.700 on val: not beaten by 0.01 (3-seed mean 0.667) | 0.718 | 0.06 | 4.8% | 87% |
| r8-s1 | R8, seed 1 (same banks) | 0.644 | ○ rejected | val 0.644, dev 0.673: passes the incumbent rule, loses to the anchor alone (0.700) | 0.673 | 0.12 | 4.2% | 83% |
| r8-s2 | R8, seed 2 (same banks) | 0.686 | ○ rejected | val 0.686, dev 0.711: passes the incumbent rule, loses to the anchor alone (0.700) | 0.711 | 0.0 | 4.8% | 90% |
| r9 | The card MPC alone behind v1.0's 6 h comfort shield: a 24 h DP on the card's 12-number linear house map, no learned parameters (R8's anchor reference row, unchanged; its val and dev numbers were seen there) | 0.700 | **◆ released** | val 0.700 vs R7 0.561 (+0.139, margin 0.01); dev 0.720 vs 0.471, dev K h 0.00 vs fixed 0.00: kept, released as v1.1 (exam then re-scored as seen) | 0.72 | 0.0 | 4.5% | 93% |

<sub>Columns after *Why*: training/ratchet_r8.json: dev gate numbers (30 episodes), then the val set's fraction of episodes cheaper than fixed.</sub>

Plan fixed in NOTEBOOK.md before anything was built. The incumbent is R7 (the v1.0 weights) re-scored on the rebuilt sets: val 0.561, dev 0.471. Reference rows, not candidates (training/ratchet_r8.json): classical MPC with forecast val 0.862, dev 0.765 (it reads the true wind and indoor humidity); card MPC alone without shield val 0.573, dev 0.572 (dev K h 0.91: it loses 0.19 dev skill to the classical MPC); the anchor, card MPC behind the 6 h shield (shield chosen on dev before training), val 0.700, dev 0.720, dev K h 0.00 (−0.045 dev skill against the classical MPC). R8 beats R7 on every seed but loses to the anchor alone on every seed (3-seed mean val 0.667 vs 0.700), so it is rejected and v1.0 stays. Dev twin test of seed 0 (anchor drives): the teacher departs from the card MPC in 5.6 % of hours, the residual in 96.7 %; correlation with the teacher's departure 0.03 heating / 0.13 cooling (0.13 / 0.46 where the teacher departs). The anchor alone would pass this ladder's rule against R7; it was a reference row in R8's run. R9 then promoted it under its own plan entry (NOTEBOOK.md, written before the decision): the same code and settings, decided by the ladder rule on the same, already seen, frozen sets. R9 is classical (no learned parameters) and was released as v1.1; the v1.0 network stays available as mode='network'.

Raw per-rung logs are in [`training/`](training/).

## Physics

BuildingSim v1 (shared/building, BuildingData + BuildingSim): ISO 13790 5R1C zone with Perez solar, LBL infiltration and EnergyPlus-derived heat-pump (capacity and COP vs outdoor temperature, defrost, backup strips), DX, furnace and boiler curves with part load, run at 1 h steps through the step API with exact state rollback. Bills from real URDB tariffs (TOU, demand charges) and ERCOT real-time prices, gas at 2025 EIA averages. The teacher is a dynamic program over the mass temperature (0.25 K grid, 24 h horizon, hourly replanning) with the true house, the true weather and the true state; on demand tariffs it re-simulates the month with a monthly peak target (two-pass). The classical MPC and the teacher use BuildingSim's own one-step 5R1C maps. v1.1's card MPC uses the same dynamic programme on the card's 12-number linear house map instead (no wind, no indoor humidity), fitted to the nominal twin's hourly maps; on three dev episodes the map reproduces them to 0.08-0.19 K RMS.

**Simulator**

- Generated homes and small commercial buildings (ResStock / ComStock / CBECS priors) and twins of RECS 2020 prior-split respondents (nearest ResStock dwelling by type, climate zone, heating system and fuel, cooling, area and vintage, with the respondent's own area, occupants and thermostat settings)
- The simulated 'true' house is a perturbed copy of the card the controller is told: envelope U × LN(0, 0.12), air leakage × LN(0, 0.25), thermal-mass class, plug loads, sizing
- Real 2016 BDG2 site weather (11 US sites) and TMYx typical years for 17 US cities; 13 residential and 5 commercial URDB tariffs, synthetic TOU / demand tariffs, ERCOT 2018 real-time prices + 10 c/kWh
- Forecast error on outdoor temperature grows with lead time; comfort bands narrower when home, wider asleep / away
- 533 episode-months (391k hours) labelled with the teacher's value of all 36 actions

**Checked against independent references**

- BuildingSim (platform checks, checks/*.json): steady conduction to 1.8e-9, RC step within 0.0035 K, ISO 13786 wall amplitude −1.6 %; BESTEST 29 of 34 values in the published ranges; ResStock twins (126 homes) site energy median 1.10×, daily r 0.92
- Numba DP sweep == numpy DP loop exactly (max difference 0)
- v1.1: the numpy card MPC in thermopilot_core.py reproduces the training-side runs exactly (closed-loop J to the cent on dev); the numpy demo replays its closed-loop setpoints in 744 / 744 hours on each of 3 exam episodes; the browser port matches Python to 8.5e-15 on 40 parity cases (20 per mode)
- v1.0 network: numpy inference == torch to 0 (float32); the JS port == Python on 2,232 / 2,232 demo hours
- Reality check (report only): on 37 WPuQ heat-pump homes, generic twins' heating slope is a median 2.7× the real one (IQR 1.4–5.4); real January profiles are flat while twins peak at night (profile r = 0.21)

## Help wanted

- Real thermostat logs: ecobee Donate Your Data (research application) would give a real exam of setpoints, run times and time constants.
- UK heat-pump field trials (RHPP, Electrification of Heat): UK Data Service registration; they would test the heat-pump load shape the WPuQ check found wrong.
- Pecan Street Dataport (paid): US homes with circuit-level HVAC data.
- EnergyPlus as a reference engine: no PyPI wheel with an engine and GitHub was blocked in the build sandbox; install it outside and compare BuildingSim's zone response.

## Next rungs

- Identify the card's house map from a thermostat's own logs (indoor temperature, HVAC runtime, weather) and test the card MPC on maps fitted that way, in simulation first.
- Find out why the card MPC beats the classical MPC on BDG2 demand tariffs (shield, demand-charge handling or model error).
- A residual that cannot act on near-tied actions (R8's failure), or one trained on closed-loop regret in the card house.
- Calibrate residential twins on RECS dev annual energy, then build a new frozen val set and re-score.
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
