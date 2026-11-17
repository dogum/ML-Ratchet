# ThermoPilot v1.0 model card

## What it does
A learned, causal thermostat / heat-pump controller for homes and small commercial buildings. Every hour it picks a
heating and a cooling setpoint from a menu of 36 pairs (each side: off, the current comfort-band edge, or the occupied
band edge shifted by 0–3 K for pre-heating / pre-cooling). Objective: monthly bill (energy + TOU and flat demand
charges + gas) plus λ × kelvin-hours outside the occupant's comfort band (λ = $3 per K h per 150 m² of floor).
Equipment limits are respected because the setpoints are met (or not) by the simulated equipment: heat-pump capacity
and COP vs outdoor temperature, defrost, backup strips, furnace / AC capacity.

## Inputs (all causal: available to a thermostat at that hour)
- indoor air temperature now and the last 3 hours; the HVAC's own heat output and the meter's electricity last hour;
- a 24 h weather forecast (outdoor temperature, solar), with lead-time-dependent error in training and exam;
- tariff: 24 h of energy prices, demand-charge windows and rates, the running monthly peak;
- the occupant comfort band for the next 24 h (narrower when home, wider asleep / away);
- a building card: UA, thermal capacitance, floor area, heating type, heat-pump / furnace / AC capacity and COP tables,
  expected other electric load. In BuildingSim this comes from the nominal twin; the simulated house differs from it.

## Outputs
`{'heat_sp', 'cool_sp'}` in °C (±100 = that side off), after a comfort shield (setpoints never let the zone fall
behind the band of the next 6 h, assuming recovery of 1 K/h heating and 0.5 K/h cooling).

## Physics and simulator
BuildingSim (ML Ratchet phase 1): ISO 13790 5R1C zone, Perez solar, LBL infiltration, EnergyPlus-derived heat-pump,
DX, furnace curves with part load and defrost; verified against analytic cases, BESTEST (29/34 in range), ResStock /
ComStock twins (BuildingSim's `checks/*.json`, in the building platform zip). ThermoPilot runs it at 1 h steps through the step API.
- Teacher check: the numba DP sweep (`training/tp/dp_fast.py`) equals the numpy DP loop exactly (max difference 0; the
  one-off check script was not shipped).
- The classical MPC and the teacher use BuildingSim's own one-step 5R1C maps (exact up to infiltration linearisation).

## Training data (simulation only)
- Buildings: generated homes and small commercial buildings (ResStock / ComStock / CBECS priors) and twins of RECS
  2020 *prior-split* respondents (nearest ResStock dwelling by type, climate zone, heating system and fuel, cooling,
  area and vintage, with the respondent's own area, occupants and thermostat settings). The simulated "true" house is a
  perturbed copy of what the controller is told (envelope U ×LN(0, 0.12), air leakage ×LN(0, 0.25), thermal-mass
  class, plug loads, sizing).
- Weather: real 2016 BDG2 site weather (11 US sites) and TMYx typical years for 17 US cities. Exam uses 2017 / 2019.
- Tariffs: 13 residential and 5 commercial URDB tariffs, synthetic TOU / demand tariffs, and ERCOT real-time prices
  (+10 c/kWh) of 2018 (exam: 2019).
- Teacher: dynamic program over the mass temperature (0.25 K grid, 24 h horizon, hourly replanning) with perfect
  weather, the true house and the true state. For demand tariffs it re-simulates the month with monthly peak targets
  (BuildingSim's exact state rollback). Each hour is labelled with the teacher's value of all 36 actions.
- Student: MLP 298 → 384 → 384 → 36 (277k parameters), loss = cross-entropy to softmax(−Q/τ) + expected regret,
  trained on 533 episode-months (391k hours): behaviour cloning, then three DAgger rounds (states visited by the
  student, labelled by the teacher). numpy == torch to 0 (float32); JS port == Python on 2,232/2,232 demo hours.

## Ratchet (frozen sim val = 60 episodes; dev gate = 30 episodes of RECS dev respondents + BDG2 dev buildings)
Skill = Σ relative saving of J vs fixed / Σ relative saving of perfect foresight (0 = fixed thermostat, 1 = perfect
foresight). Keep if val skill > incumbent + 0.01 and dev skill > 0, ≥ incumbent − 0.03, dev K h ≤ fixed + 1.
Classical MPC with forecast for reference: val 0.83, dev 0.83.

| rung | change | val skill | dev skill | dev K h/ep | decision |
|---|---|---|---|---|---|
| R1 | behaviour cloning, 173 episodes, 256×256 | 0.084 | −3.02 | 5.0 | rejected (dev comfort) |
| R2 | R1 + 4 h comfort shield | 0.377 | −0.81 | 1.5 | rejected (dev) |
| R3 | R1 + 6 h shield (1 / 0.5 K/h) | 0.419 | −0.19 | 0.6 | rejected (dev) |
| R4 | + DAgger round 1 (286 episodes) | 0.517 | 0.043 | 0.28 | **kept** |
| R5 | + DAgger round 2 (440 episodes) | 0.499 | −0.07 | 0.51 | rejected (val) |
| R6 | 384×384, 24 epochs | 0.550 | 0.200 | 0.37 | **kept** |
| R7 | + DAgger round 3 (533 episodes) | 0.562 | 0.360 | 0.29 | **kept = v1.0** |

(The oracle was upgraded to the two-pass demand version and the skill to relative savings before R1 was scored;
every rung above is scored on the same frozen set.)

## Exam (never used for any choice; run once with R7)
Twins of real exam units, perturbed true house, noisy forecasts, real weather, real tariffs. Bill change vs a fixed
thermostat (positive = cheaper), 90 % bootstrap over units in brackets; K h = kelvin-hours outside the band per episode.

**RECS 2020 exam respondents** (48 homes × January + July 2017, 12 real tariffs in 9 states, comfort band from each
respondent's own thermostat settings; first 4 of 6 drawn per tariff, decided for compute before any result):

| controller | bill saving | vs setback | K h/ep | PPD % | skill |
|---|---|---|---|---|---|
| fixed thermostat | 0 | −2.3 % | 0.7 | 22.7 | 0 |
| setback (occupant's own schedule) | 2.2 % [1.5, 3.0] | 0 | 2.5 | 23.1 | −0.55 |
| standard setback (−4 / +4 K) | 9.2 % [7.9, 10.6] | 7.2 % | 832 | 23.4 | ≪ 0 |
| in-band pre-cool rule | 2.4 % [1.7, 3.2] | 0.2 % | 1.5 | 23.3 | 0.15 |
| MPC with forecast | 3.5 % [2.7, 4.4] | 1.3 % | 0.9 | 23.7 | 0.84 |
| **ThermoPilot** | **2.2 % [1.6, 2.8]** | **−0.1 %** | **1.4** | **23.3** | **0.33** |
| perfect foresight | 3.6 % [2.9, 4.5] | 1.4 % | 0.7 | 23.8 | 1 |

**BDG2 exam buildings** (16 small offices / schools / shops, Jan + Jul 2017, PG&E B-19, OUC GSD, Xcel A15, Pepco GT):
fixed 0; setback 1.2 %; pre-cool −0.3 %; MPC 3.1 % [1.9, 4.4]; **ThermoPilot 2.8 % [1.4, 4.2]** (0.05 K h, skill
0.45); perfect foresight 7.2 %.
**HUE homes** (17 BC homes, Jan [+ Jul with AC], Vancouver 2017, PG&E E-TOU-C stand-in): setback 5.9 % (3.1 K h);
MPC 5.7 % (0.2 K h); **ThermoPilot 4.3 % (0.4 K h, skill 0.54)**; perfect foresight 5.8 %.
**WPuQ heat-pump homes** (38 German homes, Jan 2019 weather, Xcel MN TOD stand-in, generic twins): setback 5.6 %;
MPC 7.2 %; **ThermoPilot 3.1 % (skill 0.35)**; perfect foresight 7.5 %.

Reading: ThermoPilot always beats the fixed thermostat, keeps comfort better than the occupant's setback schedule,
and on commercial demand tariffs is close to the classical MPC. On homes it saves about the same money as the
setback schedule (RECS) or less (HUE, WPuQ), and captures only a third to a half of what perfect foresight could.
The classical MPC, which uses the same building model and the same noisy forecast, is better everywhere. It beats the
best rule in 28 % of RECS home-months (MPC 49 %). Standard setback saves the most money only by leaving the band for
hundreds of K h. Setback raised the bill on 2 of 12 RECS tariffs here (Duke RT, Rocky Mountain Power TOU) and on BDG2
in January (−2.5 %), versus 6 of 11 in phase 1's 1R1C house.
Coverage: the intervals are bootstrap intervals of a deterministic simulation over units, not predictive sigmas;
no coverage factor applies.

## Reality check (WPuQ, report only)
Real heat-pump electricity of 37 WPuQ homes (2019) vs generic twins: the twins' heating slope (kWh/day per K) is a
median 2.7× the real one (IQR 1.4–5.4); scaling each twin's envelope (median ×0.41) matches it. The real January
daily profile is flat; the twins peak at night (profile correlation r = 0.21). Re-running all seven controllers on
generic and envelope-scaled twins (January 2019): median Kendall tau of the controller order 0.70. Saving on cost +
comfort vs fixed, generic → scaled twins: perfect foresight 7.0 → 5.3 %, MPC 6.7 → 4.7 %, ThermoPilot 2.5 → 1.0 %,
setback 4.5 → −3.2 %. "MPC > ThermoPilot > fixed" held; ThermoPilot vs setback flipped in 18 of 37 homes, so that
comparison cannot be trusted from the simulator. Files: `training/results/wpuq/`.

## Limits
1. The exam is simulated; no real thermostat data (ecobee DYD gated, Pecan Street paid).
2. Timing of heat-pump load disagrees with real WPuQ homes; time-shifting savings are the least certain.
3. Hourly ideal thermostat: no short-cycling, no minimum run times, no humidity control; on/off counts are hourly.
4. Twins come from survey characteristics + ResStock matches, not audits; HUE / WPuQ use stand-in US tariffs.
5. Perfect foresight is a practical bound (DP on a grid with a fixed action menu, plus rules), not a proven optimum.
6. The student underperforms the classical MPC by 0.1–0.5 skill; more DAgger rounds were still helping when stopped.
7. Heat pumps are under-represented in training (most generated / RECS homes heat with gas furnaces).

## Files (in this repository)
| file | what |
|---|---|
| `thermopilot_core.py` | numpy inference: features, MLP, action menu, comfort shield |
| `thermopilot_v1.npz` | weights (298-384-384-36 MLP, input normalisation, shield = 4) |
| `demo.py`, `examples/bdg2/demo_episodes.npz` | replay of 3 exam episodes (numpy only, about 1 s) |
| `pipeline.py` | run ThermoPilot on a BuildingSim house (needs the building platform in `shared/building/`) |
| `training/` | all code as run: `tp/` (simulator wrappers, teacher, MPC, features, policy), `scripts/`, `configs/`, `tp.js`, `page_src.html`, `build_page.py`, `_paths.py`; logs `ratchet.json`, `results/exam_r7.json`, `results/wpuq/` |
| `lab/index.html` | the original page (week explorer, live lab, exam, WPuQ check, ratchet) |
| `web/export_instrument.py` | writes the site's live instrument from the lab page's data |

## Data and licences
**CC BY-SA 4.0 files (Building Data Genome 2, share-alike):** `examples/bdg2/demo_episodes.npz` (2017 BDG2 site weather
of Cockatoo, Hog and Gator in the forecasts and simulated runs; the third episode is a twin of the BDG2 building
Gator_public_Geraldine), `lab/index.html` (2017 weather of five BDG2 sites and the runs built on it) and the
instrument data written to `site/public/instruments/thermopilot/`. Each folder has a `LICENSE-DATA.md`. The code, the
weights and the logs in `training/` are MIT.

Building Data Genome 2 (Miller et al. 2020, CC BY-SA 4.0: derived data share-alike); EIA RECS 2020 (public domain);
NREL ResStock 2025 R1 / ComStock 2025 R3 (OEDI, CC BY 4.0); HUE (Makonin 2019, CC BY 4.0); WPuQ (Schlemminger et al.
2022, CC BY 4.0); URDB tariffs (OpenEI); ERCOT prices via GridData; TMYx (climate.onebuilding.org); NASA POWER;
EIA natural gas prices (2025 averages: residential $15.34, commercial $10.98 per Mcf).
