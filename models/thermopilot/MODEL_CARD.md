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

**Round 2 (17 Nov 2026, not released).** Val and dev were frozen again with reproducible RECS twins (tied ResStock
matches ranked by bldg_id); there R7 scores val 0.561, dev 0.471. Rung R8, a residual network on a numpy MPC that
plans on a 12-number linear house map from the building card, beat R7 (3-seed mean val 0.667, dev 0.701) but lost
to that card MPC alone behind the 6 h shield (val 0.700, dev 0.720), which the rule required it to beat by 0.01. It
was rejected; v1.0 stays. See `ratchet.yaml` and NOTEBOOK.md.

## Exam (never used for any choice; taken once with R7, re-scored as seen on 17 Nov 2026)
Twins of real exam units, perturbed true house, noisy forecasts, real weather, real tariffs. Bill change vs a fixed
thermostat (positive = cheaper); K h = kelvin-hours outside the band per episode; skill = share of the
perfect-foresight saving on cost + comfort. Brackets are 95 % bootstrap intervals over units (homes / buildings;
2,000 resamples, seed 20261117). The re-score (`training/results/integrity_2026-11-17/`) reproduced WPuQ exactly and
BDG2 exactly except the MPC (3.05 → 2.69 %); the RECS twins built in this repository differ from the v1.0 build's
(e.g. RECS_116893: UA 287 → 434 W/K, light → medium mass), so the RECS numbers moved. HUE was not re-run (the NASA
POWER irradiance for Vancouver is not in the repository); its numbers are v1.0's with 90 % intervals.

**RECS 2020 exam respondents** (48 homes × January + July 2017, 12 real tariffs in 9 states, comfort band from each
respondent's own thermostat settings; first 4 of 6 drawn per tariff, decided for compute before any result):

| controller | bill saving | K h/ep | PPD % | skill | peak kW | on/off per day |
|---|---|---|---|---|---|---|
| fixed thermostat | 0 | 0.36 | 23.0 | 0 | 3.88 | 1.0 |
| setback (occupant's own schedule) | 3.01 % [1.40, 4.95] | 1.77 | 23.4 | −0.18 | 4.00 | 1.3 |
| standard setback (−4 / +4 K) | 9.38 % [7.71, 10.91] | 835 | 23.7 | ≪ 0 | 4.63 | 2.1 |
| in-band pre-cool rule (3 h, 2 K) | 3.10 % [1.63, 4.72] | 1.00 | 23.6 | 0.28 | 4.05 | 1.4 |
| phase-1 pre-cool rule | 3.38 % [2.73, 4.06] | 69 | 22.6 | ≪ 0 | 4.01 | 1.3 |
| pre-cool rule tuned on dev (4 h, 3 K) | 3.07 % [1.63, 4.61] | 0.88 | — | 0.35 | — | — |
| MPC with forecast (same nominal house) | 4.10 % [2.71, 5.56] | 0.49 | 24.0 | 0.84 | 4.14 | 1.6 |
| **ThermoPilot** | **2.56 % [1.67, 3.44]** | **1.28** | **23.5** | **0.24** | **4.15** | **1.6** |
| perfect foresight | 4.25 % [2.87, 5.74] | 0.29 | 24.0 | 1 | 4.10 | 1.6 |

**ThermoPilot minus the best baseline** (bill rows count only baselines with K h ≤ fixed + 1):

| set (units) | bill saving, points | skill | K h / episode | vs setback, bill |
|---|---|---|---|---|
| RECS (48 homes) | −1.55 [−2.28, −0.93] vs MPC: loss | −0.60 [−0.85, −0.39] vs MPC: loss | +0.91 [+0.20, +2.00] vs fixed: loss | −0.46 [−1.73, +0.56]: tie |
| BDG2 (16 buildings) | +0.08 [−0.85, +1.09] vs MPC: tie | −0.00 [−0.11, +0.10] vs MPC: tie | +0.05 [0.00, +0.09] vs fixed: tie | +1.58 [+0.30, +2.81]: better |
| WPuQ (38 homes) | −4.03 [−4.66, −3.47] vs MPC: loss | −0.60 [−0.68, −0.53] vs MPC: loss | 0.00 [0.00, 0.00]: tie | −2.45 [−3.16, −1.54]: worse |
| HUE (17 homes, v1.0) | 4.29 % vs MPC 5.75 %: loss (no interval) | 0.54 vs 0.95: loss | 0.36 vs fixed 0.14: loss | −1.70 [−1.89, −1.53] (90 %) |

**BDG2 exam buildings** (16 small offices / schools / shops, Jan + Jul 2017, PG&E B-19, OUC GSD, Xcel A15, Pepco GT):
setback 1.18 %; in-band pre-cool −0.28 %; tuned pre-cool −0.03 %; MPC 2.69 % [1.07, 4.57] (0.21 K h, skill 0.45);
**ThermoPilot 2.77 % [1.19, 4.53]** (0.05 K h, skill 0.45); perfect foresight 7.23 %.
**HUE homes** (17 BC homes, Jan [+ Jul with AC], Vancouver 2017, PG&E E-TOU-C stand-in; v1.0): setback 5.9 %
(3.1 K h); MPC 5.7 % (0.2 K h); **ThermoPilot 4.3 % (0.4 K h, skill 0.54)**; perfect foresight 5.8 %.
**WPuQ heat-pump homes** (38 German homes, Jan 2019 weather, Xcel MN TOD stand-in, generic twins): setback 5.58 %;
MPC 7.16 % [6.62, 7.60]; **ThermoPilot 3.13 % [2.18, 3.96] (skill 0.35)**; perfect foresight 7.50 %.

**Fairness of the baselines (checked 17 Nov 2026).** The MPC uses the nominal twin (the same house the building card
comes from), the same noisy forecast and an observer; it never sees the true house or state. It does read the true
future wind for infiltration, which ThermoPilot does not see: with the wind at its monthly mean its dev skill drops
0.815 → 0.787. The setback is the occupant's own schedule. The in-band pre-cool rule was never tuned; the version tuned
on dev (4 h lead, 3 K deep, the edge of a 1-4 h × 1-3 K grid) is listed as its own baseline.

Reading: ThermoPilot always beats the fixed thermostat. On small commercial buildings with demand charges it ties
the classical MPC and beats the operating-schedule setback. On homes the MPC, which uses the same building model and
the same noisy forecast, is better on every set, and ThermoPilot saves no more than the occupant's setback (RECS, a
tie) or less (WPuQ, HUE). It beats the best rule in 27 % of RECS home-months (MPC 52 %). Standard setback saves the
most money only by leaving the band for hundreds of K h.

**Dev twin test (what the network adds over the cheap MPC; 30 dev episodes).** The perfect-foresight teacher departs
from the MPC in only 4.1 % of hours, with no mean offset; ThermoPilot shifts setpoints by +0.88 K (heating) and
−0.52 K (cooling) on average. The two corrections correlate 0.10 / 0.14 over all hours, 0.37 / 0.53 where the teacher
departs. This is the case for the next rung, a residual on the cheap MPC (planned in NOTEBOOK.md).
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
6. The student underperforms the classical MPC by 0.4–0.6 skill on homes (ties it on BDG2 commercial); more DAgger
   rounds were still helping when stopped.
7. Heat pumps are under-represented in training (most generated / RECS homes heat with gas furnaces).
8. The 17 Nov 2026 re-score did not reproduce v1.0 on RECS-derived twins (dev skill 0.360 → 0.482, val 0.562 → 0.556,
   RECS exam rows moved); WPuQ reproduced exactly, BDG2 except the MPC. Cause: the twin builder's np.argsort over
   tied match scores is not stable, so the chosen ResStock dwelling depends on the numpy build and CPU.

## Files (in this repository)
| file | what |
|---|---|
| `thermopilot_core.py` | numpy inference: features, MLP, action menu, comfort shield |
| `thermopilot_v1.npz` | weights (298-384-384-36 MLP, input normalisation, shield = 4) |
| `demo.py`, `examples/bdg2/demo_episodes.npz` | replay of 3 exam episodes (numpy only, about 1 s) |
| `pipeline.py` | run ThermoPilot on a BuildingSim house (needs the building platform in `shared/building/`) |
| `training/` | all code as run: `tp/` (simulator wrappers, teacher, MPC, features, policy), `scripts/`, `configs/`, `tp.js`, `page_src.html`, `build_page.py`, `_paths.py`; logs `ratchet.json`, `results/exam_r7.json`, `results/wpuq/`; the 17 Nov 2026 integrity pass: `scripts/{intervals,precool_tuned,twin_test,mpc_nowind}.py` and `results/integrity_2026-11-17/` |
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
