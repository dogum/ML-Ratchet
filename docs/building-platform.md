# The building platform

The third shared platform. **BuildingSim** is a verified building-energy simulator: zone heat balance, HVAC plant,
faults and controls. **BuildingData** is real building data, split once and frozen: hourly campus meters, national
surveys, fault runs, homes and tariffs. Four models sit on it. All four were built on 16-17 Nov 2026 by parallel
agents in a claude.ai sandbox, then brought into this repository on 17 Nov 2026.

| Model | The question |
|---|---|
| [LoadCast](../models/loadcast) | How much electricity, chilled water, steam, hot water or gas will this building use each hour next year? |
| [BuildingDetective](../models/buildingdetective) | From its meters alone, what kind of building is this: size, use, envelope, heating and cooling plant? |
| [FaultFinder](../models/faultfinder) | From one day of an air handler's trend log, is something broken, and what? |
| [ThermoPilot](../models/thermopilot) | Hour by hour, which setpoints keep the occupants comfortable for the smallest bill? *(graded in simulation)* |

Code and data: [`shared/building/`](https://github.com/dogum/ML-Ratchet/tree/main/shared/building) (BuildingSim,
BuildingData and the checks). The models' `training/_paths.py` import the code from there and read data from
`$ML_RATCHET_BUILDING` (default the same folder). The platform release (`buildingsim-buildingdata-v1.0.zip`, 230 MB)
came in on 17 Nov 2026; its 409 data files (271 MB) are git-ignored and checked by sha256. See
[What is still waiting](#what-is-still-waiting).

## BuildingSim: the simulator

- **Zones:** an ISO 13790 5R1C heat balance, with a multinode wall option; Perez solar on each facade; LBL
  infiltration; ventilation with an economizer; ground-coupled floors; zone moisture; PMV/PPD comfort. Single zone
  only.
- **Plant:** 13 HVAC systems with EnergyPlus/DOE-2 performance curves: heat pumps with defrost, DX cooling,
  furnaces, boilers, chillers with a Braun cooling tower, VAV air handlers with reheat, PTAC/RTU units, district heat
  and chilled water, domestic hot water.
- **Faults and control:** 20 fault types that write a BMS-style trend log; a controller hook with a step API and an
  exact rollback; utility tariffs, PV and a battery.
- **A generator** draws buildings from ResStock, ComStock and CBECS priors.
- **Speed:** 0.7 s per building-year at 15 min steps, 0.2 s at 1 h, 2 s with multinode walls.

Checks (`shared/building/checks/*.json` as shipped; the baseline any change must not regress). Re-run in the
repository on 17 Nov 2026, every check that could run reproduced its shipped JSON: analytic, BESTEST, faults,
controller, generator, multinode (to round-off), the BDG2 dev twins (1,233 of 1,236 numbers equal, the other 3 within
5e-6), and the ResStock/ComStock summaries re-computed from the shipped per-unit twins. Not re-run: the
psychrometrics/comfort references (they need a CoolProp environment) and the twin simulations themselves (they need
the stock samples). Details in `checks/REPRODUCED.md`:

| Check | Result |
|---|---|
| Steady conduction | to 1.8e-9 |
| RC step response | within 0.0035 K |
| Degree-day ratio | 0.999996 |
| ISO 13786 wall | amplitude −1.6 %, lag −0.01 h |
| Energy ledgers | every one closes to ≤ 1e-8 |
| BESTEST (ASHRAE 140) | 29 of 34 values inside the published ranges, all 20 annual heating and cooling values inside. The ranges were transcribed from the literature, not the official tables |
| Psychrometrics, comfort | humidity ratio within 0.55 % of CoolProp; PMV within 0.0025 of pythermalcomfort |
| ResStock twins (126 homes, simulator vs simulator) | site energy median 1.10× (IQR 0.98-1.26), daily r 0.92 |
| ComStock twins (45 buildings) | site energy 0.86× (aggregate 0.97), daily r 0.89 |
| Real BDG2 dev twins from metadata alone | electricity 0.98× (IQR 0.50-1.42, daily r 0.34); chilled water 0.28×; steam 0.29×; hot water 0.15× |

EnergyPlus could not be used as a reference engine: no PyPI wheel ships the engine and GitHub was blocked in the
sandbox. The BESTEST ranges and the ResStock/ComStock twins stand in.

## BuildingData: the real data

- **Building Data Genome 2 (BDG2):** 1,636 non-residential buildings and 3,053 meters (electricity, chilled water,
  steam, hot water, gas) on 19 sites, hourly 2016-2017, with site weather. Quality-flagged and moved to UTC.
- **Surveys:** EIA RECS 2020 (18,496 homes) and CBECS 2018 (6,436 commercial buildings).
- **Modelled stock:** NREL ResStock 2025 R1 and ComStock 2025 R3 characteristics and 3,097 hourly profiles, labelled
  MODELLED. Used as generator priors and simulator-vs-simulator checks, never as truth.
- **Faults:** 237 real or laboratory AHU and RTU fault runs (ASHRAE RP-1312 at the Iowa Energy Resource Station,
  LBNL FLEXLAB, ORNL's Flexible Research Platform, Drexel's Nesbitt Hall) on one 27-type taxonomy.
- **Homes:** HUE (28 homes in British Columbia) and WPuQ (38 German heat-pump homes).
- **Tariffs:** 19 utility tariffs from the OpenEI Utility Rate Database; TMYx typical-year weather.
- **Frozen splits** (`data/proc/splits.json`, seed 20261116):

  | Data | Split |
  |---|---|
  | BDG2 | 52 dev / 1,584 exam buildings; 2016 is history, 2017 is the forecast year |
  | RECS | 9,247 prior / 1,848 dev / 7,401 exam |
  | CBECS | 3,217 prior / 661 dev / 2,558 exam |
  | Faults | 20 dev / 217 exam runs |

  The "prior" part of each survey may shape the generator; dev may calibrate and gate; exam is used once.
- **Classical baselines on the exam:** electricity hourly CV(RMSE) 19.1 % (change-point) and 19.6 % (TOWT); fault
  rules AUC 0.69 (AHU) and 0.72 (RTU).

BDG2 quirks the models had to handle: timestamps mixed local daylight and standard time (fixed to UTC); one site's
electricity ("Gator") is daily totals spread over the hours, so it is excluded from hourly labels; another site's
("Eagle") thermal meters were 1,000× too large; four sites were located by matching their temperatures; one site's
("Peacock") weather does not match its stated location. NASA POWER runs up to 2.5 °C cold at BDG2 sites, so the
models use station temperature for loads.

## Licences: BDG2 is share-alike

BDG2 is **CC BY-SA 4.0** (its repository's LICENSE file). Everything derived from it in this repository (the
example series in `models/*/examples/bdg2/`, the instrument excerpts, the lab pages that embed BDG2 meters or
weather) is CC BY-SA 4.0, credited to Miller et al. (2020), and carries a `LICENSE-DATA.md` next to it. The code
stays MIT. LoadCast's and BuildingDetective's weights are CC BY-SA 4.0 too: their training simulations were shaped
by BDG2 dev data (real residual series as noise, calibration and error-bar factors), so they are treated as
adaptations. FaultFinder's and ThermoPilot's weights used no BDG2 data in training and are MIT. The other sources and
their terms are in [DATA_LICENSES.md](../DATA_LICENSES.md).

## What the models found

Every model was trained only on BuildingSim and examined once on real data (ThermoPilot on simulated twins of real
units). The full win/loss tables are on each model's page.

- **LoadCast** (BDG2 2017, 1,502 meters): given a year of the building's own history it ties the classical models on
  electricity (18.8 % CV against 19.1-19.6 %) and chilled water, and loses on steam, hot water and gas. With no
  history it loses to a median guess and forecasts thermal meters about 2× low: the simulator's campus district
  meters are about 3× too small and far too seasonal.
- **BuildingDetective:** wins on heating fuel (RECS 87.3 % against a rule's 85.3 %), CBECS heating type (74.7 %
  against 68.6 %) and on telling from electricity alone whether a BDG2 building has chilled water (AUC 0.75). It
  fails on primary use (11 % against 40 % for the majority class), year built (no skill) and heat pump versus
  resistance heat (AUC 0.40 on RECS, worse than chance): the simulator's heat-pump signature is not what real homes
  show.
- **FaultFinder** (217 real days): AHU fault detection AUC 0.78 against 0.69 for expert rules, and 0.78 against 0.54
  on Nesbitt Hall, a real occupied building. It raises far more false alarms (33 % of healthy days at P = 0.5),
  diagnoses no better than guessing the most common fault, and loses to a one-line rule on rooftop units (0.65
  against 0.72).
- **ThermoPilot** always beats a fixed thermostat, but the classical MPC with a forecast beats it everywhere, by 0.1
  to 0.5 skill. Against setback it is about equal on bills with better comfort. On real WPuQ heat-pump homes the
  simulator cannot settle ThermoPilot against setback: the order flips in 18 of 37 homes.

## What it gets wrong

The measured gaps are BuildingSim's next rungs. Each has a "done when" in the handoff.

| Gap | Measured |
|---|---|
| Campus district energy | default twins read chilled water, steam and hot water about 3× low; real meters are flatter (summer district heat 0.72 of the annual mean vs 0.06 simulated; chilled water night/day 0.69 vs 0.31) |
| Electricity weather sensitivity | real PRISM r² 0.07, twins 0.24: real campus electricity is mostly base load |
| Heat-pump signature | twins of WPuQ heat-pump homes overstate the heating slope 2.7× and run at night; the real ones run flat (hourly profile r 0.21) |
| Generator intensity by use type | schools 1.8×, healthcare 1.9×, labs 2.0×, food 1.5×, single-family 1.4×, lodging 0.66× |
| Single zone | no perimeter/core, no simultaneous heating and cooling (ComStock heating 0.75×) |
| Air-side detail for faults | no return fan, exhaust damper, ducts, chiller-off or chilled-water sensor faults: 13 exam fault types cannot be simulated |
| Fault logs | the simulator logs actual actuator positions; real BMS logs commands. Real dampers at 100 % command mix 81-90 % outdoor air |
| Reference engine | official ASHRAE 140 tables and EnergyPlus not yet run |

Each model wrapped BuildingSim to calibrate it on its own dev data (holiday calendars, campus operating defaults,
district scale and base loads, intensity factors, command logging, an installed damper curve, a DST-safe tariff
builder). Those wrappers live in each model's `training/` and are listed for upstreaming in the
[roadmap](roadmap.md#the-building-platform).

## Round 2 (17 Nov 2026): fair baselines, new physics, fresh exams

The second pass followed WakeNet's 17 Nov pattern: every plan written in the model's NOTEBOOK before it ran, the
strongest fair baseline (physics or a simple model tuned on the same dev data the simulator was tuned on), a dev
twin test of what each network adds, and every row with a 95 % bootstrap interval (win or loss only if the
interval sits on one side of zero). The v1.0 exams were already seen, so their re-scores are labelled seen. Three
fresh real exams were frozen and committed before any round-2 decision and scored once afterwards.

| Model | v1.0 as imported | Seen exam, fair baselines + intervals | Round-2 release | Fresh exam (scored once) |
|---|---|---|---|---|
| LoadCast | 4 wins, 6 losses | 6 wins, 16 ties, 18 losses | none (r6 physics 1.1, r7, r8 rejected) | I-BLEND, 3 Delhi buildings: 3 wins, 5 ties (with history not worse than TOWT; no history about 2.2x too high) |
| BuildingDetective | 5 wins, 4 ties, 9 losses | 0 wins, 7 ties, 21 losses | none (r10 kept by the rule, not released: real dev area 50.8 -> 69.0 %) | HEAPO, 105 Zurich homes: 4 losses, 2 ties (the dev median area beats it) |
| FaultFinder | 11 wins, 1 tie, 12 losses | 4 wins, 30 ties, 6 losses | v1.1: the unit's own earlier days (r8) | Nehasil, 4 AHUs: 2 wins, 1 tie, 1 loss; v1.1 no better than v1.0 |
| ThermoPilot | 0 wins, 6 losses | v1.1: 2 wins, 3 ties, 4 losses | v1.1: a card MPC behind a 6 h shield (classical; the network kept as a mode) | none (no open thermostat logs) |

What changed the picture:

- **Fair baselines take most wins away.** Rules tuned on FaultFinder's 20 dev days tie it on AUC; linear models fitted
  on dev beat BuildingDetective on most survey rows (RECS heating 91.9 % vs 87.3 %); the dev-calibrated BuildingSim
  twin beats LoadCast's no-history forecasts on electricity and chilled water.
- **The networks' corrections point the right way but are too large** (BuildingDetective, ThermoPilot), or point the
  wrong way (LoadCast's no-history level moves away from the twin on dev).
- **Physics 1.1-1.4** (opt-in; the default stays 1.0): district plants fix chilled water (within +-30 % even
  leave-one-site-out) but steam and hot water do not transfer between campuses; flatter electricity, the heat-pump
  signature and generator intensity all fall short of their targets on dev. No model adopted a version: LoadCast's
  physics-1.1 rung failed its dev gate, and BuildingDetective's flipped its area bias.
- **Reproducibility fixes:** RECS twins now rank tied ResStock matches by `bldg_id` (numpy's unstable sort made
  v1.0's twins machine-dependent); LoadCast banks are now deterministic; the four models' wrappers are in
  BuildingSim 1.0.1 behind flags.
- **Decisions the owner may revisit:** FaultFinder v1.1 was released by its pre-fixed rule although the seen exam
  showed AHU false alarms 33 -> 56 %; the fresh Nehasil exam then found it no better than v1.0. LoadCast's dev gate
  compares 3-seed means with v1.0's single selected seed, so a rung needs about 3 points of dev gain to tie its own
  recipe; changing that is a rule change left to the owner.

## Using it

```bash
pip install -e ".[building]"
mlratchet data building --from-zip buildingsim-buildingdata-v1.0.zip     # unpack the release, check all 409 files
mlratchet data building --list                                            # the rebuild stages and their hosts
cd shared/building/buildingsim && python demo.py
pytest -q tests/test_building.py
```

```python
import buildingsim as bs, buildingdata as bd
bd.recs(split='dev'); bd.faults(split='exam'); bd.tariff_prices('pge_e_tou_c', 2017)
```

`data/proc/splits.json` (file sha256 `524aee15…`, internal hash `7ac15e9f…`, seed 20261116) is tracked and never
changes; the rest of `data/` and `buildingsim/refdata/` are git-ignored. TMYx weather is not redistributed: its site
says "All Rights Reserved". Every physics change bumps the BuildingSim version, rebuilds each building model's frozen
simulated set and re-takes their ratchet decisions in order.

## Rule notes

No exam result was used to choose anything. Three things the build agents were unsure about are recorded on each
model's ladder:

- **LoadCast** changed its frozen-set objective twice before rung R3 and re-scored every rung each time. Its noise
  bank includes dev buildings' 2017 residuals, which makes its dev-gate numbers slightly optimistic.
- **BuildingDetective** withdrew a kept rung by hand (a 3×512 network, 1.15 M parameters) for breaking the ~500k
  parameter cap. Its real dev gate drifted from 0.331 to 0.306 across kept rungs, each step within the 0.02 tolerance.
- **ThermoPilot** used 4 of the 6 RECS respondents per tariff slot to save compute, decided before the exam.
- FaultFinder's detection rates at fixed false-alarm rates set their thresholds on the exam's own healthy days, so
  they are reported, never used.

## What is still waiting

- **A public home for the data release.** The 230 MB zip is on the owner's Google Drive; `mlratchet data building
  --from-zip` needs a local copy. A GitHub release or another public host would let anyone rebuild in minutes.
- **Splits reproduce:** `make_splits.py` re-run here gives the same internal hash (`7ac15e9f…`); the file differs only
  by Windows line endings. The BuildingData baselines re-run within 5e-5.
- **Not re-run here:** `check_components.py` (needs the CoolProp/pythermalcomfort environment), the ResStock and
  ComStock twin simulations (their summaries were re-computed from the shipped per-unit results), and every model's
  training (banks and checkpoints are not shipped; ThermoPilot's DP also needs `numba`). The models' exam numbers are
  imported from their logs; their demos reproduce the released outputs.
- **Rebuildable, not shipped:** ResStock/ComStock samples and weather (1.2 GB), the raw downloads (1.4 GB),
  BuildingSim's stock weather cache (55 MB), the reference environment (CoolProp, pythermalcomfort) and the training
  banks.
- **Gated data:** ecobee Donate Your Data (real thermostat logs; an application), the full LBNL fault datasets (a
  name and email form), UK heat-pump field trials (UK Data Service registration), Pecan Street. EnergyPlus was
  blocked in the sandbox.
