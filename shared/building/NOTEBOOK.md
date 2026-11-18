# BuildingSim / BuildingData notebook

Dated entries, newest last. `HANDOFF.md` is the v1.0 record and is not edited; this file holds the work after it.

## 2026-11-17 — P0.4 plan: upstream the Phase 2 wrappers behind flags (written before any of it runs)

HANDOFF section 4, P0.4: the four models wrapped BuildingSim in their own training code instead of changing it.
These wrappers move into `shared/building/buildingsim` (and `buildingdata`) as ports of the models' own code, each
off by default. Sources are the v1.0 training files as committed (dfafdcf): `models/loadcast/training/{lcsim,
lc_common,calib,bank}.py`, `models/buildingdetective/training/{wrappers,common,simone,calib_campus}.py`,
`models/faultfinder/training/{simbank,noise,common}.py`, `models/thermopilot/training/tp/{twins,env,mpc,dp_fast,
episode}.py`. Nothing in `models/` is edited.

**One mechanism: opt-in by name.** A v1.0 call never reaches a wrapper. A wrapper is one of:
- a key of the new `simulate(..., options=None)` / `Simulator(..., options=None)` dict, for what must happen inside
  the time loop (holiday calendar); `options=None` is v1.0;
- a new hvac spec field that is absent unless set (`district_eff_chw`); `fill_spec` does not add it;
- a function or class in the new module `buildingsim.phase2` (spec transforms, meter post-processing, trend-log
  realism, a controller), or a new function in `buildingsim.stock` / `buildingdata`, which the caller applies.
The package version becomes 1.0.1; `PHYSICS_VERSION` stays '1.0' (flags off is physics 1.0).

**What moves where.**

| Model | Wrapper | Upstream as |
|---|---|---|
| LoadCast | US/CA/UK/IE holiday calendars (`lc_common.holiday_dates`, `lcsim.use_country`) | `options={'holidays': 'auto' or a country}`, calendars in `buildingsim/holidays.py` |
| LoadCast | BDG2-style spec maker (`lcsim.make_spec`) | `phase2.spec_from_meters(base, meters, knobs)` |
| LoadCast | campus defaults from dev (`calib_summary.json`: 24x7 22/52, OA x2.5 37/52, VAV min 0.3 43/52) | `phase2.CAMPUS_KNOBS` |
| LoadCast | per-meter scale and base load (`lcsim.post`) | `phase2.scale_meters`; dev medians in `phase2.DEV_METER_SCALE` |
| LoadCast | bounded memory (`LC_MAXPARTS` restarts; the weather cache holds 25 prepared frames) | `bs.set_weather_cache(n)`, `bs.clear_caches()` |
| Detective | W1 intensity per use (`apply_intensity`) | `phase2.apply_intensity`, factors from the shipped `calibration.json` in `phase2.W1_INTENSITY` |
| Detective | W2 campus district (`make_campus`, `campus_meters`; heat 0.18, CHW 0.33) | `phase2.make_campus` + hvac field `district_eff_chw` (in the plant, so ledgers still close) |
| Detective | W3 flat base loads (`w3_base_loads`) | `phase2.flat_base_loads` |
| Detective | label cleaning (`clean_labels`) | `phase2.clean_labels`, `phase2.base_share` |
| FaultFinder | log commands, not positions (`simbank.log_commands`) | `phase2.log_commands` on the trend frame |
| FaultFinder | installed damper characteristic (`Noise.damper_command`) | `phase2.damper_command`, `phase2.installed_damper` |
| FaultFinder | larger static sensor offsets (`Noise.apply`, `big_offsets`) | `phase2.static_offsets` |
| FaultFinder | operator start/stop jitter (`simbank.Operator`) | `phase2.OperatorJitter` (a controller) |
| ThermoPilot | RECS respondent twins and comfort bands (`twins.recs_twin`, `recs_pref`) | `stock.recs_to_spec`, `stock.recs_comfort_band` |
| ThermoPilot | DST-safe tariff prices (`env.price_arrays`, `bill_arrays`) | `bd.tariff_arrays`, `bd.bill_arrays`; `bd.tariff_prices` checked on every DST hour |
| ThermoPilot | ERCOT RTP pairing (`env.rtp_prices`) | `bd.rtp_prices` |
| ThermoPilot | numba DP kernel, two-pass demand-peak oracle | judged below, before porting |

The DP kernel and the oracle are ThermoPilot's controller (its control model, action menu and comfort weight), not
building physics, and numba is not installed here. Plan: leave them in the model and say so, unless reading them shows
a part that is simulator-generic.

**Tests, in this order.**
1. **Default identity.** With every flag off, re-run analytic, bestest, faults, controller, generator, multinode,
   bdg2 and speed, then `compare.py` against the shipped JSONs. Pass = the same counts as `checks/REPRODUCED.md` (the
   same round-off residuals and nothing else). Plus a direct test: `simulate(..., options=None)` and
   `options={}` give the same arrays as the v1.0 call.
2. **Per-wrapper equivalence** (`checks/check_phase2.py` → `checks/out/phase2_equivalence.json`): the port with the
   flag on reproduces the model's own wrapper on a few cases, to round-off (max abs difference reported).
3. **Flags-on run of the BDG2 dev twins** (52 buildings, 2016, as `check_bdg2.py`), on dev only:
   - `w2`: Detective W1 + W2 at the fitted medians (heat 0.185, CHW 0.329, no per-building draw), with the campus
     physics `calib_campus.py` used (VAV min 0.45, OA x1.5 for district buildings);
   - `w2w3`: the same plus W3 flat base loads at their means (phi heat 0.70, CHW 0.50, electricity +0.50 x mean);
   - `lc`: LoadCast `spec_from_meters` with `CAMPUS_KNOBS` and `holidays='auto'`, no meter scaling.
   Recorded: annual sim/real median per meter, daily r, and for W3 the summer-level and night/day ratios. W2's factors
   were fitted on these buildings' 2017 data, so the 2016 annual ratios are an in-sample check, not a test.
   This is the dev baseline the physics rungs start from. No exam building is read.
4. ≤ 3 worker processes; anything over 2 minutes in the background.

## 2026-11-17 — P0.4 results: every wrapper ported, flags off identical, flags on recorded

Everything below followed the plan in the entry above; no code or setting was changed after a result was seen
(two later edits touched only text: a measured cache size in a docstring, and a `merge` mode in `check_phase2.py`).
Full tables: `checks/REPRODUCED.md`, last section.

- **Flags off = v1.0.** analytic, multinode, bestest, faults, controller, generator, bdg2 and speed re-run with no
  wrapper used. `compare.py` gives exactly the counts of the first re-run: 53/53, 82/84 (the same two 1e-9 round-off
  residuals), 222/222, 544/544, 34/34, 1715/1715, 1233/1236 (the same three near-zero values). `PHYSICS_VERSION`
  stays '1.0'; the package is 1.0.1.
- **Every port reproduces the model's wrapper** (`checks/phase2_equivalence.json`, models' v1.0 files unchanged from
  dfafdcf): 0.0 difference for the calendars, the holiday simulation (LoadCast monkeypatch vs `options`), 1,044
  spec-maker cases, meter scaling, W1, W3, label cleaning, damper characteristic, operator jitter, 24 RECS twins,
  20 tariffs, 608 DST frames and ERCOT RTP. Round-off only for W2 (1.2e-10 kW: the factor now sits in the plant, not
  in a meter rescale), `log_commands` (3e-8, float32) and static offsets (3.8e-6 K, float32).
- **Decisions.**
  - W2's chilled-water factor is an hvac field (`district_eff_chw`) used by the district plant, not a post-hoc
    rescale: same meters, and the
    plant ledger now includes the factor (the model rescaled the meter after the ledger was built).
  - `bd.tariff_prices`: no bug fix was needed for the tz frame. With `tz` it is right on every hour of 608 frames
    (19 tariffs, 7 time zones, 2016-2019, pandas 3.1.1); the November failure could not be reproduced. The real
    hazard is the default naive frame: it cannot be tz-localized (the spring hour does not exist) and
    `simulate(prices=)` reads it as UTC (prices shifted by the UTC offset, 4-8 h at US sites). Added behind a flag: `utc=True` returns
    the same prices on the UTC index; the default output is unchanged on every hour.
  - Not ported: ThermoPilot's numba DP kernel and two-pass demand-peak oracle. They are the model's controller (its
    control model, 36-action menu, comfort weight), not building physics, and numba is not installed here. The
    billing pieces they rely on (`tariff_arrays`, `bill_arrays`) moved to BuildingData.
  - Bounded memory: `bs.set_weather_cache(n)` / `bs.clear_caches()`. A prepared year is 3.0 MB at 1 h and 11.8 MB
    at 15 min, so the v1.0 cache (25 frames) holds at most ~300 MB; what else grew in LoadCast's killed process was
    not traced.
- **Flags on, BDG2 dev twins 2016** (annual sim/real median; v1.0 -> `w2` -> `w2w3` -> `lc`):
  electricity 0.98 -> 0.68 -> 1.02 -> 1.23; chilled water 0.28 -> 0.89 -> 0.90 -> 0.50; steam 0.29 -> 0.88 ->
  0.87 -> 0.88; hot water 0.15 -> 0.89 -> 0.87 -> 0.53. Daily r: chilled water 0.88 throughout; electricity 0.34
  except `lc` 0.17.
- **What failed or needs care.**
  - W2's 0.9x district ratios are in-sample: its factors were fitted on these buildings (2017).
  - W3 is affine per meter, so it moves no correlation. It fixes the steam summer level (0.09 -> 0.74; real 0.66)
    but is wrong for the 5 hot-water meters, which are off in summer (real 0.01; W3 0.70). A physics rung should not
    apply it to hot-water meters as is.
  - LoadCast's campus knobs without its meter scaling halve electricity's daily r and make steam and hot water peak
    at night (night/day 1.8-2.1 against 0.9 real).
  - The holiday flag changes nothing on dev: the only non-US dev buildings are two Robin lodging buildings, and the
    lodging schedule ignores the day type.
- These four JSONs (`bdg2_dev_meters__{w2,w2w3,lc}.json`, `phase2_equivalence.json`) are published in `checks/` as
  the dev baseline the physics rungs start from. No exam unit was read.

## 2026-11-17 — P1 physics rungs, round 2: plan for physics 1.1 and 1.2 (written before any fit runs)

Two rungs from HANDOFF section 4, P1 table: campus and district thermal plants (physics 1.1) and weather-insensitive
electricity (physics 1.2 = 1.1 + electricity). Both are opt-in. The default stays '1.0', so the four released models'
training reproduces. A model adopts a version only later, after the fresh-exam split files are committed; this round
builds, fits on dev and measures, and does not adopt or declare anything done.

**Selection.** `simulate(..., options={'physics': '1.1'})` or the environment variable `BUILDINGSIM_PHYSICS`
(the option wins). `options={'physics': '1.0'}`, `None`, `{}` and an unset variable are v1.0, bit for bit
(`test_building.py`). `out['physics']` records the version used. `PHYSICS_VERSION` stays '1.0' (the default);
`PHYSICS_VERSIONS = ('1.0', '1.1', '1.2')`. Package version 1.2.0.

**Data rule.** Calibration uses BDG2's 52 dev buildings only (2016 and 2017 are both dev). Never an exam building,
never HUE or WPuQ. `splits.json` is not touched.

### What was looked at before writing this (dev only)
Real dev meters, 2016 (median over building-meters; W/m2 of floor): chilled water 54 W/m2, summer level 1.53,
winter 0.41, night/day 0.80, n 20 at 4 sites; steam 5.6 W/m2, summer 0.66, night/day 0.95, n 10 at 3 sites; hot
water 16.6 W/m2, summer 0.01 (2017: 0.44), night/day 0.87, n 5 at 2 sites. Heterogeneous: 3 of 5 hot-water loops
are off in summer, 2 are not; chilled water is off in winter at the cold sites (Eagle, Hog: winter 0.00-0.14 of the
mean) except healthcare and one public building; steam runs all summer at Bull and Eagle but not at Hog_Jay.
Electricity (my daily PRISM 5P r2, same grid as the Detective's `features.prism`, on 2016 daily means): real median
0.24 [IQR 0.05-0.45], v1.0 twins 0.25 [0.04-0.61]; the twins' weekday/weekend swing is 0.26 of the mean against
0.10 real, and their day-to-day residual (after PRISM and a weekday term) is 0.09 against 0.13 real. The Detective's
0.07 vs 0.24 is a different pipeline (2017, quirked meters, W1-W3 twins); I report mine and say so.

### Rung 1, physics 1.1: a district plant model (new module `buildingsim/district.py`)
Applies only to buildings with a district plant (district heat, district chilled water, district DHW).
1. **Loop operation (control setpoints).** Hot-water heating loops run while the 7-day running mean outdoor
   temperature is below `T_hw_off`; chilled-water loops while it is above `T_chw_off`, except 24x7 uses (healthcare,
   lab), whose process cooling keeps the loop on all year; steam loops run all year (DHW, reheat, humidification,
   sterilisers). Candidate grid, chosen on dev: `T_hw_off` none / 12.8 / 15.6 / 18.3 C, `T_chw_off` none / 7.2 / 10 /
   12.8 C.
2. **Loop base load while the loop runs** (distribution and standby losses, process steam, process cooling): one
   constant per loop type, `q_steam`, `q_hw`, `q_chw` in W/m2 of floor. It is booked as its own end use (in the plant
   ledger and on the meter), not as a meter rescale.
3. **Summer reheat (in the time loop)**: VAV systems with district heat reheat a share `f_rh` of the minimum airflow
   from supply-air temperature to the heating setpoint while the air handler runs in cooling mode and the heat loop
   is on (the simultaneous heating and cooling of lightly loaded zones that a single zone cannot produce). The heat
   enters the zone through the box, is metered on the heat plant, and the zone removes it with more chilled water.
   Grid {0, 0.25, 0.5}. Only VAV with district heat, so boiler VAV (ComStock twins, generator) is unchanged.
4. In-building distribution efficiency stays 1.0 (no W2 factor).
5. Not modelled: one plant metered across several buildings (allocation); a loop lockout does not remove coil heat
   that is still asked for (the lockout gates the base and the reheat only).

**Fit (dev 2016, `buildingsim/tools/calibrate_district.py`).** For each grid point, `q_m` is set so the median
annual sim/real ratio of meter m is 1.0; the grid point is the one with the lowest sum over the three meters of the
median daily CV(RMSE) of daily energy. Free parameters: 3 continuous (`q_*`), 3 grid choices.

**Tests.**
- Gap first: `check_bdg2.py --physics 1.0` (new stats: summer level, night/day, winter level, daily CV(RMSE), PRISM r2
  for electricity) -> `bdg2_dev_meters__p1.0.json`.
- After: `check_bdg2.py --physics 1.1` -> `bdg2_dev_meters__p1.1.json`, 2016 in sample.
- Cross-checks that do not reuse the fit: (a) the 2016 fit scored on 2017 (`--year 2017`; same buildings, so weak);
  (b) leave-one-site-out over the 4 sites with thermal meters (Fox, Bull, Hog, Eagle): refit on 3, score the 4th.
- Done when (HANDOFF): median annual ratio within 0.7-1.3 and median daily r >= 0.7 for chilled water, steam and hot
  water. Judged on the leave-one-site-out numbers, since the in-sample annual median is 1.0 by construction.
- Comparison with the P0.4 wrappers (`w2`, `w2w3`: W2 factors fitted on these buildings' 2017 data, so in sample).

### Rung 2, physics 1.2: weather-insensitive electricity (1.1 + this)
1. **Always-on share of lighting and plug loads**: a share `s_on` of each building's scheduled interior lighting and
   plug energy runs flat around the clock (servers, fridges, corridors, labs left on); energy-preserving, so the annual
   total does not move. It also enters the zone as internal gain.
2. **Day-to-day activity**: a daily multiplier on occupancy, lighting and plug, lognormal AR(1) with log-sd `sigma_d`
   and lag-1 correlation `rho_d` (events, term calendar, weather-independent use). Seeded (`options['seed']`, default
   0), mean-preserving.
3. Fit on dev 2016 electricity: `s_on` so the median weekday night/day ratio matches real; `sigma_d` so the median
   day-to-day residual matches real; `rho_d` = median lag-1 autocorrelation of the real residuals. None of these is
   the PRISM r2.
4. **Done when**: real and twin daily PRISM r2 distributions match on dev: two-sample KS p > 0.05 and each of the
   p25 / p50 / p75 within 0.05; reported for 2016 (fit year), 2017 and leave-one-site-out (refit on the other sites).
   The v1.0 KS result is reported too, because with n 47 the KS test has little power.

**Unchanged where the change cannot reach.** Both versions are opt-in, so at the default every check must give the
v1.0 numbers again (analytic, multinode, bestest, faults, controller, generator, bdg2, speed, phase2 equivalence
through `compare.py`). At 1.1/1.2: analytic, BESTEST and multinode have no district plant, no VAV with district heat
and no lighting/plug schedule in the tested quantities (BESTEST has constant gains) - re-run at 1.2 to show it;
ResStock/ComStock twins are re-summaries of shipped per-unit results here and cannot be re-simulated (no stock data).
≤ 3 worker processes; anything over 2 minutes in the background.

## 2026-11-17 — P1 physics rungs 3 and 4: plan for physics 1.3 and 1.4 (written before any fit runs)

Same rules as rungs 1-2: opt-in (`options={'physics': '1.3'}` or `'1.4'`), the default stays '1.0', each version
contains the earlier ones (1.3 = 1.2 + heat pumps; 1.4 = 1.3 + generator intensity), dev data only, plan first, gap
shown failing on dev first, a cross-check that does not reuse the fit. Added after the coordinator's message that the
fresh-exam split files are committed (3d9f949). `stock.recs_to_spec(..., stable=True)` (d7d9c91) is used as is.

### Rung 3, physics 1.3: the heat-pump signature (residential heat pumps only)
Legal dev data: RECS 2020 dev respondents (annual) and the 25 HEAPO dev homes of
`models/buildingdetective/training/split_exam2_heapo.json` (Zurich, hourly whole-home electricity, site-visit
protocols; read-only from `models/buildingdetective/work/exam2/heapo/`). Never HEAPO exam homes, HUE or WPuQ.

Looked at before writing this (dev only):
- RECS dev, electric heat: 255 heat pumps, 302 resistance (electric furnace, baseboard, portable electric). Real
  annual electricity per m2 does not separate them (AUC 0.49): heat-pump homes are bigger (145 vs 111 m2), in warmer
  places (HDD18 1,580 vs 2,074) with more cooling. A logistic reader fitted on the REAL dev respondents themselves
  (5-fold CV; log kWh/m2, log HDD, log CDD, log area) reaches only AUC 0.63 (0.66 adding EIA's modelled heating kWh).
  So the HANDOFF target, AUC > 0.7 from annual meters, is above what these real data carry even for a reader fitted
  on them; I expect to report it as not reachable and measure how far physics moves the sim-trained reader.
- HEAPO dev protocols (first visit): 20 of 25 floor heating, night setback on in 11 of 24, heating limit recorded
  for 4 (17, 18.5, 19, 21 C).

Physics (all only for residential `heat_pump_air` / `heat_pump_ground`):
1. **Emitter store**: heat-pump heating electricity passes a first-order store (floor slab or buffer tank), time
   constant `tau_e` h; energy-preserving over the run (booked in the heating end use, ledgers close). Grid 0 / 3 / 6
   / 12 / 24 h.
2. **Heating limit**: no space heating while the 24-h running-mean outdoor temperature is above `T_limit` = 18.75 C
   (median of the 4 HEAPO dev protocol values; not fitted).
3. **Field COP factor** `f_cop` on air-source heat-pump compressor power (installed / rated: airflow, charge, ducts),
   grid 0.6 / 0.7 / 0.8 / 0.9 / 1.0.
Setbacks are not changed (11 of 24 HEAPO dev homes use one).

Fit and test:
- `tau_e` on HEAPO dev fold A (13 homes, sorted by id, odd positions), objective: median winter (Dec-Feb)
  hour-of-day profile r, real vs twin; scored on fold B (12 homes). Twins: residential_single, protocol heated area
  and construction year, air- or ground-source as recorded, DHW by heat pump as recorded, climate zone 5A, the home's
  station temperature and humidity, the processed GHI.
- `f_cop` on RECS dev fold A (random half, seed 20261117) by matching the real heat-pump coefficient of a
  regression log(kWh/m2) ~ log HDD + log CDD + log area + heat pump (twins: `stock.recs_to_spec` of each respondent,
  1 year of the climate-zone TMYx); scored on fold B.
- **Reader** (not the network): logistic regression on log kWh/m2, log HDD18, log CDD18, log area, fitted on the
  twins, applied to the real respondents' annual electricity and their own HDD/CDD; AUC heat pump vs resistance on
  RECS dev (fold B and all), at 1.0 and 1.3.
- **Done when** (HANDOFF): reader AUC > 0.7 on RECS dev; HEAPO dev median winter profile r > 0.6 (the WPuQ target,
  moved to the legal dev homes). Also reported: HEAPO heating slope ratio (daily kWh per K below 12 C, twin / real)
  and night/day.

### Rung 4, physics 1.4: generator intensity by use type (1.3 + this)
Measured v1.0 gap (`generator.json`, 13 per use, vs the prior split): schools 1.81x, healthcare 1.91x, labs 2.02x,
food 1.53x, single-family 1.40x, lodging 0.66x.
- `generate_buildings(..., physics=...)` (or `$BUILDINGSIM_PHYSICS`): at 1.4 every generated spec is filled and its
  per-area intensities (lighting, plug, process, cooking, people, exterior lighting, outdoor air, DHW volume; homes
  also infiltration by sqrt) are multiplied by one factor per use type (the W1 mechanism, `phase2.apply_intensity`).
- Fit on the **prior** split only: 30 generated buildings per use (seeds 1000+), simulated at 1.4, 3 log-secant
  iterations so the median site EUI equals the weighted median of CBECS 2018 / RECS 2020 prior respondents.
- **Check on dev**: a disjoint sample (30 per use, seeds 2000+) against the dev respondents' weighted medians. Done
  when every use type is within +-15 %. The v1.0 numbers on the same dev check are reported first.

## 2026-11-17 — Physics 1.1 and 1.2 results: both rungs measured on dev; neither meets its done-when

Everything followed the plan entry above; no code, grid or objective was changed after a result was seen. The fits
ran once each (`tools/calibrate_district.py`, `tools/calibrate_electricity.py`); `check_bdg2.py --physics 1.1|1.2`
reproduces the tools' offline numbers exactly. JSONs: `checks/bdg2_dev_meters__p{1.0,1.1,1.2}[_y2017].json`,
`checks/district_fit.json`, `checks/electricity_fit.json`. No exam building was read.

### Rung 1, physics 1.1 (district plants)
Fitted (dev 2016, 35 building-meters): `T_hw_off` 15.6 C, `T_chw_off` 7.2 C, `q_steam` 3.5, `q_hw` 23.5,
`q_chw` 42.1 W/m2, summer reheat `f_rh` 0 (the grid's 0.25 and 0.5 made daily CV(RMSE) worse).

| Meter (n) | annual ratio median (within 30 %) | daily r | summer level real / sim | night/day real / sim |
|---|---|---|---|---|
| chilled water (20), 1.0 | 0.28 (5 %) | 0.88 | 1.53 / 1.94 | 0.80 / 0.10 |
| chilled water, 1.1 in sample | 1.00 (35 %) | 0.83 | 1.53 / 1.31 | 0.80 / 0.60 |
| chilled water, 1.1 fit 2016 -> 2017 | 0.98 (40 %) | 0.83 | 1.52 / 1.30 | 0.79 / 0.59 |
| chilled water, 1.1 leave one site out | 1.03 (15 %) | 0.85 | 1.53 / 1.33 | 0.80 / 0.61 |
| steam (10), 1.0 | 0.29 (10 %) | 0.70 | 0.66 / 0.36 | 0.95 / 0.14 |
| steam, 1.1 in sample | 1.00 (40 %) | 0.70 | 0.66 / 0.75 | 0.95 / 0.62 |
| steam, 1.1 fit 2016 -> 2017 | 0.98 (40 %) | 0.67 | 0.69 / 0.75 | 0.92 / 0.60 |
| steam, 1.1 leave one site out | **3.29** (20 %) | 0.70 | 0.66 / 0.81 | 0.95 / 0.91 |
| hot water (5), 1.0 | 0.15 (0 %) | 0.76 | 0.01 / 0.15 | 0.87 / 0.12 |
| hot water, 1.1 in sample | 1.00 (60 %) | 0.75 | 0.01 / 0.02 | 0.87 / 0.79 |
| hot water, 1.1 fit 2016 -> 2017 | 0.91 (40 %) | **0.47** | 0.44 / 0.03 | 0.85 / 0.79 |
| hot water, 1.1 leave one site out | **8.46** (0 %) | **0.62** | 0.01 / 0.00 | 0.87 / 0.97 |

- **Done when** (median annual 0.7-1.3 and daily r >= 0.7, judged leaving one site out): met for chilled water
  (1.03, r 0.85); **not met** for steam (3.29x) or hot water (8.46x, r 0.62). The loop base per m2 does not transfer
  between campuses: real steam is 1.4-4 W/m2 at Bull, 7-45 at Hog, 15-81 at Eagle; held-out ratios by site: steam Bull
  12.1x, Hog 0.58x, Eagle 0.26x; hot water is metered at only two sites (Eagle without Fox: q_hw 229 W/m2). Hot water
  2017 also changed shape (Eagle_office_Isidro's loop ran all summer in 2017: summer level 0.44 vs 0.01 in 2016).
- **What physics adds over rescaling** (W2/W3 are in sample: fitted on these buildings' 2017 data). W2 scales
  meters (0.89 / 0.88 / 0.89, daily r 0.88 / 0.72 / 0.79) and cannot move a shape statistic; W3 flattens every district
  meter by the same share and puts hot water at 0.70 of its mean in summer (real 0.01). 1.1's loop lockouts give hot
  water 0.02 in summer and cut chilled water in cold-site winters, and the base lifts night/day from 0.10-0.14 to
  0.60-0.79 (real 0.80-0.95). Its cost: chilled-water daily r falls from 0.88 to 0.83 (a winter base where the real
  loop is lower; winter level 0.73 vs 0.41 real). Summer reheat did not help on any meter.

### Rung 2, physics 1.2 (weather-insensitive electricity)
Fitted (dev 2016, 47 meters): `s_on` 0.6 (**the top of the grid**: night/day still 0.69 vs 0.79 real), `sigma_d`
0.15, `rho_d` 0.7 (real residual lag-1 0.71). Every leave-one-site-out fold chose the same values.

| Daily PRISM r2, real vs twin | p25 | p50 | p75 | KS p | match? |
|---|---|---|---|---|---|
| 2016, 1.1 (= 1.0 for electricity) | 0.05 / 0.04 | 0.24 / 0.25 | 0.45 / 0.61 | 0.36 | no (p75) |
| 2016, 1.2 (fit year; = LOSO) | 0.05 / 0.03 | 0.24 / 0.19 | 0.45 / 0.65 | 0.36 | no |
| 2017, 1.1 | 0.02 / 0.05 | 0.07 / 0.28 | 0.25 / 0.58 | 0.054 | no |
| 2017, 1.2 (fit 2016; = LOSO) | 0.02 / 0.03 | 0.07 / 0.19 | 0.25 / 0.63 | 0.093 | no |

- 1.2 matches what it was fitted to (residual sd 0.134 vs 0.133, night/day 0.69 vs 0.79, weekday effect 0.14 vs
  0.10 real, from 0.26), and lowers the median r2. It does **not** fix the upper tail: twins with electric HVAC
  (packaged RTU, PTAC lodging) keep r2 0.6-0.9, where most real buildings sit below 0.45. That tail is the system the
  metadata twin assigns (unmetered campus plants serve many electricity-only buildings), not schedules. **Done when
  not met** (quartiles not within 0.05 in any year). KS passes at 1.0 already (p 0.36, n 47): it has little power here.
- The Detective's real 0.07 vs twin 0.24 is the 2017 number: my 2017 real median is 0.066.
- Cost: electricity daily r falls from 0.34 to 0.20 (the daily activity draws are random, by design).
- Electricity annual ratio 0.98 -> 0.96. 1.2 changes nothing for homes (non-residential only, tested).

### Physics 1.0 identity and checks the change cannot reach
`check_bdg2.py` (default and `w2`, `w2w3`, `lc`), analytic, multinode, bestest, faults, controller, generator, speed
and phase2 re-run at the default with the new code; `compare.py`: the same counts as the P0.4 entry (53/53, 82/84
with the same two 1e-9 residuals, 222/222, 544/544, 34/34, 1715/1715, 1233/1236 with the same three near-zero values,
1595/1595 for each variant). `phase2_equivalence.json` differs in two places that are not physics: FaultFinder's
`noise.py` is being edited by another agent (`unchanged_from_commit` false), and ERCOT RTP was skipped (no GridData in
this worktree). Analytic, multinode and BESTEST re-run with `BUILDINGSIM_PHYSICS=1.2`: identical to 1.0 (no
district plant, no lighting or plug schedule: BESTEST and the analytic cases have process gains only).
ResStock / ComStock twins cannot be re-simulated here (no stock data); they are re-summaries.

### Which frozen sets would need rebuilding
1.1 changes only buildings with district heat or chilled water: LoadCast (BDG2 campus twins, district meters) and
BuildingDetective (W2 campus twins; 30 % of commercial banks). 1.2 changes every non-residential twin: LoadCast,
BuildingDetective and FaultFinder (AHU/RTU banks are commercial: internal gains and occupancy change). ThermoPilot
(homes) is unchanged by 1.1 and 1.2. Nothing is adopted yet; each model decides after its own dev test.

## 2026-11-17 — Physics 1.3 results (heat-pump signature): not met; the target is above what the dev data carry

As planned; nothing changed after a result was seen. `tools/calibrate_heatpump.py`, report `checks/heatpump_fit.json`.
Dev only: 557 RECS dev electric-heat respondents (498 with a twin: `recs_to_spec`, stable ties), 25 HEAPO dev homes.

- **Fitted**: `f_cop` 0.6 (**the bottom of the grid**: the real heat-pump coefficient on fold A is +0.07, i.e. heat-pump
  homes use 7 % more electricity per m2 than resistance homes at the same HDD, CDD and area; twins -0.27 at
  `f_cop` 1.0 and still -0.13 at 0.6). `tau_e` 0 (the emitter store lowered the HEAPO fold-A profile r at every
  grid value: 0.39 at 0 h, -0.00 at 6 h, 0.33 at 24 h). `T_limit` 18.75 C (protocols).
- **RECS dev reader** (logistic on log kWh/m2, log HDD, log CDD, log area, fitted on the twins, applied to the real
  respondents): AUC 0.51 -> 0.57 on fold B (n 248, 124 heat pumps), 0.50 -> 0.56 on all 498. The same reader fitted
  on the real rows themselves (5-fold CV) reaches 0.60 on these 498. **Done when (AUC > 0.7) not met and not
  reachable from annual meters on these data.** The twin-trained reader's own AUC on twins falls 0.75 -> 0.70: 1.3
  makes twins more like reality by making heat pumps less distinguishable, which is what the real data say.
- **HEAPO dev** (whole-home electricity, twins with the protocol area, year and heat-pump type, station weather):
  winter hour-of-day profile r median 0.24 -> 0.21 (fold B 0.19 -> 0.19); heating slope ratio twin/real 1.78 ->
  2.18; annual ratio 1.59 -> 1.71. **Done when (profile r > 0.6) not met, and 1.3 is worse on HEAPO** than 1.0: the
  US field-COP factor fitted on RECS raises Swiss air-source electricity, which was already too high (the twins' US
  envelopes and defaults are the bigger gap). The WPuQ finding ("real heat pumps run flat, twins at night") does not
  hold for HEAPO: real night/day is 1.16 (Swiss low-tariff night operation), twins 1.28.
- The Detective's 0.40 is its network on the exam; this reader at 1.0 gives 0.50 on dev.
- Frozen sets affected by 1.3: only residential heat-pump homes (BuildingDetective RECS/ResStock banks, ThermoPilot's
  heat-pump twins). I would not adopt 1.3 as fitted: it trades HEAPO for RECS.

## 2026-11-17 — Physics 1.4 results (generator intensity): 7 of 12 use types within 15 % on dev; not met

As planned; nothing changed after a result was seen. `tools/calibrate_intensity.py`, report
`checks/intensity_fit.json`. 30 generated buildings per use, fit seeds 1000+ against the prior split, check seeds
2000+ against the dev split, all simulated at 1.4 (so the 1.1-1.3 physics is inside).

Factors (prior fit, best of 3 log-secant evaluations): office 0.98, warehouse 0.87, other 0.76, food 0.67, education
0.52, assembly 0.52, healthcare 0.51, residential_multi 0.48, retail 0.46, residential_single 0.45, lab 0.20 (the
clip; its prior ratio is still 1.93: most generated labs are district buildings (7 of 13 in generator.json) and
carry the 1.1 loop base, which the intensity factor does not touch), lodging 1.68. Every other use reached 1.00-1.05 of the prior median.

| Use (dev n) | v1.0 sim / dev | 1.4 sim / dev | prior / dev survey median |
|---|---|---|---|
| office (133) | 1.11 | **1.08** | 1.05 |
| education (93) | 1.02 | 0.70 | 0.63 |
| lodging (41) | 0.81 | 1.25 | 1.16 |
| retail (63) | 1.43 | **1.02** | 0.93 |
| healthcare (63) | 1.18 | 0.80 | 0.79 |
| assembly (76) | 1.44 | **1.10** | 1.06 |
| warehouse (77) | 1.09 | **1.04** | 1.01 |
| residential_single (1504) | 1.44 | 1.17 | 1.01 |
| residential_multi (344) | 1.35 | **1.06** | 0.97 |
| food (32) | 1.72 | **1.02** | 1.03 |
| lab (9) | 10.7 | 3.72 | 4.8 |
| other (64) | 1.16 | **1.01** | 0.97 |

- **Done when (every use within +-15 % on dev) not met**: 7 of 12 (v1.0: 3 of 12). Education, healthcare and lodging
  miss because the dev survey medians differ from the prior ones by 16-37 % (last column: the weighted medians of two
  random splits of the same survey), so a prior fit cannot land within 15 % of dev there. Lab dev is 9 respondents,
  one of them carrying 633 of 1,003 weight units at 55 kWh/m2 (unweighted dev median 597). Single-family misses by 2
  points (1.17).
- 1.4 acts only through `generate_buildings`; `simulate()` at 1.4 is 1.3. Frozen sets that would change: every
  model's generated banks (BuildingDetective most; LoadCast's generated pre-training buildings; FaultFinder and
  ThermoPilot only where they draw from the generator).
