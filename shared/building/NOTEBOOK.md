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
