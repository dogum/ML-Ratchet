# BuildingSim: known limits

What the simulator gets wrong, by how much where we measured it, and what to watch in Phase 2. Numbers are from
`shared/building/checks/*.json`.

## Against real buildings (BDG2 dev meters, generic twins from metadata only)

1. **Campus district energy is about 3x too low.** On the 52 BDG2 dev buildings, the median simulated/real annual
   ratio is:
   - chilled water 0.28 (n = 20, IQR 0.20–0.45);
   - steam 0.29 (n = 10);
   - hot water 0.15 (n = 5).

   The daily shape is reasonable (daily r 0.88 chilled water, 0.70 steam, 0.76 hot water). The temperature
   response has the right sign in every case. The magnitude is not right. Real campus buildings run their air
   handlers long hours, with high outdoor air, reheat, simultaneous heating and cooling, steam for humidification
   and hot water, and distribution losses. BuildingSim's defaults (setbacks, economizers, 30 % VAV minimum, code
   outdoor air) describe a well-run building. **LoadCast must calibrate this on dev** (operating hours, minimum
   flow, outdoor air, a distribution-loss factor) before simulated pre-training means anything for district meters.
2. **Electricity totals scatter widely** (median ratio 0.98, IQR 0.50–1.42, 30 % within ±25 %), because BDG2 gives
   only use, area and sometimes year. The weekday hourly profile correlates at r 0.64 but the daily series only at
   0.34. In 55 % of buildings the daily-vs-temperature slope has the wrong sign: many campus buildings with district
   cooling have flat electricity, while twins with electric HVAC respond to weather. Gas has only 2 scorable
   buildings (ratio 0.03: their gas is not space heating as the twin assumes).
3. **Generated buildings vs real surveys** (site EUI medians, generated vs CBECS/RECS prior split):
   - right for offices (0.91), warehouses (0.98), retail (1.11), multi-family (1.12) and other (1.17);
   - too high for single-family houses (1.40), food service (1.53), schools (1.81), healthcare (1.91) and labs (2.0);
   - too low for lodging (0.66).

   The generated climate mix is the stock's, but the weather is one TMYx city per climate zone; ComStock rows are
   drawn by model count, not by floor area or survey weight. Treat absolute EUIs of generated buildings as
   ±50 % by use type.

## Against other simulators (ResStock / ComStock twins, same weather; NOT reality)

4. **Residential heating loads are ~30 % above ResStock** for the same characteristics:
   - delivered heating load median 1.29 (IQR 1.04–1.51);
   - cooling load 1.22;
   - electricity 1.06; on-site fuel 1.07; site energy 1.10.

   Design heating loads agree (my UA x dT vs ResStock's autosized furnace capacity: median 0.96), so the excess is
   in mild weather. Mobile homes (1.5x), large multi-family units (1.4x heating, 1.5x cooling) and slab houses (1.3x)
   are worst. Basements and crawlspaces are within 5-15 %. Candidates, not isolated:
   - internal and furniture mass (EnergyPlus has more);
   - the ground models (Kiva vs F-factor / buffer spaces);
   - AIM-2 vs the LBL infiltration model;
   - hot-water heat gains to the space.

   ResStock uses stochastic occupant schedules, so its hourly electricity correlates with ours at only r 0.52
   (daily r 0.80).
5. **Commercial heating and fans run low vs ComStock** (held-out draw):
   - heating median 0.75 with a very wide IQR (0.17–1.01);
   - cooling 0.83; fans 0.80; pumps and heat rejection 0.43;
   - site energy 0.86, electricity 0.83 (aggregates 0.97 / 1.00).

   ComStock's envelope U-values are not in its metadata: BuildingSim uses its own code-level table for the energy
   code of construction, which is the largest single source of scatter. Lighting, water heating, exterior lighting
   and night base loads were corrected once on a separate ComStock sample (seed 11); the quoted numbers are a
   disjoint draw.

## Physics and model structure

6. **One zone by default.** There is no perimeter/core split and no multi-zone model. Simultaneous heating of
   perimeter and cooling of core zones (a big part of real VAV reheat energy) is absent. That is one reason for
   limit 1. The VAV system therefore has a lumped "heating mode" (economizer and cooling coil locked out when the
   zone calls for heat). Multi-zone was not built in Phase 1.
7. **The ISO 13790 5R1C network** couples air to the surface star node through 3.45 W/m²K x 4.5 A_f.
   - With the air temperature controlled, steady losses are 6.9 % below the naive sum of U·A (`analytic.json`).
     BuildingDetective labels should use `UA_network` (in `labels()`), which includes this.
   - Gains are split 50/50 convective/radiant (BESTEST: 40/60), and a share of radiant gains is lost through the
     windows (booked in the ledger).
   - BESTEST free-float minimum temperatures are 0.3–1.6 K too cold in light buildings.
   - The case-900 peak cooling is 3 % below range.
8. **The multi-node zone** (option) has no interior long-wave exchange between surfaces (all surfaces couple through
   the air node) and fixed solar distribution (floor/internal mass).
   - With implicit Euler it damps a 24 h wall response by 5 % at 15 min and 13 % at 1 h; use 15 min.
   - BESTEST annual loads are inside the ranges, but peak loads are 6–12 % above them.
9. **Return air is coupled implicitly; supply air is not always.** The air-side ledger closes exactly, but when a
   stuck coil or the VAV heating mode makes the supply temperature differ from the one the zone solve assumed, the
   difference is booked as `supply_lag` (up to 2 % of air-side energy in generated buildings, ~0 % when healthy).
10. **DX and heat-pump part load.** Single-speed units use one cycling-loss curve. Two-stage and variable-speed units
    use assumed curves (variable speed: COP +14 % at half load), not manufacturer data. Latent capacity uses the
    steady bypass-factor model: no moisture re-evaporation on cycling fans, so humid-climate RH is optimistic.
11. **Fault models are first-order.**
    - Refrigerant charge and condenser fouling follow published trends (Kim & Braun 2012) fitted by eye, not the
      authors' correlations.
    - A leaking valve is an air-side effectiveness toward the chilled-water temperature.
    - Sensor bias acts on control and on the trend.
    - There are no dynamics (no hunting, no short-cycling, no actuator hysteresis).
    - Real fault data will show noise and transients the simulator lacks. FaultFinder should add real
      normal-operation noise (B's normal days) and be graded only on B's real/experimental runs.
12. **Controls are idealised.** The thermostat holds the setpoint exactly each step (proportional within the step),
    with no deadband cycling and no PID dynamics. Optimal start is a fixed early-start window per use type.
    Residential natural ventilation opens windows "just enough", which real occupants do not.
13. **Infiltration** uses the LBL model with met-station wind (no terrain or eave-height correction beyond the
    shelter class). Commercial airtightness is converted from ComStock's 75 Pa flow with residential coefficients.
14. **Plant.** One or two identical chillers (no staging logic beyond capacity, no chilled-water reset). The tower
    holds max(21 °C, T_wb + 3 K). Boilers have no standby or cycling losses. District meters read delivered energy
    times a distribution efficiency (default 1.0). Pump power is 19 W/gpm.
15. **Domestic hot water** is a fixed-temperature tank with standby losses and a draw schedule: no stochastic draws,
    no tank stratification, no recovery-limited unmet hot water.
16. **Weather handling.** Hourly irradiance is held constant inside the hour at 15-min steps (sunrise/sunset steps
    are clipped to keep beam ≤ GHI). Wet bulb near 0 °C can jump by 0.4 K (ice/water branch). The 17 bundled TMYx
    files are typical years, not actual years.

## Equipment curves (assumptions to know)

17. Air-cooled chillers reuse the DOE-2 centrifugal curve shape re-centred at 35 °C. Cold-climate heat pumps get half
    the capacity slope of the single-speed curve. Condensing-boiler efficiency vs return temperature is a linear
    fit. SEER → EER and HSPF → COP47 conversions are rules of thumb. These choices are documented in `curves.py`.

## Data and blockers

18. **EnergyPlus could not be run** (no PyPI engine; the openstudio wheel only translates; GitHub blocked), so the
    DOE prototype comparison was replaced by the ResStock/ComStock twins (which are EnergyPlus-based) and BESTEST.
19. BESTEST reference ranges were transcribed from the literature, not the standard itself; confirm before quoting.
20. ResStock twins skip Alaska and Hawaii (B's sample gives their counties by name, not GISJOIN) and assume
    sea-level pressure (the county weather files have no elevation).
