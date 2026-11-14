"""Page texts (web/info.json) with numbers pulled from the exam / checks files."""
import json
from common import *

ck = 'r8_wide'
E = json.load(open(f'{DATA}/exam/{ck}.json')); o = E['overall']
P = json.load(open(f'{ROOT}/checks/physics.json')); J = json.load(open(f'{ROOT}/checks/js_vs_python.json'))
T = json.load(open(f'{ROOT}/checks/twin_{ck}.json'))
a, f, same = o['actual']['demand'], o['forecast']['demand'], o['actual_same_days_as_forecast']['demand']
beat = [ba for ba in BAS if E['table'][ba]['all']['actual']['demand']['model'] < E['table'][ba]['all']['actual']['demand']['ba_day_ahead']]
beat_fc = [ba for ba in BAS if 'forecast' in E['table'][ba]['all'] and E['table'][ba]['all']['forecast']['demand']['model'] < E['table'][ba]['all']['forecast']['demand']['ba_day_ahead']]
nfc = [ba for ba in BAS if 'forecast' in E['table'][ba]['all']]
twin_d = sum(abs(v['d']['twin'] - v['d']['real']) for v in T.values()) / len(T)
info = dict(
    nsim='182,500', startBA='ERCO',
    heroTag=f"Exam 2023–2025 · 8 grids · {a['hours']:,} demand hours · actual weather unless noted",
    scoreNote=(f"Demand: DemandCast beats the grid's own day-ahead forecast in {len(beat)} of 8 grids with actual weather. "
               f"With real day-ahead forecast weather (2024, {len(nfc)} grids) the error rises to {f['model']:.2f} % "
               f"(grid forecasts {f['ba_day_ahead']:.2f} % on the same days) and it wins in {len(beat_fc)} of {len(nfc)}."),
    examIntro=(f"Every number below comes from 2023, 2024 and 2025, which were never used to choose anything. The forecast is "
               f"issued at 10:00 local standard time the day before and covers the 24 hours of the next day. The default view "
               f"feeds the model the weather that actually happened, which makes it an <b>upper bound</b>: the grids' own "
               f"forecasts had to use forecast weather. The second view uses real archived day-ahead forecasts (Open-Meteo "
               f"previous runs), which were only reachable for 2024 and {len(nfc)} grids before the archive's rate limit stopped us. "
               f"On those same days DemandCast's demand error is {same['model']:.2f} % with actual weather and {f['model']:.2f} % "
               f"with forecast weather; the grids' own forecasts score {f['ba_day_ahead']:.2f} %."),
    examNote=("Green marks the best column in each row; the shaded column is DemandCast. Demand error is the mean absolute "
              "percentage error over hours where every method has a value. Wind and solar error is the total absolute error "
              "divided by total output (hourly percentage errors blow up at night). Persistence uses the same hour yesterday, "
              "which a 10:00 forecast cannot fully know, so it is slightly flattered. Climatology is the dev-year (2019, 2021) mean "
              "capacity factor by month and hour times today's fleet. The band columns give the share of hours inside the 68 % and "
              "95 % bands; a calibrated forecast scores 68 / 95. NYISO does not report solar in EIA-930."),
    howText=(f"<p><b>Physics checks.</b> The envelope RC lag matches an ODE solver to {P['rc_filter_vs_solve_ivp_max_abs_K']:.0e} K; the "
             f"closed-form stock average over thermostat settings matches a million-draw Monte Carlo to {P['ghinge_vs_monte_carlo_max_abs_K']:.4f} K; "
             f"with no thermal mass the heating energy equals the degree-day formula exactly; specific humidity matches PsychroLib "
             f"(ASHRAE formulas) to {100 * P['specific_humidity_vs_psychrolib_max_rel']:.1f} %; heat-pump COP is {P['cop_hp_8.3C']:.2f} at 8.3 °C and "
             f"{P['cop_hp_-8.3C']:.2f} at −8.3 °C, AC COP {P['cop_ac_35C']:.1f} at 35 °C (typical rating points). The wind and solar physics are "
             f"GridSim's, checked in phase 1 against ERCOT (hourly r 0.90 wind, 0.97–0.98 solar).</p>"
             f"<p><b>Dev calibration (2019, 2021 only).</b> The building model was fitted to each grid's real demand: in-sample error "
             f"2.2–3.5 %. Each dev year is described by the other year's fit, so dev errors include two years of staleness, like the "
             f"exam. Wind and solar get one scale per grid (reporting scope, reanalysis bias). Dev data also showed that CAISO and "
             f"PJM solar in EIA-930 lead the physics by one hour, so those series are shifted. The simulator's drift, noise and "
             f"weather-error widths were set so simulated physics errors match dev (simulator v2).</p>"
             f"<p><b>Twin test.</b> Each grid's dev year was re-simulated with the full simulator (its own weather, fleet and fitted "
             f"buildings). The first run showed simulated demand far easier than real (0.3–1.9 % vs 1.7–3.0 %): the residual-noise "
             f"draw ran over all years and left about 60 % of simulated hours without noise. Fixed in simulator v3; twin and real "
             f"demand errors now differ by {twin_d:.2f} points on average. Wind twins are still easier than real in California, "
             f"the Northwest, New York and PJM, where 50 km reanalysis winds miss terrain effects.</p>"
             f"<p><b>Browser port.</b> The page re-implements the building physics and the three networks in JavaScript (weights in "
             f"float16). Against Python on nine page days: physics within {100 * J['phys_rel']:.2f} %, network outputs within "
             f"{100 * J['mu_rel']:.2f} % (demand) and {J['w_abs']:.0e} / {J['s_abs']:.0e} capacity factor (wind / solar).</p>"),
    limits=[
        f"Actual weather is an upper bound. With real day-ahead forecast weather (2024, {len(nfc)} grids) demand error rose from {same['model']:.2f} % to {f['model']:.2f} %, and DemandCast lost to the grids' own forecasts in ERCOT, SPP, ISO New England and Bonneville. The forecast archive was rate-limited, so only population-point temperature, humidity and irradiance (and one ERCOT wind point) were graded; 2023 and 2025 have no forecast-mode exam.",
        "Solar failed where EIA-930's timestamps moved. CAISO solar led the physics by one hour in the dev years, matched in 2023 and lagged by one hour in 2024–2025 (PJM changed in 2025). The dev-calibrated shift was wrong in the exam: CAISO solar error 59 % against 11 % for persistence, and persistence beats DemandCast solar overall (19 % vs 30 %). Bonneville's 0.1–0.2 GW solar fleet is also badly served.",
        f"Error bars are too narrow on real data: {100 * a['cov68']:.0f} % of demand hours fall inside the 68 % band and {100 * a['cov95']:.0f} % inside the 95 % band (CAISO 46 / 74). One widen-only factor per output was fitted on dev (demand ×1.14), which was not enough for the exam's longer staleness.",
        "Wind is limited by the weather input. NASA POWER reanalysis (about 50 km) misses Columbia Gorge and California pass winds: Bonneville wind error is 50 %, CAISO 33 %. The network trims the physics error by about 15 %, not more.",
        "Demand relies on the last week of metered load to correct the physics (in-context calibration). Physics alone is about 5 % off in the exam years, mostly from load growth since 2019–2021.",
        "Heat-wave peaks run high (ERCOT August 2023 peak predicted 88.5 GW, actual 85.4 GW). During Winter Storm Uri it predicted about 85 GW of demand while ERCOT served 69 GW under load shedding: the model forecasts demand, not what the grid could deliver.",
        "Building parameters are effective, not surveyed: several dev fits sit on their bounds (slow envelope time constant 240 h, cooling setpoint spread 8 K), so the split into heating, cooling and base load is indicative."],
    credits=("<p><b>Data.</b> U.S. EIA Hourly Electric Grid Monitor (EIA-930) and EIA-860/923 via Catalyst Cooperative PUDL (CC-BY-4.0); "
             "NASA POWER hourly weather (MERRA-2, CERES; NASA LaRC POWER Project); ASOS station data courtesy of the Iowa Environmental "
             "Mesonet; day-ahead forecasts from the Open-Meteo Previous Runs API (Zippenfenig, Open-Meteo.com, CC-BY-4.0); EIA-861 "
             "service territories and Census county populations for the weather weights. All assembled by GridData (ML Ratchet grid phase 1).</p>"
             "<p><b>Physics and tools.</b> GridSim (phase 1) wind and solar models; pvlib (BSD-3); PsychroLib (ASHRAE Handbook formulas, MIT) "
             "as a check; PyTorch for training, numpy for the release. Turbine curves after the LBNL Land-Based Wind Market Report; "
             "PVWatts, Perez and SAPM models via pvlib.</p><p>DemandCast is part of ML Ratchet (github.com/dogum/ML-Ratchet): trained on "
             "physics, graded on reality.</p>"),
)
json.dump(info, open(f'{ROOT}/web/info.json', 'w'), indent=1)
print(info['scoreNote'])
