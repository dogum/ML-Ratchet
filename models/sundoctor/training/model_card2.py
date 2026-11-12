"""MODEL_CARD.md and README.md for SunDoctor v2, written from the logs and the page data (nothing typed by hand twice)."""
import _paths as _P  # portable paths (see _paths.py)
import json
R1 = json.load(open(_P.WORK + '/ratchet.json')); R2 = json.load(open(_P.WORK + '/ratchet2.json'))
D = json.load(open(_P.LAB + '/data.json')); ph = json.load(open(_P.WORK + '/physics_check.json'))
E = D['exam']; X = E['real']; X1 = E['real_v1']; rel = D['release']['ckpt']; n_rel = len(R1) + next(i for i, r in enumerate(R2) if r['ckpt'] == rel)
pc = lambda x, d=1: '-' if x is None else f"{100 * x:.{d}f}%"
f2 = lambda x: '-' if x is None or x != x else f"{x:.2f}"
import numpy as np
W = dict(np.load(_P.WORK + '/release/sundoctor_weights.npz')); n_par = sum(v.size for k, v in W.items() if k != 'meta')

rows2 = []
for i, r in enumerate(R2):
    v = r['val']; s = lambda h, k: f"{v[h][k]['skill']:.2f}"; a, b = r['real']['2107_home'], r['real']['9068_poa']
    rows2.append(f"| {len(R1) + i if i else 'start'} | {r['note'].split(':')[0]} | {'kept' if r['accepted'] else 'rejected'} | {r['objective']:.3f} | {v['poa']['objective']:.3f} | {v['home']['objective']:.3f} | "
                 f"{s('poa', 'snow')} / {s('home', 'snow')} | {s('poa', 'stuck')} / {s('home', 'stuck')} | {pc(a['count_acc_when_dead'], 0)} | {f2(b['snow_auc'])} | {f2(r['real']['9068_home']['outage_auc'])} |")
rows1 = [f"| {i + 1} | {r['note'].split(':')[0]} | {'kept' if r['accepted'] else 'rejected'} | {r['objective']:.3f} | {pc(r['real']['count_acc_when_dead'], 0)} |" for i, r in enumerate(R1)]

def ex_rows(keys, fmt):
    return '\n'.join(f"| {lab} | {fmt(X1['9068_poa'].get(k))} | {fmt(X['9068_poa'].get(k))} | {fmt(X['9068_home'].get(k))} |" for lab, k in keys)

P, Ph, P1h = E['episodes'], E['episodes_home'], E.get('episodes_v1_home')
I, Ih = E['inject'], E['inject_home']
card = f"""# SunDoctor model card (v2.0, rung {n_rel})

**What it is.** A 2-D convolutional network ({n_par:,} weights) that reads 14 days x 96 fifteen-minute steps (local *solar*
time) of one PV system: AC power per kW of DC nameplate, air temperature and, if the site has one, plane-of-array (POA)
irradiance. Physics-derived channels fill in the rest: clear-sky POA for the array's fixed or tracking geometry, two
PVWatts expectations (from the measured POA, and from clear sky), sun angles, a data mask, a sensor flag and the
system's own healthy level. For every day it returns, each with a 68% error bar:

- `pf`: performance factor of the clean, unshaded, unclipped array, relative to nameplate physics
- `frac`: fraction of the array working (dead strings: `round(n_strings x (1 - frac))`)
- `soil`: soiling loss; `shade`, `clip`, `outage`, `snow`, `stuck`: the share of energy lost to shading, inverter
  clipping, outages, snow cover, and trackers stuck or stowed away from their ideal angle
- `step_p`: probability that part of the array failed or was repaired that day

Plus per window: `eta0` (baseline efficiency) and `ac_limit` (inverter limit per kW DC).

**Two modes.** With an irradiance sensor, and *power only* (no `poa` column), as from a home monitoring app. The same
weights serve both; the sensor flag tells the network which it is reading.

**What changed from v1.** Single-axis trackers, snow, stuck and stowed trackers, thin-film modules, irradiance-sensor
misalignment, morning dew dropouts and power-only mode were added to the simulator; two outputs (`snow`, `stuck`) and two
input channels (sensor flag, clear-sky expectation) were added to the network; v1's weights were transplanted in and
training continued on the new simulator. A second real farm (PVDAQ 9068, Colorado trackers) joined the final exam.

## The simulator (all training data)

Vectorized physics in solar time, each block checked against pvlib 0.16:

| Block | Model | Max difference vs pvlib |
|---|---|---|
| Sun position | Spencer declination + equation of time | elevation {ph['sun_elevation_maxerr_deg']:.2f} deg, azimuth {ph['sun_azimuth_maxerr_deg']:.2f} deg |
| Clear sky | Ineichen-Perez | GHI {100 * ph['clearsky_ghi_rel_err_noon']:.2f}% |
| Decomposition | Erbs | DHI {100 * ph['erbs_dhi_rel_err']:.2f}% |
| Transposition | Hay-Davies | POA {100 * ph['poa_rel_err']:.4f}% |
| Tracker rotation | single axis, with or without backtracking | {ph['tracker_theta_maxerr_deg']:.0e} deg; AOI {ph['tracker_aoi_maxerr_deg']:.0e} deg |
| Tracker row shading | Anderson & Jensen 2024 (pvlib `shaded_fraction1d`) | {ph['tracker_shaded_fraction_maxerr']:.0e} |
| Cell temperature | Faiman | {ph['faiman_maxerr_C']:.3f} C |
| DC / inverter | PVWatts (AC clipping) | exact |

Systems: latitude 20-52 N; 60% fixed (tilt 0-40 deg, azimuth 90-270) and 40% single-axis trackers (max angle 45-60 deg,
GCR 0.28-0.5; 60% backtrack, the rest true-track and shade each other at low sun); 30% thin-film (CdTe: smaller
temperature coefficient, spectral gain under high air mass, near-linear partial-shading loss); DC/AC 0.95-1.5.
Weather: clear, hazy, broken-cloud (Markov) and overcast days; 30% of systems in cold-winter climates. Faults: 1-12
strings per inverter with failures or repairs inside the window; soiling reset by rain, snow or cleaning; row and horizon
shading; outages, trips and curtailment; snow that falls on cold overcast days and slides off over the following days
(faster on steep and tracking arrays, and in sun); trackers with 10-100% of rows stuck at a fixed angle, and stows (wind,
or flat-out for snow); morning-only string dropouts. Measurements: power noise, irradiance gain error, sensors misaligned
by up to 8 deg tilt and 15 deg azimuth (40%), satellite-like noise (20%), sensors sometimes snow-covered, timestamp
shifts, missing data, and 40% of fortnights with no sensor at all (temperature then from a weather service: extra noise).
Total over both versions: {D['release']['samples']:,} simulated fortnights.

## The ratchet

**v1** (fixed tilt, sensor present; objective on 2,000 fortnights, seed 777):

| Rung | Change | Decision | Objective | California: counted right when dead |
|---|---|---|---|---|
{chr(10).join(rows1)}

**v2** changed the exam: a new frozen set of 2,000 fortnights (seed 778), half without a sensor, with trackers, snow and
stuck rows. Objective = mean over the two halves of [mean skill over pf, frac, soil, shade, clip, outage, snow, stuck
+ 0.5 x step AUC + 0.5 x dead-string counting accuracy], skill = 1 - MAE / MAE of predicting the median. Kept only if it
rises by more than 0.01. The v1 release, run through v2's inputs (no snow or tracker outputs), is the starting point.

| Rung | Change | Decision | Objective | with sensor | power only | snow skill (sensor / none) | stuck skill | California power-only, counted right when dead | Colorado snow AUC | Colorado outage AUC, power only |
|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows2)}

Real-data numbers are reported, never used to choose. All rungs were re-scored after the Colorado answer keys were
corrected (decisions unaffected): outages are not graded on cold, dim days, when a dark module may be under snow its
neighbours shed; and snow is graded module by module, after rung 6's misses showed the first, fleet-wide key marking
modules that had already shed their snow as buried.{D.get('card_note', '')}

## Final exam 1: PVDAQ 2107 (Arbuckle, California), with and without the sensor

24 identical string inverters, each with 6 strings of 20 modules (37.2 kW DC, 30 kW AC), fixed 25 deg tilt facing south,
2017-2023. Answer key: each inverter against its 23 siblings, under the same sky. SunDoctor sees one inverter at a time.

| | v1 with sensor | v1 power only | v2 with sensor | v2 power only |
|---|---|---|---|---|
| Dead strings counted right, every inverter-day | {pc(X1['2107_poa']['count_acc'])} | {pc(X1['2107_home']['count_acc'])} | {pc(X['2107_poa']['count_acc'])} | {pc(X['2107_home']['count_acc'])} |
| ...on days with dead strings | {pc(X1['2107_poa']['count_acc_when_dead'])} | {pc(X1['2107_home']['count_acc_when_dead'])} | {pc(X['2107_poa']['count_acc_when_dead'])} | {pc(X['2107_home']['count_acc_when_dead'])} |
| Healthy days called healthy | {pc(X1['2107_poa']['healthy_called_healthy'])} | {pc(X1['2107_home']['healthy_called_healthy'])} | {pc(X['2107_poa']['healthy_called_healthy'])} | {pc(X['2107_home']['healthy_called_healthy'])} |
| String failures flagged within a day | {E['episodes_v1']['onsets_found_pm1day']}/{E['episodes_v1']['string_failures']} | {(str(P1h['onsets_found_pm1day']) + '/' + str(P1h['string_failures'])) if P1h else '-'} | {P['onsets_found_pm1day']}/{P['string_failures']} | {Ph['onsets_found_pm1day']}/{Ph['string_failures']} |
| Repairs flagged within a day | {E['episodes_v1']['repairs_found']}/{E['episodes_v1']['repairs']} | {(str(P1h['repairs_found']) + '/' + str(P1h['repairs'])) if P1h else '-'} | {P['repairs_found']}/{P['repairs']} | {Ph['repairs_found']}/{Ph['repairs']} |
| Inverter offline caught | {E['episodes_v1']['offline_found']}/{E['episodes_v1']['offline_events']} | {(str(P1h['offline_found']) + '/' + str(P1h['offline_events'])) if P1h else '-'} | {P['offline_found']}/{P['offline_events']} | {Ph['offline_found']}/{Ph['offline_events']} |
| Injected dead strings counted exactly | {pc(E['inject_v1']['dead_count_right_after'])} | - | {pc(I['dead_count_right_after'])} | {pc(Ih['dead_count_right_after'])} |
| Injected tree shade: correlation with the truth | {f2(E['inject_v1']['shade_corr'])} | - | {f2(I['shade_corr'])} | {f2(Ih['shade_corr'])} |
| Injected dust: performance-drop error | {pc(E['inject_v1']['soil_pf_drop_mae'])} | - | {pc(I['soil_pf_drop_mae'])} | {pc(Ih['soil_pf_drop_mae'])} |
| Injected clipping: loss error | {pc(E['inject_v1']['clip_mae'])} | - | {pc(I['clip_mae'])} | {pc(Ih['clip_mae'])} |

RdTools on inverter 11's healthy days (v2, with sensor): degradation {E['rdtools']['rdtools_yoy_degradation_pct_per_yr']:.2f} %/yr (YoY) vs SunDoctor
{E['rdtools']['sundoctor_degradation_pct_per_yr']:.2f} %/yr; soiling (SRR) RdTools {pc(E['soiling']['mean_loss_rdtools'])} mean loss vs SunDoctor {pc(E['soiling']['mean_loss_sundoctor'])}.

## Final exam 2: PVDAQ 9068 (Kersey, Colorado), single-axis trackers, snow

4.7 MW DC, two inverters of four power modules each (one module = one SunDoctor system, 592 kW DC), 224 horizontal
single-axis trackers on a north-south axis, 2017-2023. The trackers true-track (no backtracking; measured angles sit at
+-60 deg at low sun), GCR about 0.35. The site clock is US Mountain time with daylight saving (found by fitting the
tracker angles). Answer keys, all from the plant's own instruments:

- **Trackers:** each inverter's 112 measured tracker angles against the ideal angle; the key is the clear-sky POA lost
  by the actual angles relative to ideal (Hay-Davies on every tracker).
- **Outages:** a module dark (under 5% of the fleet's level) while the fleet produces; not graded on cold, dim days.
- **Snow:** on cold days (mean below 3 C) when the fleet is far below what the sensor implies and most modules are alike,
  each module's shortfall against the sensor.
- **The sensor:** the main POA sensor read about half the true irradiance from May 2019 to October 2022 (and nothing for
  much of 2022). A month is trusted only if its clearest days reach clear-sky levels: {D['co_sys']['sensor_ok_days']:,} of {D['co_sys']['days']:,} days. With-sensor
  grading, and the snow key, use only those days; power-only mode is graded on every day.

| Every module-day | v1 with sensor | v2 with sensor | v2 power only |
|---|---|---|---|
| Snow module-days (key: over 20% lost) | {X['9068_poa']['snow_days']} | | |
{ex_rows([('Snow days flagged (over 20%)', 'snow_found'), ('Snow days called an outage instead', 'snow_called_outage'), ('False snow alarms, May-Sep', 'snow_false_summer'), ('False snow alarms, clear winter days', 'snow_false_winter_clear')], pc)}
{ex_rows([('Snow AUC', 'snow_auc')], f2)}
| Tracker module-days losing over 10% | {X['9068_poa']['stuck_days_big']} | | |
{ex_rows([('...flagged (over 5%)', 'stuck_found_big')], pc)}
{ex_rows([('Tracker loss AUC (over 2%)', 'stuck_auc_2pct'), ('Correlation with the daily tracker loss', 'stuck_corr')], f2)}
{ex_rows([('Mean tracker loss: key', 'stuck_true_mean'), ('Mean tracker loss: SunDoctor', 'stuck_pred_mean')], pc)}
| Outage module-days | {X['9068_poa']['outage_days']} | | |
{ex_rows([('Outage days caught', 'outage_found'), ('False outage alarms', 'outage_false')], pc)}
{ex_rows([('Outage AUC', 'outage_auc')], f2)}

## Limits

{chr(10).join('- ' + x.replace('<b>', '**').replace('</b>', '**') for x in D['limits'])}
"""
open(_P.WORK + '/release/MODEL_CARD.md', 'w').write(card)
readme = f"""# SunDoctor v2.0

What's wrong with this solar array? A neural network trained only on physics simulations of PV systems reads two weeks
of a system's power and air temperature, plus plane-of-array irradiance if there is a sensor, and returns a day-by-day
diagnosis with error bars: dead strings, failures and repairs, shading, clipping, outages, soiling, snow, stuck
trackers, and the kWh lost to each. Fixed-tilt arrays and single-axis trackers.

    pip install numpy pandas
    python demo.py

    import sundoctor
    doc = sundoctor.SunDoctor()
    # fixed tilt, with a sensor
    r = doc.diagnose_csv('examples/pvdaq2107_inverter11_2019H2.csv', lat=38.996, lon=-122.134, tilt=25, azimuth=180,
                         dc_kw=37.2, tz='US/Pacific', n_strings=6)
    # power only (a CSV without a poa column, or home=True)
    r = doc.diagnose_csv('examples/pvdaq2107_inverter11_2019H2.csv', lat=38.996, lon=-122.134, tilt=25, azimuth=180,
                         dc_kw=37.2, tz='US/Pacific', n_strings=6, home=True)
    # single-axis trackers
    r = doc.diagnose_csv('examples/pvdaq9068_module2_2018winter.csv', lat=40.3864, lon=-104.5512, mount='tracker',
                         max_angle=60, gcr=0.35, backtrack=False, dc_kw=592.25, tz='US/Mountain')
    print(r['summary'])

CSV columns: a timestamp (local time in `tz`; say whether it marks the start, center or end of each interval with
`label=`), AC power (kW, or `power_unit='W'`), air temperature (C, or `temp_unit='F'`; any weather service will do) and,
optionally, POA irradiance (W/m2). Longer records help: SunDoctor compares the system with its own best level over the
past year.

MODEL_CARD.md: the simulator and its pvlib checks, both ratchets rung by rung, the real-data exams on two farms and the
limits. `training/` holds every script (PyTorch, pvlib, RdTools for the exam).

Example data: NREL PVDAQ systems 2107 and 9068, via the Open Energy Data Initiative (DOE 2023 Solar Data Prize).
"""
open(_P.WORK + '/release/README.md', 'w').write(readme); print('card', len(card), 'readme', len(readme))
