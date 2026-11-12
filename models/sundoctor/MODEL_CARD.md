# SunDoctor model card (v2.0, rung 8)

**What it is.** A 2-D convolutional network (591,455 weights) that reads 14 days x 96 fifteen-minute steps (local *solar*
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
| Sun position | Spencer declination + equation of time | elevation 0.45 deg, azimuth 0.55 deg |
| Clear sky | Ineichen-Perez | GHI 0.13% |
| Decomposition | Erbs | DHI 0.08% |
| Transposition | Hay-Davies | POA 0.0029% |
| Tracker rotation | single axis, with or without backtracking | 3e-13 deg; AOI 2e-11 deg |
| Tracker row shading | Anderson & Jensen 2024 (pvlib `shaded_fraction1d`) | 1e-15 |
| Cell temperature | Faiman | 0.000 C |
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
Total over both versions: 1,760,000 simulated fortnights.

## The ratchet

**v1** (fixed tilt, sensor present; objective on 2,000 fortnights, seed 777):

| Rung | Change | Decision | Objective | California: counted right when dead |
|---|---|---|---|---|
| 1 | Baseline | kept | 1.263 | 29% |
| 2 | Beta-NLL | kept | 1.678 | 60% |
| 3 | Polish | kept | 1.748 | 64% |
| 4 | Longer memory | rejected | 1.753 | 66% |
| 5 | Realism | rejected | 1.718 | 65% |

**v2** changed the exam: a new frozen set of 2,000 fortnights (seed 778), half without a sensor, with trackers, snow and
stuck rows. Objective = mean over the two halves of [mean skill over pf, frac, soil, shade, clip, outage, snow, stuck
+ 0.5 x step AUC + 0.5 x dead-string counting accuracy], skill = 1 - MAE / MAE of predicting the median. Kept only if it
rises by more than 0.01. The v1 release, run through v2's inputs (no snow or tracker outputs), is the starting point.

| Rung | Change | Decision | Objective | with sensor | power only | snow skill (sensor / none) | stuck skill | California power-only, counted right when dead | Colorado snow AUC | Colorado outage AUC, power only |
|---|---|---|---|---|---|---|---|---|---|---|
| start | v1 release (s3_polish) seen through v2 inputs; no snow or stuck heads | kept | 0.341 | 1.342 | -0.661 | 0.00 / 0.00 | 0.00 / 0.00 | 3% | 0.52 | 0.63 |
| 6 | v2 | kept | 1.579 | 1.664 | 1.493 | 0.87 / 0.75 | 0.40 / 0.19 | 52% | 0.99 | 0.94 |
| 7 | Polish | kept | 1.615 | 1.693 | 1.536 | 0.89 / 0.81 | 0.48 / 0.29 | 55% | 0.98 | 0.95 |
| 8 | Weather fronts | kept | 1.628 | 1.702 | 1.554 | 0.91 / 0.83 | 0.51 / 0.31 | 52% | 0.98 | 0.97 |

Real-data numbers are reported, never used to choose. All rungs were re-scored after the Colorado answer keys were
corrected (decisions unaffected): outages are not graded on cold, dim days, when a dark module may be under snow its
neighbours shed; and snow is graded module by module, after rung 6's misses showed the first, fleet-wide key marking
modules that had already shed their snow as buried.

## Final exam 1: PVDAQ 2107 (Arbuckle, California), with and without the sensor

24 identical string inverters, each with 6 strings of 20 modules (37.2 kW DC, 30 kW AC), fixed 25 deg tilt facing south,
2017-2023. Answer key: each inverter against its 23 siblings, under the same sky. SunDoctor sees one inverter at a time.

| | v1 with sensor | v1 power only | v2 with sensor | v2 power only |
|---|---|---|---|---|
| Dead strings counted right, every inverter-day | 94.1% | 84.7% | 95.9% | 94.2% |
| ...on days with dead strings | 63.8% | 2.8% | 63.1% | 52.2% |
| Healthy days called healthy | 97.7% | 96.1% | 99.9% | 99.9% |
| String failures flagged within a day | 54/56 | 1/56 | 55/56 | 47/56 |
| Repairs flagged within a day | 44/47 | 0/47 | 40/47 | 26/47 |
| Inverter offline caught | 47/49 | 0/49 | 44/49 | 39/49 |
| Injected dead strings counted exactly | 100.0% | - | 94.2% | 58.3% |
| Injected tree shade: correlation with the truth | 0.97 | - | 0.98 | 0.94 |
| Injected dust: performance-drop error | 3.3% | - | 3.6% | 3.7% |
| Injected clipping: loss error | 1.7% | - | 0.6% | 1.1% |

RdTools on inverter 11's healthy days (v2, with sensor): degradation -0.08 %/yr (YoY) vs SunDoctor
0.08 %/yr; soiling (SRR) RdTools 4.4% mean loss vs SunDoctor 1.2%.

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
  much of 2022). A month is trusted only if its clearest days reach clear-sky levels: 1,049 of 2,269 days. With-sensor
  grading, and the snow key, use only those days; power-only mode is graded on every day.

| Every module-day | v1 with sensor | v2 with sensor | v2 power only |
|---|---|---|---|
| Snow module-days (key: over 20% lost) | 166 | | |
| Snow days flagged (over 20%) | 0.0% | 54.8% | 42.2% |
| Snow days called an outage instead | 33.7% | 13.3% | 4.2% |
| False snow alarms, May-Sep | 0.0% | 0.2% | 0.0% |
| False snow alarms, clear winter days | 0.0% | 0.5% | 0.5% |
| Snow AUC | 0.52 | 0.98 | 0.94 |
| Tracker module-days losing over 10% | 234 | | |
| ...flagged (over 5%) | 0.0% | 77.4% | 86.2% |
| Tracker loss AUC (over 2%) | 0.52 | 0.50 | 0.48 |
| Correlation with the daily tracker loss | - | 0.64 | 0.30 |
| Mean tracker loss: key | 1.6% | 1.6% | 1.1% |
| Mean tracker loss: SunDoctor | 0.0% | 1.3% | 4.7% |
| Outage module-days | 183 | | |
| Outage days caught | 84.2% | 86.9% | 79.2% |
| False outage alarms | 0.1% | 0.1% | 0.1% |
| Outage AUC | 1.00 | 0.98 | 0.97 |

## Limits

- Power only is the weaker mode. On the California farm it counts dead strings right on 52% of the days they are really dead, against 63% with the sensor, and its dust and shading figures are rough: without a sensor, a hazy week and a dusty week look much alike.
- A panel buried in snow all day makes no power at all, exactly like a tripped inverter. SunDoctor can then only go on the weather, and sometimes calls the buried day an outage; it is surer once the snow starts sliding off. In Colorado it flags 55% of snow module-days with the sensor and 42% without, and calls 13% of them outages (4% without the sensor).
- Stuck trackers: it flags 77% of the days trackers cost more than 10%, but cannot rank small losses of 2 to 5% (AUC 0.50). Without a sensor it over-reports them (average 4.7% against 1.1% in the key): a cloudy morning can look like panels facing the wrong way.
- It tells dead strings from dust partly by memory: it compares the system with its own best level over the past year. A string dead from the first day it saw looks like a weaker system, not a broken one.
- Two real farms, both utility scale. Rooftops, microinverters, bifacial panels and real satellite irradiance are untested. The Colorado answer keys are physics estimates from the plant's own sensors, and its main sunlight sensor was wrong for three and a half years.
- Colorado is no longer a fully blind test for rung 8: the weather-front change was prompted by rung 6's snow misses there and in the lab on this page. It improved the simulated exam and the Colorado outage calls, but not the snow calls.
