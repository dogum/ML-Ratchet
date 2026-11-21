# Examples: three real exam fires

Three fires from the 48-fire exam replay subset (`shared/fire/data/proc/baselines/replay_subset.json`), as shipped
with the release (`demo_cases/`, bytes unchanged). They are a seeded draw (seed 20261123, `training/make_demo.py`), one
fire per tercile of the detected area, not picked by result; windows above 200,000 cells would have been skipped for
size, and none was. None is a dev fire.

| File | Fire | Period | Detected | Window |
|---|---|---|---|---|
| `AZ3395311201220190903` | River, Arizona, Sept 2019 | main 2017-21 | 220 ha | 166 × 178 cells |
| `MT4707111382020220904` | Boulder Lake, Montana, Sept 2022 | recent 2022-24 | 66 ha | 102 × 111 cells |
| `OK3464409522620180309` | Duncan Fire, Oklahoma, March 2018 | main 2017-21 | 7 ha | 97 × 130 cells |

Each fire has two files:

- `<id>.npz`: what the planner sees at the first afternoon FEDS detection, on the 90 m grid:
  - `burned0`: the detected burned area (bool)
  - `arr_f`, `fl_f`: arrival time (h after detection; 10,000 = never in 72 h) and flame length (m) from the one
    FORECAST FireSim run (real LANDFIRE landscape, real HRRR weather with the exam's seeded forecast wind error)
  - `cost`: line-building cost per metre relative to timber litter, from the fuel map (`firepilot.cost_raster`)
  - `clock_t`, `clock_w`: clock hours since detection and cumulative crew working hours (08:00-20:00 local solar
    time, from 2 h after detection). The 300 m/h budget's clock; the 1000 m/h budget uses the same crew clock.
- `<id>.json`: fire name and period, the hidden-draw index `k` (the demo's planner seed), the forecast head azimuth,
  the detected and forecast no-action areas, and for each budget (300 and 1000 m of line per working hour) the plan
  FirePilot chose in the exam and the exam's scored outcomes: the share of the no-action 72 h growth stopped in the
  FireSim truth world for no action, the fixed plans, MPC, FirePilot and the perfect-foresight bound (3 decimals).

The truth columns are simulated outcomes, stored from the one exam run. They need FireSim's truth world (a hidden
ROS factor the planner never sees), so the demo prints them and does not recompute them.

## Sources and credit

- **FEDS-MTBS** fire progression (detected burned area, Event_ID): Chen et al. 2022 FEDS method, MTBS-constrained;
  Zenodo 20187963; CC BY 4.0.
- **MTBS** fire perimeters and Event_IDs: USGS / USFS, public domain.
- **LANDFIRE** fuels, canopy and terrain (the cost raster and the forecast run): US public domain.
- **HRRR** analyses (the forecast run's weather): NOAA, via the University of Utah hrrrzarr archive on AWS Open Data.
- **gridMET** daily precipitation where HRRR has none (Abatzoglou 2013): public domain.
- Processed by FireData and FireSim (Wildfire Phase 1, `shared/fire`). None of these sources is share-alike.
