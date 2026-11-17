# BuildingData: real data for the buildings project (Phase 1, agent B)

Everything Phase 2 building models are graded on lives here. It is frozen: the splits, cleaning rules and baselines below
were decided once (seed 20261116) and must not change after Phase 2 starts. Build scripts are in `shared/building/data/scripts/`
(all resumable); the loader package is `shared/building/buildingdata/` (pandas / numpy / pyarrow only).

```python
import sys; sys.path.insert(0, 'shared/building')
import buildingdata as bd
sp  = bd.splits()                                     # frozen split file (proc/splits.json)
m   = bd.bdg2_meta(split='exam')                      # BDG2 buildings: spec fields, meters, q_ok per meter, split
e   = bd.bdg2('electricity', split='dev')             # hourly kWh (UTC index x buildings); bad hours are NaN
y   = bd.bdg2('steam', building='Hog_office_Lizzie', local=True)   # one building in its local time
f   = bd.bdg2_flags('electricity'); q = bd.bdg2_quality('electricity')   # hourly bitmask / per-meter quality table
s   = bd.bdg2_spec('Hog_office_Lizzie')               # building-spec dict (unknowns None + defaults_source)
w   = bd.site_weather('Hog'); wm = bd.site_weather_meta('Hog'); t = bd.tmy('Hog')   # actual-year + TMYx weather
r   = bd.resstock(columns=['bldg_id', 'weight', 'in.vintage']); c = bd.comstock()   # whole-stock metadata (MODELLED)
h   = bd.resstock_hourly(bldg_id=[...]); ch = bd.comstock_hourly(); sw = bd.stock_weather('G0800310')
x   = bd.recs(split='exam'); z = bd.cbecs(split='dev'); xf = bd.recs(full=True)   # EIA surveys (REAL, annual)
runs = bd.faults(split='exam'); ts = bd.fault_data('ers_rp1312', run_id='ERS_20070828_AHU-A'); bd.taxonomy()
o   = bd.other('list'); hue = bd.other('hue')         # other real datasets (HUE, WPuQ) + gated list
tf  = bd.tariffs(); p = bd.tariff_prices('pge_e_tou_c', 2017, tz='America/Los_Angeles'); b = bd.bill('pge_b19_sec', load_kW)
wp  = bd.wholesale_prices('ERCOT', 'HB_HOUSTON', '2023-01-01', '2024-01-01')   # GridData prices, read in place
bl  = bd.baselines(); bd.baselines('load_bdg2')        # exam numbers / per-unit tables
pri = bd.priors('office', '4A')                        # ComStock-based priors for unknown spec fields
```

**Rule for Phase 2:** only `dev` may be looked at for calibration, gates and any fitting. `exam` is for the final report
only. Simulated training may use real *inputs* (weather, tariffs) freely, and real-building *metadata* only from
non-exam units (BDG2 dev buildings; RECS/CBECS `prior` + `dev` respondents). NREL ResStock/ComStock are MODELLED data:
use them for priors and simulator-vs-simulator checks, never as a real exam.

## 1. What is where

| Path (under `shared/building/data/`) | What |
|---|---|
| `proc/bdg2/meters/<meter>.parquet` | BDG2 hourly readings, UTC (17,552 h x buildings), float32; kWh per hour (water, irrigation: litres) |
| `proc/bdg2/meters/<meter>_flags.parquet`, `_info.json` | uint16 quality bitmask per hour (section 3); per-meter findings (breakpoints, duplicates, T correlation) |
| `proc/bdg2/meter_quality.parquet` | one row per building-meter (3,053): flag fractions, coverage, annual totals, native resolution, q_ok |
| `proc/bdg2/buildings.parquet`, `specs.json` | 1,636 buildings in the building spec (unknowns None, `defaults_source`); q_ok per meter |
| `proc/bdg2/site_points.json` | 19 sites: coordinates (and how they were found), time zone, timestamp conventions |
| `proc/bdg2/weather/<site>.parquet/.json`, `index.json` | contract weather per site 2016-17 (UTC) + meta (sources, ASOS match, NASA bias) |
| `proc/tmy/<site>.parquet/.json` | TMYx 2011-2025 typical year for each site (climate.onebuilding.org) |
| `proc/stock/resstock_meta.parquet` | ResStock 2025 R1 baseline, whole stock: 549,971 dwellings x 324 columns (in.*, annual out.*) |
| `proc/stock/comstock_meta.parquet` | ComStock 2025 R3 baseline (national aggregate file): 149,272 rows = 91,464 unique models x their state allocations, 464 columns |
| `proc/stock/<name>_sample.parquet`, `<name>_hourly.parquet` | stratified samples (1,191 ResStock, 1,906 ComStock): hourly end uses for 2018 |
| `proc/stock/weather/<county>.parquet` | NREL AMY2018 county weather of the sampled buildings |
| `proc/stock/county_climate_zones.parquet`, `weatherfile_climate_zones.parquet` | IECC 2004 zone per county (from ResStock) |
| `proc/eia/recs2020.parquet`, `cbecs2018.parquet` | surveys mapped to the spec + annual energy by fuel (kWh) + weights |
| `proc/eia/recs2020_full.parquet`, `cbecs2018_full.parquet` | every public variable as published (codes, replicate weights) |
| `proc/faults/runs.parquet`, `data/<dataset>.parquet` | fault runs (unit-days) and their trend logs in the AHU/RTU contract |
| `proc/faults/taxonomy.json`, `channel_map.json` | unified fault taxonomy; source-column -> contract mapping |
| `proc/other/` | HUE (BC homes), WPuQ (German heat-pump homes), `catalogue.json` (incl. gated list) |
| `proc/tariffs/tariffs.json` | 19 URDB tariffs (residential TOU, commercial TOU + demand) |
| `proc/splits.json` | frozen splits + sha256 |
| `proc/baselines/` | per-unit baseline tables + `summary.json` (section 9) |
| `raw/` | downloads kept for rebuild (BDG2 CSVs gzipped, ASOS, NASA POWER, EIA, faults, URDB, TMYx, HUE, WPuQ 60-min) |

Disk: 2.7GB in total (budget 5 GB); hourly stock samples are the largest part (ComStock 584 MB, ResStock 220 MB).

## 2. Sources, licences, citations

| Source | Licence | Citation / URL |
|---|---|---|
| Building Data Genome 2 (meters, metadata, weather) | **CC BY-SA 4.0** (repository LICENSE file; derived data must stay share-alike) | C. Miller et al., "The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition", Sci. Data 7, 368 (2020), doi:10.1038/s41597-020-00712-x; github.com/buds-lab/building-data-genome-project-2 (LFS files via media.githubusercontent.com) |
| BDG2 weather (inside BDG2) | NOAA public | NOAA NCEI Integrated Surface Database, ISD-Lite |
| ResStock 2025 Release 1 (AMY2018) | CC BY 4.0 | NREL (2025), ResStock 2025 R1, OEDI doi:10.25984/1876417, s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2025/resstock_amy2018_release_1/; Reyna et al. (2025) ResStock Technical Reference Documentation |
| ComStock 2025 Release 3 (AMY2018) | CC BY 4.0 | NLR (2025), ComStock 2025 R3, OEDI doi:10.25984/1876417 (…/2025/comstock_amy2018_release_3/); Parker et al. (2023) ComStock Reference Documentation, NREL/TP-5500-83819 |
| EIA RECS 2020 public microdata v7 | US public domain | U.S. EIA, eia.gov/consumption/residential/data/2020/ (csv/recs2020_public_v7.csv + codebook) |
| EIA CBECS 2018 public microdata | US public domain | U.S. EIA, eia.gov/consumption/commercial/data/2018/ (xls/cbecs2018_final_public.csv + codebook) |
| LBNL FDD data synthesis (MZVAV, SZCAV, SZVAV, RTU) | CC0 | J. Granderson, G. Lin et al., LBNL, figshare doi:10.6084/m9.figshare.11752740 (inventory PDF documents every set) |
| Labeled AHU fault datasets (ASHRAE RP-1312 ERS, Drexel Nesbitt Hall, Modelica sets) | CC BY 4.0 | N. Ghalamsiah (2025), figshare doi:10.6084/m9.figshare.29297999.v3; ERS data from J. Wen & S. Li, ASHRAE RP-1312 (2011); Nesbitt from Y. Chen, PhD thesis, Drexel (2019) |
| Korean office AHU BMS data | CC BY 4.0 | S. Wang (2024), figshare doi:10.6084/m9.figshare.25932409.v2 |
| HUE hourly residential electricity, BC | CC BY 4.0 | S. Makonin, "HUE: The Hourly Usage of Energy Dataset for Buildings in British Columbia", Data in Brief 23 (2019); Harvard Dataverse doi:10.7910/DVN/N3HGRN |
| WPuQ heat-pump households, Germany | CC BY 4.0 | M. Schlemminger et al., Sci. Data 9, 56 (2022); Zenodo doi:10.5281/zenodo.5642902 |
| NASA POWER hourly (MERRA-2, CERES SYN1deg) | free, cite NASA POWER | power.larc.nasa.gov/api/temporal/hourly/point (NASA LaRC POWER Project) |
| ASOS / METAR hourly | free use, credit IEM | Iowa Environmental Mesonet, mesonet.agron.iastate.edu/cgi-bin/request/asos.py |
| TMYx 2011-2025 EPW | free with attribution (17 Nov 2026 note: the site's footer says "All Rights Reserved" and states no reuse licence, so the repository does not redistribute TMYx files; `buildingsim/refdata/` and `proc/tmy/` are git-ignored) | climate.onebuilding.org (D. Lawrie, D. Crawley), derived from NOAA ISD |
| U.S. Utility Rate Database (bulk CSV) | CC0 (OpenEI) | OpenEI URDB, apps.openei.org/USURDB/download/usurdb.csv.gz (downloaded 16 Nov 2026) |
| ERCOT / CAISO wholesale prices | as in GridData | read in place from `shared/grid/data/proc/prices` via `bd.wholesale_prices` (no copy) |

Attribution text required by NREL: "Data includes information from the ResStock™ / ComStock™ datasets developed by the
National Renewable Energy Laboratory (NREL; now National Laboratory of the Rockies) with funding from the U.S. DOE."

## 3. BDG2 meters (LoadCast, BuildingDetective)

- **Coverage.** 1,636 non-residential buildings at 19 sites (North America and Europe), 3,053 meters, 2016-2017 hourly:
  electricity 1,578, chilled water 555, steam 370, hot water 185, gas 177, water 146, irrigation 37, solar 5. Primary
  use (mapped to the spec): education 617, office 307, other 268, assembly 207, lodging 168, healthcare 29,
  warehouse 15, retail 12, lab 7, food 6. Floor area for all; year built for 50%; floors 27%; heating type 13%.
- **Units.** BDG2 converted everything to kWh (water, irrigation in litres). **Eagle chilled water and hot water are
  ~1000x too large** (EUI medians 195,709 and 97,305 kWh/m2/yr vs 50-800 elsewhere): consistent with kBTU readings run
  through BDG2's mmBTU factor. Divided by 1,000 here (`unit_factor_applied` = 0.001 in the quality table).
- **Time (checked, not assumed).**
  - Meter stamps are **local clock time with DST** at every site except **Eagle, Moose, Shrew (local standard time)**.
    Evidence: at DST sites the non-existent spring hour is NaN and the fall-back hour shows doubled readings; a per-building
    vote on the weekday profile shift across each DST change (two weeks either side) gives "clock" for 14 sites and
    "standard" for Eagle (272 vs 14), Moose (37 vs 1), Shrew (15 vs 4). **Rat** and **Wolf** have DST hallmarks
    (clock-stamped) but most of their buildings' HVAC schedules stay on standard time (BMS clocks not adjusted).
  - **Fox is Arizona** (no DST) although BDG2 says US/Mountain: tz America/Phoenix.
  - BDG2 weather stamps are **local standard time** at all sites (BDG2 wiki: "DST ignored"): Panther vs KMCO in UTC
    matches to 0.03 degC at a constant +5 h in winter and summer.
  - Everything is stored on one **UTC hour-beginning** index (2016-01-01 00:00 .. 2018-01-01 07:00 UTC). The spring-forward
    reading is dropped; the fall-back reading is put on the first (DST) UTC hour and both UTC hours are flagged DST.
- **Native resolution.** `native_res` per building-meter: a day (month) counts as coarse when its readings take <= 2
  distinct values. **All 71 Gator electricity meters are daily totals spread evenly over 24 hours** (and 5 electricity
  meters are monthly); Panther irrigation/water, Cockatoo steam (22) and a few others are daily too. These are not
  hourly labels: they are excluded from hourly q_ok but kept for daily tasks (`q_ok_daily`).
- **Hourly flags** (bitmask, `bd.FLAGS`): MISSING 1, ZERO 2, ZERO_RUN 4 (zero for >= 24 h), NEGATIVE 8, SPIKE 16
  (> 8 robust sigma above the +-12 h median and > 2x it, or > 10x the meter's 99.9th percentile), FLATLINE 32 (identical
  non-zero value >= 24 h), INTERP 64 (exactly linear for >= 12 h: gap fill), LEVEL_SHIFT 128 (14-day median > 3x or < 1/3
  of the whole-record median; electricity and water only, possible meter swap / replacement / unit change), BDG2_OUTLIER
  256 (BDG2 authors' cleaned set removed a non-zero reading: none found - their cleaning removed zeros only), DST 512,
  SEASON_SIGN 1024 (whole meter: chilled water anti-correlated with temperature or heating meter correlated, r > 0.3:
  a swapped meter), ZERO_RUN_INSEASON 2048 (thermal meter zero >= 24 h while it should run: chilled water on days
  > 18 degC, heating meters on days < 10 degC).
- **Bad hours per meter type** (`bd.BAD_MASK`): electricity = missing, zero, zero runs, negative, spike, flatline,
  interp, level shift, DST; thermal meters = the same minus single zeros and level shifts, with in-season zero runs
  only (seasonal shutdowns are real behaviour); irrigation/solar: zeros are normal.
- **Per-meter quality** (`meter_quality.parquet`): `q_ok_<year>` = >= 85% good hours in that local year, native hourly,
  < 30 level-shift days, no season-sign flag, not a duplicate series, EUI plausible (electricity 10-2000, thermal
  < 3000 kWh/m2/yr). `q_ok` = both years ok AND 2017/2016 ratio (same calendar days) within 0.4-2.5. `q_ok_daily` =
  the same rules for daily use (flatlines allowed, native daily accepted).

| meter | n | q_ok (both years, hourly) | q_ok_daily | q_ok 2016 / 2017 | median bad hours | notes |
|---|---|---|---|---|---|---|
| electricity | 1,578 | 903 | 987 | 1,056 / 1,124 | 4.9% | 257 with level-shift days, 40 implausible EUI, 198 with 2017/2016 jump |
| chilled water | 555 | 343 | 350 | 370 / 481 | 2.8% | 10 implausible EUI (after the Eagle fix) |
| steam | 370 | 235 | 238 | 251 / 299 | 4.7% | 3 season-sign suspects |
| hot water | 185 | 77 | 83 | 107 / 108 | 7.0% | Fox hot water EUI ~2,250 (possibly ~10x high; not changed) |
| gas | 177 | 28 | 56 | 88 / 32 | 15% | Lamb gas (119 meters) mostly fails 2017; Panther/Wolf gas EUI high |
| water | 146 | 8 | 8 | 9 / 10 | 29% | unreliable; 75 with level shifts |
| irrigation, solar | 37, 5 | 2, 2 | 2, 2 | | | seasonal / night zeros are normal |

## 4. Building specs, sites, weather

- **Spec mapping** (`bd.bdg2_spec`): id, site_id, lat/lon (site), climate_zone, primary_use (BDG2 primary space usage ->
  spec: Education->education, Office->office, Entertainment/public assembly and Religious worship->assembly,
  Lodging/residential->lodging, Healthcare->healthcare, Warehouse/storage->warehouse, Retail->retail,
  Technology/science->lab, Food sales and service->food, Public services/Parking/Manufacturing/Services/Utility/Other->
  other; BDG2 strings kept in `primary_use_bdg2`, `sub_use_bdg2`), floor_area_m2, n_floors, year_built, occupants,
  energystar_score, `meters` (which whole-building meters exist), `hvac_hint` (e.g. "central plant chilled water +
  steam" when a building has those meters), heating_type_bdg2. Envelope, internal gains, schedules, setpoints,
  ventilation, HVAC system and DHW are unknown in BDG2: None, with `defaults_source` pointing to `bd.priors()`
  (ComStock 2025R3 distributions by building type and climate zone).
- **Site coordinates.** BDG2 gives one coordinate per site (city level). Four sites have none; they were located by
  matching BDG2's ISD temperatures to candidate ASOS airports in the site's time zone (2016, exact-value matching):
  **Bobcat = Ogden UT (KOGD, 96% of hours identical)**, **Eagle = Trenton NJ (KTTN, 93%)**, **Gator = Orlando FL
  (KMCO, 93%; same weather as Panther)**, **Swan = Mountain View CA (KNUQ, 96%)**. **Wolf's longitude has the wrong sign**
  in BDG2 (+6.26 = North Sea): fixed to -6.26 (Dublin).
- **Climate zone.** US sites: IECC 2004 zone of the site's county (ResStock county table): Bear 3C, Swan 3C, Bobcat 5B,
  Bull 2A, Cockatoo 6A, Eagle 5A, Peacock 5A, Fox 2B, Gator 2A, Panther 2A, Hog 6A, Rat 4A. Others from 2016-17 site
  weather with ASHRAE 169 thermal thresholds (moisture regime A assumed): Crow/Moose (Ottawa) 6A, Lamb (Cardiff) 4A,
  London sites 4A, Wolf (Dublin) 4A.
- **Weather frames** (`bd.site_weather`, UTC): `T_out, dewpoint, wind_speed, wind_dir, cloud, precip_mm,
  pressure_sealevel` from BDG2 (ISD-Lite), gaps filled from the best-matching ASOS station (IEM) and then NASA POWER
  (`T_out_src` 0/1/2); `RH` from T and dewpoint; `GHI, DNI, DHI, GHI_clearsky, LW_down, pressure` (surface, Pa) from
  NASA POWER (CERES irradiance). Kept for checks: `T_out_asos`, `T_out_nasa`, `RH_nasa`. The matched ASOS station is
  evidently the ISD station BDG2 used at 11 US sites (MAE 0.05-0.12 degC, 83-99% of hours identical); Ottawa and the
  European METARs report whole degrees (MAE 0.34-0.71 degC); Peacock matches nothing nearby (2.5 degC, see below).
- **NASA POWER is cold-biased** at these sites: mean T_nasa - T_bdg2 = -0.02 (Orlando) to -2.5 degC (Ogden), MAE 1.1-3.1
  degC. Use the station T for loads; NASA only for irradiance and fill.
- **Peacock weather does not belong to Princeton.** Its BDG2 weather averages 15.3 degC (Trenton TMYx: 13.1 degC) and
  matches no nearby station (TTN MAE 2.95 degC at the best time offset; best of ~60 eastern airports is Richmond VA at
  1.5 degC). Either the weather or the site location is wrong. Peacock (47 buildings) is kept but flag it in any
  weather-driven grading.
- **TMYx** (`bd.tmy(site)`): TMYx 2011-2025 of the matched airport (Oakland, Moffett, Ogden-Hill AFB, Austin Camp Mabry,
  Ithaca, Trenton, Phoenix, Orlando, Minneapolis, Reagan National, Ottawa, Cardiff, Heathrow, Dublin), EPW hour-ending
  LST -> `time_lst` hour-beginning in dummy year 2001.

## 5. Other real datasets

| name | what | units | licence | grading use |
|---|---|---|---|---|
| `hue` | 28 homes, Metro Vancouver / Victoria, hourly electricity 2012-2020 (615,841 house-hours), house type, facing, heating/cooling system codes; airport weather YVR/WYJ | kWh, degC, %, Pa | CC BY 4.0 | BuildingDetective (main heat: 6 electric baseboard, 1 air-source HP with gas backup, 1 ground-source HP; 10 gas furnace, 9 radiant - mostly gas boiler; 1 unknown), LoadCast residential |
| `wpuq` | 38 single-family homes with heat pumps, Lower Saxony, Germany, 2018-2020: 80 hourly series (38 heat pumps, households, transformer, 3 PV inverters; P_TOT active power), 26,023 h of local weather (T, RH, GHI, wind, pressure hPa) | W, degC | CC BY 4.0 | heat-pump electricity vs temperature (ThermoPilot / Detective sensitivity), real noise for residential loads |
| `korea_office` (in faults) | 20 AHUs of a large Korean office, hourly BMS, Nov 2023 - May 2024 | degC, 0-1 | CC BY 4.0 | **no labels in the public files** (the description promises expert labels): real background operation only |

HUE times are converted from local hour-ending to UTC hour-beginning. HUE meters electricity only (gas from FortisBC is
not in the data). House 7 has no metadata. WPuQ: 10-s and 1-min files (2-9 GB each) were not taken; heat output is not
metered.

**Gated or not taken** (for the owner; `bd.other('list')['_gated_or_not_taken']`): ecobee Donate Your Data (research
application), Pecan Street Dataport (paid / licence), UK RHPP heat-pump trial (UK Data Service SN 8151, registration),
UK Electrification of Heat (UKDS SN 9050, registration), Irish CER smart meters (ISSDA application), IDEAL homes
(open, CC BY, but >10 GB), Low Carbon London smart meters (open, ~10 GB, no per-home weather), LBNL FDD full sets
(faultdetection.lbl.gov form; section 7).

## 6. NREL ResStock / ComStock (MODELLED - priors and simulator check only)

- **Releases:** the latest documented ones on OEDI (16 Nov 2026; the `2026/` prefix is empty): ResStock 2025 Release 1
  and ComStock 2025 Release 3, both actual meteorological year 2018. Baseline (upgrade 0) only.
- **Metadata, whole stock:** ResStock 549,971 dwelling models (324 columns: every `in.*` characteristic, out.params,
  annual energy by fuel and end use, loads, capacities, unmet hours, bills); weights sum to the 2018 US housing stock.
  ComStock national aggregate file 149,272 rows (464 columns; **one row per model and state allocation: 91,464 unique `bldg_id`s**, each model's weight split over the states it represents; the same model's time series exists under every state): building type, HVAC system, envelope, schedules, setpoints,
  annual energy by fuel/end use, monthly electricity, peak demand, bills). Read over HTTPS with column projection; strings
  dictionary-encoded.
- **Hourly sample:** stratified by (RECS building type x IECC 2004 zone) for ResStock (1,191 homes in 85 strata, 589
  counties) and (ComStock building type x IECC 2006 zone) for ComStock (1,982 draws in 251 strata -> 1,906 unique models after removing models drawn twice under two state allocations; `n_sampled_allocations`);
  per stratum n = clip(round(share x 1500), 4, 60), drawn with probability proportional to weight (seed 20261116).
  ResStock columns: electricity/gas/oil/propane totals, heating/cooling/fans/HP-backup/DHW end uses, PV, delivered
  heating/cooling load, indoor temperature and RH, heating/cooling setpoint schedules, outdoor dry bulb, occupancy,
  unmet hours. ComStock: every end-use column (kWh). 15-min period-ENDING local standard time -> hourly sums (states:
  means) over (h, h+1], labelled h (`time_lst`). `std_utc_offset_h` from the state (approximate for split-zone states).
- **Weather:** the NREL AMY2018 county files for every sampled county (`bd.stock_weather`), hour-beginning LST: 1,014
  counties (3 Alaska county files are not published on OEDI). ResStock gives Alaska/Hawaii counties by NAME in
  `in.county`; their GISJOIN (first part of `in.county_and_puma`) was put in the sample's `county` column.

## 7. EIA RECS 2020 and CBECS 2018 (REAL, annual)

- **RECS 2020 v7:** 18,496 households (weights sum to 123.5 M homes). Mapped: state, division, IECC code, Building
  America zone, HDD/CDD (2020 and 30-yr), housing type -> residential_single / residential_multi, floor area (total,
  heated, cooled; m2), stories, year-built range, foundation, wall material, glazing, window frame, number of windows,
  reported insulation, draftiness, heating equipment and fuel -> spec `system` (split_ac_furnace 8,885, heat_pump_air
  2,179, radiant 1,510, baseboard_resistance 1,377, heat_pump_ground 193, none 218, unmapped 4,134), cooling equipment,
  equipment ages, water-heater fuel, thermostat control, **thermostat settings (home / away / night, winter and
  summer, degC)**, household size, PV; annual electricity, gas, propane, oil, wood (kWh) and EIA's end-use estimates.
- **CBECS 2018:** 6,436 buildings (weights sum to 5.92 M). Mapped: region, division, PUBCLIM climate class (5 classes;
  finer IECC zone not public), HDD/CDD 2018, principal activity -> spec primary_use, floor area, floors (top-coded 10-14 /
  15+), construction year range, wall/roof material, glass %, window type, cool roof, main heating/cooling equipment ->
  spec `system` (packaged_rtu 3,084, vav_chiller_boiler 957, split_ac_furnace 375, heat_pump_air 360, ptac 359,
  district_steam_chw 282, none 237, unmapped 767), equipment presence flags (boiler, chiller, packaged, heat pump,
  furnace), economizer, BAS, occupancy sensors, hours per week, workers; annual electricity, gas, oil, district heat,
  total (kWh) and EIA's modelled end uses.
- **Read the labels right:** fuel totals are EIA's billing-based estimates (RECS `PERIODEL`/`PERIODNG` say how many
  billing days exist; `elec_from_bills`); end-use splits are EIA MODEL estimates, not measurements. CBECS district heat
  is derived. Both surveys are annual only: no hourly or monthly data.

## 8. HVAC fault data (FaultFinder)

- **Real or experimental (graded):**
  - `ers_rp1312` - ASHRAE RP-1312 at the Iowa Energy Resource Station: 49 fault days on AHU-A (summer 2007, winter 2008,
    spring 2008), each with the twin AHU-B running fault-free the same day (98 runs, 1-min, 162 points -> contract).
  - `lbnl_mzvav2_ers_exp` - the same ERS AHU-A on 10 unfaulted days 2008-09 (its 3 leaking-valve days are byte-identical
    to ERS days and are marked `duplicate_of`).
  - `lbnl_szcav_flexlab`, `lbnl_szvav_flexlab` - LBNL FLEXLAB X3A single-zone AHU, March and September 2017 (26 days).
  - `lbnl_rtu_ornl` - ORNL Flexible Research Platform 2 with a 12.5-ton Trane RTU and 10 VAV zones, 2017-18 (19 gradable
    days): condenser fouling, setback/schedule errors, infiltration, thermostat bias.
  - `nesbitt` - Nesbitt Hall, Drexel University (a real occupied building), 20 fault days (3 natural, 17 injected) and
    22 baseline days, 5-min, AHU-1 and AHU-2 kept separately (84 runs); fault windows from the dataset's table.
- **Simulated (never graded, `sim_reference`):** ERS HVACSIM+ twins (68), LBNL MZVAV-1 (PNNL model, 189 days), MZVAV-2-2
  (26), Ghalamsiah Modelica sets 03 RBC-5wk, 05 G36-5wk, 07 G36-Cyber (indexed; raw Modelica point names), 08 G36-HIL
  (hardware-in-the-loop with a simulated building: not counted as real). 04 G36-1wk (5 GB) and 06 G36-Degrad (0.8 GB)
  were not downloaded (simulated).
- **Contract channels** (`channel_map.json`; proposed by B for A's trend logs): `T_oa T_ra T_ma T_sa T_sa_sp`
  (`T_sa_sp_heat/cool` for single-zone units) degC; `RH_oa RH_ra RH_sa` %; `damper_oa damper_ra damper_ea valve_cc
  valve_hc fan_sa_speed fan_ra_speed` 0..1; `fan_sa_status fan_ra_status occ comp1_status comp2_status` 0/1;
  `fan_sa_power_kW fan_ra_power_kW power_kW`; `flow_sa_m3s flow_oa_m3s flow_ra_m3s`; `p_duct_Pa p_duct_sp_Pa`;
  `T_chw_in T_chw_out T_hw_in T_hw_out T_zone` degC; extras prefixed `x_`. Per row: `fault` (0/1; -1 unlabelled).
  Conversions found the hard way: ERS heating-valve and return-damper signals are inverted (100 = closed; checked
  against LBNL's normalised copy of the same day, r = -1.00); RTU electricity is Wh per minute (x 0.06 -> kW).
- **Taxonomy** (`bd.taxonomy()`): oa/ea damper stuck or leaking, cooling/heating valve stuck or leaking, supply duct leak,
  heating coil fouling / reduced capacity, return fan failure / fixed speed, air filter blockage, unstable control,
  reverse-acting valve, sensor bias (T_sa, T_oa, T_chw, dp_chw, OA flow, zone T), chiller off, HVAC schedule fault,
  lighting schedule fault, condenser fouling, excessive infiltration (+ cyber attack and degradation for sims).
  Gradable exam counts by type: cc_valve_stuck 15, oa_damper_stuck 14, schedule_fault 8, ea_damper_stuck 7, hc_valve_leak
  6, chiller_off 6, control_unstable 5, oa_damper_leak 5, hc_valve_stuck 5, and 1-4 each for 18 others.
- **Label problems:** the RTU file's ground-truth column is 1 on every row (also on unfaulted days): labels come from the
  inventory's dated table; 2017-12-03 is listed both as a fault and as unfaulted (excluded), 2017-12-10 is in the file
  but in no table (excluded; probably the "12/3" unfaulted day). Nesbitt faults are active only inside their windows;
  plant- and system-level faults (chiller off, CHW setpoint, schedule) are labelled on both AHUs.
- **LBNL's own site** (faultdetection.lbl.gov) puts every download behind a name/email form, and OEDI submission 5763
  (its DOI) returns no data to this client: the full LBNL sets (RTU field data, SD-AHU, chiller, boiler, FCU, VAV
  simulations) were not taken. If the owner fills the form, the field RTU data would be a second real RTU exam.

## 9. Splits (`proc/splits.json`, seed 20261116)

- **BDG2 - unit = building, stratified by (primary_use, climate_zone).** dev = 2 buildings drawn per stratum that has
  >= 8 buildings with at least one q_ok meter: **52 dev buildings** in 26 strata (assembly, education, lodging, office,
  other in 2A/2B/3C/4A/5A/6A; healthcare 5A). exam = the other 1,584. Hourly-label coverage: electricity 47 dev / 856
  exam q_ok meters; chilled water 20 / 323; steam 10 / 225; hot water 5 / 72; gas 2 / 26.
  *Why:* a handful per stratum is enough for calibration and dev gates; everything else stays unseen. Building, not
  site, because there are only 19 sites (two of which share weather); site-level effects (time conventions, Gator's daily
  data) are documented instead.
- **BDG2 time split** for models that need history: **history = local calendar 2016, forecast = local calendar 2017**
  (each building's own time zone). Grade only hours that are good in the label meter's flags.
- **RECS / CBECS - unit = respondent**, stratified by (division, housing type) / (division, principal activity):
  prior 50% (may feed generator priors / simulated training), dev 10%, exam 40%. RECS 9,247 / 1,848 / 7,401; CBECS
  3,217 / 661 / 2,558. Weights refer to the whole sample: multiply by 1/fraction for population totals within a split.
- **Faults - unit = run (unit-day), grouped by facility + date** so a day never spans splits (an ERS fault day and its
  AHU-B baseline move together). dev = seeded draw: ERS one fault day per season (2007-08-24 unstable cooling valve
  control, 2008-02-03 and 2008-05-10 exhaust damper stuck) + their AHU-B days, 1 ERS normal day, FLEXLAB SZCAV 1 fault
  day, SZVAV 1 fault + 1 normal, ORNL RTU 1 fault + 1 normal, Nesbitt 2 fault days + 2 baseline days (both AHUs) =
  **20 dev runs; 217 exam runs** (141 experimental-lab, 76 real-building). Simulated / HIL = `sim_reference`; Korea =
  `noise_unlabelled`; ambiguous = `excluded`.
- **NREL stock samples:** reference only (modelled), free for priors.

## 10. Data problems Phase 2 must know

1. **Mixed timestamp conventions in BDG2** (section 3): fixed to UTC here; Rat and Wolf schedules follow standard time.
2. **Gator electricity (71 meters) is daily data** spread over 24 h; 5 more meters are monthly. Not hourly labels.
3. **Eagle chilled/hot water unit error** (/1000 applied). Fox hot water, Panther and Wolf gas EUIs remain high.
4. **Peacock weather does not match its location** (section 4). Wolf longitude sign fixed; 4 sites located by weather.
5. **Thermal meters are hard labels:** seasonal shutdowns, campus plants serving several buildings, low q_ok for gas
   (28 meters) and hot water (77). 198 electricity meters jump by > 2.5x between 2016 and 2017.
6. **NASA POWER is 0-2.5 degC cold** vs station data at BDG2 sites.
7. **RECS/CBECS are annual:** end uses are EIA model estimates; CBECS climate is a 5-class field.
8. **Fault labels:** RTU ground-truth column unusable (table used), one ambiguous and one unlisted day excluded; the
   ERS fault days appear twice in public sources (deduplicated); fault presence is day-level except Nesbitt (windows).
   Only 2 real "real-building" facilities (Nesbitt; Korea unlabelled).
9. **ResStock/ComStock are EnergyPlus models**, not measurements. ComStock's aggregate file repeats a model once per state
   allocation (149,272 rows, 91,464 models): de-duplicate on `bldg_id` before counting models. Hourly sums match the
   annual metadata to < 0.1% (checked on 6 buildings); stored floats keep 12 mantissa bits (<= 0.024% rounding).
10. **URDB tariffs are as entered:** some are delivery-only (e.g. NYSEG energy 0.066 $/kWh), APS E-32 L shows 0.57 $/kWh
    energy rates (likely an entry error); `bd.bill` uses the first tier only.

## 11. Tariffs (ThermoPilot)

`bd.tariffs()` - 19 current URDB rates (start dates 2024-2026), each with URDB label/URI, utility, sector, climate zone,
BDG2 sites in the territory, energy periods/tiers with 12x24 weekday/weekend schedules, TOU and flat demand charges,
fixed charges. Residential TOU: PG&E E-TOU-C (3C), SCE TOU-D-PRIME and TOU-D-5-8PM (3B), APS R-3 with demand charge and
SRP E-26 (2B), Austin Energy TOU (2A), Duke RT (2A), Xcel CO RE-TOU and Rocky Mountain Power UT Schedule 1 TOU (5B),
Xcel MN A04 (6A), NYSEG SC-12 (6A), PSE&G RS-TOU-3P (5A), Con Edison SC-1 Rate III (4A). Commercial TOU + demand:
PG&E B-19, SCE TOU-GS-3 B, APS E-32 L, OUC GSD option B, Xcel MN A15, Pepco GTLV. `bd.tariff_prices(key, year, tz)`
gives hourly energy prices and period ids; `bd.bill(key, load_kW)` the bill. Wholesale ERCOT/CAISO prices come from
GridData in place (`bd.wholesale_prices`).

## 12. Baselines (exam numbers; `proc/baselines/summary.json`)

### Hourly load, BDG2 exam building-meters with q_ok, fit on local 2016, graded on local 2017
ASHRAE Guideline 14 CV(RMSE)% and NMBE% per building-meter (n - p in denominators); median [IQR] over meters.

| meter (n) | persistence 1 h | persistence 24 h | same hour last week | TOWT (fit 2016) | G14 change-point, hourly | G14 change-point, daily | G14 change-point, monthly |
|---|---|---|---|---|---|---|---|
| electricity (856) | 9.6 [6.5-14.2] | 18.6 [12.3-26.4] | 19.5 [13.1-27.3] | 19.6 [13.7-27.9] | 19.1 [13.4-27.3] | 17.2 [12.5-25.2] | 11.0 [7.1-17.7] |
| chilled water (323) | 18.8 | 40.3 | 51.8 | 45.7 | 42.4 | 32.4 | 21.6 |
| steam (225) | 24.9 | 41.9 | 56.6 | 39.8 | 40.5 | 30.6 | 24.2 |
| hot water (72) | 52.8 | 69.4 | 93.8 | 83.1 | 82.6 | 59.5 | 49.6 |
| gas (26) | 64.3 | 87.7 | 94.1 | 95.7 | 95.8 | 62.0 | 41.5 |

- Change-point model: daily mean kW vs daily mean outdoor T on 2016, best of 2P / 3P heating / 3P cooling / 4P / 5P by
  BIC (electricity picks 4P 353, 2P 246, 3PC 152, 5P 57, 3PH 48); hourly = daily prediction x the 2016 time-of-week
  shape. TOWT = 168 time-of-week indicators + piecewise-linear temperature (knots 4.4/12.8/18.3/26.7/32.2 degC).
  Median |NMBE| for electricity: 4.6% (CP) / 4.7% (TOWT); persistence ~0.
- G14 calibration criteria met (hourly CV <= 30% and |NMBE| <= 10%): electricity 71% of meters (CP), 70% (TOWT);
  chilled water 24%; steam 23%; hot water and gas ~0%. Monthly criteria (15% / 5%): electricity 38%.
- Dev (47 electricity meters, for reference): persistence 24 h 16.4, TOWT 22.3, CP 22.4.
- **A useful LoadCast must beat TOWT / change-point on the day-ahead-with-actual-weather task (19-20% for electricity)
  and must be compared with persistence 24 h (18.6%), which is hard to beat for weather-insensitive buildings.**

### Detective
- **PRISM** change-point fits on 2017 daily data for every q_ok_daily building-meter (`baselines('prism_bdg2')`):
  balance points, heating/cooling slopes (W/m2K of floor), base load (W/m2). Exam medians, electricity: Tb_heat 15.5
  degC, Tb_cool 16.5 degC, heating slope 0.02, cooling slope 0.15 W/m2K, base 12.9 W/m2; steam heating slope 1.34
  W/m2K (Tb_heat 14.5 degC); chilled water cooling slope 4.3 W/m2K (Tb_cool 13.0 degC). Predicting a building's PRISM
  parameters by the **dev median of its use type**: MdAE Tb_heat 5.0 K, Tb_cool 5.5 K, electric cooling slope 0.29
  W/m2K, steam heating slope 1.76 W/m2K - i.e. use type alone says little.
- **BDG2 floor area** from annual electricity / dev median EUI of the use type: MdAPE 46%, 66% within a factor 2 (1,133
  exam buildings); dumb dev median area: 64%.
- **BDG2 primary use** from the 2017 weekly load shape (nearest dev centroid): accuracy 34% / macro-F1 0.20 vs majority
  class (education) 39% / 0.06 (1,082 exam buildings, 10 classes).
- **BDG2 year built:** dev median 1969, MAE 24.9 years (797 exam buildings).
- **RECS (7,401 exam homes):** floor area from total energy / dev EUI by type x IECC zone MdAPE 33% (dumb dev median by
  type: 29% - homes vary little in size); annual energy from area + HDD + CDD (per-type dev regression) MdAPE 36%;
  main heating fuel by "largest non-electric fuel used, else electricity" 84% vs majority (gas) 52%; **heat pump vs
  electric resistance from annual kWh/m2/HDD: balanced accuracy 0.50 (no skill)** - annual data cannot separate them.
- **CBECS (2,558 exam):** floor area MdAPE 39% (dumb 76%); annual energy from characteristics 38%; heating fuel rule
  73% vs majority 51%.

### Faults (run = unit-day; threshold tuned on dev runs only)
APAR-style rules (Schein et al. 2006; 12 rules: heating/economizer/mechanical-cooling mode rules, simultaneous heating
and cooling, mixed-air temperature outside the OA-RA range), each must hold 15 min while the fan runs (and occupied);
score = share of operating samples with any rule; fault if score > 0.232 (dev).

| exam | runs (faulty) | accuracy | TPR | FPR | F1 | balanced acc. | ROC AUC |
|---|---|---|---|---|---|---|---|
| AHU, all real/experimental | 200 (91) | 0.68 | 0.32 | 0.02 | 0.48 | 0.65 | 0.69 |
| majority class ("no fault") | 200 | 0.545 | 0 | 0 | 0 | 0.50 | - |
| ERS RP-1312 | 92 | | | | | 0.66 | 0.74 |
| FLEXLAB SZCAV / SZVAV | 14 / 9 | | | | | 0.77 / 0.75 | 0.77 / 0.67 |
| Nesbitt Hall (real building) | 76 | | | | | 0.57 | 0.54 |
| RTU (ORNL), rule "fan on while unoccupied" | 17 (12) | 0.47 | 0.33 | 0.20 | 0.47 | 0.57 | 0.72 |
| RTU majority ("fault") | 17 | 0.71 | 1 | 1 | 0.83 | 0.50 | - |

Rules catch stuck valves (cc 79%, hc 100%) but not sensor biases, schedule faults, damper stuck at a plausible
position, or most Nesbitt faults. FaultFinder should beat ROC AUC 0.69 (AHU) with a low false-alarm rate.

### Thermostat (ThermoPilot)
1R1C house (UA 200 W/K, C 20 MJ/K, tau 28 h, 4 m2 effective solar aperture, 500 W internal gains) with an air-source
heat pump limited to 8 kW thermal (COP_heat 1.8 + 0.07 (T_out + 10) in [1.5, 4.5]; COP_cool 5.5 - 0.1 (T_out - 20) in
[2, 5.5]); real 2017 site weather and real tariffs; annual bill incl. demand and fixed charges. Not a calibrated home:
ThermoPilot should re-run the same three controllers on BuildingSim. Controllers: **fixed** 20 / 24 degC; **setback**
heat 20 home / 16 night and weekday-away, cool 24 / 26 night / 28 away; **pre-cool rule** 22 degC for 2 h before the
summer weekday peak window, 26.5 during it, else 24.

| tariff (weather) | fixed $ / kWh | setback $ / kWh | pre-cool $ / kWh |
|---|---|---|---|
| PG&E E-TOU-C (Berkeley) | 681 / 1,275 | 661 / 1,157 | 711 / 1,407 |
| SRP E-26 (Tempe) | 917 / 4,454 | 950 / 4,144 | 811 / 4,331 |
| APS R-3 demand (Tempe) | 895 / 4,454 | 1,044 / 4,144 | 639 / 4,487 |
| Austin Energy TOU (Austin) | 551 / 3,022 | 527 / 2,778 | 527 / 3,063 |
| Duke RT (Orlando) | 448 / 2,335 | 552 / 2,101 | 486 / 2,561 |
| RMP UT Sched. 1 TOU (Ogden) | 846 / 6,106 | 1,001 / 5,372 | 807 / 6,192 |
| Xcel MN A04 (Minneapolis) | 1,155 / 7,644 | 1,091 / 6,751 | 1,089 / 7,603 |
| NYSEG SC-12 (Ithaca) | 788 / 6,453 | 736 / 5,667 | 788 / 6,453 |
| PSE&G RS-TOU-3P (Trenton) | 366 / 4,448 | 457 / 3,920 | 342 / 4,522 |
| Con Ed SC-1 R III (Trenton weather) | 1,399 / 4,448 | 1,301 / 3,920 | 1,255 / 4,354 |
| Xcel CO RE-TOU (Ogden weather) | 1,018 / 6,106 | 1,128 / 5,372 | 974 / 6,156 |

Setback saves 7-12% of energy but **raises the bill on 6 of 11 TOU tariffs** (recovery lands in priced hours or sets
demand peaks) and costs comfort (median 1,155 K h outside 20-24.5 degC while home, vs 0 fixed); the pre-cool rule cuts
the bill on 8 of 11 for 68 K h discomfort. A good ThermoPilot should beat the best rule per tariff at no more discomfort.

## 13. Rebuild

```
scripts/fetch_bdg2.sh; python3 -I scripts/build_bdg2.py meters summary
python3 -I scripts/weather_fetch.py screen; python3 -I scripts/build_bdg2.py points
python3 -I scripts/weather_fetch.py power; python3 -I scripts/weather_fetch.py asos; python3 -I scripts/build_bdg2.py weather
python3 -I scripts/stock_meta.py resstock|comstock; python3 -I scripts/build_bdg2.py spec
python3 -I scripts/stock_ts.py sample; ... fetch resstock|comstock; ... weather
python3 -I scripts/build_recs_cbecs.py; python3 -I scripts/fetch_ghal.py; python3 -I scripts/build_faults.py
python3 -I scripts/fetch_tmy.py; python3 -I scripts/build_tariffs.py; python3 -I scripts/build_other.py
REFREEZE=1 python3 -I scripts/make_splits.py    # only before Phase 2
python3 -I scripts/baselines.py load|detective|faults|thermostat|summary
```
