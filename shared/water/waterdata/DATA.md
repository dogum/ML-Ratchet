# WaterData: real watershed data for the Watershed project (Phase 1, agent B)

Everything the Phase 2 water models are graded on lives here. The splits, cleaning rules and baselines below were
decided once (seed 20261118) and must not change after Phase 2 starts. Build scripts are in `water/data/scripts/`
(all resumable: re-run a script and it skips what is already on disk). The loader package is `water/waterdata/`
(pandas / numpy / pyarrow; numba + scipy only for the baseline models).

```python
import sys; sys.path.insert(0, '/home/claude/water')
import waterdata as wd
from waterdata import metrics, signatures, models
sp  = wd.splits()                                         # frozen splits (data/proc/splits.json, sha256 inside)
ids = wd.basins('camels', split='dev')                    # 'camels_01022500', ... (also 'exam', 'exam_531')
a   = wd.attributes('camels', split='exam')               # Caravan + HydroATLAS + NWIS + derived attributes
s   = wd.spec('camels_01022500', approx_bands=True)       # contract basin spec (unknowns None + defaults_source)
f   = wd.forcing('camels_01022500', '1989-10-01', '1999-09-30')   # P,T_mean,T_min,T_max,SRAD,vp,wind,PET + extras
m   = wd.forcing_meta('camels_01022500')
q   = wd.streamflow('camels_01022500', 'best')            # Q_mm, Q_cms, src (nwis|caravan), flag (USGS qualifiers)
n   = wd.nwm('camels_01022500')                           # NWM v3.0 retrospective, daily mean (local standard day)
nh  = wd.nwm_hourly('camels_01022500')                    # NWM hourly m3/s, UTC, 2015-10..2023-01 only
pk  = wd.peaks('camels_01022500')                         # USGS annual instantaneous peaks
gb  = wd.forcing('camelsgb_10002'); wd.streamflow('camelsgb_10002')   # other regions (GB, AUS, CL, BR)
st  = wd.snotel_meta(split='dev'); sw = wd.snotel(st.station_id.iloc[0]); near = wd.snotel_near('camels_12381400')
rm  = wd.reservoir_meta(split='exam'); r = wd.reservoir('grand_132'); c = wd.reservoir_cdec('SHA')
bl  = wd.baselines(); fs = wd.baselines('flow_scores')    # exam summary / per-unit score tables
metrics.summary(obs, sim)                                 # NSE, KGE(r, alpha, beta), FHV, FLV, peak bias, timing
```

**Rules for Phase 2** (also inside `splits.json['rules']`):
- Only `dev` basins / stations / reservoirs may be used to calibrate, gate or choose anything. `exam` is for the final
  report only. Exam streamflow (any period) is never a training target.
- Gauged exam: an exam basin may use its own calibration-period Q (1999-10-01..2008-09-30) as model *input* or for
  per-basin classical calibration; it is scored on the validation period 1989-10-01..1999-09-30 (and the recent exam).
- Ungauged exam: no streamflow from the scored basin in any period. Donors only from dev basins (protocol **U-dev**,
  the same information an ML model gets) or, for the literature-style regionalisation baseline, from the other 11
  folds of `ungauged_folds_k12` (protocol **U-k12**).
- Simulated training may use real forcing of any basin from non-exam periods (1951-01-01..1988-09-30 and
  1999-10-01..2015-09-30) and real attributes only from dev basins (plus regions not used in any exam).

## 1. What is where

| Path (under `water/data/`) | What |
|---|---|
| `proc/splits.json` | frozen splits + rules + sha256 (sections frozen as written; later runs only add sections) |
| `proc/caravan/camels/<id>.parquet` | CAMELS-US (671 basins): daily ERA5-Land basin forcing 1951-01-01..2023-12-30 + `Q_mm` (1980/81..2014) |
| `proc/caravan/<camelsgb, camelsaus, camelscl, camelsbr>/<id>.parquet` | other regions: forcing 1981-01-01..2023-12-30 + `Q_mm` (fewer extras: no sm2-sm4) |
| `proc/attributes/<region>.parquet` | Caravan (other + climate) + HydroATLAS (196 columns) + derived; CAMELS-US also NWIS site info, `in_531`, implied area |
| `proc/specs/<region>_specs.json` | contract basin specs (`wd.spec`) |
| `proc/nwis/daily/<id>.parquet` | USGS daily mean discharge 1980-01-01..2026-11 (Q_mm, Q_cms, Q_cfs, flag) for all 671 gauges |
| `proc/nwis/peaks.parquet`, `coverage.parquet` | USGS annual instantaneous peaks; per-gauge coverage/flag fractions |
| `proc/nwm/daily_local.parquet`, `daily_utc.parquet`, `crosswalk.csv` | NWM v3.0 retrospective daily mean m3/s at 650 CAMELS gauges (1980-01..2023-01) |
| `raw/nwm/chanobs_hourly/<YYYY-MM>.parquet` | NWM hourly m3/s at the 650 gauges (kept for the recent period; see 4) |
| `proc/snotel/stations.parquet`, `daily/<id>.parquet` | all 934 SNOTEL stations (SWE, P, T, snow depth; metric) |
| `proc/reservoirs/reservoirs.parquet`, `daily/grand_<id>.parquet` | ResOpsUS v2: 678 dams, daily storage/inflow/release/elevation/evaporation + GRanD attributes |
| `proc/reservoirs/cdec/<STA>.parquet`, `crosswalk.csv` | CDEC 1990-2025 for 18 California reservoirs (extends ResOpsUS past 2020) |
| `proc/baselines/` | exam baselines: `summary.json`, `flow_*`, `snow_*`, `detective_*`, `reservoir_*`, `gr4j/` params, `sims/` |
| `raw/` | downloads kept for rebuilds (Caravan attributes + licences, NWIS RDB, SNOTEL CSV, ResOpsUS zip, CDEC CSV, lists) |
| `scripts/` | fetch_*, build_*, make_splits.py, run_* (baselines) |

## 2. Sources, licences, citations

| Source | Licence | Citation / URL |
|---|---|---|
| Caravan v1.6 (netCDF) | CC BY 4.0 | F. Kratzert et al., "Caravan - A global community dataset for large-sample hydrology", Sci. Data 10, 61 (2023), doi:10.1038/s41597-023-01975-w; data doi:10.5281/zenodo.15529786 (Caravan-nc.tar.gz, streamed, never stored) |
| CAMELS (US) streamflow inside Caravan | CC BY 4.0 | A. Newman et al., HESS 19, 209-223 (2015); dataset doi:10.5065/D6MW2F4D; N. Addor et al., HESS 21, 5293 (2017) |
| CAMELS-GB inside Caravan | OGL v3 (see `raw/caravan/licenses/camelsgb`) | G. Coxon et al., ESSD 12, 2459 (2020) |
| CAMELS-AUS v2 inside Caravan | CC BY 4.0 | K. Fowler et al., ESSD 13, 3847 (2021); v2: Fowler et al. (2024) |
| CAMELS-CL inside Caravan | CC BY 4.0 | C. Alvarez-Garreton et al., HESS 22, 5817 (2018) |
| CAMELS-BR inside Caravan | CC BY 4.0 | V. Chagas et al., ESSD 12, 2075 (2020) |
| ERA5-Land (all forcing) | Copernicus licence (free, attribution) | J. Munoz-Sabater et al., ESSD 13, 4349 (2021), via Caravan |
| HydroATLAS / BasinATLAS attributes | CC BY 4.0 | S. Linke et al., Sci. Data 6, 283 (2019), via Caravan |
| USGS NWIS daily values, site info, annual peaks | US public domain | waterservices.usgs.gov/nwis/dv, /nwis/site; nwis.waterdata.usgs.gov/nwis/peak (downloaded 18 Nov 2026) |
| CAMELS 531-basin list | CC BY / MIT (repo) | neuralhydrology repo, examples/06-Finetuning/531_basin_list.txt (Newman et al. 2017 subset) |
| NOAA National Water Model retrospective v3.0 | US public domain (NOAA Open Data Dissemination) | s3://noaa-nwm-retrospective-3-0-pds, CONUS/netcdf/CHANOBS + chrtout.zarr gage_id crosswalk |
| NRCS SNOTEL | US public domain | USDA NRCS National Water and Climate Center, wcc.sc.egov.usda.gov/reportGenerator (downloaded 18 Nov 2026) |
| ResOpsUS v2 | CC BY 4.0 | J. Steyaert, L. Condon, S. Turner, N. Voisin, "ResOpsUS, a dataset of historical reservoir operations in the contiguous United States", Sci. Data 9, 34 (2022); doi:10.5281/zenodo.6612040 |
| GRanD attributes (via ResOpsUS+CARS) | CC BY 4.0 | B. Lehner et al., Front. Ecol. Environ. 9, 494 (2011); J. Casado-Rodriguez et al., ResOpsUS+CARS v1.0, doi:10.5281/zenodo.15978041 (only `attributes/*.csv` read by HTTP range) |
| California CDEC | public (California DWR) | cdec.water.ca.gov/dynamicapp/req/CSVDataServlet, sensors 15/76/23 daily |

## 3. Units, time, and conventions

| Quantity | Unit / convention | Notes |
|---|---|---|
| `P` | mm/day | ERA5-Land total precipitation, basin average (Caravan) |
| `T_mean`, `T_min`, `T_max` | degC | 2 m air temperature (daily mean / min / max of hourly ERA5-Land) |
| `SRAD` | W/m2 | **NET** surface shortwave (incoming x (1 - albedo)), daily mean. Not incoming. `LRAD_net` = net longwave (negative = loss) |
| `vp` | Pa | from daily mean 2 m dewpoint: 610.94 exp(17.625 Td / (Td + 243.04)) |
| `wind` | m/s | magnitude of the daily mean 10 m wind vector (a lower bound of the mean speed; calm-day directions cancel) |
| `PET` | mm/day | FAO-56 Penman-Monteith (Caravan v1.5+). The ERA5-Land `potential_evaporation` band is NOT used: it is ~10x too high (mean 21 mm/day at a Texas basin) |
| `pressure` | kPa | surface pressure |
| `SWE_era5`, `sm1..sm4` | mm, m3/m3 | ERA5-Land model states, not observations (sm2-4 only for CAMELS-US) |
| `Q_mm` | mm/day | over `spec['area_km2']` (see gotcha 1) |
| `Q_cms` | m3/s | `Q_mm * area_km2 / 86.4`; NWIS: cfs x 0.028316846592 |
| SNOTEL `SWE`, `P`, `P_accum`, `snow_depth_mm` | mm | inches x 25.4; `T_*` degF -> degC |
| Reservoir `storage_hm3` | hm3 (= million m3 = MCM) | CDEC acre-feet x 0.00123348 |
| Reservoir `inflow_cms`, `release_cms` | m3/s | ResOpsUS `outflow` = release (incl. spill); CDEC cfs x 0.0283168 |

**What a "day" is.** Caravan forcing: UTC calendar day (ERA5-Land hourly aggregated in UTC). USGS daily values and
CAMELS Q: local standard-time day of the gauge (5-8 h behind UTC). So forcing day D covers 00-24 UTC while Q day D
covers 05-13 UTC of D to 05-13 UTC of D+1: rain late on UTC day D shows up in Q on day D, not D+1. Models should allow
a same-day response. NWM daily is given both ways (`day='local'` default, `'utc'`); local-day NWM scores slightly better
against USGS (median NSE 0.549 vs 0.533, 1989-1999). SNOTEL SWE / P_accum are start-of-day (midnight local standard
time) values; temperatures are that day's statistics; `P` for day D = accumulation(D+1) - accumulation(D).
ResOpsUS days are operator days (local); storage is end-of-day for 238 of 277 dams with a mass-balance check and
start-of-day for 39 (column `storage_stamp`, found by which alignment makes dS match (I - O)).
**Water year** WY N = 1 Oct (N-1) .. 30 Sep N.

## 4. CAMELS-US basins (FlowCast, BasinDetective)

- **671 basins** (Caravan `camels_<USGS id>`), 531 of them in the standard CAMELS-531 subset (`in_531`, from the
  neuralhydrology list). Forcing 1951-01-01..2023-12-30 (26,662 days); Caravan Q 1980/81..2014-12-31 (CAMELS end).
- **NWIS extension.** Daily values 1980-01-01..2026-11-17 for all 671 gauges (`wd.streamflow(id, 'nwis')`). Coverage
  (median over gauges) is 100% in every period; 39 exam gauges have < 90% coverage in the recent exam (17 gauges have
  no data at all after 2015: discontinued), 32 have < 50% in WY2024-2026. Qualifier `flag`: `A` approved, `P`
  provisional (1.2% of days, recent), `e` estimated (8.2% of days on average, up to 44%; this is where ice-affected
  winter flows end up after approval: only 0.007% of days still carry an explicit `Ice` code), `+alt` = filled from a
  second daily series at the same site. `wd.streamflow(id, 'best')` = NWIS where present, else Caravan.
- **Annual peaks** (`proc/nwis/peaks.parquet`, 48,769 peaks at 669 gauges, 1844-2026): instantaneous annual peak, date, gage height and USGS peak codes
  (e.g. 1 = daily max only, 2 = estimate, 5/6 = regulation/diversion, 7 = historic peak, C = urbanisation, E = estimate
  only). Dates with unknown day/month (`-00`) are set to the 1st and flagged `date_partial`.
- **NWM retrospective v3.0** at 650 of the 671 gauges (21 have no NWM gauge reach), 1980-01-01..2023-01-31 daily
  (`wd.nwm`); hourly kept for 2015-10..2023-01 (`wd.nwm_hourly`). Extracted from the hourly CHANOBS files with one HTTP
  range request per hour (the streamflow chunk always starts at byte 162458): 385k requests, ~32 KB each.
- **Attributes / spec.** Caravan climate indices (from ERA5-Land 1981-2020), HydroATLAS (196 columns, area-weighted over
  the basin polygon: `_sav` average, `_sse` extent %, `_smj` majority, `_syr` annual, `_pva`/`_usu` at the pour point /
  upstream), NWIS site info (drainage and contributing area, HUC8, state, time zone). The contract spec maps: slope =
  `slp_dg_sav`/10 (degrees), forest = `for_pc_sse`/100, irrigation `ire_pc_sse`/100, urban `urb_pc_sse`/100, carbonate
  ~ karst extent `kar_pc_sse`/100, porosity and Ksat from SoilGrids sand/clay via Cosby et al. (1984),
  `reservoir_storage_frac` = upstream reservoir volume (`rev_mc_usu`) / WaterGAP natural annual flow (`dis_m3_pyr`).
  Unknown (None, with `defaults_source`): `elev_bands` (no DEM; `approx_elev_bands` gives a triangular-hypsometry
  stand-in from min/mean/max), `lai`, `soil_depth_m`, `max_water_content_mm`, `geology_permeability`, `dam_count`.
  `reg_class` = none / low (< 0.1) / mid (0.1-0.5) / high (> 0.5) of `reservoir_storage_frac`.

## 5. Other regions (Caravan)

| Region | Basins | Q record (median start..end) | Dev basins (one per aridity tercile) |
|---|---|---|---|
| CAMELS-GB | 671 | 1981-01-01..2015-09-29 | camelsgb_17004, camelsgb_18003, camelsgb_33035 |
| CAMELS-CL | 505 (5 without any Q) | 1982-04..2018-03 (latest 2020-06) | camelscl_3404001, camelscl_7330001, camelscl_8332001 |
| CAMELS-BR | 870 | 1981-01-01..2018-11 | camelsbr_45480000, camelsbr_72680000, camelsbr_76750000 |
| CAMELS-AUS v2 | 561 | (pending: see STATUS.md) | frozen by `make_splits.py other` once the files are on disk |

Forcing 1981-01-01..2023-12-30, same columns as CAMELS-US except sm2-sm4. No NWIS/NWM equivalents. Everything not dev
is exam. Meant for out-of-domain checks (other continents, other forcing biases); no classical baselines were run on
them in Phase 1. CAMELS-AUS files in Caravan v1.6 are netCDF4 (the others netCDF3); the first stream pass crashed on
them at byte 23.84 GB of 24.8 and a second pass was started.

## 6. Splits (`proc/splits.json`, seed 20261118; frozen 18 Nov 2026)

**Temporal (CAMELS standard protocol, Newman et al. 2017 / Kratzert et al. 2019).**
| Period | Dates | Use |
|---|---|---|
| calibration | 1999-10-01..2008-09-30 (spin-up from 1998-10-01) | per-basin calibration, gauged-mode inputs |
| validation = main exam | 1989-10-01..1999-09-30 (spin-up from 1988-10-01) | scored |
| recent exam | 2015-10-01..2023-09-30 (WY2016-2023) | scored; Q from NWIS; forcing ends 2023-12-30; NWM only to WY2022 |
| free forcing | 1951-01-01..1988-09-30, 1999-10-01..2015-09-30 | simulated training may use any basin's forcing here |

**CAMELS-US spatial.** 52 dev / 619 exam (487 exam basins are in the CAMELS-531 set: `exam_531`).
- Eligible for dev: >= 90% Q coverage in both CAMELS periods and |Caravan polygon area - NWIS area| < 10% (72 basins
  ineligible, all exam).
- Per HUC2 region n = max(1, round(50 x N_region / 671)) (49 basins, every one of the 18 regions); within a region,
  eligible basins sorted by snow class (frac_snow < 0.1 / 0.1-0.3 / >= 0.3) then aridity rank and picked by systematic
  sampling with a seeded random start; plus 3 extra regulated basins (reg_class mid/high, distinct regions).
- Dev mix: HUC2 01:2 02:6 03:8 04:2 05:3 06:1 07:2 08:1 09:1 10:5 11:2 12:4 13:1 14:1 15:1 16:1 17:7 18:4; 10 with
  frac_snow > 0.3; 14 with aridity > 1; 5 regulated (4 high, 1 mid); 44 in the 531 set.
- Balance dev vs exam (KS p-values): aridity 0.87, frac_snow 0.96, p_mean 0.33, area 0.34, elevation 0.66, forest 0.17,
  regulation 0.96, lat 0.56, lon 0.99. No significant difference.
- `ungauged_folds_k12`: every basin in one of 12 random folds (for U-k12 regionalisation / cross-validation).

**Other regions.** 3 dev basins each in GB / AUS / CL / BR (one per aridity tercile, first basin of a seeded
permutation with >= 80% Q coverage 1990-2009); rest exam.

**SNOTEL.** 880 stations with SWE data (of 934 listed; 61 in Alaska); 811 eligible (>= 10 complete water years:
>= 330 SWE days, >= 300 P and T days). Dev = 39 stations (per state n = max(1, round(40 x N_state / 811)), systematic
by elevation, seeded) drawn only from stations whose HUC8 holds no exam CAMELS gauge; exam = 772. Classical per-station
calibration WY2001-2010, scoring WY2011-2025.

**Reservoirs.** 678 ResOpsUS dams. Core = >= 10 years of simultaneous storage + inflow + release and a known capacity
(265 dams). Dev = 15 (stratified by main use fcon/irri/elec/supp/other x capacity above/below median), exam = 250 core,
aux = 413 (incomplete records; usable for single-process calibration, never scored). Each exam reservoir: first half of
its record (from 1980) calibrates demand / rule curve / channel capacity, second half (max 10 years) is scored. Recent
exam: 15 CDEC reservoirs WY2021-2025 (demand etc. from the whole ResOpsUS record).

## 7. Exam baselines (`proc/baselines/summary.json`; per-unit tables `*_scores.parquet`)

Metrics (`waterdata.metrics`): NSE; KGE (2009); FHV = % bias of the top 2% flows; FLV = % bias of the bottom 30% in
log space relative to each series' own minimum, computed exactly as Yilmaz et al. (2008) and neuralhydrology,
including their leading minus sign (so read FLV only as "0 is perfect; larger |FLV| is worse"); peak = mean / median-absolute % error of annual-maximum daily flow; timing = mean |days|
between observed and simulated peaks (top 3 peaks per year, +-3 d window). All are medians over exam basins.

**Streamflow, CAMELS-US exam basins** (gauged = may use the basin's own calibration-period Q):
| Method | Mode | Val NSE | Val KGE | Val NSE (531) | Val FHV % | Val peak abs err % | Val timing d | Recent NSE | Recent KGE |
|---|---|---|---|---|---|---|---|---|---|
| climatology (own DOY mean) | gauged | 0.065 | -0.054 | 0.062 | -80.7 | 84.5 | 2.58 | 0.053 | -0.065 |
| persistence, 1-day lead | gauged forecast | 0.475 | 0.738 | 0.443 | 0 | 0 | 1.03 | 0.480 | 0.740 |
| GR4J + snow, per-basin calibrated | gauged | **0.473** | 0.602 | 0.478 | -18.0 | 38.0 | 0.76 | **0.517** | 0.604 |
| NN regionalisation, dev donors (U-dev) | ungauged | **0.372** | 0.433 | 0.402 | -27.3 | 47.3 | 0.82 | **0.433** | 0.469 |
| NN regionalisation, 12-fold donors (U-k12) | ungauged | 0.385 | 0.462 | 0.401 | -21.0 | 45.5 | 0.82 | 0.443 | 0.469 |
| NOAA NWM v3.0 retrospective (n=600) | operational | **0.550** | 0.596 | 0.566 | -13.0 | 37.4 | 0.86 | **0.687** | 0.802 |
| *Literature LSTM (context only, not reproduced)* | | *~0.74-0.76 gauged; ~0.69 ungauged* | | | | | | | |

n = 619 exam basins (val), 603 (recent, gauges with >= 365 valid days). Recent NWM = WY2016-2022 (NWM ends Jan 2023).
Fraction of exam basins with NSE < 0 (val): GR4J 6%, U-dev 14%, NWM 9%, climatology 16%. Dev-basin medians (val) for
the gates: GR4J 0.462, U-dev 0.407, NWM 0.536, persistence 0.497. Literature: Kratzert et al. 2019 HESS (EA-LSTM,
531 basins, Daymet forcing: median NSE ~0.74 single / ~0.76 ensemble) and Kratzert et al. 2019 WRR (ungauged 12-fold:
~0.69); different forcing, so not comparable row-for-row.

**Flood peaks, CAMELS-US exam basins** (USGS annual *instantaneous* peaks; codes 5/6/7/C and partial dates
excluded; per-basin median |% error|, then median over basins; `run_baselines_peaks.py`):
| Method | Val WY1990-99: abs err % / bias % / rho | Recent WY2016-23: abs err % / bias % / rho |
|---|---|---|
| own calibration-years median peak (gauged, dumb) | 42.2 / -14.0 / 0.85 | 43.5 / -8.1 / 0.85 |
| GR4J annual daily max x 1.38 (dev inst/daily ratio) | 46.5 / -31.3 / 0.86 | 47.8 / -32.0 / 0.86 |
| NWM daily max x 1.38 | 42.4 / -23.8 / 0.85 | 34.1 / -12.2 / 0.92 (WY16-22) |
| NWM hourly annual max | | **30.6 / -2.3 / 0.93** (WY16-22) |

**BasinDetective, CAMELS-US exam basins** (target = signature from the basin's validation-period Q; Spearman rho / R2 /
median |error|):
| Signature | dev median (dumb) | kNN on attributes (ungauged) | Budyko-Fu (forcing only) | GR4J sim (gauged) | own calibration-period Q (stability) |
|---|---|---|---|---|---|
| runoff ratio | - / -0.07 / 0.148 | 0.78 / 0.63 / 0.082 | 0.74 / 0.42 / 0.087 | 0.97 / 0.95 / 0.043 | 0.97 / 0.96 / 0.038 |
| BFI (Lyne-Hollick) | - / 0.00 / 0.115 | 0.55 / 0.30 / 0.095 | | 0.79 / 0.52 / 0.065 | 0.98 / 0.96 / 0.018 |
| BFI (Eckhardt) | - / -0.06 / 0.079 | 0.60 / 0.39 / 0.060 | | 0.82 / 0.55 / 0.038 | 0.98 / 0.96 / 0.013 |
| recession K (days) | - / -0.04 / 3.67 | 0.52 / 0.19 / 3.25 | | 0.46 / <0 / 8.94 | 0.98 / 0.94 / 0.69 |
| recession exponent b | - / 0.00 / 0.177 | 0.46 / 0.19 / 0.160 | | 0.26 / <0 / 0.52 | 0.85 / 0.71 / 0.069 |
| half-flow date (d) | - / -0.03 / 23.1 | 0.72 / 0.58 / 14.2 | | 0.91 / 0.84 / 6.9 | 0.87 / 0.77 / 8.3 |
| FDC slope | - / -0.10 / 1.06 | 0.28 / 0.15 / 1.12 | | 0.51 / 0.04 / 0.94 | 0.79 / 0.15 / 0.35 |
| Q mean (mm/d) | - / -0.11 / 0.61 | 0.80 / 0.74 / 0.32 | | 0.98 / 0.98 / 0.13 | 0.97 / 0.96 / 0.19 |
| P elasticity | - / -0.04 / 0.71 | 0.07 / <0 / 0.76 | | 0.49 / 0.16 / 0.49 | 0.18 / <0 / 0.76 |

All signatures for every basin, both periods, obs and GR4J: `proc/baselines/signatures.parquet`. Independent labels
for detective tasks that do not come from Q: `aridity`, `frac_snow` (forcing), `reservoir_storage_frac` / `reg_class`,
`kar_pc_sse`, `for_pc_sse`, `ire_pc_sse`, `urb_pc_sse`, `gwt_cm_sav`, soil texture (HydroATLAS).

**SnowSense, SNOTEL exam stations** (n = 621 with >= 5 complete calibration and scoring years; WY2011-2025):
| Method | NSE | RMSE mm | peak SWE abs err % | peak SWE bias % | melt-out MAE d |
|---|---|---|---|---|---|
| climatology (own WY2001-2010 DOY median) | 0.701 | 84.6 | 28.2 | +2.1 | 11.4 |
| degree-day, calibrated per station (own P, T) | **0.925** | 41.2 | 12.5 | -3.6 | 6.1 |

(dev medians: 0.746 / 0.917.) The degree-day baseline uses the station's own precipitation gauge, which is a large
advantage over any model driven by gridded forcing; a SnowSense model driven by ERA5-Land should also be compared with
the degree-day model driven by the same gridded inputs.

**ReservoirPilot, ResOpsUS exam reservoirs** (n = 244 scored; cost J = mean daily (shortage/demand)^2 + 10 (excess over
Qmax / Qmax)^2, lower is better; demand = calibration-half day-of-year median release, Qmax = its 99th percentile):
| Policy | J median | shortage ratio | days/yr above Qmax | release NSE | storage NSE |
|---|---|---|---|---|---|
| historical operations (observed) | 0.183 | 0.285 | 1.9 | (1) | (1) |
| SOP (release demand if available) | 0.108 | 0.010 | 2.2 | 0.06 | -11.8 |
| rule curve (track median storage, tau 30 d) | 0.143 | 0.186 | 0.0 | 0.43 | -0.16 |
| perfect-foresight DP (near lower bound) | **0.006** | 0.030 | 0.0 | | |

Mass-balance-consistent subset (dS vs (I-O) correlation >= 0.9, n = 151): historical 0.200, SOP 0.099, rule 0.151,
DP 0.007. Recent CDEC exam (15 California reservoirs, WY2021-2025): historical 0.108, SOP 0.095, rule 0.240, DP 0.012.
Real operators score worse than SOP under this cost because they hedge, keep flood space and meet targets this cost
does not see; J is a well-defined control benchmark, not a judgement of the operators. Means of J are dominated by a
few dams with near-zero demand; use medians.

## 8. Gotchas (read before modelling)

1. **Which area.** CAMELS/Caravan converted discharge to mm/day with the NWIS *contributing* drainage area (GAGES-II
   convention), not the Caravan polygon and not the NWIS total drainage area. Recovered from the data
   (`derive_area.py`): implied area / NWIS area = 0.9995 (median); 36 gauges differ by > 2%, prairie gauges by up to 6x
   (e.g. 08079600: contributing 632 km2 vs drainage 3,797 km2). `spec['area_km2']` is this implied area and every
   mm/day <-> m3/s conversion here uses it; before this fix, NWIS mm/day disagreed with Caravan at about 10 gauges (NSE 0.3-0.7).
   The forcing is still averaged over the Caravan polygon (`area_polygon_km2`): for those prairie basins the
   forcing area is 2-6x the contributing area. Dev basins exclude all basins with > 10% polygon/NWIS area mismatch.
2. **ERA5-Land precipitation bias.** 50 of 671 basins have validation-period runoff ratio > 1 (mostly Pacific
   Northwest / mountain basins: ERA5-Land undercatches orographic precipitation). Models need a precipitation
   multiplier or must learn the bias. This also caps the classical GR4J baseline (median val NSE 0.47 here vs ~0.6-0.7
   typical with CAMELS Daymet forcing) and pushes 37% of GR4J calibrations to at least one parameter bound
   (most often ddf 15%, tt 9%, x2 8%, x1 7%).
3. **SRAD is net shortwave**, not incoming (ERA5-Land `surface_net_solar_radiation`). Penman-Monteith with SRAD as
   incoming would double-count albedo. The ERA5-Land `potential_evaporation` band is unusable (~10x); use `PET`
   (FAO-56 PM) or compute your own.
4. **Time zones.** Forcing days are UTC, Q days local standard time (section 3). A 1-day lag in a model is a real
   choice, not a given.
5. **Ice and estimates.** USGS approved records carry ice-affected winter flows as estimated (`e`): 8% of all days,
   up to 44% at northern gauges. Treat winter low flows at northern basins as uncertain (flag-weighted metrics are
   possible with the `flag` column). Provisional (`P`) values are only in the last ~1-2 years.
6. **Regulated / impacted basins.** CAMELS is "minimally impacted" but 58 basins (5 dev) have reg_class low/mid/high
   (upstream reservoir volume > 0) and 20 (4 dev) have reg_class high (> 50% of annual flow). Several prairie basins have
   non-contributing areas (gotcha 1). Peak codes 5/6/C in the peak file flag regulation, diversion, urbanisation.
7. **Missing data.** Caravan Q starts 1980-01-01 (153 basins), 1981-01-01 (440) or later (up to 1989-10-01) and ends
   2009-09-30 .. 2014-12-31 (15 basins end before 2013-09-30); CAMELS -999 were set to NaN by Caravan. 17 gauges stopped before 2015 (no recent exam), 39 exam gauges have < 90% recent
   coverage. SNOTEL: 54 listed stations have no SWE at all; SNOTEL temperatures have a known inhomogeneity from the
   network-wide sensor change in the 2000s (Oyler et al. 2015, GRL 42, 153), so do not read trends from them; P is
   accumulation-differenced where `PRCP` is missing (`P_src`).
8. **Persistence and peaks.** 1-day persistence scores high KGE (0.74) and 0 peak error by construction (annual
   maxima are shifted copies); judge forecast models on NSE, timing and multi-day leads, not peak magnitude alone.
9. **NWM is not a clean baseline.** NOAA calibrated NWM v3.0 on many of the same USGS gauges, with years overlapping our
   calibration window, and its AORC forcing is independent of ERA5-Land. It is the operational bar, not an ungauged
   method. It covers 650/671 gauges and ends 2023-01-31. NWM daily is the mean of 24 hourly instantaneous values.
10. **Reservoirs.** 189 dams have negative inflows somewhere (back-calculated from mass balance); kept as published.
    Only 171 of the 277 dams with all three variables close the daily mass balance well (corr >= 0.9); evaporation
    units differ by agency (hm3 or m3/s, see `raw/resopsus/docs/agency_attributes.csv`) and evaporation is ignored in the
    baselines. Capacity = GRanD `CAP_MCM`, raised to the observed maximum storage where that is larger.
11. **SNOTEL is a point; basins are areas.** Do not compare SNOTEL SWE with basin-mean ERA5-Land `SWE_era5` without
    an elevation correction. `wd.snotel_near(basin)` is a proximity rule (same HUC8 or within 50 km), not
    point-in-polygon (Caravan basin shapefiles were not kept).
12. **Other regions** have no extended or operational data; Q periods vary by country (section 5) and none reach
    the CAMELS-US recent-exam years, so they are spatial (out-of-domain) tests only.

## 9. Rebuild

Order: `fetch_caravan.py` (streams 24.8 GB once; ~2-3 h) -> `fetch_nwis.py` -> `build_attributes.py` ->
`make_splits.py core` (refuses to overwrite frozen sections) -> `build_nwis.py` -> `derive_area.py` ->
`build_attributes.py` -> `build_nwis.py` (now with the implied area) -> `fetch_nwm.py` -> `build_nwm.py` ->
`fetch_snotel.py` -> `build_snotel.py` -> `make_splits.py snotel` -> `build_reservoirs.py` -> `make_splits.py reservoirs`
-> `fetch_cdec.py` -> `make_splits.py other` -> `run_gr4j.py` -> `run_baselines_flow.py`, `run_baselines_detective.py`,
`run_baselines_snow.py`, `run_baselines_reservoir.py`, `run_baselines_reservoir_recent.py` -> `build_summary.py`.
All fetchers are resumable (skip files on disk). Downloaded archives were treated as data only (no code from them was
run). Disk: 2.9 GB (proc/caravan 1.5 GB; ResOpsUS zip 0.3 GB; NWM hourly 0.2 GB; baselines 0.17 GB).
