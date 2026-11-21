# Wildfire real-data package (FireData, Phase 1 agent B)

Everything real that the Wildfire models are checked against: daily fire progression, landscapes, weather, frozen
dev/exam splits, an exam guard, and classical baselines with exam numbers. Built 18-19 Nov 2026.

```python
import sys; sys.path.insert(0, 'shared/fire')
import firedata as fd
fd.fires('dev')                          # 40 dev fires (table: region, fuel group, size class, window, FEDS stats)
pr = fd.progression('CA4150012192920210801')   # daily PM->PM burned masks, 90 m EPSG:5070 grid
L  = fd.landscape('CA4150012192920210801')     # LANDFIRE contract stack (fuel, cc, ch, cbh, cbd, elev, slope, aspect)
W  = fd.weather('CA4150012192920210801')       # hourly HRRR (series + 3 km grid) + daily gridMET
fd.guard.check_landscape(bbox); fd.guard.check_weather(bbox, t0, t1)   # raise ExamLeakError
fd.metrics.nextday(now, nxt, prob, cell_km2, window=fd.metrics.score_window(now))  # the exam metric
```
Exam fires raise `ExamLeakError` in every loader unless `for_exam_scoring=True` is passed (scoring code only).

## 1. Sources (all public, no keys)

| Data | What we keep | Period | Licence / credit |
|---|---|---|---|
| **FEDS-MTBS** (Zenodo 20187963; Chen et al. 2022 FEDS method, MTBS-constrained) | 12-hourly fire perimeters, active fire lines, VIIRS S-NPP 375 m fire pixels for 7739 MTBS fires (>= 1000 ac W / 500 ac E) | 2012-2024 | CC BY 4.0 |
| **MTBS** final perimeters (inside FEDS-MTBS fireslist) | final perimeter, ignition date, Incid_Type | 2012-2024 | public domain (USGS/USFS) |
| **GOFER-Combined** (Zenodo 14642378; Liu et al.) | hourly GOES perimeters, 28 California fires | 2019-2021 | CC BY 4.0 |
| **WildfireSpreadTS** (Zenodo 8006177; Gerard et al. 2023) | zip index, tile headers for all 607 fires; daily VIIRS AF + gridMET/GFS weather streamed per fire (in progress) | 2018-2021 | CC BY 4.0 |
| **LANDFIRE** (lfps.usgs.gov ImageServer exportImage) | FBFM40, CC, CH, CBH, CBD (LF2016/2022/2023), Elev/SlpD/Asp (LF2020) | — | public domain (LANDFIRE) |
| **HRRR analyses** (hrrrzarr, U. Utah MesoWest on AWS Open Data) | hourly 2 m T, RH, 10 m wind, gust, DSWRF, cloud; 1 h precip from short-lead forecasts | 2016-08+ | NOAA, public |
| **gridMET** (Abatzoglou 2013; northwestknowledge.net HTTP files) | daily erc, bi, fm100, fm1000, vs, th, rmin, rmax, tmmx, tmmn, pr, vpd at points | 2012-2024 | public domain |
| **FPA-FOD 6th ed.** (Short 2022, RDS-2013-0009.6) | 1.59 M CONUS wildfire records, 2001-2020 (we use 2012-2020) | 2001-2020 | USDA Forest Service RDS |
| **Globe-LFMC 2.0** (Yebra et al. 2024, figshare 25413790) | 239k CONUS live fuel moisture samples | 1977-2023 | CC0 |

Not obtained: FIRMS archive (not needed: FEDS ships the VIIRS pixels it used; a MAP_KEY would be needed for more),
GOES FDC (not needed in Phase 1), NFMD dead-fuel sticks (see section 7), RAWS (Synoptic needs a key).

## 2. Units and conventions (gotchas first)

- **Grid:** every fire has a window = MTBS final-perimeter bbox + max(3 km, 30 % of the longer side), snapped to
  x, y = 15 + 90k m in **EPSG:5070** (NAD83 CONUS Albers), so 90 m cells nest exactly on LANDFIRE's 30 m cells.
  The window is derived from the FINAL perimeter, so **never score the whole window** (it tells you where the fire
  stops). The standard scored region is cells within 10 km of the burned area at issue time (`metrics.score_window`).
  1.6 % of dev new-burn cells fall outside it (long jumps / merges).
- **FEDS time labels** `t` are nominal local half-days: `00:00` = S-NPP night overpass (~01:30 local solar time),
  `12:00` = day overpass (~13:30). `t_utc` ~ label + 1.5 h - lon/15 h. Daily truth = PM-to-PM perimeters.
- **FEDS units:** raw `farea` is **m2** (the readme does not say); `fperim` and `flinelen` are km. Loaders return
  `farea_km2`, `fperim_km`, `flinelen_km`. FEDS perimeters are cumulative and only grow; steps above 99.9 % of the final
  area were removed by the authors, so the last days of long fires are missing.
- **How good is the truth?** On the 4 dev GOFER fires, FEDS PM perimeters vs GOFER (GOES) perimeters within 30 min
  agree with Dice 0.77-0.93 (median per fire: Antelope 0.88, CZU 0.86, Creek 0.93, W-5 Cold Springs 0.77) and area
  ratio 0.80-1.22. Two satellite products disagree by ~10-20 % of the burned area, so day-to-day new-area scores have
  a real noise floor.
- **FEDS vs MTBS area:** FEDS area / MTBS area has a wide spread (VIIRS 375 m pixels over-cover small fires, miss
  low-intensity burning). Dev fires were required to have ratio 0.5-1.5; exam fires were not.
- **Landscape** (`proc/landscapes/<id>.npz`, 90 m): fuel = 3x3 mode of FBFM40 30 m (91-99 non-burnable, -9999 nodata);
  cc %, ch m, cbh m, cbd kg/m3 = 3x3 means of LANDFIRE values scaled (/10, /10, /100); elev m; slope degrees;
  aspect degrees from N (-1 flat). **Version:** the latest LANDFIRE before the fire year (pre-fire): fires 2017-2022
  use LF2016 Remap, 2023 use LF2022, 2024 use LF2023. For 2012-2016 fires only LF2016 exists, which already shows
  their own burn scar (**post-fire fuels**, flag `landfire_postfire`). LF2016 for a 2021 fire is also 5 years stale
  (fires, treatments and growth in between are not in it).
- **HRRR** (`proc/weather/hrrr/<id>.npz`): hourly UTC, window +1 cell at 3 km. T degC (2 m), RH % (2 m),
  wind_speed m/s at **10 m** (rotated from grid-relative to earth-relative), wind_dir degrees FROM true north, gust m/s,
  srad W/m2, cloud %. 20-ft (6.1 m) wind ~ 10 m wind / 1.15 (BehavePlus convention). Period = first FEDS step - 2 d to
  min(last step, first + 40 d) + 1 d. Values are stored float16 in the archive (T steps ~0.25 K).
  **Precip:** the analysis PRATE chunks do not exist in hrrrzarr, so precip = 1 h accumulation from the 00z/12z
  forecast cycles (leads f01-f12). hrrrzarr has no forecasts before ~mid-2018, so **precip is NaN for 2017 and
  early-2018 fires** (14 of 40 dev, 12 of 48 replay): use gridMET daily `pr`. 1-5.5 % of hours are NaN on 2017-18
  fires: these are gaps in the hrrrzarr archive itself (a rebuild returned the same gaps), logged as
  `frac_hours_missing`; interpolate short gaps. hrrrzarr switched from float16 to float32 chunks during 2024; the
  reader handles both. HRRR-Zarr starts 2016-08-25: no hourly weather for 2012-2016 fires.
- **gridMET** daily at the fire centroid for the whole ignition year (`proc/weather/gridmet/fires_<year>.parquet`;
  `fd.weather(id)['daily']`), at FPA-FOD points on the discovery day and the day before (`fod_<year>`), and at
  Globe-LFMC sites (`lfmc_<year>.npz`). Temperatures converted to degC; vs m/s at 10 m; nearest 4 km cell.

## 3. Splits (frozen 18 Nov 2026 23:44Z, seed 20261118, `proc/splits.json`, sha256 d73e2d9f...)

**Unit = the fire** (MTBS Event_ID). Every time step of a fire shares its split. GOFER and WildfireSpreadTS fires
inherit the split of the MTBS fires they overlap (tile + date match); mixed matches are excluded.

| Split | Fires | Rule |
|---|---|---|
| dev | 40 | CONUS Wildfire 2017-2021 with >= 4 PM steps, >= 2 growth days, FEDS/MTBS area 0.5-1.5, < 10 % of window covered by other fires, fuel group known. 4 GOFER fires first (seeded, one per year first), then per region n = round(36 x share), systematic sampling with seeded start over (fuel group, size class, year) |
| exam | 4197 | all other CONUS Wildfire/Unknown fires 2012-2024. Periods: early 2012-16 (1647; post-fire LANDFIRE, no HRRR), main 2017-21 (1859), **recent 2022-24 (691, temporal exam: whole seasons)** |
| aux_rx | 2859 | prescribed fires: may be used like dev for simulator calibration, never scored |
| excluded_nonCONUS | 643 | Alaska, Hawaii, Puerto Rico |

Dev mix: regions NR 9, CA 8, SW 7, NW 5, RM 4, SP 3, SE 2, GB 2; fuel groups TL 12, TU 11, GS 9, GR 7, SH 1; sizes
1k-5k ac 17, 5k-50k 14, >50k 9; years 2017 10, 2018 10, 2019 5, 2020 10, 2021 5. Dev fires are longer than typical
exam fires (median 10.5 PM steps vs 3) because of the eligibility rule.

GOFER dev: Antelope 2021, CZU Lightning Complex 2020, Creek 2020, W-5 Cold Springs 2020; the other 24 are exam.
WildfireSpreadTS: 4 dev, 568 exam, 35 excluded (tile overlaps both).

**FireSize** (FPA-FOD CONUS 2012-2020, 664,782 fires): an FOD fire with an MTBS_ID inherits that fire's split; other
fires: dev = seeded 1.5 % of 2012-2019 (8913 dev in total), exam = the rest (655,869; recent = all of 2020).
**Moisture** (Globe-LFMC sites): seeded 10 % of sites dev (`proc/moisture_split.json`), rest exam.
**Replay subset** (`proc/baselines/replay_subset.json`): 48 exam fires 2017-2024 (32 main, 16 recent) drawn once by
region share and systematic sampling; their landscapes and HRRR are built for physics-replay scoring only.

### What simulated training may use (enforced by `firedata.guard`)
- Landscapes: any window that does not touch an exam footprint (MTBS final perimeter + 2 km, any year, because the
  landscape IS the fire site; WildfireSpreadTS-only exam fires use their whole tile). `check_landscape(bbox)` raises;
  `exam_mask(grid)` returns the cells to blank if you want to reuse a window.
- Weather: not within 50 km of an exam footprint during that fire's active window [ignition - 3 d, last FEDS step + 3 d].
  `check_weather(bbox, t0, t1)` raises.
- Dev fires' landscapes and weather are free for calibration and replays. **But many dev windows touch older or nearby
  exam fires** (e.g. the 2021 Moore fire window overlaps a 2016 exam fire): for training banks, run the guard or blank
  `exam_mask` even on dev windows.
- Random non-fire places at any time are free (subject to the guard).

## 4. Next-day spread baselines (exam numbers)

Fire-day = FEDS PM perimeter (issue) -> next PM perimeter exactly 24 h later, rasterised on the 90 m grid (cell-centre
rule). Scored: candidate cells (unburned at issue) within 10 km of the burned area. Only growth days (observed new
area > 0) are averaged. Metrics (`firedata.metrics`): AP over candidate cells (score ranking), F1 at threshold 0.5
(on candidate cells F1 equals Dice on the new area), Dice on total burned area, log10 ratio of (predicted + 1 cell) /
(observed + 1 cell) new area. Means with 95 % bootstrap CIs over fires.

Methods (all distance buffers from the burned edge; score = 1 / (1 + (dist/d)^4), 0.5 at the buffer distance d):
`persist` no growth; `grow_prev` d = yesterday's equivalent-radius growth; `grow_prev_k` same x k_dev = 3.0 (chosen on
dev: max mean Dice-new); `clim` d = dev median next-day radial growth by size bin (6-58 m: most days barely grow, so
this is the "guess the median" baseline); `fl_prev` yesterday's growth measured from the FEDS **active fire line**
instead of the whole perimeter. Distance buffers that differ only in d share the same ranking, so the same AP.

**Exam, all periods: 3030 fires, 25,105 growth fire-days**

| method | AP mean [95% CI] | AP median | F1 (= Dice new) mean | Dice total median | log10 area ratio median |
|---|---|---|---|---|---|
| persist | 0.013 [0.012, 0.013] | 0.005 | 0.000 | 0.964 | -2.20 |
| clim (dumb) | 0.204 [0.201, 0.208] | 0.159 | 0.004 | 0.964 | -2.14 |
| grow_prev | 0.191 [0.187, 0.194] | 0.144 | 0.140 | 0.924 | -0.30 |
| grow_prev_k (dev k = 3) | 0.191 [0.187, 0.194] | 0.144 | 0.146 | 0.856 | +0.47 |
| **fl_prev** (best) | **0.252 [0.247, 0.257]** | **0.216** | **0.184** | 0.947 | -0.38 |

Dev (40 fires, 535 growth days): persist 0.013, clim 0.185, grow_prev 0.171, fl_prev 0.252 AP mean; fl_prev F1 0.182.
By period (fl_prev AP mean): early 0.249, main 0.257, recent 0.242. By region: SW 0.304, CA 0.281, SE 0.257, NW 0.251,
NR 0.235, RM 0.222, GB 0.220, NE_MW 0.199, SP 0.173. Full tables: `proc/baselines/spread_summary.json`; per fire-day
rows: `spread_{dev,exam}.jsonl`.

**Physics replays (uncalibrated and dev-calibrated Rothermel):** agent A runs them with firesim on dev fires; the exam
replay subset (48 fires) has landscapes + HRRR ready for the final scoring run.

**Published context (not comparable number for number):** WildfireSpreadTS (Gerard et al. 2023, Table 5, test AP
mean +- sd over 12 year folds, 375 m, target = next-day **active fire** pixels, not new burned area): persistence
0.193 +- 0.065, logistic regression 0.286 +- 0.092, Res18 U-Net 1-day 0.341 +- 0.086, UTAE 5-day 0.372 +- 0.088.

## 5. FireSize baselines (FPA-FOD exam, 655,869 fires; fitted on 8913 dev fires)

Targets: final size (ha, log10); escape = final size >= 100 acres (2.7 %); large = >= 1000 acres (0.6 %).

| method | size MAE log10 (median abs) | size Spearman rho | escape AUC | escape AP | large AUC | large AP |
|---|---|---|---|---|---|---|
| median / prevalence (dumb) | 0.699 | — | 0.500 | 0.027 | 0.500 | 0.006 |
| climatology region x cause | 0.505 | 0.369 | 0.688 | 0.061 | 0.767 | 0.025 |
| ERC + BI + wind at discovery (linear / logistic) | 0.645 | 0.201 | 0.634 | 0.042 | 0.748 | 0.016 |
| gradient boosting, all features (real labels fitted on dev) | 0.510 | 0.533 | 0.777 | 0.099 | 0.803 | 0.029 |

Recent (2020 only, 72,883): gbm escape AUC 0.789, AP 0.092; clim 0.711 / 0.067. File: `proc/baselines/firesize_exam.json`.
Note: FOD has no records after 2020, and the FireSize "exam" is dominated by tiny fires; for "large fire from an
ignition" questions use the MTBS-matched subset or the FEDS fires.

## 6. Live fuel moisture baselines (Globe-LFMC exam sites, 138,113 samples, 2012-2022)

| method | MAE (% dry mass) | RMSE | bias | r |
|---|---|---|---|---|
| dev median (dumb) | 29.1 | 44.0 | -8.9 | — |
| climatology functional type x month | 27.2 | 42.1 | -8.7 | 0.30 |
| LFMC ~ 7-day ERC per functional type | 29.7 | 41.9 | +1.7 | 0.25 |
| gradient boosting on gridMET drought features (dev labels) | 24.7 | 37.0 | +0.3 | 0.52 |

Dev: 15,361 samples at 10 % of 1167 sites. File: `proc/baselines/moisture_exam.json`.

## 7. Missing and why (help_wanted)

- **Dead fuel moisture observations (NFMD 10-h / 100-h / 1000-h).** `wfas.net/nfmd/public/` now returns 404; the
  National Fuel Moisture Database moved into FEMS (fems.fs2c.usda.gov), whose download path we did not find as a
  documented public file. Not worked around. Needed for the Nelson-model check against real sticks.
- **RAWS hourly / fuel sticks:** Synoptic needs a key (skipped per brief); WRCC RAWS pages are interactive forms only.
- **FIRMS:** not used (FEDS already carries the S-NPP pixels); archive downloads need a free MAP_KEY.
- **Hourly weather for 2012-2016 fires** (no HRRR-Zarr) and hourly precip before mid-2018.
- **Pre-fire LANDFIRE for 2012-2016 fires** (only LF2016 on the service).
- **WildfireSpreadTS full stream** is slow (~2 fires/min over range requests; dev first, then exam in a seeded order,
  `proc/wsts/order.txt`); it was still running at hand-off. WSTS `af` = UTC hhmm of the VIIRS detection, -1 none.
- **WildfireSpreadTS static layers** (elevation, land cover, NDVI) are kept only for 299 fires (`wsts/meta/*.npz`);
  the full stream keeps daily AF and weather only.

## 8. Files

`proc/`: `splits.json`, `fires.parquet`, `firesize_split.parquet`, `moisture_split.json`, `guard/`, `feds/`
(perims, firelines, firepix per year; final_perims; fires_raw), `gofer/`, `fpafod/fod.parquet`, `lfmc/`,
`landfire/fuelgroups.parquet`, `landscapes/` (dev + replay), `weather/hrrr/` (dev + replay), `weather/gridmet/`,
`wsts/` (meta, fires), `baselines/` (spread_*, firesize_exam.json, moisture_exam.json, fire_days_{dev,exam}.parquet,
replay_subset.json, spread_params.json). `scripts/` rebuild everything (each resumable, run with `python3 -I`).
