# GridData: real data for the power-grid project (Phase 1)

Everything Phase 2 grid models are graded on lives here. It is frozen: the splits, cleaning rules and baselines below
were decided once (seed 20261113) and must not change after Phase 2 starts. Build scripts are in `shared/grid/data/scripts/`
(all resumable); the loader package is `shared/grid/griddata/` (pandas / numpy / pyarrow only).

```python
import griddata as gd   # pip install -e ".[grid]" from the repo root; data: `mlratchet data grid`
import griddata as gd
sp  = gd.splits()                                        # frozen split file (proc/splits.json)
d   = gd.eia930('ERCO', '2023-01-01', '2024-01-01')       # hourly demand, DA forecast, generation by fuel, interchange
d   = gd.eia930('PJM', split='dev')                       # dev years only (2019, 2021)
fl  = gd.fleet('ERCO')                                    # generator-spec dicts (GRID_BRIEF contract), src per field
fl  = gd.fleet('ERCO', as_frame=True, plant_safe=True, existing_on='2024-07-01')   # PlantDetective-safe fleet
u   = gd.campd_units(ba='ERCO', split='exam')             # CAMPD units with EIA links, split, 2018-22 characteristics
h   = gd.campd_unit('3470_WAP8', '2019-01-01', '2020-01-01')   # hourly gross load, heat input, CO2, operating time
ev  = gd.events('GB', split='exam'); er = gd.events('ERCO')   # frequency events (GB from 1 s data, ERCOT from reports)
meta, t, f = gd.event_trace('GB_20190809T155233')        # 1 s trace (contract: t [s], f [Hz])
meta, t, f = gd.frequency('GB', '2019-08-09 15:40', '2019-08-09 16:10')   # full 1 s months (see section 5)
m   = gd.frequency_minutes("2024-01-01", "2024-02-01")    # per-minute GB frequency summary, 2018-01..2026-10
I   = gd.gb_inertia(); mix = gd.gb_genmix()              # NESO outturn inertia (GVA.s) and generation mix
p   = gd.prices('ERCOT', 'HB_HOUSTON', '2023-01-01', '2024-01-01', kind='rt')   # also 'da', 'rt15'; CAISO
w   = gd.weather('ERCO', '2023-01-01', '2023-02-01')      # BA-level weather; gd.weather_points('ERCO') for weights
net = gd.network('case_ACTIVSg2000')                      # MATPOWER-style network dict (+ approximate bus lat/lon)
b   = gd.baselines()                                      # exam numbers of the classical baselines
```

**Rule for Phase 2:** only `dev` may be looked at for calibration, gates and any fitting. `exam` is for the final
report only. Simulated training may use real *inputs* (weather, demand shapes, prices, quiet frequency noise) from the
`train_inputs` years only (dev-year inputs also for calibration). `reserve` (2026) is untouched in Phase 2.

## 1. What is where

| Path (under `shared/grid/data/`) | What |
|---|---|
| `proc/eia930/ba/<BA>.parquet` | EIA-930 hourly per BA, 2018-01 .. 2026-11 (all 71 BAs in the files; focus BAs keep reported fuel columns too) |
| `proc/eia930/interchange.parquet`, `subregion.parquet` | hourly BA-to-BA interchange (23.4 M rows), subregion demand (6.1 M rows) |
| `proc/eia930/quality_summary.json` | per-BA coverage, imputation shares, closure statistics (by year for focus BAs) |
| `proc/fleet/fleet_<BA>.json/.parquet` | per-generator fleet in the generator spec, 8 focus BAs; `defaults.json` = every default + source |
| `proc/campd/hourly/tx_<year>.parquet` | EPA CAMPD hourly unit data, every Texas unit, 2018-2025 (28.0 M unit-hours) |
| `proc/campd/units.parquet` | 452 CAMPD units: names, type, fuel, EIA plant/generator crosswalk, BA, hours |
| `proc/campd/unit_characteristics*.parquet` | per unit (2018-2022 only): pmax, min stable load, observed ramps, run/off lengths, starts, heat-input fit |
| `proc/frequency/GB/events/*.npz/.json` | 1 s GB frequency windows (-120 s .. +540 s) around every detected event (contract format) |
| `proc/frequency/GB/1s/<YYYY-MM>.npz/.json` | full months of 1 s GB frequency (int16 mHz offsets from 50 Hz) |
| `proc/frequency/GB/minute_summary.parquet` | per-minute min/max/mean and largest 10 s fall/rise of GB frequency |
| `proc/frequency/GB/inertia.parquet`, `genmix.parquet` | NESO outturn + market inertia per settlement period; GB generation mix |
| `proc/frequency/gb_events.parquet`, `ercot_events.parquet` | event tables (section 5) |
| `proc/prices/ercot_dam.parquet`, `ercot_rtm.parquet`, `ercot_rtm_hourly.parquet` | ERCOT hub + load-zone SPPs 2018-01 .. 2026-11 |
| `proc/prices/caiso_dam.parquet`, `caiso_rtpd.parquet` | CAISO hub + DLAP LMPs (with components) 2023-08 .. 2026-11 |
| `proc/weather/<BA>.parquet`, `GB.parquet`, `points.json`, `asos_choice.json` | BA-level hourly weather and its weights |
| `proc/networks/<case>.json` + `_bus/_gen/_branch.parquet`, `index.json` | 11 test networks |
| `proc/splits.json` | frozen splits + sha256 |
| `proc/baselines/` | baseline predictions and `summary.json` |
| `proc/checks/qc_summary.json` | cross-source checks (section 10) |
| `raw/` | downloads kept for rebuild (PUDL tables, ERCOT zips, ERCOT ops reports, NASA POWER/ASOS points, NESO pieces) |

Disk: about 1.7 GB in total (budget 6 GB).

## 2. Sources, licences, citations

| Source | Licence | Citation / URL |
|---|---|---|
| EIA-930 Hourly Electric Grid Monitor, six-month bulk CSVs (BALANCE, INTERCHANGE, SUBREGION) | US public domain | U.S. EIA, eia.gov/electricity/gridmonitor/sixMonthFiles/ (no key) |
| EIA-860 / EIA-860M / EIA-923, via PUDL parquet tables | PUDL data CC-BY-4.0; EIA public domain | Catalyst Cooperative, Public Utility Data Liberation project, s3.us-west-2.amazonaws.com/pudl.catalyst.coop/stable/ (out_eia__yearly_generators, core_eia860__scd_generators_wind/solar/energy_storage, out_eia923__yearly_generation_fuel_by_generator, out_eia923__monthly_fuel_receipts_costs, out_eia__yearly_plants, out_eia861__yearly_balancing_authority_service_territory, out_censusdp1tract__counties) |
| EPA-EIA crosswalk | CC-BY-4.0 (PUDL), from EPA's camd-eia-crosswalk | PUDL core_epa__assn_eia_epacamd |
| EPA CAMPD hourly emissions bulk files | US public domain | EPA Clean Air Markets Program Data, api.epa.gov/easey/bulk-files/emissions/hourly/state/emissions-hourly-<year>-tx.csv (file route needs no key; the file *listing* endpoint needs an api.data.gov key and was not used) |
| RTS-GMLC generator data (defaults by unit type) | open (GridMod GitHub) | C. Barrows et al., "The IEEE Reliability Test System: A Proposed 2019 Update", IEEE Trans. Power Syst. 35(1), 2020; github.com/GridMod/RTS-GMLC |
| NESO system frequency (1 s), system inertia, historic generation mix | NESO Open Data Licence | National Energy System Operator data portal (api.neso.energy CKAN datastore API) |
| NESO/ESO and Ofgem 9 Aug 2019 reports | Crown / NESO copyright, cited | "Technical Report on the events of 9 August 2019", National Grid ESO, 6 Sep 2019; Ofgem, "9 August 2019 power outage report", Jan 2020 |
| Elexon BMRS REMIT messages and B1610 metered output | Elexon BMRS licence (free use with attribution) | data.elexon.co.uk/bmrs/api/v1/datasets/REMIT, /B1610 |
| ERCOT monthly Operations Reports (frequency events) | public ERCOT documents | ercot.com/files/docs/... (85 monthly reports, list in raw/ercot_ops/urls.tsv) |
| ERCOT historical DAM / RTM settlement point prices (report types 13060, 13061) | public ERCOT market data | ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId=13060/13061 |
| CAISO OASIS LMPs (PRC_LMP DAM, PRC_RTPD_LMP) | public CAISO data | oasis.caiso.com/oasisapi/SingleZip |
| NASA POWER hourly (MERRA-2 meteorology, CERES SYN1deg irradiance) | free, cite NASA POWER | power.larc.nasa.gov/api/temporal/hourly/point (NASA LaRC POWER Project) |
| ASOS hourly METARs | free use, credit IEM | Iowa Environmental Mesonet, mesonet.agron.iastate.edu/cgi-bin/request/asos.py |
| MATPOWER test cases (IEEE 9/14/30/39/57/118/300, RTS-GMLC) | case data: see each file header (IEEE cases public; MATPOWER code BSD-3) | R. D. Zimmerman et al., MATPOWER, github.com/MATPOWER/matpower data/*.m |
| Texas A&M ACTIVSg200/500/2000 (MATPOWER copies) | CC-BY-4.0 | A. B. Birchfield et al., "Grid Structural Characteristics as Validation Criteria for Synthetic Networks", IEEE TPWRS 32(4), 2017 |
| US Census 2020 Gazetteer places (TX, IL, SC) | US public domain | census.gov Gazetteer files (used to place ACTIVSg buses) |

## 3. EIA-930 (DemandCast, DispatchNet)

- **Coverage.** Every BA in the files, 2018-01-01 .. 2026-11-14 (UTC), 77,736 hours. Fuel mix starts 2018-07-01 (the
  2018 Jan-Jun files have no fuel columns). Subregion demand starts 2018-07 (the 2018 Jan-Jun SUBREGION file is not
  published). The 2026 Jul-Dec file was partial at download (to mid-November).
- **Time.** EIA gives "UTC time at end of hour"; we store hour-BEGINNING UTC (`time_utc` = end - 1 h).
- **Columns.** `d` demand, `ng` net generation, `ti` total interchange (EIA "Adjusted" = reported value after EIA's
  screening and imputation); `df` day-ahead demand forecast (as reported); `d_rep/ng_rep/ti_rep` as reported;
  `ng_COL, ng_NG, ng_NUC, ng_OIL, ng_WAT, ng_SUN, ng_WND, ng_OTH, ng_UNK` (MW). From 2024-07 EIA split hydro / solar / wind
  and added storage and geothermal; we keep the new columns (`ng_WAT_HY, ng_WAT_PS, ng_SUN_NOBAT, ng_SUN_BAT,
  ng_WND_NOBAT, ng_WND_BAT, ng_BAT, ng_OES, ng_UES, ng_GEO`) and rebuild `ng_WAT/SUN/WND` as their sums. Before 2024-07,
  batteries and geothermal sit inside `ng_OTH` (EIA's earlier category); after, they are separate. (EIA misspells one
  header "SOLAR_WITHO_INTEGRATED..."; handled.)
- **Flags.** `d_imp, ng_imp, ti_imp, fuel_imp` = EIA imputed the value (EIA's own anomaly screening replaced a missing
  or anomalous report). Our checks: `q_missing_row`, `q_d_missing`, `q_d_nonpos`, `q_d_spike` (> 20% off the +-3 h
  median and not imputed), `closure_resid = ng - ti - d` and `q_closure_bad` (|resid| > max(2% of d, 50 MW)),
  `fuel_resid = ng - sum(fuels)` and `q_fuel_sum_bad`, `q_df_missing`. **`q_ok`** (demand usable as a label) = present,
  positive, not a spike, not imputed.
- **Quality of the focus BAs** (exam years 2023-25; `proc/checks/qc_summary.json`): `q_ok` 99.6-99.98%; demand
  imputed 0-0.36%; DF missing <= 0.4%.
- **Balance closure (demand = generation - interchange)** holds exactly in ERCO, ISNE, NYIS, SWPP (by construction:
  these BAs' demand is derived from the other two), but **not** in CISO (resid median -354 MW in exam years, 62% of
  hours fail), MISO (+1,555 MW, 54%), BPAT (-3,810 MW, 81%) and PJM (13% fail; and PJM's `ti` differs from the sum of
  its BA-to-BA interchange by a median 1.8 GW). These BAs report demand from their own metering and their NG/TI scope
  differs (pseudo-ties, dynamic schedules, BPAT federal hydro serving other BAs). Use `d` as the demand label; do not
  derive demand from NG and TI there.
- **ERCOT is the cleanest BA** (islanded, closes exactly, 0.09% imputed): the natural primary target.

## 4. Generator fleet (simulator input)

- **Units.** One entry per EIA-860 generator that was "existing" in any EIA-860/860M report 2017-2026 and not retired
  before 2018-01-01 (retired and new units included, with `online_date`, `retired_date`, `status_latest`). BA = the
  plant's latest EIA-860 BA code. ERCO 2,393 generators (175.7 GW existing at the latest report, incl. 40.6 GW wind,
  33.8 GW solar, 18.9 GW batteries, 39.6 GW gas CC, 13.6 GW coal, 5.0 GW nuclear); CISO 3,206; PJM 4,663; MISO 5,590;
  SWPP 1,837; ISNE 2,482; NYIS 2,160; BPAT 410.
- **Fields** follow the GRID_BRIEF spec; unknowns are None. Each entry has `src` (where each filled value came from).
  Extras: `nameplate_MW`, `winter_MW`, `eia_technology`, `prime_mover`, `cc_part` (CT / CA / CS for combined-cycle
  parts; aggregate a CC block with `unit_id_pudl`), `lmp_node`, `hr_eia923_2018_2022_MMBtu_per_MWh`,
  `ramp_obs_p99_MW_per_h`, `starts_per_year_obs`, `campd_linked`, `cold_start_code_eia`, `design_wind_speed_mph`.
- **Sources in order of preference** (all defaults with their source in `proc/fleet/defaults.json`):
  - `pmax_MW` = EIA-860 summer capacity (else nameplate). `pmin_MW` = EIA-860 minimum load (1,299 of 2,393 ERCO units),
    else CAMPD min stable load (2018-22), else RTS-GMLC PMin/PMax by type. Batteries: pmin = -max charge rate.
  - `heat_rate_curve` ([MW, MMBtu/MWh] average heat rate points): CAMPD quadratic fit of hourly heat input vs gross load
    on steady hours 2018-22, only where one CAMPD unit maps to exactly one generator (185 ERCO units; gross basis); else
    the RTS-GMLC curve shape scaled so its 80%-load value equals the generator's EIA-923 2018-22 annual average
    (net basis; 1,076 ERCO units).
  - Ramp capability, min up/down, start time, start fuel: RTS-GMLC medians by unit type (CC 1.2 %/min, CT 6.7 %/min,
    coal steam 1.9 %/min; gas steam and biomass use the coal STEAM row; gas_ice the oil CT row - choices, not data).
    Min up/down from CAMPD run lengths where linked. The CAMPD hourly ramp (`ramp_obs_p99_MW_per_h`) is what units
    actually did hour to hour - a lower bound on capability, not used as the limit.
  - `start_cost_$` = RTS-GMLC warm-start fuel per MW x pmax x fuel cost. `fuel_cost_$_per_MMBtu` = EIA-923 receipts
    2018-22, receipt-weighted by state and fuel group (nuclear 0.65 $/MMBtu). `co2_kg_per_MMBtu` = 40 CFR 98 Table C-1.
  - `inertia_H_s` = RTS-GMLC by type (CC 5.0, CT 2.8, coal/gas steam 3.0, nuclear 5.0, hydro 3.5 s); IBR 0; geothermal
    3.0 (low end of Kundur Table 3.2). `mva_base` = nameplate / EIA-860 power factor (default 0.85).
  - `governor` = droop 5%, deadband 0.036 Hz (NERC BAL-001-TRE-2 / NERC PFR guideline), nominal T_s by tech
    (steam 7 s, CC 3 s, CT/ICE 0.5 s, hydro 1 s); nuclear headroom 0. `ffr` for ERCO batteries: trigger 59.85 Hz,
    delay 0.25 s, MW unknown.
  - Storage: `energy_MWh` from EIA-860, `eta_rt` 0.85 (battery, assumption) / 0.78 (pumped, assumption).
  - Wind: `hub_height_m` (EIA-860), `power_curve` = "manufacturer|model" string (no curve table), design wind speed.
    Solar: `tracking` (fixed / single / dual), `dc_ac`, `tilt`, `azimuth` (EIA-860 where reported).
- **PlantDetective leakage guard:** CAMPD-derived values are exam truth for exam plants. `gd.fleet(ba, plant_safe=True)`
  replaces them (pmin, heat-rate curve, min up/down, observed ramps/starts) with values that use no CAMPD data. Use it
  for anything PlantDetective trains on.
- **Limits.** EIA-860 has no reservoir / run-of-river split: all conventional hydro is `hydro_res`. Ramp, min up/down,
  start costs and inertia are type defaults, not unit data. Battery FFR MW per unit is unknown.

## 5. Frequency (GridPulse)

### GB 1 s frequency (NESO)
- NESO's direct CSV downloads redirect to a Cloudflare R2 storage host that this sandbox's proxy refuses (403). All
  NESO data were taken through NESO's documented CKAN datastore SQL API on api.neso.energy, one query at a time.
- **Every month 2018-01 .. 2026-10** was summarised per minute on the server (min, max, mean, largest 10 s fall and rise):
  `gd.frequency_minutes()`. Full 1 s months stored: see `gd.frequency_months()` (2019-08 plus train-year months used as
  quiet noise: 2018-07, 2020-07, 2022-01; 2018-07 lacks its last UTC hour, which sits in NESO's August local-time file).
  Around every event an 11-minute 1 s window is stored.
- Timestamps: 2018-2019 files are local time with a UTC offset suffix (+0000/+0100); 2020+ files are UTC (checked:
  86,400 s on the DST change days). Everything is converted to UTC. The Dec 2019 file stops on 15 Dec 16:59 UTC.
- 1 s values are averages / samples at 1 s with 1 mHz resolution; the true initial RoCoF (over 100-500 ms) is
  smoothed. Data gaps are NaN.

### GB events
- **Detection.** A minute is a candidate when its largest 10 s fall is <= -0.15 Hz; candidates within 10 min are
  merged. 470 events 2018-01 .. 2026-10 (27 in 2018 .. 78 in 2024; the count rises as inertia falls). For each, a 1 s
  window (-120 s .. +540 s) is stored and measured: onset `t0_utc` (start of the steepest fall), `f0_Hz` (mean of the
  60 s before), `nadir_Hz` (minimum in 90 s; 90 s so that 9 Aug 2019's second trip and 48.787 Hz nadir at +76 s are
  caught) and `nadir_15s_Hz`, `t_nadir_s`, `rocof_1s_Hz_s` / `rocof_2s_Hz_s` (largest 1 s / 2 s fall in the first
  5-6 s), `f_60s_Hz`. Median nadir 49.81 Hz; 8 events below 49.6 Hz. Not every event is a trip: some are fast
  demand / interconnector ramps.
- **Context per event:** NESO outturn and market-provided inertia of the settlement period (GVA.s; 439 events, the
  dataset stops 2026-07-26), and the NESO generation mix of the half hour (`mix_*_MW`).
- **Loss labels (170 events)** - read `loss_confidence`:
  - *high* (1): 9 Aug 2019, from the NESO (ESO) technical report: 1,131 MW at onset (~150 MW embedded vector-shift +
    737 MW Hornsea + 244 MW Little Barford ST), 1,481 MW after embedded RoCoF trips, 1,691 MW after Little Barford GT1a
    (+58 s), 1,878 MW in total; LFDD shed ~931 MW at 48.8 Hz. `loss_MW` = 1,131 (`loss_MW_total` 1,878).
  - *medium* (98): an interconnector import that fell by >= 300 MW to <= 30% across the onset in Elexon FUELINST
    (5-min flows; 71 events, e.g. 22 Dec 2023 IFA 998 MW -> 0, nadir 49.28 Hz), and/or Elexon REMIT unplanned outages
    starting within 2 min of the onset whose unit produced >= 50 MW in the settlement period before (Elexon B1610).
    Loss = import lost + sum of min(B1610 output, capacity lost).
  - *low* (71): REMIT matches within -5 .. +8 min of the onset, or without B1610 output (capacity lost used).
  - The matching window was set from the REMIT-start-minus-onset histogram (peak at -2..0 min; matches 10-20 min away
    are mostly coincidental), not from the frequency response. REMIT times are often rounded (Hornsea's 9 Aug message
    says 16:00 for a 15:52 trip). Labels remain noisy: implied inertia E = loss f0 / (2 RoCoF) differs from NESO's
    outturn by a median 37% on the exam events (1 s RoCoF smoothing explains part of this).
- **Exam subsets:** 412 exam events (2018-2025 minus dev), 150 with a loss label (86 medium), 410 with NESO inertia.
- **Label corrections** (`data/sources/gb_event_label_corrections.csv`, committed; applied by `gd.events` to the
  shipped v1.0 parquet and by `gb_events.py build` to a rebuild; `loss_MW_v1` keeps the v1.0 label, `loss_corrected`
  marks the row, `loss_note` gives the reason and source; the v1.0 files are not edited, MANIFEST.json still matches).
  - GB_20191203T200030 (dev), 14 Nov 2026: **1,787.7 -> 854.7 MW**. The v1.0 label was 854.7 MW from Spalding CCGT
    (T_SPLN-1: REMIT unplanned "Plant Fault" 870 -> 0 MW from 20:00Z, published 20:07:30Z; B1610 427.35 MWh in
    settlement period 40, 19:30-20:00 UTC, i.e. 854.7 MW; 7.4 MWh in SP41) plus 933 MW "interconnector import lost
    (INTFR)". FUELINST shows IFA ramping 1,900 -> 1,751 -> 1,262 -> 809 -> 329 -> 204 MW over 19:45-20:10, the same
    shape as its hour-boundary ramps at 17:00, 19:00 and 22:00 that evening; the rule compared 1,262 MW (19:55 slot)
    with 329 MW (20:05 slot). No interconnector REMIT message. The other REMIT match, T_SEEL-1 (Spalding expansion
    OCGT, 299 MW), was producing -0.5 MW and was already excluded. Physics check: 2 x 333 GVA.s x 0.057 Hz/s / 49.98 Hz
    = 760 MW from the smoothed 1 s RoCoF (a lower bound). No event-specific NESO report was found. Source: Elexon
    BMRS REMIT, B1610, FUELINST (data.elexon.co.uk/bmrs/api/v1/datasets), re-queried 14 Nov 2026. Confidence stays
    *medium*; the splits are unchanged (`make_splits.py` chose this event as one of the three largest medium losses
    of 2019/2021 using the v1.0 label; the frozen split is kept).
  - Same failure mode, **not corrected** (exam events; listed for a later check): the FUELINST rule also fires on
    hour-boundary ramps and on exports that grow (it tests `before - after >= 300` and `after <= 0.3 x before`, which
    a negative `before` passes). FUELINST shows ramps or reversals, not steps, at GB_20200214T085638 (INTNED 650 ->
    -511, INTNEM 307 -> -590 MW), GB_20200522T185814 (INTFR export -183 -> -561), GB_20200810T195818 (INTNEM export
    -439 -> -805), GB_20220625T230149 (INTFR export -581 -> -1,031), GB_20230707T005707 (INTNEM export -252 -> -713),
    GB_20240828T195646 (INTFR 549 -> -239 over 15 min). Interconnector-only labels of ~1,000 MW that drop in one
    5-min slot to ~0 look like real trips.

### ERCOT events
- From ERCOT's public monthly Operations Reports ("Frequency Events"): 85 of 102 months 2018-01 .. 2026-06 were found
  (missing months listed in `proc/frequency/ercot_events_coverage.json`). Reported criteria: delta f >= 60 mHz or
  loss >= 350 MW. Per event: time (Central -> UTC; minute resolution, seconds from 2023), delta f, min (or max) f,
  duration, PMU oscillation mode and damping, MW loss (negative = load loss), load, wind (early reports) / IRR share,
  inertia (MW*s; the 2018 header says "GW-s" but the numbers are MW*s).
- 329 events (326 generation losses, 3 load losses): median loss 541 MW, largest 2,519 MW on 4 Jun 2022 (the NERC
  "Odessa 2022" solar disturbance; min 59.70 Hz), 1,378 MW on 15 Feb 2021 (Winter Storm Uri), 1,299 MW on 9 May 2021
  (Odessa 2021). Inertia 130-387 GW*s. These are official labels (loss and inertia from ERCOT) and the cleanest
  event labels in this package - but there is no trace, only delta f and min f.
- No public 1 s ERCOT frequency trace was found; ERCOT events have summary metrics only. NERC/Texas RE annual
  assessments show event scatter plots but no tables.

## 6. EPA CAMPD (PlantDetective truth)

- Hourly data of every Texas unit 2018-2025 (8 annual state files, 766-875 MB CSV each, streamed to parquet):
  `op_time` (fraction of the hour operating), `gross_load_MW` (average while operating), `heat_input_MMBtu`,
  `co2_short_tons`, `nox_lbs`, `so2_lbs`, `steam_klb_h`, and the heat-input / CO2 measurement indicators (`hi_meas`:
  Measured, Calculated, Substitute, ...). 452 units; 373 in ERCO, 43 SWPP, 23 MISO, 13 EPE. CAMPD hours are local
  standard time; converted to UTC with the plant's EIA timezone (no DST).
- **Linking:** 447 of 452 units link to EIA plant/generator ids through the EPA-EIA crosswalk (PUDL). A CAMPD unit can
  map to several generators (a CC's CT + steam part) and vice versa.
- **Gross vs net:** CAMPD gross load is at the generator terminals; EIA-923/EIA-930 are net. ERCO coal + gas:
  EIA-930 net / CAMPD gross = 0.92-0.95 per year (2019-2025) - consistent with station service.
- **Combined cycles:** many CC units report only the combustion turbine's output as gross load (the heat-recovery
  steam turbine is a separate or unmonitored generator). Their CAMPD heat rate is therefore CT-like (~9-10 MMBtu/MWh),
  not the CC block's ~7. This is the main reason the EIA-923 baseline is 22% low for gas CC (section 11).
- Per-unit characteristics from 2018-2022 hours (`unit_characteristics.parquet`): median pmin fraction 0.50, median
  observed p99 hourly ramp 0.35%/min of pmax (21%/h), p5 run length 2.9 h, starts 74/yr (median unit), heat-input
  quadratic fit residual 1.5% (median).
- TX total CO2 in CAMPD: 215 (2018), 201, 184, 189, 196, 191, 191, 194 Mt (2025).

## 7. Prices (BatteryPilot)

- **ERCOT** settlement point prices at the hubs (HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_WEST, HB_PAN, HB_BUSAVG, HB_HUBAVG)
  and load zones (LZ_*): DAM hourly and RTM 15-min, 2018-01-01 .. 2026-11-08, from ERCOT's public annual archives.
  Central Prevailing Time with a repeated-hour flag -> UTC, interval-beginning. `ercot_rtm_hourly` = mean of the four
  15-min SPPs. One RTM row in 2019 had no time key and was dropped. Prices include scarcity adders (e.g. 9,000 $/MWh
  in Feb 2021).
- **CAISO** OASIS LMPs (with energy/congestion/loss components) at TH_NP15/SP15/ZP26 and DLAP PGAE/SCE/SDGE:
  DAM hourly and RTPD 15-min. **OASIS only serves about 3 years**: data start 2023-08-03. So CAISO has no 2019/2021
  dev years; its split is dev = 2023-08..12, exam = 2024-25 (section 9).

## 8. Weather and test networks

- **Weather points.** Per US focus BA: 8 population points (weighted k-means of the BA's EIA-861 service-territory
  counties, weight = county population, location = population-weighted county internal points), 6 wind points and 6
  solar points (weighted k-means of the BA's EIA-860 wind / solar plants by capacity). GB: 12 cities weighted by
  approximate built-up-area population (ONS 2021 / NRS, rounded), 7 wind clusters and 5 solar areas, equal weights.
  Points and weights: `gd.weather_points()`.
- **Series.** NASA POWER hourly at every point (T2M, RH2M, QV2M, WS10M, WS50M, WD50M, PS, all-sky and clear-sky GHI),
  2018-01-01 .. 2026-10-31 UTC; BA series = weighted means (temperature/humidity on population points, 50 m wind on
  wind points, GHI on solar points). US also: ASOS station nearest each population point (archive since <= 2016, still
  active, >= 85% coverage), population-weighted T, dew point, RH, wind.
- **Checks.** NASA POWER vs ASOS population-weighted temperature: MAE 1.3-1.9 degC, corr 0.96-0.99 (ERCO MAE 1.65,
  bias -0.24 degC); bias up to -1.2 degC in ISNE/NYIS and -2.4 degC in BPAT (MERRA-2 cells average over terrain).
  CERES irradiance ends ~2026-08 (lag): GHI is NaN after that (98.1% coverage overall). Use ASOS temperatures for
  demand where possible; NASA POWER for wind/irradiance.
- **Networks** (`gd.network`, `gd.networks()`): IEEE 9, 14, 30, 39, 57, 118, 300, RTS-GMLC (73 bus), ACTIVSg200
  (Illinois), ACTIVSg500 (South Carolina), ACTIVSg2000 (Texas, 2,000 buses, 544 gens, 3,206 branches, 67.1 GW load).
  MATPOWER units (MW, MVAr, p.u. on baseMVA, kV, degrees). Bus/generator/branch counts and loads of case39/118 match
  pandapower's built-in copies (case300 load differs by 1.4% in pandapower's own conversion).
  The Texas A&M site gates its downloads behind a sign-up form (email), so the ACTIVSg cases come from MATPOWER's
  copies (same CC-BY-4.0 data) and have no published coordinates. **Approximate bus coordinates** were added from the
  bus names (Texas A&M synthetic buses are named after real towns): Census Gazetteer place centroid; 94.5% of ACTIVSg2000
  buses, 99% of ACTIVSg200, 93.4% of ACTIVSg500 matched; several buses share one town point.

## 9. Splits (`proc/splits.json`, seed 20261113)

- **Time** (demand, dispatch, prices, weather inputs): train_inputs = 2018, 2020, 2022; **dev = 2019, 2021**;
  **exam = 2023, 2024, 2025**; reserve = 2026 (not used in Phase 2; a later exam). Calendar years in UTC.
  *Why:* alternating years spread seasons and fleet changes over every split; dev includes Winter Storm Uri (Feb 2021,
  ERCOT) and 2019's 9 Aug GB event, so calibration sees extremes; the exam is the three most recent full years, which
  is how the models would be used, and it has a much larger battery/solar fleet than dev (a real test of
  extrapolation). CAISO prices: dev 2023-08..12, exam 2024-25 (OASIS history limit).
- **Plants (PlantDetective):** split by EIA plant, never by hour. dev = 10 ERCOT plants drawn by seed, stratified by the
  plant's dominant technology among plants with >= 8,000 CAMPD operating hours still reporting in 2025 (2 coal, 3 gas
  CC, 3 gas CT, 2 gas steam): W A Parish, J K Spruce (coal); Odyssey Energy Altura Cogen, Baytown Energy Center, South
  Houston Green Power (CC); Sweeny Cogeneration, Port Comfort Peaking, Mark One (CT); Trinidad, Roland C. Dansby (gas
  steam) - 40 units. exam = every other CAMPD Texas plant (407 units, all years 2018-2025).
- **Events:** GB dev = 9 Aug 2019 (named) + the three largest medium-confidence losses in the dev years:
  GB_20190701T072734 (Nemo Link import 1,000 MW), GB_20190711T075329 (Peterhead 1,153 MW), GB_20191203T200030
  (1,788 MW in v1.0; corrected to 854.7 MW, section 5). ERCO dev = 5 seeded events from 2019/2021: ERCO_20190421T203000, ERCO_20190523T214100,
  ERCO_20210117T033500, ERCO_20210501T031400, ERCO_20210519T170600. exam = every other event before 2026 (GB 412,
  ERCO 301); 2026 events = reserve (GB 54, ERCO 23). *Why:* 9 Aug 2019 is the simulator's reference case (GridSim
  reproduces it), so it cannot be an exam item; a handful of large, well-documented dev events is enough to calibrate
  the loss/RoCoF/nadir relation; everything else stays unseen.
- **Noise for simulated frequency events:** quiet 1 s data from train_inputs years, >= 1 h from any event.

## 10. Data problems Phase 2 must know

1. **Balance closure fails for CISO, MISO, BPAT, PJM** (section 3); ERCO/ISNE/NYIS/SWPP close by construction.
2. **CISO day-ahead forecast is misaligned and biased:** in 2023-25 it matches demand best when shifted by +1 h (MAPE
   5.5/7.0/8.1% unshifted vs 3.6/5.5/7.1% shifted) and runs 2-6% low; 2019 has a +30% bias (different footprint).
   SWPP's DF is 6.5% high in 2025. The baseline uses the published DF as is.
3. **Fuel categories changed in 2024-07** (section 3). Fuel mix is absent before 2018-07.
4. **CAMPD CC units:** gross load often excludes the steam turbine (section 6). Compare heat rates within the CAMPD basis.
5. **Gross vs net:** CAMPD gross is 5-8% above EIA net.
6. **Fleet defaults are type-level** (ramps, start costs, inertia, governors). Only pmax, pmin (55% of ERCO units),
   heat rates and dates are unit data.
7. **CAISO prices only from 2023-08.** ERCOT prices are complete.
8. **NASA POWER temperature bias** vs stations up to -2.4 degC (BPAT); GHI missing after ~2026-08.
9. **GB 1 s data smooth RoCoF;** 2018-19 local-time stamps; Dec 2019 ends on the 15th; NESO's inertia dataset updates
   are "temporarily paused" (data to 2026-07-26).
10. **Event loss labels** (GB) are REMIT-based except 9 Aug 2019: REMIT event times are often rounded or late, and the
    capacity lost is not always the output lost (B1610 half-hour output is used when available). The FUELINST
    interconnector rule can count an hour-boundary ramp as a trip (3 Dec 2019 corrected; 6 exam suspects listed in
    section 5).
11. ERCOT event times have minute resolution before 2023 and are local times.
12. **EIA-930 solar timestamps move between years** (added 14 Nov 2026; `scripts/qc_solar.py`,
    `gd.eia930_solar_qc()`): the shift that best aligns reported solar (`ng_SUN`) with the irradiance at the BA's solar
    points. CISO leads the sun by 1 h in 2018-2021 (correlation 0.88-0.92 unshifted, 0.90-0.93 shifted), is aligned in
    2022-2023, and lags by 1 h in 2024-2025 (0.86-0.88 vs 0.92-0.94); PJM leads by 1 h through 2024 and is aligned
    from 2025; the other six focus BAs are aligned in every year (NYIS has no solar column). CISO also reports negative
    solar at night in 61-97 % of night hours (BPAT 21-60 % since 2020). DemandCast's solar loss is explained by this
    (its dev fit learned CISO's 2019/2021 offset). A model may use the flags for training-input and dev years; using
    exam-year flags to choose anything would use exam data.

## 11. Baselines (exam numbers; `proc/baselines/summary.json`)

### Demand, day-ahead (EIA-930 `d` where `q_ok`, exam 2023-25, hours where every predictor exists)
MAPE %:

| BA | persistence 24 h | same hour last week | BA's own day-ahead forecast | dev-year climatology (scaled) |
|---|---|---|---|---|
| ERCO | 4.56 | 8.38 | **2.41** | 7.97 |
| CISO | **4.91** | 6.47 | 6.87 | 7.66 |
| PJM | 4.95 | 8.32 | **3.48** | 7.81 |
| MISO | 4.18 | 7.01 | **2.71** | 6.47 |
| SWPP | 4.33 | 8.22 | **4.32** | 7.58 |
| ISNE | 7.27 | 10.65 | **2.56** | 10.06 |
| NYIS | 5.35 | 8.22 | **2.63** | 7.85 |
| BPAT | 3.70 | 6.43 | **2.00** | 5.98 |

Dev (2019, 2021) for reference, ERCO: 5.44 / 9.72 / 2.61 / 8.10. RMSE, MAE, bias and by-year numbers are in the
summary. DemandCast must beat the BA's own forecast to be useful.

### Heat rate (PlantDetective; CAMPD steady full hours; split by plant, all years)
Predict hourly heat input = heat rate x gross load. Unit-years with >= 50 steady measured hours and a matching
EIA-923 record. Exam: 2,308 unit-years, 367 units, 125 plants.

| exam | EIA-923 annual average (same year) | dev-plant median by tech |
|---|---|---|
| all: median abs. error of annual heat input | **13.4%** (bias +1.9%) | 14.9% |
| all: median hourly APE | 13.6% | - |
| coal steam (168) | 8.9% (bias +8.5%: net vs gross) | 4.5% |
| gas CC (966) | 22.1% (bias -21.9%: CT-only gross load) | 18.1% |
| gas CT (625) | **4.7%** | 8.3% |
| gas steam (410) | 11.9% (bias +11.4%) | 16.2% |

### Battery (BatteryPilot): 1 MW / 2 MWh, round-trip 0.85, one daily cycle window, $/kW-year
Perfect foresight (LP, upper bound) on RT and DA prices; daily-spread rule (charge in the 2 cheapest DAM hours,
discharge in the 2 dearest later DAM hours, settled at RT, no foresight); threshold rule (dev-period RT quartiles).

| node | year | PF RT | PF DA | daily spread | threshold |
|---|---|---|---|---|---|
| ERCOT HB_HOUSTON | 2023 / 2024 / 2025 | 152.6 / 57.3 / 45.0 | 135.6 / 52.2 / 38.7 | 94.5 / 37.9 / 25.8 | 10.2 / 11.3 / 8.9 |
| ERCOT HB_NORTH | 2023 / 2024 / 2025 | 151.2 / 59.6 / 49.1 | 135.4 / 54.1 / 41.7 | 95.7 / 38.7 / 27.8 | 9.1 / 13.0 / 10.1 |
| ERCOT HB_WEST | 2023 / 2024 / 2025 | 160.0 / 69.2 / 54.6 | 140.1 / 61.3 / 46.2 | 99.5 / 42.5 / 31.8 | 13.0 / 16.1 / 13.6 |
| ERCOT HB_SOUTH | 2023 / 2024 / 2025 | 137.3 / 59.2 / 45.3 | 129.1 / 52.3 / 38.6 | 83.8 / 39.5 / 26.3 | 9.6 / 11.5 / 9.6 |
| CAISO TH_NP15 | 2024 / 2025 | 37.6 / 27.5 | 27.3 / 21.2 | 23.1 / 16.8 | -13.0 / -16.8 |
| CAISO TH_SP15 | 2024 / 2025 | 49.3 / 33.5 | 43.7 / 29.2 | 35.0 / 22.4 | -3.3 / -9.6 |

A good BatteryPilot should land between the daily-spread rule and the perfect-foresight bound.

### Frequency (GridPulse)
GB exam events with a loss label (150; 148 with NESO inertia). RoCoF formula dP = 2 E RoCoF / f0, RoCoF = largest
1 s fall in the first 5 s of the 1 s data.

| GB exam | loss MdAPE | inertia MdAPE |
|---|---|---|
| RoCoF formula, assumed E = 230 GVA.s (median NESO outturn, dev years) | **31.3%** (medium labels: 27.1%) | - |
| RoCoF formula with the event's own NESO outturn inertia (oracle input) | 27.8% | - |
| E from (label loss, RoCoF) | - | 37.4% |
| constant E = 230 GVA.s | - | **19.7%** |
| dumb: dev median loss | 63.4% (52.4% with the corrected dev label) | - |

The table is the v1.0 `summary.json`. With the corrected 3 Dec 2019 dev label (re-run 14 Nov 2026, same script) the
dev median loss falls from 1,142 to 1,066 MW and only the dumb row moves on the exam (63.4 -> 52.4%); on the 4 dev
events: RoCoF @230 33.4 -> 30.7%, RoCoF @own inertia 22.8 -> 12.9%, dev median 7.6 -> 7.1%, E from (label, RoCoF)
vs NESO 28.6 -> 12.7%.

ERCOT exam (292 generation-loss events): loss = k x delta f with k = 4,668 MW/Hz (median over the 5 dev events):
MdAPE **20.1%**; dumb dev-median loss 24.5%. (Dev: 9.7% / 7.9% on 5 events.)

## 12. Blocked or not found

- NESO direct file downloads (Cloudflare R2 host refused by the proxy): used the NESO datastore API instead.
- Texas A&M electricgrids download form (asks for an email): used MATPOWER copies; coordinates approximated.
- CAMPD bulk-file *listing* needs an api.data.gov key (not used; the file URLs follow CAMPD's documented pattern and
  need no key).
- CAISO OASIS history before 2023-08 (not served). ERCOT ops reports for 21 months not found by search.
- No public 1 s ERCOT frequency data; NERC FRAA reports on nerc.com returned an HTML page instead of the PDF.
- nationalgrideso.com (old ESO domain) returned 502; the same reports were taken from ofgem.gov.uk / neso.energy.

## 13. Rebuild

One resumable command runs everything below in order and ends with the hash check (`build.py`; stages with `--list`):

```
mlratchet data grid --from-zip gridsim-griddata-v1.0.zip   # the shipped v1.0 data, verified file by file (MANIFEST.json)
mlratchet data grid                                        # or download every source and rebuild (hours; resumable)
mlratchet data grid --only verify
```

The data root is `$ML_RATCHET_GRID/data` (default `shared/grid/data`). The stages, as originally run:

```
python3 -I scripts/eia930_fetch.py; scripts/build_eia930.py
python3 -I scripts/campd_fetch.py; scripts/campd_units.py; scripts/build_fleet.py
python3 -I scripts/neso_fetch.py inertia|genmix|scan; scripts/gb_events.py detect windows attribute interconnectors build; scripts/neso_months.py
python3 -I scripts/ercot_events.py   (after scripts/ercot_ops_fetch.sh; URL list in sources/ercot_ops_urls.tsv)
python3 -I scripts/ercot_prices.py; scripts/caiso_prices.py; scripts/weather.py points|power|asos|build
python3 -I scripts/networks.py; scripts/finalize.py; scripts/make_splits.py; scripts/baselines.py demand heatrate battery frequency; scripts/qc.py
```

`build.py`'s `fetch` stage also downloads what the original build fetched by hand: the PUDL tables, RTS-GMLC
`gen.csv`, the MATPOWER cases, the Census Gazetteer places, NESO's package lists and the ERCOT reports, and fills
`gridsim/refdata/`. Licences as checked on 14 Nov 2026 are in the repository's `DATA_LICENSES.md`.
