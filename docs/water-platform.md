# The water platform

The fourth shared platform. **WaterSim** is a verified hydrology simulator: snow, soil, groundwater, river channels
and reservoirs, closing its water balance to round-off. **WaterData** is real watershed data, split once and frozen:
671 CAMELS-US basins with ERA5-Land weather and USGS flow, 880 SNOTEL snow pillows, 678 ResOpsUS dams, the National
Water Model's retrospective run, and four other countries' basins for later out-of-domain tests. Four models sit on
it. All four were built on 18 Nov 2026 by parallel agents in a claude.ai sandbox, then brought into this repository
the same day.

| Model | The question |
|---|---|
| [FlowCast](../models/flowcast) | How much water will this ungauged river carry each day, from weather and the basin's attributes alone? |
| [SnowSense](../models/snowsense) | How much water is stored in the snowpack at this station today, from its rain gauge and thermometer alone? |
| [BasinDetective](../models/basindetective) | From nine years of a river's weather and flow, how will its flow change in another decade, and is a dam upstream? |
| [ReservoirPilot](../models/reservoirpilot) | How much should this dam release today, knowing only what an operator knows today? |

Code and data: [`shared/water/`](https://github.com/dogum/ML-Ratchet/tree/main/shared/water) (WaterSim, WaterData,
the checks and the data scripts). The models' `training/_paths.py` import the code from there and read data from
`$ML_RATCHET_WATER` (default the same folder). The platform release (`watersim-waterdata-v1.0.zip`, 10 MB) holds
the code, the attributes, the per-unit baseline scores and the frozen splits, but **no time series**: those are
rebuilt from the public sources (about 2.9 GB on disk, see [Using it](#using-it)).

**The weather caveat comes first.** Every model and every baseline here is driven by ERA5-Land, the only forcing that
could be fetched (NCAR GDEX, which serves the original CAMELS Daymet, Maurer and NLDAS forcing, and the Daymet API
both timed out from the build sandbox). ERA5-Land under-catches mountain and Pacific Northwest precipitation: 50 of
the 671 basins carry more water out than the forcing puts in (runoff ratio above 1). It also drifts between decades,
by about 6 % between 1980-89 and 1999-2008. That caps every method, classical and learned alike: calibrated GR4J
reaches a median NSE of 0.47 here, against about 0.6-0.7 in the CAMELS literature. Published LSTM numbers (median NSE
0.69 ungauged to 0.76 gauged, Kratzert et al. 2019) used Daymet forcing and are **not comparable** row for row.
Compare the models only with baselines run on the same forcing, as every table here does.

## WaterSim: the simulator

- **Snow:** degree-day with refreeze and liquid water over elevation bands (HBV-light), or a daily energy balance
  (radiation, longwave, turbulent fluxes, ground heat, cold content).
- **Runoff:** four structures, so a network does not learn one model's quirks: HBV, GR4J (a line-by-line port of
  airGR), SAC-SMA (the NWS FLAND1 routine) and VIC-lite (variable infiltration capacity).
- **Evaporation:** FAO-56 Penman-Monteith, Priestley-Taylor, Hargreaves, Oudin; soil-moisture stress for actual ET.
- **Routing:** unit hydrographs, Muskingum, Muskingum-Cunge and a kinematic wave, plus multi-reach networks.
- **Reservoirs and water use:** storage-area-elevation curves, spill, evaporation, hydropower, generic operating
  policies (SOP, rule curve, Hanasaki) or any policy function; irrigation withdrawals with return flows.
- **Training data:** a random-basin generator (attribute priors corrected on the 52 dev basins), a stochastic weather
  generator, and reuse of real forcing from non-exam years only (`realdata` raises on exam years).
- **Speed:** 0.08-0.27 ms per basin-year in the numba core; 1,000 parameter sets x 10 years in 1.2-3.9 s. Without
  numba the same code runs about 100x slower and gives the same numbers.

Checks (`shared/water/checks/*.json` as shipped; the baseline any change must not regress):

| Check | Result |
|---|---|
| GR4J vs airGR 1.7.9 (compiled from CRAN in an isolated R) | max \|ΔQ\| 1.0e-7 mm; NSE 0.79882 vs the published 0.7988 |
| HBV vs TUWmodel 1.1-1 | snow, soil and ET identical to 1e-12 mm; runoff NSE 0.9990 (0.9998 with routing) |
| Water balance, 121 runs (all models, snow routines, routings, reservoirs, cascades, irrigation, a network) | all close; worst 1.4e-15 relative |
| Linear-reservoir recession; UH S-curve; Nash cascade | 4e-15; 6e-13; 2e-10 |
| Muskingum-Cunge vs the analytic diffusion wave; kinematic wave | NSE 0.9999998; NSE 0.9998 |
| FAO-56 Example 18 (Brussels) | ET0 3.88 vs 3.9 mm/day |
| Reservoir storage-elevation power law on the 15 dev dams | median R² 0.9993 |
| Weather generator vs each dev basin's real forcing | wet-day frequency, persistence, annual variability and dry spells at ratio about 1.00 |
| Generator vs the 52 dev basins' observed signatures | mean KS 0.14-0.18; 2-4 % of real basins outside the simulated cloud |
| Snow vs SNOTEL, 33 dev stations, WY2011-25 | degree-day NSE 0.89 calibrated / 0.70 default; energy balance 0.93 / 0.82 |
| Calibrated dev benchmarks, 52 basins, non-exam validation | median NSE 0.47-0.58 (snow basins 0.8-0.9, humid rain basins about 0.4, arid ≤ 0) |

Re-run in the repository on 18 Nov 2026: the analytic checks reproduce `analytic.json` exactly (with and without
numba), and GR4J vs airGR and HBV vs TUWmodel reproduce from the shipped reference outputs. Both run in `pytest`
(`tests/test_water.py`, about 15 s), together with water-balance closure on generated basins for all four runoff
models. One small finding on import: with channel routing, a run restarted from `final_state` differs from one
continuous run by about 4e-6 mm/day, because the channel unit-hydrograph buffer is not carried over (the shipped
restart check uses no channel routing, where it is exact). The dev calibrations, the SNOTEL check and the speed
check were not re-run: they need the time series.

## WaterData: the real data

- **CAMELS-US** (671 basins, via Caravan v1.6): daily ERA5-Land basin weather 1951-2023 and USGS flow, with Caravan,
  HydroATLAS and NWIS attributes. NWIS extends the flow to November 2026 for every gauge.
- **Other regions** (via Caravan): CAMELS-GB 671, CAMELS-CL 505, CAMELS-BR 870, CAMELS-AUS 561 basins, split and
  frozen, for out-of-domain exams. No model has used them yet.
- **NOAA National Water Model v3.0** retrospective at 650 of the 671 gauges, daily 1980-2023 (hourly 2015-2023).
- **SNOTEL:** daily SWE, precipitation and temperature at 880 stations.
- **Reservoirs:** ResOpsUS v2 (678 dams: daily storage, inflow, release) with GRanD attributes, plus 18 California
  reservoirs from CDEC to 2025.
- **Frozen splits** (`data/proc/splits.json`, seed 20261118):

  | Data | Dev | Exam |
  |---|---|---|
  | CAMELS-US basins | 52 | 619 (487 in the CAMELS-531 set) |
  | GB, CL, BR, AUS basins | 3 per region | the rest |
  | SNOTEL stations | 39 | 772 (621 scored WY2011-2025) |
  | ResOpsUS dams | 15 (plus 413 aux, never scored) | 250 (244 scored) |

  Time windows: calibration 1999-10..2008-09; main exam 1989-10..1999-09; recent exam WY2016-2023; simulated training
  may use any basin's forcing before 1988-10 and in 1999-10..2015-09. A network that reads only weather and attributes
  is **ungauged** and is compared with regionalisation and NWM; a mode that reads the basin's own calibration-period
  flow is **gauged** and is compared with per-basin calibrated GR4J.
- **Classical baselines on the exam** (medians): ungauged regionalisation NSE 0.372 (dev donors) and 0.385 (12-fold
  donors); calibrated GR4J 0.473; NWM 0.550; persistence 0.475. SNOTEL: per-station calibrated degree-day 0.925
  (it reads the station's own snow record), climatology 0.701. Reservoirs (cost J, lower is better): SOP 0.108, rule
  curve 0.143, historical operations 0.183, perfect-foresight DP 0.0063.

Quirks the models had to handle (all in `waterdata/DATA.md`, section 8): CAMELS flow is per **contributing** area,
not the polygon or the total drainage area (prairie gauges differ up to 6x); ERA5-Land `SRAD` is **net** shortwave;
forcing days are UTC and USGS days are local; 8 % of USGS days are estimates (ice); NWM was calibrated on many of
the same gauges, so it is the operational bar, not an ungauged method; 189 dams have negative back-calculated
inflows somewhere, and only 171 of 277 close their daily mass balance well.

## Licences

Caravan is CC BY 4.0 (Kratzert et al. 2023), with each national dataset credited (CAMELS-US Newman et al. 2015 and
Addor et al. 2017; CAMELS-GB under the Open Government Licence v3; CAMELS-AUS, CL and BR CC BY 4.0). The forcing is
ERA5-Land (Muñoz-Sabater et al. 2021, Copernicus, attribution required); the attributes include HydroATLAS (Linke et
al. 2019, CC BY 4.0). ResOpsUS (Steyaert et al. 2022) and GRanD (Lehner et al. 2011) are CC BY 4.0. USGS NWIS, NOAA's
National Water Model and NRCS SNOTEL are US public domain; CDEC is published by California DWR. **None of the sources
used is share-alike**, so unlike BDG2 on the building platform, the example excerpts and the instruments carry
attribution only and every model's weights are MIT like the code. The airGR and TUWmodel reference inputs and outputs
(`checks/ref/*.csv`) come from GPL R packages' examples and are not redistributed: `mlratchet data water --from-zip`
unpacks them. Details in [DATA_LICENSES.md](../DATA_LICENSES.md).

## What the models found

Every model was trained only on WaterSim and examined once on real data. On import (18 Nov 2026) the released models
were re-scored on rebuilt data with nothing changed, to get paired bootstrap intervals over units (10,000 resamples,
seed 20261118): every shipped exam number reproduced. Two fairer baselines were added at the same time, tuned on dev
only and frozen before scoring: a dev-tuned energy balance for snow and a dev-tuned reservoir rule. Those re-scores
are labelled seen. Each model's page has the full table, every row with its interval and verdict.

- **FlowCast** (ungauged daily flow, 619 basins): median NSE 0.341 against 0.372 for regionalisation from the dev
  basins (paired difference −0.031 [−0.056, −0.014]) and 0.550 for NWM: a loss. It is level with regionalisation in
  arid, regulated and large basins and loses most in snowy (0.40 against 0.56) and humid ones. Peaks run about 46 %
  low and the 90 % band holds 97 % of days (too wide). Its gauged nowcast mode beats persistence at every lead (NSE
  0.569, 0.474, 0.426 at 1, 3 and 7 days against 0.475, −0.19, −0.47), ties NWM at one day and loses to it at three
  and seven.
- **SnowSense** (621 SNOTEL stations, WY2011-2025): median NSE 0.888 against 0.853 for the energy-balance model with
  WaterSim's default parameters, which it beats at 75 % of stations. Against the same energy balance tuned on the dev
  stations (0.895) it **ties** (−0.007 [−0.018, +0.001]) and loses on peak SWE, which runs 9 % low. It still beats
  the tuned physics below 1,500 m. The per-station calibrated degree-day model, which reads the station's own snow
  record, stays ahead (0.925).
- **BasinDetective** (619 basins): predicts how six flow signatures change from 1999-2008 to 1989-99 about as well as
  calibrated GR4J, with no per-basin fitting: level on five of six changes, behind on mean flow (Spearman 0.76
  against 0.79, paired difference −0.036 [−0.063, −0.008]). Its error bars hold but are too wide (68 % bars cover
  73-83 %). Regulation does not transfer: AUC 0.49 for a reservoir storing over 10 % of
  annual flow, against 0.91 in simulation.
- **ReservoirPilot** (244 ResOpsUS dams): median cost J 0.0154 against 0.0285 for a classical hedging rule tuned in
  the same simulator (−0.013 [−0.020, −0.007]), 0.0322 for the same rule tuned on the 15 dev dams and 0.108 for SOP. **J charges nothing for storage**, and the pilot exploits that: it holds
  reservoirs at a median 38 % of capacity against SOP's 59 %, and ends at least 20 % of capacity lower on 34 % of
  dams. It loses to SOP on 15 % of dams, mostly large supply reservoirs whose scored decade was drier. On the
  15-reservoir CDEC exam (WY2021-2025) it only ties the tuned rule (better on 8 of 15).

## What it gets wrong

| Gap | Measured |
|---|---|
| Forcing | ERA5-Land only: mountain and Pacific Northwest under-catch (50 basins with runoff ratio > 1), about 6 % drift between decades, about 10 % year-to-year precipitation error (BasinDetective's dev twins); forcing ends 2023-12-30 |
| Water yield | the dev-measured ERA5-Land multiplier runs from 0.53 (Texas, North Dakota plains) to 1.71 (Olympic Peninsula); two different fits live in two model folders |
| Gauge noise | the default gauge model is too jittery (dev fit: AR(1) ρ 0.999, cv 0.02, 3 significant figures in cfs) and the default forcing error too small (a calibrated twin matches its clean run at NSE 0.88; reality allows about 0.56) |
| Point snow | energy-balance defaults put peaks about 20 % low at SNOTEL points (dev calibrations: turbulent transfer about 0.2x the default and albedo decay 1.4-2.4 d against 12 d; the handoff's "GHEAT 0.32" is a prior mean, the per-station median is 0.06); degree-day TT is about +1 °C at points; cold-shifted weather can grow perennial packs |
| Flashiness | the half prior correction leaves simulated flashiness at 0.62x observed; the full correction fixes it but cost FlowCast NSE |
| Structure | lumped soil and groundwater per basin; no glaciers, wetlands, lakes or karst; daily energy balance only; no floodplains; single reservoirs (no cascades of real operating rules, no minimum flows); no groundwater pumping, urban drainage, water quality or temperature |
| Hourly | HBV, SAC-SMA and VIC-lite run hourly, but no hourly real dataset is wired up (only NWM hourly 2015-2023) |
| Restart | channel-routing state is not carried by `final_state` (about 4e-6 mm/day; found on import) |
| `realdata.random_forcing` | `span*365` days is shorter than `span` years, so a long draw can end a few days inside an exam window; the guard raises `PermissionError` (no exam data leaked), but the arithmetic should be fixed |

Each model wrapped WaterSim with the physics it measured (water-yield multipliers, gauge noise, forcing drift,
point-snow priors, a reservoir-task generator). Those wrappers live in each model's `training/` and are listed for
upstreaming into WaterSim v1.1 in section 4 of `shared/water/HANDOFF.md` and on the
[backlog](backlog.md) (`watersim-v11`).

## Using it

```bash
pip install -e ".[water]"                       # add numba for speed (".[water-data]" for a rebuild)
mlratchet data water --from-zip watersim-waterdata-v1.0.zip   # attributes, baselines, checks data: 1,170 files, 21 MB
mlratchet data water --list                     # the rebuild stages, their hosts and costs
mlratchet data water --only snotel,reservoirs,cdec            # the smaller time series
mlratchet data water --only caravan             # streams 24.8 GB once (about 2-3 h), keeps 1.7 GB
pytest -q tests/test_water.py
```

The full rebuild runs, in order: the Caravan stream (Zenodo 15529786, 24.8 GB, never stored), NWIS daily values and
peaks (about 2,000 requests), NWM CHANOBS (about 385,000 HTTP range requests of 32 KB from NOAA's S3 bucket, many
hours), SNOTEL, ResOpsUS (0.3 GB) and CDEC, then per-basin GR4J calibration and the baselines (hours on one core,
needs numba). Every fetcher skips what is already on disk. Hosts that fail are listed at the end; record them, don't
route around them.

```python
import sys; sys.path[:0] = ['shared/water', 'shared/water/watersim']
import watersim as ws, waterdata as wd
specs = ws.generate_basins(100, rng=0)                  # random basins with parameters for all four models
f = ws.generate_weather(specs[0], '2000-01-01', '2009-12-31', rng=1)
out = ws.simulate(specs[0], f, '2001-01-01', model='gr4j', snow='energy_balance', routing='muskingum')
out['ledger']['ok'], out['labels']['bfi_lyne_hollick']
wd.basins('camels', split='dev'); wd.attributes('camels'); wd.baselines()   # with the release unpacked
```

`data/proc/splits.json` (file sha256 `33f88ca0…`, internal hash `ed8109e5…`, seed 20261118) is tracked and never
changes; the rest of `data/`, `checks/out/` and `checks/ref/*.csv` are git-ignored and checked against
`data/MANIFEST.json`. Every physics change bumps the WaterSim version, rebuilds each water model's frozen simulated
set and re-takes their ratchet decisions in order.

## Rule notes

No exam result was used to choose anything. What the build agents were unsure about is recorded on each model's
ladder and card:

- **Dev data fitted physics and also gated rungs** (partly in-sample): FlowCast's four water-yield coefficients,
  BasinDetective's decadal-drift mean, SnowSense's parameter prior (dev SWE up to WY2010, gate on WY2011-25). The exam
  was untouched in every case.
- **Free-window forcing from exam basins** went into training banks, as the brief allows. No exam flow, snow or
  reservoir record was used for training.
- **Climate statics at exam time:** FlowCast computes them from the forcing of the window it predicts (exam-period
  weather, no flow).
- **ReservoirPilot** set its generator's ranges from aux-dam statistics (6 complete dams, inflow ratios on 29 aux
  dams); its climatology features use each exam dam's calibration-half inflow, as WaterData's rule curve does; two
  void training runs (the optimiser step was missing) were removed and logged.
- **SnowSense**'s released rung R6 was initialised from the weights of the rejected R5; its demo ran on three exam
  stations before the final exam, for code testing only.
- **Ratchet noise:** simulated-metric differences of about 0.01 are noise-level for FlowCast; a 300-basin subset and
  the full 800 frozen basins disagreed on rung r5.

## What is still waiting

- **The recent exam and flood peaks.** The re-scores on import used Caravan (CAMELS-US), SNOTEL, ResOpsUS and CDEC.
  NWIS was not fetched, so FlowCast's recent exam (WY2016-2023) and instantaneous flood peaks, and BasinDetective's
  peak-code regulation row, keep their shipped medians with binomial intervals on the shipped win fractions.
- **Independent forcing** (Daymet or NLDAS) to separate method skill from ERA5-Land bias: the single biggest gap
  (backlog `water-forcing`).
- **WaterSim v1.1:** the upstream physics in section 4 of the handoff (backlog `watersim-v11`).
- **Not re-run here:** the dev calibrations, the SNOTEL and speed checks, the GR4J calibrations (BasinDetective's
  re-score re-made the daily GR4J runs from the shipped parameters) and every model's training (no banks or
  checkpoints are shipped). The exams were re-scored, not re-trained.
- **USGS is retiring the legacy NWIS web services** (`waterservices.usgs.gov`) that `fetch_nwis.py` calls, in favour
  of the Water Data APIs (`api.waterdata.usgs.gov`). A rebuild may need that script ported.
- **Not shipped, rebuildable:** the time series (about 2.9 GB), the R reference environment (`watersim/.refenv`,
  1.3 GB) and the training banks.
