# Data licences and credits

The code, weights and docs in this repository are MIT-licensed (see [LICENSE](LICENSE)). **The real data is not ours.**
Small excerpts of it are bundled so every demo runs offline in seconds (`models/*/examples/`), and the interactive
lab pages (`models/*/lab/index.html`) embed processed excerpts. They stay under their providers' terms below.
Please cite the providers whenever you use a model or its data.

Training data is never bundled — every model learned only from simulations. Where a simulator was laid into real
instrument noise, the fetch scripts in `models/*/training/` download that noise from the documented public source.

| Model | Source | Terms | Bundled here |
|---|---|---|---|
| ChirpNet | [Gravitational Wave Open Science Center](https://gwosc.org) — LIGO Scientific Collaboration and Virgo Collaboration | CC BY 4.0 | 8 × 32 s strain files (GW150914, GW151012, GW151226, GW170104) |
| ChirpNet | GWOSC O3 event files (GWTC-2.1, GWTC-3) for the O3 exam | CC BY 4.0 | No — `training/o3_exam.py` downloads them |
| ChirpNet | [LIGO DCC P1700349 / P1700337](https://dcc.ligo.org/LIGO-P1700349/public) (GW170817) | CC BY 4.0 | No — download links in `models/chirpnet/examples/README.md` |
| SunDoctor | [NREL PVDAQ via OEDI](https://data.openei.org/submissions/4568) (systems 2107, 9068) | CC BY 4.0 | Two CSV excerpts |
| CellSense | [Severson et al. 2019](https://data.matr.io/1/) (Toyota Research Institute, Stanford, MIT) | CC BY 4.0 | One cell (b3c33), seven readings |
| CellSense (NMC811) | [Pozzato, Allam & Onori 2022](https://doi.org/10.1016/j.dib.2022.107995), Data in Brief 41, 107995 — LG INR21700-M50T aging data on [OSF (qsabn)](https://osf.io/qsabn/) (through the view-only link printed in the article) | CC BY 4.0 (stated on the OSF project) | One cell (W9), six C/20 capacity tests on a voltage grid; `training/nmc_fetch.py` downloads the processed files |
| TransitNet | [TESS](https://archive.stsci.edu/missions-and-data/tess) via MAST; [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/); [ExoFOP](https://exofop.ipac.caltech.edu/tess/) | NASA public data; acknowledge TESS, MAST, the archives | Three light curves |
| QuakeNet | [USGS ComCat](https://earthquake.usgs.gov/); [EarthScope / IRIS DMC](https://ds.iris.edu/) — networks IU (GSN, IRIS/USGS) and II (GSN, IRIS/IDA); [Syngine](https://ds.iris.edu/ds/products/syngine/) | USGS public domain; EarthScope open data — cite the networks | Four one-station records |
| BearingNet | [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) | Public download; acknowledge CWRU | Eight excerpts |
| BearingNet | [Paderborn KAt-DataCenter](https://groups.uni-paderborn.de/kat/BearingDataCenter/) — Lessmeier et al., PHM Society European Conference 2016 | Terms on the provider's page; cite the paper | Six excerpts |
| SwellCast | [NOAA NDBC](https://www.ndbc.noaa.gov/) spectral wave data | U.S. public domain | Three yearly files for buoy 51004 and five windows |
| SwellCast | [IBTrACS v04r01](https://www.ncei.noaa.gov/products/international-best-track-archive) — Knapp et al. 2010, BAMS 91 | U.S. public domain; cite the paper | Storm truth for the examples |
| BeatNet | [PhysioNet](https://physionet.org) — MIT-BIH Arrhythmia, Noise Stress Test, Supraventricular Arrhythmia and QT databases | Open Data Commons Attribution 1.0 | Four 60 s ECG excerpts |
| StarQuake | [Kepler](https://archive.stsci.edu/missions-and-data/kepler) via MAST | NASA public data; acknowledge Kepler and MAST | Three light curves |
| StarQuake | Yu et al. 2018, ApJS 236, 42 | Journal tables; cite the paper | Truth values for the examples |
| StarQuake | [Gaia DR3](https://www.cosmos.esa.int/web/gaia/dr3) (ESA/Gaia/DPAC) | CC BY-SA 3.0 IGO | No |
| WakeNet | [Kelmarsh Wind Farm Data](https://doi.org/10.5281/zenodo.16807551) (2025 version; concept DOI 10.5281/zenodo.5841834) — Charlie Plumley / Cubico Sustainable Investments | CC BY 4.0 | Four single-timestamp 6-turbine excerpts (`examples/`, instrument, `lab/`) |
| WakeNet | [Penmanshiel Wind Farm Data](https://doi.org/10.5281/zenodo.16807304) (2025 version; concept DOI 10.5281/zenodo.5946808) — Charlie Plumley / Cubico Sustainable Investments | CC BY 4.0 | Two single-timestamp 14-turbine excerpts (`examples/`, instrument, `lab/`) |


### The vehicle platform (`shared/vehicle`) and its models (EnergyNet, TripDetective, HybridPilot)

| Source | Terms | Bundled here |
|---|---|---|
| [Vehicle Energy Dataset (VED)](https://github.com/gsoh/VED) — Oh, LeBlanc, Peng, IEEE T-ITS 2020 | Apache-2.0 | Example trips in the three models' `examples/` and RoadData's baseline tables; the 1.2 GB of processed trips is rebuilt by `shared/vehicle/data/scripts/` |
| US EPA test-car data, drive cycles, ALPHA engine/motor maps; fueleconomy.gov | U.S. public domain | Processed tables in `shared/vehicle/data/proc/`, the maps used in `roadsim/roadsim/data/` |
| [USGS 3DEP lidar](https://www.usgs.gov/3d-elevation-program) | U.S. public domain | Grades inside the example trips |
| [Iowa Environmental Mesonet ASOS](https://mesonet.agron.iastate.edu/) | Free use with credit | Weather inside the example trips |
| [Downloadable Dynamometer Database (D3)](https://www.anl.gov/taps/downloadable-dynamometer-database) — Argonne National Laboratory | Use with attribution: "This data is from the Downloadable Dynamometer Database and was generated at the Advanced Mobility Technology Laboratory (AMTL) at Argonne National Laboratory under the funding and guidance of the U.S. Department of Energy (DOE)" | Not redistributed. `shared/vehicle/data/scripts/d3_fetch.py` downloads the 10 Hz chassis-dynamometer tests into the git-ignored `data/raw/d3/`; only derived calibration and exam results are published |
| [ZTBus](https://doi.org/10.3929/ethz-b-000626723) — Widmer et al., ETH Zurich | CC BY 4.0 | Processed legs in `shared/vehicle/data/proc/hd/ztbus/` |
| Mersin bus trips — Abut, Zenodo | CC BY 4.0 | Trip totals in `shared/vehicle/data/proc/hd/` |
| ICCT 2018 tractor-trailer tests (Rodriguez, Delgado, Muncrief) | Cite the report | Six published numbers |

### The grid platform (`shared/grid`) and its models (GridPulse, BatteryPilot, PlantDetective, DemandCast)

None of GridData's tables are in git: `mlratchet data grid` downloads them from the sources below (or unpacks the v1.0
release) into the git-ignored `shared/grid/data/`. Only the four models' `examples/`, their instrument excerpts
(`site/public/instruments/<id>/`) and their lab pages carry real data. Terms checked on the providers' pages on
14 Nov 2026.

| Source | Terms | Bundled here |
|---|---|---|
| [EIA-930 Hourly Electric Grid Monitor](https://www.eia.gov/electricity/gridmonitor/) six-month bulk files (U.S. Energy Information Administration) | U.S. public domain | DemandCast's 16 example days and lab; BatteryPilot's lab (demand forecast input) |
| EIA-860, EIA-860M, EIA-861, EIA-923 via [PUDL](https://catalyst.coop/pudl/) (Catalyst Cooperative) | EIA data U.S. public domain; PUDL's processed tables CC BY 4.0 — cite Catalyst Cooperative, Public Utility Data Liberation project | No (fleets, heat rates and fuel costs rebuild from PUDL) |
| [EPA Clean Air Markets Program Data](https://campd.epa.gov/) hourly emissions bulk files; EPA-EIA crosswalk (via PUDL) | U.S. public domain | PlantDetective's 3 example units and lab |
| [NESO data portal](https://www.neso.energy/data-portal): system frequency (1 s), system inertia, historic generation mix | [NESO Open Data Licence v1.0](https://www.neso.energy/data-portal/neso-open-licence) (based on OGL v3.0): commercial reuse and redistribution allowed; attribution "Supported by National Energy SO Open Data"; no implied endorsement | GridPulse's example windows, instrument and lab (frequency excerpts, inertia) |
| National Grid ESO, *Technical Report on the events of 9 August 2019* (6 Sep 2019); Ofgem, *9 August 2019 power outage report* (Jan 2020) | Cited, not redistributed | Published numbers only (GridSim's reference event) |
| [Elexon BMRS](https://bmrs.elexon.co.uk/) REMIT, B1610, FUELINST (GB loss labels) | [BMRS data licence](https://www.elexon.co.uk/bsc/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/): commercial reuse allowed; attribution required: "Contains BMRS data © Elexon Limited copyright and database right 2026."; no implied endorsement | Loss labels in GridPulse's examples and lab |
| [ERCOT](https://www.ercot.com/) monthly Operations Reports (frequency events) and settlement point prices (DAM 13060, RTM 13061) | [ERCOT terms of use](https://www.ercot.com/help/terms): public content may be reused and redistributed unmodified with its notices; raw data may be used in compilations, charts and analyses; "ERCOT" is a trademark | BatteryPilot's 2 ERCOT example node-years and lab; GridPulse's ERCOT summary numbers |
| [CAISO OASIS](http://oasis.caiso.com/) LMPs (PRC_LMP, PRC_RTPD_LMP) | Public market data; cite CAISO OASIS (terms page not reachable on 14 Nov 2026 — check before reuse outside this project) | BatteryPilot's CAISO example node-year and lab |
| [NASA POWER](https://power.larc.nasa.gov/) hourly (MERRA-2, CERES SYN1deg) — NASA Langley Research Center POWER Project | No licence stated; cite the POWER reference and data reference ("The data was obtained from the POWER Project's Hourly 2.x.x version on YYYY/MM/DD") | Weather inside DemandCast's and BatteryPilot's examples and labs |
| [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) ASOS (Iowa State University) | Public domain per the IEM disclaimer; credit appreciated | Station temperatures inside DemandCast's examples |
| [Open-Meteo](https://open-meteo.com/) Previous-Runs API (archived day-ahead forecasts: 2024-2025 all variables, 2022 temperature) | CC BY 4.0; the free API is for non-commercial use; credit Open-Meteo.com | No (DemandCast's forecast-weather exam; `mlratchet data grid --only forecastwx` fetches it into `shared/grid/data/raw/openmeteo/`, not shipped) |
| [MATPOWER](https://github.com/MATPOWER/matpower) test cases (IEEE 9-300, RTS-GMLC) | Case data per file header (IEEE cases public); MATPOWER code BSD-3 | No (fetched into `gridsim/refdata/`) |
| Texas A&M synthetic grids ACTIVSg200/500/2000 (Birchfield et al., IEEE TPWRS 2017), MATPOWER copies | CC BY 4.0 | No |
| [RTS-GMLC](https://github.com/GridMod/RTS-GMLC) generator data (Barrows et al., IEEE TPWRS 2020) | BSD-3-Clause | No (fleet defaults by unit type) |
| U.S. Census 2020 Gazetteer places | U.S. public domain | No (approximate ACTIVSg bus coordinates) |
| EIA Henry Hub daily spot price | U.S. public domain | No (`gridsim/refdata/fuel/`) |

### The building platform (`shared/building`) and its models (LoadCast, BuildingDetective, FaultFinder, ThermoPilot)

**BDG2 is share-alike.** Building Data Genome 2 is CC BY-SA 4.0. Every file in this repository that holds BDG2 meter
readings, BDG2 site weather or BDG2 building metadata is an adaptation and stays CC BY-SA 4.0. Those files sit in
folders marked with a `LICENSE-DATA.md` notice, apart from the MIT code:

- `models/*/examples/bdg2/` (the demos' BDG2 example series),
- `models/*/lab/` where the page embeds BDG2 data,
- `site/public/instruments/<id>/` where the instrument's excerpts come from BDG2.

**Weights.** All four were trained on BuildingSim simulations only. LoadCast's and BuildingDetective's training banks
were shaped by BDG2 data: real BDG2 dev residual series added as noise (LoadCast), and calibration and error-bar
factors fitted on BDG2 dev buildings (both). Those two models' weights are therefore treated as adaptations and are
**CC BY-SA 4.0** (`models/loadcast/LICENSE-WEIGHTS.md`, `models/buildingdetective/weights/LICENSE-DATA.md`; decided
17 Nov 2026 on the cautious reading). FaultFinder's and ThermoPilot's weights used no BDG2 data in training
(ThermoPilot uses BDG2 buildings only in its dev gate) and are MIT like the code.

None of BuildingData's tables are in git (`shared/building/data/`, git-ignored; `splits.json` is tracked). Terms
checked on the providers' pages on 17 Nov 2026 where a page answered; otherwise as recorded by BuildingData.

| Source | Terms | Bundled here |
|---|---|---|
| [Building Data Genome 2](https://github.com/buds-lab/building-data-genome-project-2) — Miller, C., et al. (2020), "The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition", *Scientific Data* 7, 368, [doi:10.1038/s41597-020-00712-x](https://doi.org/10.1038/s41597-020-00712-x) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (the repository's LICENSE file, checked 17 Nov 2026). Share alike, credit the paper | Example meters and weather in LoadCast's and BuildingDetective's `examples/bdg2/`, instruments and labs; ThermoPilot's BDG2 site weather (see each folder's `LICENSE-DATA.md`) |
| U.S. EIA [RECS 2020](https://www.eia.gov/consumption/residential/) and [CBECS 2018](https://www.eia.gov/consumption/commercial/) microdata | U.S. public domain ([EIA copyright and reuse](https://www.eia.gov/about/copyrights_reuse.php)); acknowledge "Source: U.S. Energy Information Administration" | Survey respondents in BuildingDetective's examples; ThermoPilot's twins are built from RECS characteristics |
| NREL [ResStock and ComStock End-Use Load Profiles](https://data.openei.org/submissions/4520) on OEDI (2025 R1 / R3) | CC BY 4.0 per BuildingData (OEDI's dataset page did not load on 17 Nov 2026; the OEDI site footer says CC BY 4.0 unless noted) | No (generator priors and simulator-vs-simulator checks; rebuilt by `data/scripts`) |
| LBNL fault detection data synthesis (LBNL FLEXLAB, ORNL Flexible Research Platform RTU, MZVAV-2 at the Iowa Energy Resource Station) — Granderson, Lin et al., figshare [doi:10.6084/m9.figshare.11752740](https://doi.org/10.6084/m9.figshare.11752740) | CC0 (as credited in FaultFinder's release; faultdetection.lbl.gov returned HTTP 521/522 on 17 Nov 2026, so not re-checked) | FaultFinder's FLEXLAB, Iowa ERS and ORNL example days, instrument and lab |
| ASHRAE RP-1312 (Iowa Energy Resource Station; Wen and Li 2011) and Drexel Nesbitt Hall (Chen 2019) AHU data — Ghalamsiah (2025), figshare [doi:10.6084/m9.figshare.29297999.v3](https://doi.org/10.6084/m9.figshare.29297999.v3) | CC BY 4.0 (as credited in FaultFinder's release); cite the package and ASHRAE RP-1312 | FaultFinder's Nesbitt Hall example day (168 of its 217 exam days), instrument and lab |
| Korean office AHU trend logs (unlabelled) | CC BY 4.0 (as credited in FaultFinder's release) | No (noise statistics only) |
| [WPuQ](https://zenodo.org/records/5642902) heat-pump homes — Schlemminger, Ohrdes, Schneider, Knoop (2022), Zenodo, doi:10.5281/zenodo.5642902 | CC BY 4.0 (Zenodo record, checked 17 Nov 2026) | ThermoPilot's reality-check results (summary numbers) |
| [HUE](https://doi.org/10.7910/DVN/N3HGRN): Hourly Usage of Energy, British Columbia homes — Makonin, Harvard Dataverse | CC BY per BuildingData (the Dataverse page returned an empty response on 17 Nov 2026; check before reuse) | Summary numbers only |
| [OpenEI Utility Rate Database](https://openei.org/wiki/Utility_Rate_Database) (19 tariffs, bulk CSV) | CC0 (OpenEI, as recorded in BuildingData's DATA.md) | Tariff structures inside ThermoPilot's examples and lab |
| [TMYx](https://climate.onebuilding.org) typical-year weather — Lawrie, L. K. and Crawley, D. B., "Development of Global Typical Meteorological Years (TMYx)" | The site says "All Rights Reserved" and gives a citation, with no reuse licence. **Not redistributed:** used for simulation in training only | No |
| [NASA POWER](https://power.larc.nasa.gov/) hourly irradiance | No licence stated; cite the POWER Project | Irradiance inside BDG2-site weather excerpts |
| [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) ASOS and NOAA ISD station weather | Public domain; credit appreciated | Station temperatures inside the weather excerpts |
| ERCOT 2018 real-time prices | ERCOT terms of use (as in the grid section) | No (ThermoPilot's RTP tariff in training) |
| [HEAPO](https://doi.org/10.5281/zenodo.15056919) heat-pump homes, Zurich — Brudermueller et al. (2025), Zenodo | CC BY 4.0 | No: BuildingDetective's fresh exam (round 2); only home IDs, hashes and counts are in `training/split_exam2_heapo.json` |
| [I-BLEND](https://doi.org/10.6084/m9.figshare.6007637.v1) IIIT Delhi campus energy — Rashid et al. (2019), figshare; floor areas from Sci Data [doi:10.1038/sdata.2019.15](https://doi.org/10.1038/sdata.2019.15) | CC0 1.0 (data); the paper CC BY 4.0 | No: LoadCast's fresh exam (round 2); IDs and hashes only |
| NOAA ISD-Lite, Delhi IGI (421810-99999) | U.S. government, no restrictions | No (weather for the I-BLEND exam) |
| [Nehasil AHU fault data](https://doi.org/10.17632/jkw85sm4v4.1), CTU Prague — Mendeley Data | CC BY 4.0 | No: FaultFinder's fresh exam (round 2); IDs, hashes and counts only |
| [Wang et al. 2024](https://doi.org/10.6084/m9.figshare.25909813.v1) Korean office AHUs — figshare | CC BY 4.0 | No: downloaded and rejected as an exam (it repeats the Korean office data behind v1.0's noise model) |

### The water platform (`shared/water`) and its models (FlowCast, SnowSense, BasinDetective, ReservoirPilot)

**No share-alike source.** Every source below is public domain or attribution-only (CC BY 4.0, OGL v3, the
Copernicus terms), so the example excerpts, instrument packs and labs carry attribution, not a share-alike licence,
and all four models' weights are MIT like the code (checked 18 Nov 2026 against the licences recorded in WaterData's
`DATA.md` and Caravan's `licenses/` folder). The excerpts are listed in each model's `examples/` README.

None of WaterData's tables are in git (`shared/water/data/`, git-ignored; `splits.json` is tracked): `mlratchet data
water` unpacks the release or rebuilds from the sources. The airGR and TUWmodel reference inputs and outputs that
WaterSim's checks compare against (`shared/water/checks/ref/*.csv`) come from the examples of GPL-licensed R
packages; they are not redistributed here (git-ignored, unpacked from the release).

| Source | Terms | Bundled here |
|---|---|---|
| [Caravan v1.6](https://doi.org/10.5281/zenodo.15529786) — Kratzert, F., et al. (2023), "Caravan - A global community dataset for large-sample hydrology", *Scientific Data* 10, 61, [doi:10.1038/s41597-023-01975-w](https://doi.org/10.1038/s41597-023-01975-w) | CC BY 4.0; cite Caravan and each national dataset below | ERA5-Land forcing and CAMELS flow excerpts in FlowCast's and BasinDetective's `examples/`, instruments and labs |
| CAMELS (US) streamflow and attributes inside Caravan — Newman, A., et al. (2015), HESS 19, 209-223, [doi:10.5194/hess-19-209-2015](https://doi.org/10.5194/hess-19-209-2015); Addor, N., et al. (2017), HESS 21, 5293-5313, [doi:10.5194/hess-21-5293-2017](https://doi.org/10.5194/hess-21-5293-2017) | CC BY 4.0 (as distributed in Caravan) | Flow excerpts for 7 exam basins (above) |
| CAMELS-GB inside Caravan — Coxon, G., et al. (2020), ESSD 12, 2459 | Open Government Licence v3 (attribution) | No (out-of-domain exam basins, unused so far) |
| CAMELS-AUS v2, CAMELS-CL, CAMELS-BR inside Caravan — Fowler et al. (2021, 2024); Alvarez-Garreton et al. (2018), HESS 22, 5817; Chagas et al. (2020), ESSD 12, 2075 | CC BY 4.0 | No (out-of-domain exam basins, unused so far) |
| ERA5-Land (all water forcing, via Caravan) — Muñoz-Sabater, J., et al. (2021), ESSD 13, 4349, [doi:10.5194/essd-13-4349-2021](https://doi.org/10.5194/essd-13-4349-2021); Copernicus Climate Change Service | Copernicus terms: free use with attribution ("Contains modified Copernicus Climate Change Service information") | Inside the Caravan excerpts |
| HydroATLAS / BasinATLAS attributes (via Caravan) — Linke, S., et al. (2019), *Scientific Data* 6, 283 | CC BY 4.0 | Static attributes inside FlowCast's examples; the upstream reservoir storage behind BasinDetective's regulation truth |
| [USGS NWIS](https://waterdata.usgs.gov/) daily values, site information and annual peaks | U.S. public domain; acknowledge the U.S. Geological Survey | Flow excerpts (recent exam) where an example uses NWIS |
| [NOAA National Water Model v3.0 retrospective](https://registry.opendata.aws/nwm-archive/) (AWS Open Data, s3://noaa-nwm-retrospective-3-0-pds) | U.S. public domain (NOAA Open Data Dissemination) | No (a baseline; rebuilt by `mlratchet data water --only nwm`) |
| [NRCS SNOTEL](https://www.nrcs.usda.gov/wps/portal/wcc/home/) daily SWE, precipitation and temperature (Report Generator) | U.S. public domain; acknowledge USDA NRCS National Water and Climate Center | Three exam stations' records in SnowSense's `examples/`, instrument and lab |
| [ResOpsUS v2](https://doi.org/10.5281/zenodo.6612040) — Steyaert, J., Condon, L., Turner, S., Voisin, N. (2022), "ResOpsUS, a dataset of historical reservoir operations in the contiguous United States", *Scientific Data* 9, 34 | CC BY 4.0; cite the paper | Six exam dams' inflow, storage and release in ReservoirPilot's `examples/`, instrument and lab |
| GRanD v1.3 attributes (via ResOpsUS+CARS, [doi:10.5281/zenodo.15978041](https://doi.org/10.5281/zenodo.15978041)) — Lehner, B., et al. (2011), *Front. Ecol. Environ.* 9, 494 | CC BY 4.0 | Dam capacity and main use inside ReservoirPilot's examples (BasinDetective's regulation truth uses HydroATLAS's GRanD-derived reservoir volume) |
| [California CDEC](https://cdec.water.ca.gov/) daily reservoir storage, inflow and outflow (California Department of Water Resources) | Public data of the State of California; no licence text found; credit DWR CDEC | No (ReservoirPilot's recent-exam scores only) |
| CAMELS-531 basin list — neuralhydrology repository (Kratzert et al.) | BSD-3-Clause (repository) | No (a list of IDs) |
| airGR 1.7.9 (Coron et al. 2017) and TUWmodel 1.1-1 (Parajka et al. 2007) example data and outputs | GPL (the R packages) | No (`checks/ref/*.csv`, git-ignored) |

### The fire platform (`shared/fire`) and its models (FireCast, FireSize, FireDetective, FirePilot)

**No share-alike source.** Every source below is public domain, CC0 or CC BY 4.0, so the example excerpts,
instrument packs and labs carry attribution, not a share-alike licence, and all four models' weights are MIT like the
code (checked 21 Nov 2026 against the licences recorded in FireData's `data/DATA.md` and the build scripts' headers).
The excerpts are listed in each model's `examples/` README.

None of FireData's tables are in git (`shared/fire/data/`, git-ignored; `proc/splits.json` is tracked): `mlratchet data
fire` unpacks the release or rebuilds from the sources. The US Forest Service reports FireSim's published-table check
reads (GTR-143, GTR-153 and GTR-371, as PDFs and text extracts in `shared/fire/firesim/refdata/`) are public domain but
are not redistributed here: download them from the Forest Service Treesearch pages.

| Source | Terms | Bundled here |
|---|---|---|
| [FEDS-MTBS](https://zenodo.org/records/20187963) (Zenodo 20187963): 12-hourly fire perimeters, active fire lines and VIIRS pixels for 7,739 MTBS fires, 2012-2024 — FEDS method of Chen, Y., et al. (2022), *Scientific Data* 9, 249 | CC BY 4.0; cite the dataset and the FEDS paper | Exam fire-day excerpts (burned masks, fire lines, VIIRS pixels) in FireCast's, FireDetective's and FirePilot's `examples/`, instruments and labs |
| VIIRS S-NPP 375 m active fire (NASA LANCE / [FIRMS](https://firms.modaps.eosdis.nasa.gov/)), as carried inside FEDS-MTBS | NASA open data; acknowledge LANCE/FIRMS | Inside the FEDS excerpts. The FIRMS archive itself was not used |
| [MTBS](https://www.mtbs.gov/) final perimeters, ignition dates and burn severity (USGS and USDA Forest Service) | U.S. public domain | Inside the FEDS excerpts; FireDetective's severity truth (MTBS severity read through LANDFIRE Annual Disturbance) |
| [GOFER-Combined](https://zenodo.org/records/14642378) (Zenodo 14642378): hourly GOES perimeters for 28 California fires, 2019-2021 — Liu et al. | CC BY 4.0 | No (truth-quality check on 4 dev fires; 24 exam fires unused so far) |
| [WildfireSpreadTS](https://zenodo.org/records/8006177) (Zenodo 8006177) — Gerard, S., Zhao, Y., Sullivan, J. (2023), NeurIPS Datasets and Benchmarks | CC BY 4.0 | No (benchmark context; split-matched, not scored by a model yet) |
| [FPA-FOD 6th edition](https://doi.org/10.2737/RDS-2013-0009.6) — Short, K. C. (2022), *Spatial wildfire occurrence data for the United States, 1992-2020*, USDA Forest Service Research Data Archive | U.S. public data; cite Short (2022) | Eight exam ignitions' records and features in FireSize's `examples/`; exam ignitions in its instrument and lab |
| [Globe-LFMC 2.0](https://doi.org/10.6084/m9.figshare.25413790) (figshare) — Yebra, M., et al. (2024), *Scientific Data* 11, 332 | CC0 | No raw samples; FireDetective's live-moisture truth and scores (FireDetective's card cites the 2019 Globe-LFMC paper as CC BY) |
| [LANDFIRE](https://landfire.gov/) fuel models (FBFM40), canopy layers, elevation, slope and aspect, via the LFPS service (LF2016 Remap, LF2020, LF2022, LF2023) | U.S. public domain (USGS and USDA Forest Service) | Landscape crops in FireDetective's `examples/`; landscape summaries in FireSize's; fields derived from them in FirePilot's |
| [HRRR](https://registry.opendata.aws/noaa-hrrr-pds/) analyses (NOAA), read through the University of Utah MesoWest `hrrrzarr` archive on AWS Open Data | U.S. public data (NOAA Open Data Dissemination) | No raw series; FirePilot's examples hold simulated arrival times computed from it |
| [gridMET](https://www.climatologylab.org/gridmet.html) daily weather and NFDRS indices — Abatzoglou, J. T. (2013), *Int. J. Climatol.* 33, 121-131 | U.S. public domain | Daily weather in FireCast's, FireSize's and FireDetective's `examples/` |
| US Forest Service General Technical Reports: Andrews (2018) RMRS-GTR-371, Rothermel (1983) INT-GTR-143, Scott & Burgan (2005) RMRS-GTR-153 | U.S. public domain | No (`firesim/refdata/`, git-ignored); the check's results are in `checks/published_examples.json` |

Every model card (`models/*/MODEL_CARD.md`) carries the full citation list, including the physics papers.

## StormNet

NASA SPDF OMNIWeb and its contributing instrument teams supply the hourly solar wind. Credit King & Papitashvili (2005). The World Data Center for Geomagnetism, Kyoto, and the contributing observatories supply hourly Dst. Kyoto permits scientific use with acknowledgement and prohibits commercial application of its geomagnetic indices. The four short StormNet excerpts, browser packs and offline lab carry those data terms. Annual source files are downloaded into the model’s ignored work directory. [OMNI documentation](https://omniweb.gsfc.nasa.gov/html/ow_data.html) · [OMNI paper](https://doi.org/10.1029/2004JA010649) · [Dst DOI](https://doi.org/10.17593/14515-74000) · [Kyoto terms](https://wdc.kugi.kyoto-u.ac.jp/wdc/Sec3.html)

## Prior Wars (method experiment, experiments/priorwars)

- Built with PriorLabs-TabPFN. TabPFN-v2 weights (Prior Labs License 1.1: Apache 2.0 plus an attribution clause)
  were used to produce the results and are not redistributed here; `tabpfn` 10.2.0 code is Apache-2.0.
- The exam rows it scores come from the models' own data, under the licences listed above for each model.

## Before redistributing

- Some providers ask for a specific citation or acknowledgement sentence; copy it from their page.
- Paderborn and CWRU publish terms on their own pages rather than a standard licence; read them before reusing the
  excerpts outside this project.
- BDG2-derived files are CC BY-SA 4.0: anything you build from them and share must carry the same licence and the
  credit to Miller et al. (2020). The MIT licence of the code does not extend to them.
- If you add a model, add its rows here, and only bundle data whose terms allow redistribution. Otherwise ship a
  fetch script.
