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

Every model card (`models/*/MODEL_CARD.md`) carries the full citation list, including the physics papers.

## Before redistributing

- Some providers ask for a specific citation or acknowledgement sentence; copy it from their page.
- Paderborn and CWRU publish terms on their own pages rather than a standard licence; read them before reusing the
  excerpts outside this project.
- BDG2-derived files are CC BY-SA 4.0: anything you build from them and share must carry the same licence and the
  credit to Miller et al. (2020). The MIT licence of the code does not extend to them.
- If you add a model, add its rows here, and only bundle data whose terms allow redistribution. Otherwise ship a
  fetch script.
