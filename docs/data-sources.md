# Data source atlas

Public endpoints that have worked (or not) for this collection. Reachability was checked in October–November 2026,
first from the shared cloud build machine (behind a proxy), later from the repository's own machine; your network may
differ. "Blocked" means it refused scripted requests from the build machine — never try to route around a refusal;
find another documented source or download it yourself and note it.

## Used by released models

| Source | What | Used by | Notes |
|---|---|---|---|
| [GWOSC](https://gwosc.org) | LIGO/Virgo strain, event files | ChirpNet | Blocked from the cloud build machine; reachable from a home connection (v2 API `/api/v2/events/<name>`), which is how the O3 exam was run |
| [LIGO DCC](https://dcc.ligo.org) | GW170817 data products | ChirpNet | |
| [OEDI / PVDAQ](https://data.openei.org/submissions/4568) | PV system time series | SunDoctor | S3 bucket reachable |
| [data.matr.io](https://data.matr.io/1/) | Severson 2019 battery cycling | CellSense | |
| [BatteryArchive](https://www.batteryarchive.org) | Sandia multi-chemistry cycling | (CellSense, waiting) | No public download link; their FAQ says to email. Don't probe the database API. |
| [MAST](https://archive.stsci.edu) + stpubdata S3 | TESS and Kepler light curves | TransitNet, StarQuake | `archive.stsci.edu` itself had TLS errors; astroquery + S3 worked |
| [NASA Exoplanet Archive TAP](https://exoplanetarchive.ipac.caltech.edu/) | Planet parameters | TransitNet | |
| [ExoFOP](https://exofop.ipac.caltech.edu/tess/) | TOI list and dispositions | TransitNet | |
| [EarthScope FDSN + Syngine](https://ds.iris.edu) | Waveforms, Green's functions | QuakeNet | Syngine needs `model=ak135f_2s`, `sourcewidth ≥ 2.1`; strip `+` from exponents |
| [USGS ComCat](https://earthquake.usgs.gov/fdsnws/event/1/) | Earthquake catalogue, moment tensors | QuakeNet | |
| [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) | Bearing vibration | BearingNet | Normal-baseline files are 48 kHz |
| [Paderborn KAt](https://groups.uni-paderborn.de/kat/BearingDataCenter/) | Bearing vibration | BearingNet | `.rar` archives; read with libarchive-c |
| [NOAA NDBC](https://www.ndbc.noaa.gov/) | Buoy spectral wave data | SwellCast | |
| [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) | Tropical-cyclone tracks | SwellCast | |
| [PhysioNet](https://physionet.org) | MIT-BIH, NSTDB, SVDB, QT DB | BeatNet | `wfdb` reads format 212 and annotations directly |
| [Gaia TAP](https://gea.esac.esa.int/archive/) | DR3 FLAME radii | StarQuake | Upload joins worked |
| Journal machine-readable tables | Yu et al. 2018 red giants | StarQuake | Used because VizieR was blocked |
| [Vehicle Energy Dataset (VED)](https://github.com/gsoh/VED) on GitHub | 33,015 real Ann Arbor trips (2017-18), 1 Hz | Vehicle platform: EnergyNet, TripDetective, HybridPilot | `raw.githubusercontent.com` worked; github.com HTML pages did not from the cloud build machine |
| [Argonne D3](https://www.anl.gov/taps/downloadable-dynamometer-database) via its public Box folders | 10 Hz chassis-dynamometer tests: 65 vehicles, 1,661 usable tests | RoadSim dev and exam (split by vehicle) | anl.gov refuses scripts (HTTP 403, then a browser challenge); the vehicle-to-Box-folder table was read once in a browser, then `d3_fetch.py` downloads from Box. The 2020 Model 3 (905 files, 90 GB) was not downloaded |
| [ETH Research Collection](https://doi.org/10.3929/ethz-b-000626723) | ZTBus: 1,409 Zurich trolleybus missions, 1 Hz | EnergyNet, TripDetective, HybridPilot, RoadSim's bus check | HTTP 429 from the cloud build machine (the Zenodo mirror had 10 missions); the full 2.0 GB zip downloaded from the repository's own machine on 13 Nov 2026 |
| [EPA test-car lists](https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy), drive schedules, ALPHA maps | 80,227 dynamometer tests (MY2008-2026; the MY2000-2007 files for RoadSim's exam-3); 24 cycles; engine and motor maps | RoadSim calibration and checks, all three vehicle models | |
| [fueleconomy.gov](https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip) `vehicles.csv` | Window-sticker labels | EnergyNet (model-era prior), RoadSim twins | |
| [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) | 1 m lidar elevation for Ann Arbor | Road grades of the VED trips | |
| [Iowa Environmental Mesonet ASOS](https://mesonet.agron.iastate.edu/request/download.phtml) | Airport weather on the day of each trip | Vehicle platform | |
| [EIA-930 bulk files](https://www.eia.gov/electricity/gridmonitor/) | Hourly demand, forecasts, generation by fuel, interchange | Grid platform: DemandCast, BatteryPilot, GridSim checks | Six-month CSVs need no key; the api.eia.gov API needs one |
| [PUDL](https://catalyst.coop/pudl/) parquet tables (S3) | EIA-860/861/923, EPA-EIA crosswalk | Grid platform: fleets, heat rates | `s3.us-west-2.amazonaws.com/pudl.catalyst.coop/stable/<table>.parquet` |
| [EPA CAMPD](https://campd.epa.gov/) bulk files | Hourly unit output, heat input, CO2 | PlantDetective, GridSim checks | The file route `api.epa.gov/easey/bulk-files/...` needs no key (0.77 GB per state-year); the file listing needs an api.data.gov key |
| [NESO data portal](https://www.neso.energy/data-portal) CKAN API | GB 1 s frequency, inertia, generation mix | GridPulse, GridSim checks | Direct file downloads redirect to a storage host the build machine's proxy refused (403); the datastore SQL API worked |
| [Elexon BMRS](https://bmrs.elexon.co.uk/) API | REMIT, B1610, FUELINST | GridPulse (loss labels) | |
| [ERCOT](https://www.ercot.com/) MIS documents and monthly reports | Prices; frequency events | BatteryPilot, GridPulse | 17 monthly operations reports 2018-2026 not found |
| [CAISO OASIS](http://oasis.caiso.com/) | LMPs | BatteryPilot | Serves only about 3 years (from 2023-08) |
| [MATPOWER](https://github.com/MATPOWER/matpower) cases | IEEE and ACTIVSg test grids | GridSim | The Texas A&M download form asks for an email; MATPOWER's copies are the same data |
| [Building Data Genome 2](https://github.com/buds-lab/building-data-genome-project-2) | 1,636 buildings, 3,053 hourly meters, 2016-17, site weather | Building platform: LoadCast, BuildingDetective, ThermoPilot | CC BY-SA 4.0. Mixed DST/standard timestamps; one site's electricity is daily totals spread hourly; one site's thermal meters 1,000× too large |
| [EIA RECS 2020 / CBECS 2018](https://www.eia.gov/consumption/) microdata | Home and commercial-building surveys | BuildingDetective, ThermoPilot, BuildingSim's generator | |
| [OEDI](https://data.openei.org/submissions/4520) ResStock / ComStock | Modelled stock characteristics and hourly profiles | BuildingSim priors and twin checks | The OEDI dataset pages answered "No data found" to a script on 17 Nov 2026; the S3 files were fetched in the build sandbox |
| LBNL curated fault datasets (RP-1312, FLEXLAB, ORNL, Nesbitt Hall) | 237 AHU/RTU fault runs | FaultFinder | The full datasets sit behind a name/email form at faultdetection.lbl.gov (HTTP 521/522 on 17 Nov 2026) |
| [WPuQ](https://zenodo.org/records/5642902) (Zenodo), [HUE](https://doi.org/10.7910/DVN/N3HGRN) (Harvard Dataverse) | Heat-pump homes; BC homes | ThermoPilot, BuildingDetective | Dataverse returned an empty body to a script |
| [OpenEI URDB](https://openei.org/wiki/Utility_Rate_Database) | Utility tariffs | ThermoPilot | |
| [climate.onebuilding.org](https://climate.onebuilding.org) TMYx | Typical-year weather | BuildingSim training weather | "All Rights Reserved": used, not redistributed |
| [Caravan v1.6](https://doi.org/10.5281/zenodo.15529786) (Zenodo) | ERA5-Land basin forcing, flow and attributes for CAMELS-US, GB, AUS, CL, BR | Water platform: FlowCast, BasinDetective, WaterSim calibration | One 24.8 GB `Caravan-nc.tar.gz`; `fetch_caravan.py` streams it once and keeps 1.7 GB. CAMELS-AUS members are netCDF4 (the rest netCDF3); the first pass crashed on them at 23.84 GB |
| [USGS NWIS](https://waterservices.usgs.gov/) daily values, site info, annual peaks | Flow to Nov 2026 for 671 gauges; 48,769 peaks | FlowCast's recent exam and peaks; WaterData's implied-area fix | The legacy `waterservices` endpoints are being retired in favour of `api.waterdata.usgs.gov`; `fetch_nwis.py` still uses the legacy ones |
| [NOAA NWM v3.0 retrospective](https://registry.opendata.aws/nwm-archive/) (S3) | Simulated flow at 650 CAMELS gauges, 1980-2023 | FlowCast's operational baseline | One HTTP range request per hour (the streamflow chunk starts at byte 162,458 of each CHANOBS file): about 385,000 requests |
| [NRCS SNOTEL Report Generator](https://wcc.sc.egov.usda.gov/reportGenerator/) | Daily SWE, precipitation, temperature at 934 stations | SnowSense, WaterSim's snow check | SWE and accumulated precipitation are start-of-day; temperatures carry the 2000s sensor-change inhomogeneity |
| [ResOpsUS v2](https://doi.org/10.5281/zenodo.6612040) (Zenodo) + ResOpsUS+CARS GRanD attributes | Daily storage, inflow, release for 678 US dams | ReservoirPilot | Negative back-calculated inflows at 189 dams; only 171 of 277 close their daily mass balance (corr ≥ 0.9). CARS attribute CSVs read from the remote zip by HTTP range |
| [California CDEC](https://cdec.water.ca.gov/) | Daily reservoir storage, inflow, outflow, 18 reservoirs | ReservoirPilot's recent exam | |
| [FEDS-MTBS](https://zenodo.org/records/20187963) (Zenodo) | 12-hourly perimeters, active fire lines and VIIRS S-NPP 375 m pixels for 7,739 MTBS fires, 2012-2024 | Fire platform: FireCast, FireDetective, FirePilot, the dev replays and the next-day exam | CC BY 4.0. `farea` is in m² (the readme does not say); half-day labels are nominal (`12:00` is the ~13:30 local overpass); the last steps above 99.9 % of the final area were removed by the authors. 171 MB processed |
| [GOFER-Combined](https://zenodo.org/records/14642378) (Zenodo) | Hourly GOES perimeters, 28 California fires 2019-2021 | FireData's truth-quality check (FEDS against GOES on 4 dev fires: Dice 0.77-0.93) | CC BY 4.0 |
| [WildfireSpreadTS](https://zenodo.org/records/8006177) (Zenodo) | Daily VIIRS active fire, weather, fuel and terrain for 607 US fires, 2018-2021 | Benchmark context for FireCast (published AP numbers, a different target) | CC BY 4.0. One large zip streamed fire by fire with HTTP range reads (about 2 fires a minute); 269 MB kept |
| [FPA-FOD 6th ed.](https://doi.org/10.2737/RDS-2013-0009.6) (USDA Forest Service Research Data Archive) | 1.59 M CONUS wildfire ignition records 2001-2020 (2012-2020 used) | FireSize | Sizes are heavily rounded (0.01, 0.1, 1 acre); no records after 2020 |
| [Globe-LFMC 2.0](https://doi.org/10.6084/m9.figshare.25413790) (figshare) | 239k CONUS live fuel moisture samples | FireDetective's live-moisture truth | CC0 |
| [LANDFIRE LFPS](https://lfps.usgs.gov/) ImageServer `exportImage` | FBFM40, canopy cover, height, base height, bulk density (LF2016/2022/2023); elevation, slope, aspect (LF2020) | Fire platform landscapes (40 dev and 48 replay fires); training windows | Works per window at 180 m with 4 threads (network-bound). LF2016 is post-fire for 2012-2016 fires; no pre-fire layer exists for them |
| [HRRR](https://registry.opendata.aws/noaa-hrrr-pds/) via `hrrrzarr` (University of Utah, AWS Open Data) | Hourly 2 m T, RH, 10 m wind, gust, radiation, cloud; 1 h precipitation from short-lead forecasts | FireSim replays and training weather | Starts 25 Aug 2016; analysis precipitation chunks do not exist and forecasts start mid-2018, so precipitation is NaN for 2017-early 2018 fires; 1-5.5 % of hours missing on 2017-18 fires (archive gaps); float16 chunks switch to float32 during 2024 |
| [gridMET](https://www.northwestknowledge.net/metdata/data/) HTTP files | Daily weather and NFDRS fm100, fm1000, ERC, BI at fires, ignitions and moisture sites | All four fire models; FireData's baselines | The THREDDS server on port 8080 timed out; the plain HTTP files worked |

## Checked for backlog ideas

Answered: NESO API, OMNIWeb, Kyoto Dst, INTERMAGNET/BGS, NOAA NCTR/DART (landing), NGL GNSS (landing),
Crystallography Open Database, CHIME/FRB, Baseball Savant, IRSA ZTF TAP, JPL SBDB, HITRAN, NIST WebBook, CERN Open
Data, Iowa Mesonet ASOS, NASA POWER, LISIRD, Open-Meteo archive, RCSB PDB, PubChem, UCI, OpenML, Hugging Face API,
EIA, Zenodo. On 18 Nov 2026, from the claude.ai build sandbox: NIFC WFIGS, FIRMS, GOES-16 FDC on S3, MTBS, Copernicus
DEM on S3, WRCC, WFAS, CDAWeb, NOAA CO-OPS tides API, NDBC, NOAA NGDC tsunami event search, NGL over https, Hugging Face
Prior-Labs.

## Blocked or unreliable from the cloud build machine

| Host | What happened |
|---|---|
| CDS / VizieR | Connection reset — Claret limb-darkening tables and many catalogues unavailable |
| rruff.info | 403 |
| opensky-network.org | 403 |
| SDSS SkyServer, nrel.gov, jsoc.stanford.edu | Unreachable |
| NREL Fleet DNA (nrel.gov, data.nrel.gov; OpenEI only links there) | Timed out; still blocked. Real truck and bus traces with fuel: the vehicle platform's heavy-duty exam waits on them |
| Open-Meteo (forecast, archive, marine) | Timeouts / daily limit exceeded on a shared IP |
| ds.iris.edu file listings | Unreachable (FDSN web services worked) |
| MFPT bearing data | No longer offers downloads (contact only) |
| Open-Meteo Previous-Runs API | HTTP 429 after a few requests on the shared IP (DemandCast's forecast-weather archive: 2024 only). One request from the repository's own machine answered 200 on 14 Nov 2026 |
| fnetpublic.utk.edu (FNET/GridEye) | Timed out from the cloud build machine; the public pages answered from the repository's own machine on 14 Nov 2026 |
| NERC reliability reports (nerc.com) | Returned an HTML page instead of the PDF |
| nationalgrideso.com | 502 (the old ESO domain; the same reports are on neso.energy and ofgem.gov.uk) |
| github.com (from the claude.ai build sandbox, Nov 2026) | Blocked, so no EnergyPlus binary: BuildingSim has no EnergyPlus reference run |
| ecobee Donate Your Data, UK Data Service (RHPP, Electrification of Heat), Pecan Street | Gated by application or registration: no real thermostat logs and no UK heat-pump trials yet |
| NCAR GDEX (original CAMELS Daymet/Maurer/NLDAS forcing), Daymet API (ORNL) | Timed out from the claude.ai build sandbox on 13 and 18 Nov 2026, so every water model runs on ERA5-Land. Not yet tried from the repository's own machine |
| wfas.net/nfmd (National Fuel Moisture Database) | 404 on 18 Nov 2026: NFMD moved into FEMS (fems.fs2c.usda.gov), and no documented public download was found. No real dead-fuel stick data, so FireSim's Nelson model is checked only against gridMET NFDRS |
| raws.dri.edu; thredds.northwestknowledge.net:8080; Synoptic API | RAWS pages did not answer (WRCC's are interactive forms only); gridMET THREDDS timed out (HTTP files used instead); Synoptic needs a key. No hourly RAWS or fuel-stick records |
| github.com/pyregence | 403 from the claude.ai build sandbox; pyretechnics was installed from PyPI instead for FireSim's cross-check |
| NASA FIRMS archive; MTBS per-fire severity mosaics; ICS-209 | FIRMS archive downloads need a free MAP_KEY (not needed: FEDS ships the VIIRS pixels it used); the severity mosaics go through an e-mail queue (FireDetective read severity through LANDFIRE Annual Disturbance); no validated suppression records were fetched |

## Python packages that helped

pvlib, PyBaMM, FLORIS, highspy (HiGHS MILP with duals), pandapower / PYPOWER / ANDES (GridSim checks only), batman-package, astroquery, wfdb, LALSuite (training only), libarchive-c, pyretechnics and xclim (FireSim checks only, in an isolated environment).
