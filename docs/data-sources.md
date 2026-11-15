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

## Checked for backlog ideas

Answered: NESO API, OMNIWeb, Kyoto Dst, INTERMAGNET/BGS, NOAA NCTR/DART (landing), NGL GNSS (landing),
Crystallography Open Database, CHIME/FRB, Baseball Savant, IRSA ZTF TAP, JPL SBDB, HITRAN, NIST WebBook, CERN Open
Data, Iowa Mesonet ASOS, NASA POWER, LISIRD, Open-Meteo archive, RCSB PDB, PubChem, UCI, OpenML, Hugging Face API,
EIA, Zenodo.

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

## Python packages that helped

pvlib, PyBaMM, FLORIS, highspy (HiGHS MILP with duals), pandapower / PYPOWER / ANDES (GridSim checks only), batman-package, astroquery, wfdb, LALSuite (training only), libarchive-c.
