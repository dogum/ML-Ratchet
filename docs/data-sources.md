# Data source atlas

Public endpoints that have worked (or not) for this collection. Reachability was checked from a cloud sandbox in
October–November 2026 behind a proxy; your network may differ. "Blocked" means it refused that sandbox — never try
to route around a refusal; find another documented source or download it yourself and note it.

## Used by released models

| Source | What | Used by | Notes |
|---|---|---|---|
| [GWOSC](https://gwosc.org) | LIGO/Virgo strain, event files | ChirpNet | Blocked from the cloud sandbox; reachable from a home connection (v2 API `/api/v2/events/<name>`), which is how the O3 exam was run |
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

## Checked for backlog ideas

Answered: NESO API, OMNIWeb, Kyoto Dst, INTERMAGNET/BGS, NOAA NCTR/DART (landing), NGL GNSS (landing),
Crystallography Open Database, CHIME/FRB, Baseball Savant, IRSA ZTF TAP, JPL SBDB, HITRAN, NIST WebBook, CERN Open
Data, Iowa Mesonet ASOS, NASA POWER, LISIRD, Open-Meteo archive, RCSB PDB, PubChem, UCI, OpenML, Hugging Face API,
EIA, Zenodo.

## Blocked or unreliable from the sandbox

| Host | What happened |
|---|---|
| CDS / VizieR | Connection reset — Claret limb-darkening tables and many catalogues unavailable |
| rruff.info | 403 |
| opensky-network.org | 403 |
| SDSS SkyServer, nrel.gov, jsoc.stanford.edu | Unreachable |
| Open-Meteo (forecast, archive, marine) | Timeouts / daily limit exceeded on a shared IP |
| ds.iris.edu file listings | Unreachable (FDSN web services worked) |
| MFPT bearing data | No longer offers downloads (contact only) |

## Python packages that helped

pvlib, PyBaMM, FLORIS, batman-package, astroquery, wfdb, LALSuite (training only), libarchive-c.
