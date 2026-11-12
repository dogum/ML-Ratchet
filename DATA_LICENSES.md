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
| ChirpNet | [LIGO DCC P1700349 / P1700337](https://dcc.ligo.org/LIGO-P1700349/public) (GW170817) | CC BY 4.0 | No — download links in `models/chirpnet/examples/README.md` |
| SunDoctor | [NREL PVDAQ via OEDI](https://data.openei.org/submissions/4568) (systems 2107, 9068) | CC BY 4.0 | Two CSV excerpts |
| CellSense | [Severson et al. 2019](https://data.matr.io/1/) (Toyota Research Institute, Stanford, MIT) | CC BY 4.0 | One cell (b3c33), seven readings |
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

Every model card (`models/*/MODEL_CARD.md`) carries the full citation list, including the physics papers.

## Before redistributing

- Some providers ask for a specific citation or acknowledgement sentence; copy it from their page.
- Paderborn and CWRU publish terms on their own pages rather than a standard licence; read them before reusing the
  excerpts outside this project.
- If you add a model, add its rows here, and only bundle data whose terms allow redistribution. Otherwise ship a
  fetch script.
