<!-- Generated from docs/backlog.yaml by `mlratchet readme --write`. Edit that, not this file. -->
# The idea backlog

Candidates for the next models. Each has physics to simulate, public real data, and a real exam lined up.
Data reachability was checked from a cloud sandbox in November 2026 ([data-sources.md](data-sources.md)).
Want to build one? Read [adding-a-model.md](adding-a-model.md) and open an issue first.

| Idea | The question | Physics | Real data | Exam | Status |
|---|---|---|---|---|---|
| **GridPulse**<br><sub>Energy</sub> | How many MW were lost, and how much inertia does the grid have, from the frequency dip after a plant trips? | Swing equation with governor and load response | [GB system operator (NESO) data portal API](https://www.neso.energy/data-portal) | Real trip events with known MW lost | checked<br><sub>Few real events to grade on.</sub> |
| **RoomSense**<br><sub>Acoustics</sub> | How big is this room and how long does it ring, from one recorded clap? | Image-source and ray-tracing room acoustics (pyroomacoustics) | [Measured room impulse responses on Zenodo](https://zenodo.org) | Rooms with measured geometry and RT60 | candidate<br><sub>Impulse-response sets not yet checked.</sub> |
| **StormNet**<br><sub>Space weather</sub> | How strong will this geomagnetic storm get, from the solar wind arriving now? | Burton-type ring-current response to the solar wind | [OMNIWeb solar wind; Kyoto Dst; INTERMAGNET / BGS magnetometers](https://omniweb.gsfc.nasa.gov/) | Real Dst and storm onsets | checked |
| **TsunamiNet**<br><sub>Geophysics</sub> | How big is the tsunami coming, from the first deep-ocean pressure readings? | Shallow-water propagation from Okada seafloor sources | [NOAA NCTR / DART buoys](https://nctr.pmel.noaa.gov/Dart/) | Real tsunamis against USGS finite-fault sources | checked<br><sub>Landing page answered; file path still to confirm.</sub> |
| **GPSQuake**<br><sub>Geophysics</sub> | How far did the ground move, and was it an earthquake, slow slip or the seasons? | Okada dislocations, slow-slip transients, seasonal loading | [Nevada Geodetic Laboratory GNSS time series](http://geodesy.unr.edu/) | Known earthquakes' offsets; Cascadia slow-slip events | checked<br><sub>Landing page answered; tenv3 path to confirm.</sub> |
| **XRDNet**<br><sub>Materials</sub> | Which crystal phases are in this powder, from its X-ray diffraction pattern? | Kinematic powder diffraction from crystal structures | [Crystallography Open Database (500k+ structures)](https://www.crystallography.net/cod/) | Measured diffraction patterns | blocked<br><sub>RRUFF (measured patterns) returned 403; another source of measured patterns is needed.</sub> |
| **FRBNet**<br><sub>Astrophysics</sub> | How much plasma did this fast radio burst cross, and how much was it scattered? | Cold-plasma dispersion and multipath scattering | [CHIME/FRB catalogue](https://www.chime-frb.ca/catalog) | Catalogue DM and scattering times | checked |
| **SpinNet**<br><sub>Sports physics</sub> | How fast and which way is this pitch spinning, from its flight alone? | Drag, Magnus force and seam-shifted wake | [Baseball Savant pitch tracking](https://baseballsavant.mlb.com/) | Measured Hawk-Eye spin | checked<br><sub>Check the terms of use before redistributing anything.</sub> |
| **SpinRock**<br><sub>Planetary science</sub> | How fast is this asteroid spinning, and what shape is it, from sparse survey photometry? | Asteroid light-curve inversion (shape, spin axis, phase effects) | [ZTF via IRSA TAP; JPL Small-Body Database](https://irsa.ipac.caltech.edu/) | Known spin periods | checked |
| **SpectraNet**<br><sub>Chemistry</sub> | Which gases, and how much of each, are in this infrared spectrum? | HITRAN line-by-line absorption | [HITRAN; NIST Chemistry WebBook](https://hitran.org/) | Measured gas spectra | checked |
| **WakeNet**<br><sub>Energy</sub> | How much power is each wind turbine losing to its neighbours' wakes? | FLORIS wake models | [Kelmarsh and Penmanshiel SCADA (Zenodo)](https://zenodo.org) | Real farm SCADA with wake-steering periods | candidate |
| **JetNet**<br><sub>Particle physics</sub> | Which particle started this jet? | Parton showers and hadronisation (Pythia-style) | [CERN Open Data](https://opendata.cern.ch/) | Real-data control regions | candidate<br><sub>Harder to grade on real data than the others.</sub> |

Model upgrades (the next rungs for existing models) live in each model's `model.yaml` under `next`, and blocked exams under `help_wanted`. Collection-level plans: [roadmap.md](roadmap.md).
