# Data and licences

| Data | Source | Licence / terms | Role |
|---|---|---|---|
| Earthquake catalogue, locations, origin times, magnitudes, moment tensors | USGS ComCat, https://earthquake.usgs.gov/fdsnws/event/1/ | Public domain (U.S. Government) | truth (dev, exam), dev twins |
| Waveforms, IU network (Global Seismograph Network, ASL/USGS), doi:10.7914/SN/IU | EarthScope / IRIS DMC FDSN dataselect, https://service.earthscope.org/ | Open; cite the network | dev, exam, noise windows, examples/ |
| Waveforms, II network (IRIS/IDA), doi:10.7914/SN/II | EarthScope / IRIS DMC | Open; cite the network | dev, exam, noise windows, examples/ |
| Green's functions ak135f_2s | EarthScope Syngine, https://ds.iris.edu/ds/products/syngine/ (QuakeNet v1.2's grid) | Open; cite Syngine (Krischer et al. 2017) | physics |
| ak135 travel times | ObsPy TauP (Kennett, Engdahl & Buland 1995 model) | ObsPy LGPL; model published | physics (`tt_ak135.npz`) |
| 1990–2011 catalogue (source mix prior) | USGS ComCat | Public domain | simulator prior |
| Land outlines (lab page) | Natural Earth 1:110m | Public domain | page only |

`examples/` holds the full 3-component LH records of 4 exam earthquakes at 10 stations each (response removed, 1 sps).
