# StormNet exam excerpts

The four recordings are a fixed, stratified draw from the primary exam. Seed 931 selected one distinct block and one issue in each nonempty severity band. Each archive includes up to six preceding consecutive eligible issues from that block. The last frame is the draw. Issue timestamps in `records.json` are UTC hour ends.

| Array | Shape per frame | Units and use |
|---|---|---|
| `wind` | 36 × 3 | Oldest first. GSM Bz (nT), solar-wind speed (km/s), dynamic pressure (nPa). |
| `last_dst` | scalar | Last completed hourly Dst, nT. Available to every method. |
| `past_dst` | 36 | Historical Dst for the display. Only the last value enters inference. |
| `future_dst` | 3 | Observed subsequent hourly Dst for verification. |

**Data terms.** NASA SPDF OMNIWeb and its original instrument teams supplied the wind measurements. The Dst index is supplied by the World Data Center for Geomagnetism, Kyoto, and contributing observatories. WDC Kyoto permits scientific use with credit and prohibits commercial application of its geomagnetic indices. These terms cover the excerpts in this directory, the instrument packs and the lab. Please cite the index DOI and the OMNI publication. [Kyoto terms](https://wdc.kugi.kyoto-u.ac.jp/wdc/Sec3.html) · [Dst DOI](https://doi.org/10.17593/14515-74000) · [OMNI methodology](https://doi.org/10.1029/2004JA010649)
