# SnowSense offline lab

Open `index.html` in a browser. It holds the released network (float16), the three shipped exam stations, embedded
fonts and the exam and ratchet tables. Pick a station and water year, then warm every day, scale the gauge
precipitation, move the station up or down (with or without the lapse rate), remove the biggest storm or knock out the
gauge for 30 days. SnowSense and WaterSim's uncalibrated energy-balance and degree-day routines rerun locally on the
changed weather; the observed pillow is shown for the recorded weather.

The site uses the same source in `site/src/instruments/snowsense/`. To rebuild:

```bash
python models/snowsense/web/export_instrument.py   # packs and parity cases -> site/public/instruments/snowsense/
node models/snowsense/web/build_lab.mjs            # this page (needs site/node_modules)
cd site && npm test                                # browser network vs Python
```

This page is new on import (18 Nov 2026). The original claude.ai page and its data file were not in the release zip.
SNOTEL data: USDA NRCS, US public domain.
