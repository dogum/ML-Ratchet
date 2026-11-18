# BasinDetective offline lab

Open `index.html` in a browser. It holds the released 5-member network (float16 matrices), the four real exam basins
from `examples/`, embedded fonts and the exam tables. Pick a basin, then change the input: the target decade's
precipitation and temperature, a linear reservoir on the gauged flow, gauge jitter and rounding, the number of gauged
years. Every change recomputes the 150 features and reruns the ensemble locally.

The site uses the same numerical and UI source in `site/src/instruments/basindetective/` (`features.ts` is a port of
`basindetective/features.py`, `net.ts` of `basindetective/model.py`, `whatif.ts` of the exporter's what-ifs). Rebuild:

```
python web/export_instrument.py      # from models/basindetective/: writes site/public/instruments/basindetective/
node web/build_lab.mjs               # needs the site's npm dependencies (esbuild, fonts) installed in site/
```

`build_lab.mjs` reads `training/exam.json`, `training/ratchet.json`, `training/twin_final.json` and, when the water
platform data are unpacked, `shared/water/data/proc/baselines/detective_summary.json` (the kNN column; '–' otherwise).
Data credits are in the page footer and in `examples/README.md`.
