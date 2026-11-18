# ReservoirPilot offline lab

Open `index.html` in a browser. It holds the released policy, the six real exam dams, embedded fonts and the exam and
ratchet tables. Pick a dam, then change the decade's inflow, add a flood pulse, change the capacity, add a storage-gauge
error or raise demand: ReservoirPilot, SOP and the sim-tuned rule re-run at once, and WaterData's perfect-foresight DP
re-solves about half a second later. Nothing is precomputed except the dams' records.

The site uses the same numerical and UI source in `site/src/instruments/reservoirpilot/`. Rebuild from the repository
root: `python models/reservoirpilot/web/export_instrument.py` (writes `site/public/instruments/reservoirpilot/` and the
parity cases), then `node models/reservoirpilot/web/build_lab.mjs` with the site's npm dependencies installed. The
tables come from `model.yaml`, `ratchet.yaml` and `training/exam_results.json`. `lab/src/` holds the page template and
the entry script. The shipped sandbox page's sources are kept in `training/` (`engine.js`, `template.html`,
`export_web.py`, `texts.json`).
