# How QuakeNet v2 sits beside QuakeNet v1.2

- **Separate model folder:** `models/quakenet-v2/`. Nothing in `models/quakenet/` changes.
- **v1.2 is inside v2, unchanged.** v2 reads each station with v1.2's network: `quakenet.py`, `pipeline.py` and
  `quakenet_weights.npz` are byte-for-byte copies of `models/quakenet/`, so the v2 folder still runs on its own.
  If the repo prefers one copy, v2 can import them from `../quakenet/` instead; only the three imports change.
- **The same exam earthquakes.** v2 keeps v1.2's dev and exam events exactly (`splits.json`, built from v1.2's
  `real_meta.json`: 40 dev events, 1,734 exam events, seed 2026). An event never crosses splits.
  v2 adds records at more stations for all 40 dev events and for a fixed 400-event subset of the exam events
  (seed 7, drawn within depth strata). The planned station order is in `training/` results (`station_order.json`).
- **What v2 adds:** location (epicentre), origin time, and multi-station Mw and depth, from 2 to 10 stations.
  v1.2 answers per station: distance, back-azimuth, Mw, depth.
- **What v2 inherits:** v1.2's long-period window (LH, 25–95°, M ≥ 5.5), its P pick, and its biases. The twin test
  measured those biases on dev records, and v2's physics versions 2 and 3 build them into the simulated readings.
- **Model families page:** this is the `quakenet` × `gnn` entry (`site/src/lib/model-families.ts`, "Multi-station
  reading"), and the `quakenet-multistation` backlog entry (`docs/backlog.yaml`), which this release fills.
- **Training needs v1.2's work folder** (Green's functions, noise windows, catalogue, real records) and v1.2's
  rung-7 torch checkpoint (`q7_zrt.pt`); `training/_paths.py` points at them.
