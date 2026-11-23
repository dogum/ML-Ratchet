# QuakeNet v2 — status (v1.0, 22 Nov 2026)

> Kept as written on 22 Nov 2026 in the cloud project, with names of people and build roles made neutral and the claude.ai page links replaced by each model's `lab/` in this repository. Paths and status refer to that project.

Several seismometers, one earthquake: from 2–10 GSN stations' 41-minute long-period records, a graph network gives the epicentre (with a 68% ellipse), depth, Mw and origin time, each with an error bar. QuakeNet v1.2 reads each station, unchanged. The graph network corrects the classical combination of those readings, using the station geometry and the P arrival times. It was trained only on simulations and graded once on real USGS earthquakes. B batch, project 1. It fills the `quakenet` × graph-network slot on the Model families page.

- Page (the network runs live in the browser; 24 seeded exam quakes, station toggles, a pick-shift lab, the exam, the ratchet and the twin test): `models/quakenet-v2/lab/`
- Release: `quakenet-v2-v1.0.zip` (15.6 MB). It contains:
  - numpy + scipy inference: `quakenet2.py`, with v1.2's `quakenet.py`, `pipeline.py` and weights copied unchanged
  - graph weights with the dev widening factors, and the ak135 table
  - `demo.py` on 4 exam quakes (2 shallow, 2 deep), each at 2 stations and at 10; it runs in about 2 s
  - model card, notebook, `model.yaml`, `ratchet.yaml`, data licences, all training scripts and the result JSONs
  - `for_repo/models/quakenet-v2/` in the repo's layout (lab page, `_paths.py`, `splits.json`), plus `for_repo/tests/golden/quakenet-v2.txt`
  - `BESIDE_V1.2.md` on how v2 sits beside v1.2

## Honest headline

- **Epicentre, median miss:** v2 wins against every baseline at every station count.

  | Stations | QuakeNet v2 | v1.2 combined classically | Travel-time locator | Station centroid |
  |---|---|---|---|---|
  | 2 (1,456 quakes) | **427 km** | 521 | 4,139 | 6,190 |
  | 3 (400 quakes) | **279** | 410 | 669 | – |
  | 5 (400 quakes) | **188** | 320 | 276 | – |
  | all, up to 10 (400 quakes) | **127** | 240 | 173 | 4,223 |

- **Origin time, median error:**
  - all stations: 4.3 s, against 5.8 s for v1.2 combined and 14.4 s for the travel-time locator
  - 2 stations: 13.5 s, against 15.0 s and 56.5 s
- **Depth, median error:** with all stations, 17.6 km against 29.1 km for v1.2 combined. On shallow quakes, 11.7 km against 29.3 km.
- **Losses:**
  - Mw: 0.097 against 0.089 for the plain inverse-variance average of v1.2's per-station Mw, and it loses at every station count.
  - Deep quakes (> 70 km) are read too shallow: 38 km median depth error against 28 km.
- **Coverage of the 68% bars**, after dev widening (epicentre ×1.31, others about ×1.0):
  - all stations: epicentre ellipse 62%, depth 60%, Mw 64%, origin time 74%
  - 2 stations: 65%, 60%, 70%, 65%

## Recipe as built

- **Splits:** frozen first, in `splits.json`. These are v1.2's 40 dev and 1,734 exam events exactly (its `real_meta.json`, seed 2026). We fetched extra stations (`station_order.json`, seeded, written before any record was read) for:
  - all 40 dev events
  - a 400-event exam subset (seed 7, depth strata)

  That gave 3,523 new LH records, and 436 of 440 events reached 10 stations.
- **Physics:**
  - Per station: v1.2's simulator, physics v7 with layering.
  - Shared per event: epicentre, depth, Mw, mechanism, source duration and ocean layer.
  - Exact distances: the nearest Green's function is stretched so that S−P matches ak135.
  - Absolute timing: ak135 P time plus a station residual N(0, 1.5 s), with records cut as the real ones were.
- **Checks:**
  - ak135 table vs TauP: 0.078 s; vs iasp91: 0.24 s
  - distances vs ObsPy: 6e-14°
  - numpy vs torch: 6e-6
  - JS vs Python: 0.14 km, 0.05 km in depth, 2e-5 in Mw
  - release inference equals the training path on dev
- **Noise:** v1.2's 1,284 real quiet-time windows, from the same station 80% of the time.
- **Bank:**
  - 10,000 simulated events at 2–10 real stations (60,093 station records), giving 26,683 training graphs
  - frozen set: 2,000 events (3,785 graphs)
- **Network:** 115k weights, 3 rounds of message passing, width 64. It predicts corrections to the anchor, with a bivariate epicentre Gaussian and β-NLL on the other outputs.
- **Dev twins:** all 40 dev quakes were re-simulated from their USGS moment tensors, in real station noise. Compared with the twins, v1.2 reads real records:
  - +1.5° farther
  - with more back-azimuth scatter (8.6° against 5.5°)
  - with shallow quakes ×2.4 too deep

  Real P picks are also tighter than simulated ones (8.9 s against 11.8 s). Physics v2 and v3 put these differences into the simulated readings.

## Rungs

Rule: keep a rung if the frozen simulated objective beats the incumbent by more than 0.01 **and** the 40 dev quakes get no more than 5% worse. The objective is the mean over epicentre, log depth, Mw and origin time of 1 − error / rung-0 error. Every rung was re-scored on val1, then val2, then val3, and every decision was re-taken; none flipped. The exam columns were reported after the fact.

| Rung | Change | Sim val3 | Dev epi (mean) | Exam epi 2 / all | Exam depth all | Exam Mw all | Exam t0 all | Decision |
|---|---|---|---|---|---|---|---|---|
| r0 | v1.2 per station, combined classically | 0.000 | 533 km | 521 / 240 | 29.1 | 0.089 | 5.8 s | baseline |
| r1 | graph net: readings + geometry | 0.025 | 487 | 512 / 205 | 27.8 | 0.111 | 5.9 | kept |
| r2 | + P pick times | 0.070 | 375 | 452 / 114 | 25.3 | 0.104 | 5.7 | kept |
| r3 | + v1.2 256-d summary | 0.071 | 373 | 451 / 122 | 26.9 | 0.102 | 5.8 | rejected (+0.001) |
| r4 | + P cross-correlation lags | 0.075 | 371 | 446 / 117 | 26.7 | 0.098 | 6.1 | rejected (+0.006) |
| r5 | physics v2: distance / back-azimuth / pick mismatch from twins | 0.108 | 331 | 417 / 121 | 28.4 | 0.110 | 5.9 | kept |
| r6 | physics v3: shallow-depth and Mw mismatch from twins | 0.219 | 322 | 427 / 127 | 17.6 | 0.097 | 4.3 | **released** |

r2 scored better than r6 on the all-station exam epicentre (114 km against 127 km). The ratchet had no way to see that before the exam, and the exam does not change the decision.

## Limits

- Mw loses to plain averaging. The network's Mw correction helps in simulation and hurts on real records.
- Deep quakes are over-corrected toward shallow: the dev twin depth ratio was 0.74.
- Two stations still miss by a median 427 km, with a mean of 918 km.
- Inputs inherited from v1.2:
  - LH records only, at 25–95°
  - a ±30 s P search around a guess; in the exam, as in v1.2, the guess came from the catalogue time
  - record start times were rounded to 1 s when the archive cut the records
- The dev-twin calibrations are fixed shifts fitted on 40 quakes.

## Next ideas

1. Pass the classical Mw through (or learn only an event bias), and calibrate depth by depth class on dev.
2. A second pass that re-linearises at the network's own epicentre.
3. BH channels with a sharper P picker for timing, and regional distances.

## Sandbox gotchas

- `setsid … &` returns immediately, so `$!` is not the python PID. Use `ps` to find it, and kill only your own PIDs. Two duplicate fetchers ran for a few minutes because of this.
- EarthScope returned "too many requests" for 51 of about 4,200 requests at 3 threads. We did not retry; the events reached 10 stations anyway.
- Green's-function loading dominates simulation until the cache is warm: 86 ms per file on the shared disk. With all ~1,900 files in RAM (~370 MB), it runs at about 25 ms per station record.
- 3 real exam records give non-finite v1.2 outputs (empty windows) and are dropped before anything is scored.
