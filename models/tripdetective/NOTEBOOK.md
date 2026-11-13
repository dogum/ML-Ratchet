> **Lab notebook.** TripDetective was built in a claude.ai cloud session (12–13 Nov 2026) on the shared RoadSim + RoadData
> platform, now in `shared/vehicle/`. The first entry is that session's status document as written; its paths
> (`/home/claude/…`, `/mnt/user-data/…`, `src/`) refer to the cloud sandbox. `MODEL_CARD.md` holds every number.

## 12–13 Nov 2026 — build notes (tripdetective-status.md, as written)

From a trip log alone (speed + whatever the OBD logger has), infer the vehicle's architecture, mass, engine displacement, CdA/Crr, hybrid control behaviour and the trip's energy ledger, each with an error bar; pools trips of the same car.

- **Page:** https://claude.ai/artifact/2DFuvMBBAwREG9rx5e3tHg (real-car explorer with the network running in JS, lab with precomputed RoadSim sweeps, exam, ratchet, limits)
- **Zip:** `/mnt/user-data/outputs/tripdetective-v1.0.zip` (numpy/scipy inference, weights, demo on 4 real exam cars, MODEL_CARD, training/ scripts). Inference does not need RoadSim.
- **Workspace:** `/home/claude/road/detective/` (src/, bank/ v1–v3, ckpt/, web/, release/, STATUS.md, ratchet.json, exam.json)

### Recipe as built
- **Physics:** RoadSim (Phase 1, unedited). All changes are wrappers: `priors.py` (v1 generator as is; v2 dev-twin calibrated; v3 + VED-era mass prior) and `obdlog.py` (OBD emulator).
- **Real conditions:** 1,154 real Ann Arbor route fragments (17 unlabelled train_noise cars), 10 cycles, synthetic routes. OBD emulator with per-signal cadences measured on dev raw rows (speed 1 s, MAF/rpm 2–2.4 s, load/fuel PID 5 s), holds, quantisation, MAF-estimated fuel (enrichment missed, motoring air in fuel cut), dropouts, random channel sets; resampled with RoadData's own resampler.
- **Dev twin test:** older engines burn ~12 % more than modern maps; idle low; real hybrids run the engine ~2x as often while moving; Leaf heating far too strong in sim. Network twin test: a heavy-fleet prior (generator median 2,200 kg vs VED-era 1,720 kg) made the network read cars +230 kg heavy → v3 mass prior.
- **Network:** 1-D CNN per trip + 16 classical per-trip estimates (least-squares road-load fit, idle fuel/rpm, rpm per speed…) → trip embeddings averaged per car → heads. 0.40 M params. numpy == torch 8e-6; JS == Python 1e-4.

### Ratchet (frozen val_v3 after rescoring; +0.01 margin; dev gate on 16 cars)
| rung | change | sim skill | dev mass/car | dev engine/car | decision |
|---|---|---|---|---|---|
| r0_v1 | baseline, generator as is | 0.305 | 140 kg (+106 bias) | 0.31 L | kept |
| r1_v2 | dev-twin priors v2 | 0.362 | 128 | 0.52 | rejected (dev +23 %) |
| r2_v2full | v2, 2,000 cars | 0.366 | 116 | 0.51 | rejected (+15 %) |
| r3_v2phys | + physics channels | 0.363 | 129 | 0.53 | rejected |
| r4_v2cls | + classical estimates | 0.368 | 136 | 0.49 | rejected (+17 %) |
| r5_v1cls | classical estimates on v1 bank | 0.317 | 139 | 0.34 | kept |
| r6_v3cls | v3 mass prior + classical estimates | **0.402** | **110 (−9 bias)** | 0.44 | **kept (final)** |

### Exam (351 real VED exam cars, ≤40 trips each)
- VED type per car 98.9 % (rule baseline 99.4 %, majority 68.6 %); per trip 92.9 % (rule 96.7 %). **Loses to the rule.**
- Curb mass per car: median |err| 116 kg (7.8 %), 49 % inside the ±113 kg bin; physics LSQ 205 kg; dev median 227 kg. Coverage 73 % / 98 %.
- Displacement per car: 0.50 L (22 %), bias −0.48 L; idle-fuel law 0.64 L; dev median 0.60 L (25 %). Narrow win.
- Road-load force vs EPA targets (25 high-confidence cars): 16/17/18 % at 20/40/60 mph vs physics LSQ 66/35/19 %.
- EPA exam families (no real logs; real ETW/ABC, simulated logs, 150 vehicles): mass 7.8 %, F60 9.4 %, type 100 %.
- ZTBus not graded. Ledger: simulated twins only (beats median guess except net grade work).

### Top limits
1. Reads car "kind" more than dynamics: lab sweep +800 kg → +200 kg estimate; CdA changes barely register. Pooling 32 trips: 153 → 126 kg.
2. Engines read small; every v2-prior rung hurt dev displacement (cause not isolated).
3. Leafs read +460 kg (RoadSim Leaf twin uses ~1.5x real energy).

### Next rungs
- Isolate which v2 prior hurts displacement (ablate eff_scale spread vs gearing) and build v4 on v1+v3 mass prior.
- Make dynamics identifiable: train with sharper fuel channels / longer pooled sets, or a physics-regression head with explicit mass–efficiency separation; fix BEV HVAC energy in a wrapper.
- Add buses for ZTBus (relative mass vs passenger load).

### Sandbox gotchas
- `pgrep -f pattern` inside a `bash -c` watcher matches the watcher itself (deadlocked a queue for 30 min).
- `cd dir && cmd &` backgrounds the cd too; the shell stays where it was.
- 3 agents on 2 cores: bank of 100 cars × 8 trips took 1.5–3.5 min; training 0.3–0.8 s/step.

## 13 Nov 2026 — integrated into ML Ratchet

- Moved from `tripdetective-v1.0.zip` into `models/tripdetective/`. Sandbox paths now go through `training/_paths.py`
  (work folder `models/tripdetective/work/`, the vehicle platform at `shared/vehicle/` or `$ML_RATCHET_VEHICLE`).
- The demo reproduces the release's output exactly from the repository (checked against the zip's own run).
- Caveat carried from the handoff: built before RoadData's 13 Nov fixes (EPA city/highway labels were swapped; VED
  mass double-counted the 136 kg driver). Re-score after the fixes are upstreamed into RoadSim.

## 13 Nov 2026 — original lab and live instrument

- The original claude.ai page is now `lab/index.html`, unchanged.
- The site's model page has a live instrument (`web/export_instrument.py`, `site/src/instruments/tripdetective/`). The page's 13 real exam cars (8 trips each) with trip pooling, log length and channel removal, plus the simulated-car lab. Parity with Python: 58 cases, worst 2.6e-5.
