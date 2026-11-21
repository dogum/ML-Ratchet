# Watershed Phase 2 brief: four ratchet projects on WaterSim + WaterData

## Read first
- `/home/claude/AGENT_BRIEF.md`: recipe, deliverables, shared-machine rules. **Four** agents share 2 cores and ~6.3 GB RAM: small networks, modest banks, one thread (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`, `NUMBA_NUM_THREADS=1`), `nice`, `setsid nohup`, resumable checkpoints (container restarts happen). Generate banks first, then train (bank generation alongside training starves the trainer). Keep each process ≤ ~1.2 GB RAM.
- `/home/claude/water/WATER_BRIEF.md`: the shared contract.
- Phase 1 docs (read fully before designing):
  - `/home/claude/water/watersim/README.md`, especially "Using it in Phase 2" and "Known limitations"
  - `/home/claude/water/waterdata/DATA.md`, especially the splits, the baseline tables and section 8 "Gotchas"
  - the check JSONs in `/home/claude/water/checks/`

## Calling Phase 1
```python
import sys; sys.path.insert(0,'/home/claude/water/watersim'); sys.path.insert(0,'/home/claude/water')
import watersim as ws, waterdata as wd
wd.basins('camels', split='dev'|'exam'); wd.spec(id); wd.forcing(id); wd.forcing_meta(id); wd.streamflow(id)
wd.nwm(id); wd.peaks(id); wd.snotel_meta(split=); wd.snotel(sid); wd.snotel_near(id)
wd.reservoir_meta(split=); wd.reservoir(rid); wd.reservoir_cdec(cid); wd.splits(); wd.baselines(name)
ws.simulate(...); ws.simulate_batch(...); ws.simulate_many(...); ws.generate_basins(n, rng, prior=...)
ws.generate_weather(...); ws.realdata.random_forcing(...); ws.noise.*; ws.calibrate.calibrate(...); ws.reservoir.Reservoir
```
- `waterdata/metrics.py` holds NSE, KGE, FHV, FLV, peak error and the reservoir cost J. Use them for every score so numbers are comparable with B's baseline table.

## Frozen splits (`/home/claude/water/data/proc/splits.json`, seed 20261118; NEVER change)

| Data | Dev | Exam |
|---|---|---|
| CAMELS-US basins | 52 | 619 (487 in the 531 set) |
| GB, CL, BR basins (AUS when frozen) | 3 per region | the rest |
| SNOTEL stations | 39 | 772 (621 scored WY2011–2025) |
| ResOpsUS core dams | 15 | 250 (244 scored) |

- **Time windows:**
  - Calibration (history) 1999-10-01..2008-09-30.
  - Main exam 1989-10-01..1999-09-30.
  - Recent exam WY2016–2023.
  - Free windows for simulated training forcing: before 1988-10, and 1999-10..2015-09. `ws.realdata` enforces this.
- **What may be used:**
  - Train only on simulations.
  - Real attributes only from dev basins/stations/dams; real forcing may be reused from any basin in free windows.
  - Dev units (any period except the exam windows) are for calibration of physics, gates and sigma fitting.
  - Exam units/windows are for the final report only. Never select checkpoints, hyperparameters or physics on exam.
- **Ungauged protocol:** a network that reads only forcing and attributes is *ungauged* by construction (no Q from the scored basin), so compare it to B's U-dev regionalisation and to NWM. Any mode that ingests a basin's own calibration-period Q is *gauged*: compare it to per-basin calibrated GR4J, and label the protocol in every table.

## Exam baselines to beat (B's medians; full tables in DATA.md)
- **Streamflow, main exam (validation):**

  | Baseline | NSE | KGE |
  |---|---|---|
  | Climatology | 0.065 | |
  | Persistence, 1-day lead | 0.475 | |
  | GR4J + snow, gauged calibrated | 0.473 | 0.60 |
  | NN regionalisation (ungauged) | 0.372 (U-dev) / 0.385 (U-k12) | |
  | NWM v3.0 | 0.550 | 0.60 |

  Recent exam: GR4J 0.517, NWM 0.687. NWM was calibrated on many of these gauges, so it is not a clean ungauged reference.
- **Flood peaks** (abs err % / bias %): own-record median 42 / −14; GR4J × 1.38 47 / −31; NWM hourly recent 31 / −2.
- **Literature context only (Daymet forcing, different setup):** EA-LSTM NSE ~0.74 gauged, ~0.69 ungauged. Our forcing is ERA5-Land, which is biased low in mountains and the Pacific Northwest, so don't claim comparability.
- **SNOTEL:**

  | Baseline | NSE | RMSE | Melt-out error |
  |---|---|---|---|
  | Climatology | 0.70 | | 11.4 d |
  | Degree-day calibrated per station on its own SWE | 0.925 | 41 mm | 6.1 d |
  | Uncalibrated degree-day (A, dev) | ~0.70 | | |
  | Uncalibrated energy balance (A, dev) | ~0.82 | | |

  The per-station calibrated model uses the target station's own SWE. A sim-trained network that never sees it is the "ungauged" protocol, so report both.
- **Detective (Spearman rho on validation-period signatures):**

  | Signature | kNN on attributes | Budyko-Fu | GR4J from calibration Q | Same signature from own calibration Q |
  |---|---|---|---|---|
  | Runoff ratio | 0.78 | 0.74 | 0.97 | ~0.98 |
  | BFI | 0.55 | | 0.79 | ~0.98 |
  | Recession K | 0.52 | | 0.46 | |

  Pick targets where the classical answer is weak or unavailable. Don't just re-predict a signature the basin's own record already gives at 0.98.
- **Reservoirs (cost J median, lower is better):** historical 0.183, SOP 0.108, rule curve 0.143, perfect-foresight DP 0.006. On the CDEC recent exam: SOP 0.095, DP 0.012.

## The four projects (one agent each; folder `/home/claude/water/<name>/`)
1. **flowcast: FlowCast.** Daily streamflow for **ungauged** basins from forcing plus static attributes, with calibrated uncertainty (e.g. quantiles or Gaussian/log-space sigma).
   - Train on large simulated banks: all four models, both snow routines, `prior='camels'` plus `'default'`, observation and forcing noise. An LSTM or TCN over ~1 year of forcing with static inputs is the obvious architecture; keep it small enough for 2 shared cores.
   - Exam: 619 basins, main and recent windows: NSE, KGE, FHV, peak error and interval coverage, compared with U-dev regionalisation, NWM and gauged GR4J.
   - Optional second mode: a gauged/nowcast mode that assimilates recent observed Q (lead 1–7 days), compared with persistence and NWM.
   - Ratchet rungs are physics/simulator improvements found by twin tests on dev basins (e.g. ERA5 precipitation undercatch in mountains, flashiness, intermittent streams, the contributing-area mismatch).
2. **snowsense: SnowSense.** Daily SWE at SNOTEL stations from weather alone (station P/T and/or ERA5-Land basin forcing; state which and keep it consistent). Also peak SWE, peak date and melt-out date, with uncertainty.
   - Train on the snow-only simulator (`model='none'`), both snow routines, randomised parameters, gauge undercatch and forcing noise.
   - Exam: 772 stations (621 scored WY2011–2025), compared with climatology, uncalibrated degree-day and energy balance, and the per-station-calibrated degree-day as a gauged ceiling.
   - Optional: does the SWE network improve FlowCast-style streamflow in snow basins? Only if time allows.
3. **detective: BasinDetective.** From a basin's calibration-period forcing plus Q, infer hidden properties with error bars.
   - Possible targets:
     - storage/recession constants
     - groundwater share
     - soil water capacity
     - snow fraction and melt factor
     - human influence (regulated? from B's dam counts and reservoir storage fraction)
     - the basin's *out-of-period* response, e.g. validation-period signatures under different climate
   - Trained on simulated basins with known labels, it must recover parameters on sim twins and be graded on real labels: attributes, regulation flags, validation-period signatures.
   - Choose targets where real ground truth exists and the classical method is weak. Explain the choice in the model card, compare with kNN, Budyko, recession analysis, filter BFI and GR4J calibration, and report Spearman rho, MAE and coverage.
4. **reservoirpilot: ReservoirPilot.** A learned release policy that sees only causal information: storage, current and recent inflow, day of year, capacity, demand. No future inflows.
   - Train in the simulator on generated reservoirs and inflows (simulated or real free-window inflows to dev dams; do NOT replay exam dams' inflows for training).
   - Graded with B's cost J on the 244 exam dams (median) and the CDEC recent exam, compared with SOP, rule curve, historical and DP.
   - The objective is fixed by B's J; report every component (shortage ratio, days above Qmax, storage behaviour) so a "win" isn't a loophole.
   - Methods: behaviour cloning of DP on simulated reservoirs, RL (PPO or similar), or MPC with a learned inflow forecaster. Explain the choice.
   - Watch the data-quality gotchas: mass-balance closure, start-of-day vs end-of-day stamps, negative back-calculated inflows.

## Shared code rules
- Do NOT edit `/home/claude/water/watersim/`, `/home/claude/water/waterdata/` or `/home/claude/water/data/`. Physics changes and calibrations go in YOUR folder as wrappers, versioned in your ratchet, for the coordinator to upstream later.
- numba cache: if you wrap watersim, keep your numba cache inside your folder (`NUMBA_CACHE_DIR=/home/claude/water/<name>/.numba`) so agents don't collide.
- Your folder: `/home/claude/water/<name>/`. Keep a `STATUS.md` there. Put screenshots and temp files in a subfolder of your own folder; never use generic names in the shared scratchpad.
- Disk: ≤ 1.5 GB each (~13 GB free in total).
- Never `pkill -f`; only kill your own PIDs.
- Never disable TLS verification or unset HTTPS_PROXY. If a host is blocked, don't retry or circumvent it.
- B's CAMELS-AUS download may still be running (PID 16790 plus a watcher). Leave it alone.

## Ratchet (as in AGENT_BRIEF)
- A frozen simulated validation set, with a +0.01 margin on your primary sim metric.
- A small real dev gate: no more than 5% worse, or a stated threshold.
- Version the frozen set whenever the physics changes and re-score every rung on it.
- Widen-only sigma calibration on dev.
- Twin tests: simulate dev units with their calibrated/attribute twins and compare the network on twin vs real to find the biggest sim-to-real gap.
- Log every rung, including rejected ones.

## Deliverables (each agent)
- Release zip `/mnt/user-data/outputs/<name>-v1.0.zip`. Prefer a standalone numpy model; vendor small feature code. Include a `demo.py` that runs without `python3 -I` problems, a model card and the scripts.
- A showpiece Artifact page titled with your project name. Load the `artifact-design` skill first, publish with the Artifact tool (icon on the first publish only), and put any interactive inference in-browser if feasible. External hosts are blocked except Google Fonts, so inline all JS/CSS and data.
- The status doc as a FILE: `/home/claude/water/<name>/<name>-status.md`, in the same style as `/home/claude/quake/docs/quakenet-status.md`. The coordinator uploads it to the project; don't write to the project yourself.
- A compact final report covering:
  - the rung table with sim and exam numbers against baselines
  - the artifact URL and zip path
  - upstream candidates (physics fixes for watersim)
  - rule notes (anything you're unsure was allowed)
  - the best next rungs, and what a full version would still need (for HANDOFF.md)
