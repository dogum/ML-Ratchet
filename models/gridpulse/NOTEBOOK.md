> **Lab notebook.** GridPulse was built in a claude.ai cloud session (14 Nov 2026) on the shared GridSim + GridData
> platform, now in `shared/grid/`. The first entry is that session's status document as written; its paths
> (`src/`, `grid/gridpulse/`) refer to the cloud sandbox. `MODEL_CARD.md` holds every number.

## 14 Nov 2026 — build notes (gridpulse-status.md, as written)

Phase 2 GridSim project in ML Ratchet. From the 1 s grid frequency after a trip: MW lost, inertia, MW/Hz response (NERC point B), headroom/FFR (weak), and a nadir and 60 s forecast from the first N seconds, each with a 1σ band. GB (50 Hz) and ERCOT (60 Hz) as conditioned inputs, plus optional transmission demand.

- Page: the original lab, now `lab/` (in-browser GridSim port + live network, explorer of all 412 GB exam events + 4 dev events, exam, ratchet)
- Release: gridpulse-v1.0.zip (numpy/scipy; demo runs in <1 s from the unzipped copy)
- Working folder: the sandbox workspace grid/gridpulse/ (STATUS.md, ratchet.json, exam.json, widen.json, checks/)

## Recipe as built
- Physics: GridSim frequency_event (Phase 1 checks). Wrapper `src/gpsim.py` rescales generate_states fleets to real GB/ERCOT operating points (GB joint demand/inertia/IBR from NESO half-hours of 2018/2020/2022), adds GB Dynamic Containment/Moderation/Regulation from NESO's service curve, ERCOT load resources at 59.7 Hz, FFR at 59.85 Hz, battery droop. Check: initial RoCoF = ΔP·f0/2E within 0.1 %. JS port == GridSim to 1e-14 Hz; JS net == numpy to 9e-8; numpy == torch to 1.6e-6.
- Noise: 56k quiet 140 s NESO 1 s slices (2018-07, 2020-07, 2022-01; ≥1 h from events). Same onset rule and detection cut as the real catalogue.
- Dev calibration (4 GB 2019 events, 5 ERCOT): prior-predictive twins. p0 generic services → real GB nadirs at 2-6th pct (sims too shallow) → p1 (2019-era response + DC-era mixture) 13-20th pct. ERCOT twins 8-29 % too shallow at Phase 1's D = 2.5 %/% → D median 1.0. GB_20191203 loss label 1,788 MW replaced by metered 854 MW on dev.
- Network: MLP 512-512-256 Gaussian heads, 0.44 M params; 28,201 detected events from 48,000 simulated states.

## Rungs (frozen set p2s, 1,387 sims; keep if +0.01 and dev ≤ +10 %)
| rung | change | score | decision |
|---|---|---|---|
| r1 | base MLP, 1 s-mean measurement | 0.476 | kept |
| r2 | instantaneous 1 s samples | 0.478 | rejected (+0.002) |
| r3 | + demand input | 0.524 | kept |
| r4 | 2× data | 0.542 | kept |
| r5 | wider net | 0.555 | kept = release |
| r6 | p2 physics: 30 % ramped losses | 0.589 | rejected by dev gate (+14 %, inertia 15→20 %) |

## Exam (never used for choosing)
- GB inertia vs NESO outturn (410): 16.4 % MdAPE (14.5 % at N=10) vs constant 230 GVA·s 20.4 %, but a dev-fitted 7.0 s × demand rule gets 15.1 %; trace-only 20.3 %. Overestimates 2023-25 by ~30 % (20-31 % error those years). Coverage 63/89 %.
- GB MW lost: medium labels (86) 25.9 % vs RoCoF formula 27.1 % (assumed E) / 27.6 % (NESO E), dumb 32.7 %; low (64) 34.8 % vs 35.6/29.3 %. Coverage 48/79 % (too narrow).
- Forecasts: nadir at 5 s 54 vs 58 mHz (min-so-far) — small win; at 10 s 37 vs 26 mHz — loses; 60 s from 10 s 59 vs 39 mHz persistence — loses.
- ERCOT (292, reports only): MW lost from Δf+inertia+load 15.7 % vs ERCOT MW/Hz constant 20.1 % (2023-25: 24-33 % vs 37-59 %); forward Δf forecast 29.5 % (bias +29 %, GridSim too deep in exam years) vs 23.1 %; inertia 11.7 % vs 11.2 % (s × load).

## Limits
1. Inertia skill comes from demand, not the trace (1 s data blur RoCoF; response arrives in 1-2 s).
2. Learned 2018-22 inertia per MW; 2023-25 GB and ERCOT are stiffer/lower-inertia than the sim.
3. Real MW-lost error bars too narrow; 4 dev events can't widen properly.
4. 30-60 s behaviour (secondary, operator actions) not modelled well.

## Upstream candidates
- `gridsim.frequency._as_events`: a plain list of per-state loss lists is read as one event list (pass object array).
- GB Dynamic Containment knee as a two-segment droop; ERCOT effective D ≈ 1.0 %/% for generate_states-like states (2.5 for dispatch-derived states); per-tech aggregation in generate_states for speed (fleets of ~1,300 units pad batches).

## Next rungs
1. Era-aware priors: condition on year (or DC volume / battery fleet) so 2023-25 inertia per MW and ERCOT FFR are represented; train-input 2022 inertia mix weighted up.
2. Use sub-second data where available (NESO publishes 1 s only; PMU-grade data would carry inertia in the first 500 ms) or add the ambient-noise spectrum as an inertia cue with sims that drive noise through the swing equation.
3. A better 30-60 s model (secondary response as droop + delayed reserve) before trusting the 60 s forecast; more GB dev events from 2021 if the splits are ever revisited.

## Sandbox gotchas
- `setsid nohup sh -c "a && b"` survived; a bash script launched the same way died silently once.
- generate_states fleets up to 1,300 units pad frequency_event batches → aggregate per tech (40× faster).
- 4 parallel builds on 2 cores: banks of 24k states took 15-30 min.

## 14 Nov 2026 — brought into the repository

- **Layout.** The release zip became `models/gridpulse/`: `gridpulse.py`, both weight files (`gridpulse_weights.npz`,
  the main network; `ercot_head.npz`, GridPulse-Summary), `pipeline.py`, `demo.py`, `examples/`, `MODEL_CARD.md`,
  every training script in `training/`, the raw ratchet log as `training/ratchet.json`, and the release's `checks/`
  JSONs in `training/checks/`. The original page is in `lab/` unchanged (index.html, gp.js, model.json, events.json,
  noise.json, results.json). `model.yaml` and `ratchet.yaml` were written from the status document, the card, the raw
  ratchet log and `checks/exam.json`.
- **Changes to code.** `training/common.py` now takes its paths from `training/_paths.py` (WORK = `work/`, GridSim and
  GridData from `shared/grid/`); `check_js.py` and `export.py` load `gp.js` from `lab/`; the three shell queues `cd`
  to their own folder and log to `../work/logs/`. `pipeline.py` runs its command line under `if __name__ ==
  '__main__'` so the module imports cleanly; its output is unchanged (same bytes on `examples/trace_example.csv`).
  `demo.py`'s docstring promised "three ERCOT report rows"; it prints one illustrative summary, and now says so.
- **Demo.** The repository demo prints exactly the same text as the release demo run from the unzipped zip (diff
  empty). Golden: `tests/golden/gridpulse.txt`.
- **Exam re-scored from the shipped weights.** `exam.py` loads torch checkpoints that were not shipped, so
  `training/rescore_release.py` points its `predict` at `gridpulse_weights.npz` (numpy forward, stored widening set
  to 1 because `exam.py` applies `widen.json` itself) and its ERCOT head at `ercot_head.npz`; the rest is `exam.py`
  on `shared/grid/data` (6 s). Every headline number reproduced to the printed precision:

  | metric | released | re-scored |
  |---|---|---|
  | GB inertia MdAPE, N = 60 (410) | 16.44 % | 16.44 % |
  | GB inertia MdAPE, N = 10 | 14.54 % | 14.54 % |
  | GB inertia coverage 1σ / 2σ | 63.4 / 88.8 % | 63.4 / 88.8 % |
  | constant 230 GVA s / 7.0 s × demand / trace only | 20.42 / 15.07 / 20.35 % | 20.42 / 15.07 / 20.35 % |
  | GB MW lost, medium labels (86) | 25.91 % | 25.91 % |
  | RoCoF formula at 230 / at NESO inertia | 27.14 / 27.61 % | 27.14 / 27.61 % |
  | GB MW lost, low labels (64) | 34.78 % | 34.78 % |
  | nadir from 5 s / 10 s (nadir later), MAE | 54.06 / 36.65 mHz | 54.06 / 36.65 mHz |
  | frequency at 60 s from 10 s, MAE | 58.57 mHz | 58.57 mHz |
  | ERCOT MW lost (292) / dev MW/Hz × drop | 15.73 / 20.11 % | 15.73 / 20.11 % |
  | ERCOT drop forecast / ERCOT inertia | 29.49 / 11.67 % | 29.49 / 11.67 % |

  The largest difference anywhere in the table is 1e-4 percentage points (float32 checkpoint vs float32 npz).
- **Verdicts in model.yaml.** A row is a win only if it beats every baseline on the row. GB inertia at N = 60 s is a
  loss (16.4 % vs 15.1 % for the dev-fitted 7.0 s × demand rule); at N = 10 s it is a win (14.5 % vs 15.1 %). Both
  are in the exam file; the N = 60 s row is the release headline. ERCOT inertia (11.7 % vs 11.2 %) and the ERCOT drop
  forecast are losses. One detail the card did not state: on the 86 medium labels the trace-only reading (no demand)
  gets 24.8 %, slightly better than with demand (25.9 %).
- **Live instrument.** `web/export_instrument.py` exports the network (float16) and real GB exam windows for the site;
  the browser port runs the same forward pass (see `site/src/instruments/gridpulse/`).

## 14 Nov 2026 — dev label of 3 Dec 2019 corrected in GridData (HANDOFF P0.7)
- **Where 1,788 MW came from.** GridData's v1.0 label for GB_20191203T200030 was 854.7 MW (Spalding CCGT T_SPLN-1,
  REMIT unplanned "Plant Fault" 870 -> 0 MW from 20:00Z; Elexon B1610 427.35 MWh in SP40 = 854.7 MW) plus 933 MW
  "interconnector import lost (INTFR)" from FUELINST. FUELINST shows IFA ramping 1,900 -> 204 MW over 19:45-20:10 UTC,
  an hour-boundary ramp like those at 17:00, 19:00 and 22:00 that evening, not a trip. The card's earlier wording
  ("REMIT capacity") was wrong about the source; the 854 MW used on dev was right.
- GridData now applies a cited correction (`shared/grid/data/sources/gb_event_label_corrections.csv`):
  `gd.events` gives 854.7 MW, with `loss_MW_v1` = 1,787.7.
- **Dev work:** twins, the dev gate and widening used `evaluate.DEV_LOSS` / `twin_gb.LOSSES` (854 MW) and are unchanged.
- **Exam re-scored** with `training/rescore_release.py` (release weights, output to a scratch work folder, not
  `training/checks/exam.json`): every GridPulse number is identical to the release. Only the dumb "dev median loss"
  baseline moves, because the dev median falls from 1,142 to 1,066 MW: medium labels 32.7 % -> **26.8 %**, low labels
  112 % -> **97.8 %**. Verdicts unchanged (medium: 25.9 % still beats 26.8 %, 27.1 %, 27.6 %; low: still a loss to
  the RoCoF formula at NESO inertia, 29.3 %). model.yaml and MODEL_CARD.md keep the released baseline and add the
  corrected one, dated.

## 14 Nov 2026 — the GridSim outage-rate leak (HANDOFF P0.5): GridPulse is not exposed

GridSim v1.0's default forced and planned outage shares for coal, combined cycles, gas turbines and gas steam came from
Texas RE's 2024/2025 assessments (data from the exam years). GridSim physics 1.1 replaces them with pre-2023 sources
(`shared/grid/gridsim/gridsim/fleet.py`, `OUTAGE_DEFAULTS`). GridPulse never read them:

- Its training states come from `gridsim.generate_states` through `training/gpsim.py`. `generate_fleet` and
  `generate_states` take only pmin, ramp, minimum up/down and start heat from the technology defaults, never the outage
  shares, and GridPulse runs no dispatch.
- The one GridSim number that was fitted with the leaked availability is ERCOT's effective damping (2.5 %/% in v1.0,
  fitted on the 5 ERCOT dev events with states from the calibrated v1.0 dispatch; 2.75 %/% when re-fitted on v1.1).
  GridPulse's p0 prior started from it, but every rung (r1-r6) used p1 or p2, whose ERCOT damping (median 1.0 %/%) was
  re-fitted on GridPulse's own dev twins of `generate_states` states, which do not depend on it.
- The ERCOT summary head is trained on the same states.

Nothing is retrained and no number changes.
