# GridPulse v1.0 — model card

Part of ML Ratchet (github.com/dogum/ML-Ratchet, "Trained on physics. Graded on reality."), Phase 2 of the GridSim work.

## What it does
After a sudden loss of generation, the whole grid's frequency falls. From the 1 s frequency record (20 s before to up
to 60 s after the trip), GridPulse estimates, each with a 1-sigma band:
- **MW lost**
- **system inertia** (GVA·s = GW·s of kinetic energy in spinning machines)
- **frequency response** in MW/Hz (MW lost / mean drop 20-52 s after the trip, NERC "point B")
- governor **headroom** and triggered **FFR** (weakly identifiable; no real truth to grade them)
- a **forecast of the nadir** (lowest frequency in 0-60 s) and of the **frequency at 60 s** (mean 55-65 s), from the
  first N seconds (N = 2-60). This is the early-warning use.

GB (50 Hz) and ERCOT (60 Hz) are a conditioned input, and so is transmission demand (optional; NaN = unknown).
A separate small model, **GridPulse-Summary**, reads ERCOT operations-report numbers (delta f, inertia, load,
renewable share), because ERCOT publishes no frequency traces.

## Inputs and outputs
`GridPulse().read(t, f, system='GB', demand_MW=..., N=10)` with t, f at 1 s and >= 60 s before / 60 s after the trip.
Onset = griddata's rule (steepest 1 s fall, walked back while falling >= 10 mHz/s). The window is 81 samples of
deviation from the 60 s pre-event mean. Outputs in `read()`'s dict: `loss_MW(_lo,_hi)`, `E_GWs(_lo,_hi)`,
`beta_MW_Hz(_lo,_hi)`, `headroom_MW`, `ffr_MW`, `nadir_Hz`, `nadir_sigma_Hz`, `settle_Hz`, `settle_sigma_Hz`.

## Physics and simulator
- **GridSim** (Phase 1): aggregated swing equation 2E/f0·dΔf/dt = ΣΔPm + P_FFR + P_shed − P_lost − D·P_load·Δf/f0
  with per-technology governors (deadband, lag, rate limit, headroom clip, reheat / CCGT / hydro lead-lags), FFR,
  load damping, UFLS, secondary ramp. Phase 1 checks: RoCoF = ΔP·f0/2E to 4e-14, closed-form nadir to 0.03 mHz,
  ANDES Kundur two-area nadir within 24 mHz (`shared/grid/checks/`).
- **GridPulse wrapper** (`training/gpsim.py`, version p1): every `generate_states` fleet is rescaled to a real
  operating point and given that grid's services.
  - GB: demand, outturn inertia and wind/solar share drawn jointly from NESO half-hours of the training-input years
    2018/2020/2022 (70 %) or wide ranges (30 %: 16-46 GW, 90-420 GVA·s); losses log-uniform 150-2,000 MW; governor
    headroom 300-1,500 MW; in 60 % of states Dynamic Containment from NESO's service curve (5 % by 0.2 Hz, 100 % at
    0.5 Hz, deadband 0.015 Hz), Moderation (0-500 MW) and Regulation (0-600 MW); static FFR at 49.7 Hz; damping
    median 0.75 %/% (Phase 1 fit); secondary 0-30 % of the loss over 30-90 s.
  - ERCOT: load from EIA-930 train-input years, inertia ≈ 5 s × load (log-sd 0.22); losses 250-2,800 MW; governor
    headroom 1,000-3,500 MW; load resources 500-1,500 MW at 59.7 Hz; FFR 0-750 MW at 59.85 Hz; battery 5 % droop in
    half the states; BAL-001-TRE deadbands; damping median 1.0 %/% (dev-twin refit).
  - Checks (`training/checks/physics.json`): initial RoCoF of rescaled states = ΔP·f0/2E within 0.1 % (100 GB, 100 ERCOT);
    the DC knee reproduces NESO's curve at 0.1/0.2/0.3/0.5 Hz.
- Units are aggregated per technology with GridSim's `aggregate_units` (exact for the linear parts).
- Upstream note: `gridsim.frequency._as_events` treats a plain list of per-state loss lists as one event list; pass an
  object array (see `gpsim.run`).

## Noise and measurement
Simulated deviation (0.1 s) → 1 s samples with a random sub-second phase (1 s means in training; a dev check slightly
favoured instantaneous samples, see rung r2) → added to a real quiet 140 s slice of NESO 1 s frequency (56,090 slices
from 2018-07, 2020-07, 2022-01; >= 1 h from any detected event; no 10 s change beyond 0.10 Hz; training uses all but
2022-01 days 21+, which are held for the frozen set) → griddata's onset rule → window. ERCOT noise = GB noise × 0.8-1.4
(no ERCOT 1 s data exist). Catalogue selection as for real events: GB 10 s fall <= −0.15 Hz; ERCOT drop >= 60 mHz or
loss >= 350 MW. 48,000 simulated states → 28,201 detected training events.

## Network
MLP 86 → 512 → 512 → 256 → 2×7 (Gaussian mean and log-sigma per target), GELU, 0.44 M parameters, Gaussian NLL,
AdamW + one-cycle, 120 epochs, re-augmented (phase, noise slice, N, 25 % demand dropped) every epoch. numpy == torch to
1.6e-6, JS == numpy to 9e-8 (`training/checks/js_parity.json`). GridPulse-Summary: three 4→64→64→2 MLPs on the ERCOT sims.

## Dev calibration (dev only: 4 GB events of 2019, 5 ERCOT events of 2019/2021)
- GB prior-predictive twins (each event's NESO inertia, demand, loss; rest drawn): with generic services the real
  nadirs sat at the 2-6th percentile (sims too shallow); 2019-era response put them at 13-20th.
- ERCOT twins were 8-29 % too shallow at the Phase 1 damping 2.5 %/% with these random states; 1.0 %/%.
- GB_20191203T200030: griddata's 1,788 MW (REMIT capacity) is ~2× what its 0.29 Hz drop at 333 GVA·s implies; dev truth
  set to Elexon metered output 854 MW.
- Twin read test (`training/checks/twin_read_r5.json`): percentile of the real reading among 80 twin readings — MW lost
  5-99 % (median ~70 %), inertia 0-93 %, MW/Hz 7-38 % (real grids respond less stiffly than the typical sim).
- Sigma widening (widen-only, 68th percentile of |z| on dev): loss ×1.11, settling ×1.28, others ×1.0.

## Ratchet (frozen set = 1,387 simulated events of physics p2, seed 98; objective = mean skill of MW lost, inertia,
MW/Hz at 60 s and nadir, settling at 10 s; keep if +0.01 and dev errors no more than 10 % worse)

| rung | change | score | MW lost | inertia | nadir@10 | dev | decision |
|---|---|---|---|---|---|---|---|
| r1 | MLP on 1 s window, p1 physics, 1 s-mean measurement | 0.476 | 0.48 | 0.17 | 0.89 | – | kept |
| r2 | instantaneous 1 s samples | 0.478 | 0.47 | 0.18 | 0.88 | +10 % | rejected (+0.002) |
| r3 | + transmission demand input | 0.524 | 0.55 | 0.30 | 0.89 | +7 % | kept |
| r4 | 2× data (48k states) | 0.542 | 0.53 | 0.32 | 0.91 | −11 % | kept |
| r5 | wider net 512-512-256, 120 epochs | 0.555 | 0.52 | 0.34 | 0.92 | +3 % | **kept = release** |
| r6 | physics p2: 30 % of trips ramp out over 0.5-3 s | 0.589 | 0.61 | 0.36 | 0.92 | +14 % | rejected (dev gate: inertia MdAPE 15→20 %) |

## Final exam (real, never used for choosing; `training/checks/exam.json`)
**GB, 412 exam events (NESO 1 s, 2018-2025)**

| test | n | GridPulse | baselines | 1σ / 2σ coverage |
|---|---|---|---|---|
| inertia vs NESO outturn, N=60 | 410 | 16.4 % MdAPE (N=10: 14.5 %) | constant 230 GVA·s 20.4 %; 7.0 s × demand (dev fit) **15.1 %**; trace only (no demand) 20.3 % | 63 / 89 % |
| inertia, events with loss label | 148 | 15.3 % | RoCoF formula with label loss 37.4 % | |
| MW lost, medium labels, N=60 | 86 | **25.9 %** | RoCoF @230 GVA·s 27.1 %; RoCoF @NESO inertia 27.6 %; dev median 32.7 % | 48 / 79 % |
| MW lost, low labels, N=60 | 64 | 34.8 % | RoCoF @230 35.6 %; RoCoF @NESO **29.3 %**; dev median 112 % | 41 / 66 % |
| nadir forecast at 5 s (nadir later) | 353 | **54 mHz** MAE | lowest value so far 58 mHz | 72 / 92 % |
| nadir forecast at 10 s (nadir later) | 280 | 37 mHz | lowest value so far **26 mHz** | 59 / 81 % |
| frequency at 60 s from 10 s | 412 | 59 mHz | value at 10 s held **39 mHz** | 61 / 95 % |

Inertia by year (NESO median → GridPulse median, MdAPE): 2018 266→257 6.9 %; 2019 238→249 10.4 %; 2020 205→244 15.0 %;
2021 240→254 12.9 %; 2022 243→258 11.2 %; **2023 189→242 28.2 %; 2024 180→244 30.6 %; 2025 160→216 19.6 %**.
Inertia error falls with drop size: 20.9 % (<0.2 Hz, 161), 13.8 % (0.2-0.3), 10.4 % (0.3-0.45), 6.7 % (>0.45, 6).

**ERCOT, 292 exam generation-loss events (operations reports; no traces).** These test GridSim's ERCOT physics through
GridPulse-Summary, not the trace network:

| test | GridPulse-Summary | baseline | 1σ coverage |
|---|---|---|---|
| MW lost from reported Δf, inertia, load, IBR share | **15.7 %** | ERCOT dev MW/Hz (4,668) × Δf 20.1 %; dev median 24.5 % | 71 % |
| Δf forecast from MW lost, inertia, load (forward simulator test) | 29.5 % (bias +29 %, too deep) | MW lost ÷ 4,668 **23.1 %** | 70 % |
| inertia from Δf, MW lost, load | 11.7 % | dev median 19.9 %; dev s × load **11.2 %** | 85 % |

MW-lost MdAPE by year: 9-14 % in 2018-2022 vs 12-19 % for the constant; 24-33 % in 2023-2025 vs 37-59 %.
ERCOT reports no RoCoF, so "inertia from RoCoF" cannot be tested.

## Where it fails
1. GB inertia comes mostly from the demand input; the trace alone matches the constant guess. 1 s sampling blurs the
   first-second slope and response arrives within 1-2 s.
2. It learned 2018-2022 GB inertia per MW of demand and overestimates 2023-2025 by ~30 %.
3. Error bars on real MW lost are too narrow (48 % inside 1σ); four dev events cannot widen them reliably.
4. By 10 s the nadir forecast loses to "lowest value so far", and the 60 s forecast loses to persistence: real 30-60 s
   behaviour (secondary response, operator actions, further events) is outside GridSim's single secondary ramp.
5. GridSim's ERCOT drop is ~29 % too deep on exam years (2023-2025 battery FFR/ECRS); ERCOT noise is scaled GB noise.
6. MW/Hz, headroom, FFR: sim skill 0.5-0.7, no real truth.
7. Some detected GB "events" are demand/interconnector ramps, not trips. Loss labels are noisy.

## Data and licences
NESO system frequency 1 s, outturn inertia, historic generation mix (NESO Open Data Licence); Elexon BMRS REMIT/B1610
(loss labels by GridData); ERCOT monthly Operations Reports (public); EIA-930 (public domain); National Grid ESO
technical report on 9 Aug 2019. Splits frozen by GridData (`shared/grid/data/proc/splits.json`, sha256 895b4e9a…, seed 20261113).
Outage-rate defaults were not used.

## Files and reproduction (added in the repository, 14 Nov 2026)
- `gridpulse.py` (numpy + scipy inference and the onset rule), `gridpulse_weights.npz` (main network, dev widening
  factors stored in `widen`), `ercot_head.npz` (GridPulse-Summary), `pipeline.py` (command line on a CSV trace),
  `demo.py` (six GB exam events, about 1 s), `examples/` (`gb_exam_examples.npz`: the six windows, demand, NESO
  inertia, labels, measured nadir and settling; `trace_example.csv`: a 1 s trace for `pipeline.py`).
- `training/`: every script as run (paths through `training/_paths.py`: outputs in `work/`, the platform in
  `shared/grid/`), the raw ratchet log `training/ratchet.json`, and the release checks in `training/checks/`.
- The demo prints the same numbers in the repository as from the release. `training/rescore_release.py` re-runs
  `exam.py` on the shipped weights (the torch checkpoints were not shipped) and reproduces every headline number above
  to the printed precision.
- Live instrument: `web/export_instrument.py` writes the in-browser version (site/public/instruments/gridpulse/).
