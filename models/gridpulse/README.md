<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# GridPulse <sub>v1.0</sub>

> **How big was the trip, and how stiff is the grid, from one second of frequency at a time?**

*After a power station trips, how many MW were lost, how much inertia is spinning, and how low will the frequency go?*

GridPulse reads the 1 s grid frequency from 20 s before to up to 60 s after a sudden loss of generation. It returns the MW lost, the system inertia (GVA s), the frequency response (MW/Hz), and a forecast of the nadir and the frequency at 60 s from the first N seconds, each with a 1-sigma band. GB (50 Hz) and ERCOT (60 Hz) are a conditioned input, and so is transmission demand (optional). It learned only from 28,201 GridSim events laid into real quiet NESO frequency noise. It was examined on 412 real GB events (NESO 1 s data, 2018-2025) and, through a small summary head, on 292 ERCOT events from ERCOT's operations reports.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/gridpulse/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/gridpulse/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | 1 s grid frequency (Hz) from >= 60 s before to 60 s after a trip; the system (GB 50 Hz or ERCOT 60 Hz); optionally the transmission demand (MW); N, the seconds after the trip to use (2-60). GridPulse-Summary reads ERCOT report numbers: frequency drop (Hz), inertia (MW s), load (MW), renewable share. |
| **Returns** | MW lost, with a 1-sigma band (dev-widened ×1.11)<br>System inertia (GVA s) with a 1-sigma band<br>Frequency response (MW/Hz, NERC point B), governor headroom and triggered FFR (weak; no real truth)<br>Forecast of the nadir (lowest frequency in 0-60 s) and of the frequency at 60 s, from the first N seconds, each with a 1-sigma band (settling ×1.28)<br>GridPulse-Summary (ERCOT reports): MW lost from the drop, inertia and load; the drop from the MW lost; inertia from the drop and loss |
| **Learned from** | The shared GridSim frequency model (shared/grid/gridsim, Phase 1): an aggregated swing equation 2E/f0 · dΔf/dt = ΣΔPm + P_FFR + P_shed − P_lost − D·P_load·Δf/f0 with per-technology governors (deadband, lag, rate limit, headroom clip, reheat / CCGT / hydro lead-lags), fast frequency response, load damping, under-frequency load shedding and a secondary ramp. A wrapper (training/gpsim.py, version p1) rescales GridSim's generated fleets to real GB and ERCOT operating points and adds each grid's services. |
| **Noise** | Real quiet NESO 1 s frequency: 56,090 slices of 140 s from 2018-07, 2020-07 and 2022-01, at least 1 h from any detected event (2022-01 days 21+ held for the frozen set). The simulated deviation is turned into 1 s means with a random sub-second phase and added to a slice; then griddata's onset rule picks the window, as on the real events. ERCOT noise = GB noise × 0.8-1.4 (no ERCOT 1 s data exist). |
| **Network** | MLP 86 → 512 → 512 → 256 → 2×7 (Gaussian mean and log-sigma per target), GELU, 0.44 M parameters, Gaussian NLL, AdamW one-cycle, 120 epochs, re-augmented every epoch (phase, noise slice, N, demand dropped 25 % of the time); trained on CPU. GridPulse-Summary: three 4 → 64 → 64 → 2 MLPs on the ERCOT simulations. |
| **Physics also enters** | only through the simulated training data |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | GridData's frozen event split (shared/grid/data/proc/splits.json, sha256 895b4e9a…, seed 20261113): GB 9 Aug 2019 plus the three largest medium-confidence losses of 2019 (1 Jul, 11 Jul, 3 Dec); 5 seeded ERCOT events from 2019/2021 |
| **Exam** | Every other event before 2026: 412 GB events 2018-2025 (410 with NESO inertia, 150 with a loss label) and the 292 ERCOT generation-loss events with drop, inertia and load reported; 2026 events are held in reserve |

## Run it

```bash
cd models/gridpulse
pip install numpy scipy
python demo.py        # about 1 s
```

or from anywhere in the repo: `mlratchet demo gridpulse`.

```python
import numpy as np
from gridpulse import GridPulse
gp = GridPulse()
d = np.loadtxt("examples/trace_example.csv", delimiter=",", skiprows=1)   # t [s], f [Hz] at 1 s
r = gp.read(d[:, 0], d[:, 1], system="GB", demand_MW=30000, N=10)
print(r["loss_MW"], r["E_GWs"], r["nadir_Hz"])
print(gp.ercot_summary(delta_f_Hz=0.11, inertia_MWs=230e3, load_MW=45000, ibr_share=0.3, mw_loss=600))
```

## The exam

Real GB events from NESO 1 s frequency (2018-2025) against NESO's outturn inertia and GridData's loss labels, and ERCOT operations-report events through GridPulse-Summary. Median absolute % error (MAE in mHz for the forecasts). Run once, after the release choice was frozen; re-scored from the shipped weights in the repository with the same numbers. Every number here is on real data that never chose anything.

| Metric | GridPulse | Baselines | |
|---|---|---|---|
| GB inertia vs NESO outturn, N = 60 s (410 events)<br><sub>Bias +11 %. The trace alone (no demand input) gets 20.3 %: the skill comes from demand. 2023-25 overestimated by ~30 % (MdAPE 20-31 %).</sub> | **16.4%** | 7.0 s × demand (fit on the 4 dev events) (classical): 15.1%<br>Constant 230 GVA s (dumb baseline): 20.4% | ❌ loses |
| GB inertia vs NESO outturn, N = 10 s (410 events)<br><sub>Reading the first 10 s only; trace alone 18.9 %.</sub> | **14.5%** | 7.0 s × demand (fit on the 4 dev events) (classical): 15.1%<br>Constant 230 GVA s (dumb baseline): 20.4% | ✅ wins |
| GB inertia, events with a loss label (148) | **15.3%** | RoCoF formula E = ΔP·f0 / 2 RoCoF with the label loss (classical): 37.4% | ✅ wins |
| GB MW lost, medium-confidence labels (86)<br><sub>Bias +11 %. Without the demand input: 24.8 %. 14 Nov 2026: GridData corrected dev event 3 Dec 2019 from 1,788 to 854.7 MW; the dev median loss falls from 1,142 to 1,066 MW and this dumb baseline from 32.7 % (as released) to 26.8 %. GridPulse's numbers do not change and it still wins.</sub> | **25.9%** | RoCoF formula at 230 GVA s (classical): 27.1%<br>RoCoF formula at the NESO outturn inertia (classical): 27.6%<br>Dev median loss (dumb baseline): 32.7%<br>Dev median loss, corrected 3 Dec 2019 dev label (14 Nov 2026) (dumb baseline): 26.8% | ✅ wins |
| GB MW lost, low-confidence labels (64)<br><sub>14 Nov 2026: with the corrected 3 Dec 2019 dev label (854.7 MW) the dev median baseline is 97.8 % (112 % as released). The verdict is unchanged: the RoCoF formula at NESO inertia (29.3 %) still beats GridPulse.</sub> | **34.8%** | RoCoF formula at 230 GVA s (classical): 35.6%<br>RoCoF formula at the NESO outturn inertia (classical): 29.3%<br>Dev median loss (dumb baseline): 112%<br>Dev median loss, corrected 3 Dec 2019 dev label (14 Nov 2026) (dumb baseline): 97.8% | ❌ loses |
| GB nadir forecast from the first 5 s (353 events whose nadir comes later) | **54 mHz** | Lowest value so far (dumb baseline): 58 mHz | ✅ wins |
| GB nadir forecast from the first 10 s (280 events whose nadir comes later) | **37 mHz** | Lowest value so far (dumb baseline): 26 mHz | ❌ loses |
| GB frequency at 60 s, forecast from the first 10 s (412 events) | **59 mHz** | Value at 10 s held (mean of the last 3 s) (dumb baseline): 39 mHz | ❌ loses |
| ERCOT MW lost from reported drop, inertia, load and renewable share (292)<br><sub>GridPulse-Summary. By year: 9-14 % in 2018-2022 vs 12-19 %; 24-33 % in 2023-2025 vs 37-59 %.</sub> | **15.7%** | Dev MW/Hz (4,668) × drop (classical): 20.1%<br>Dev median loss (dumb baseline): 24.5% | ✅ wins |
| ERCOT frequency drop forecast from MW lost, inertia, load (292)<br><sub>A test of GridSim's ERCOT physics: bias +29 %, the simulated drop is too deep in the exam years.</sub> | **29.5%** | MW lost ÷ dev MW/Hz (4,668) (classical): 23.1% | ❌ loses |
| ERCOT inertia from drop, MW lost, load (292) | **11.7%** | Dev inertia per MW of load × load (classical): 11.2%<br>Dev median inertia (dumb baseline): 19.9% | ❌ loses |

Coverage of the 68% error bars on the exam: GB inertia, N = 60 s (410) **63%**, GB MW lost, medium labels (86) **48%**, GB MW lost, low labels (64) **41%**, GB nadir from 5 s (353) **72%**, GB nadir from 10 s (280) **59%**, GB frequency at 60 s from 10 s (412) **61%**, ERCOT MW lost (292) **71%**, ERCOT drop forecast (292) **70%**, ERCOT inertia (292) **85%**.

### Where it wins

- GB MW lost on medium-confidence labels: 25.9 % against 27.1 % for the RoCoF formula and 32.7 % for the dev median (26.8 % with the dev label corrected on 14 Nov 2026).
- GB inertia from the first 10 s: 14.5 % against 15.1 % for a dev-fitted 7.0 s × demand rule and 20.4 % for a constant; on the 148 labelled events 15.3 % against 37.4 % for the RoCoF formula.
- Nadir forecast from the first 5 s: 54 mHz against 58 mHz for the lowest value so far.
- ERCOT MW lost from report numbers: 15.7 % against 20.1 % for a dev MW/Hz constant, and 24-33 % vs 37-59 % in 2023-2025.

### Where it loses

- GB inertia at N = 60 s: 16.4 %, behind the 7.0 s × demand rule (15.1 %). The trace alone (20.3 %) only matches the constant (20.4 %): the skill comes from the demand input.
- GB MW lost on low-confidence labels: 34.8 % against 29.3 % for the RoCoF formula at NESO's inertia.
- By 10 s the nadir forecast (37 mHz) loses to the lowest value so far (26 mHz), and the 60 s forecast (59 mHz) loses to holding the 10 s value (39 mHz).
- ERCOT drop forecast 29.5 % vs 23.1 % (GridSim too deep by 29 %); ERCOT inertia 11.7 % vs 11.2 % for dev s × load.

## Limits

- 1 s data blur the first-second slope and response arrives within 1-2 s, so the trace carries little inertia information; the demand input carries most of it.
- It learned 2018-2022 GB inertia per MW of demand and overestimates 2023-2025 by about 30 % (2023 189 → 242 GVA s, 2024 180 → 244, 2025 160 → 216).
- Error bars on real MW lost are too narrow: 48 % inside 1 sigma on medium labels, 41 % on low. Four dev events cannot widen them reliably.
- 30-60 s behaviour (secondary response, operator actions, further events) is outside GridSim's single secondary ramp.
- ERCOT publishes no frequency traces: ERCOT is tested through report numbers only, and ERCOT noise is scaled GB noise.
- MW/Hz, headroom and FFR have simulated skill 0.5-0.7 but no real truth. Some detected GB 'events' are demand or interconnector ramps, and loss labels are noisy.

## The ratchet

**Objective:** Mean skill on the frozen simulated set p2s (1,387 GridSim events of physics p2, seed 98) over MW lost, inertia and MW/Hz read at 60 s, and the nadir and 60 s settling forecast from 10 s; skill = 1 - MAE / MAE of guessing the median, floored at 0; GB and ERCOT pooled  
**Rule:** Keep if the objective beats the incumbent by more than 0.01 AND, on the 4 GB dev events, the mean of four errors (MW lost MdAPE, inertia MdAPE, nadir@10 s MAE, settling@10 s MAE), each relative to the incumbent's, is no more than 10 % worse.

| Rung | Change | Frozen-set score | Decision | Why | MW lost @60 s | Inertia @60 s | Nadir @10 s | Settling @10 s | Dev vs incumbent | Dev inertia |
|---|---|---|---|---|---|---|---|---|---|---|
| r1 | MLP 256-256-128 with Gaussian heads on the -20..+60 s 1 s window, truncated after N s; 13k selected GridSim events (p1 priors: dev-twin calibrated GB response + Dynamic Containment-era services, ERCOT damping refit); NESO 1 s modelled as the mean of the preceding second; real quiet GB noise | 0.476 | ● kept | first rung | 0.478 | 0.169 | 0.886 | 0.271 | – | 16.6% |
| r2 | measurement model: NESO 1 s values as instantaneous samples, not 1 s means (the dev events' first-second share sat at the 32-68th percentile of simulated events for 'sample' vs 50-92th for 'mean') | 0.478 | ○ rejected | score 0.478 vs 0.476 (needs +0.01); dev errors +10% vs the incumbent (1.098, inside the gate) | 0.475 | 0.181 | 0.879 | 0.271 | +10% | 18.9% |
| r3 | conditioned input: GB/ERCOT transmission demand (NESO generation mix / ERCOT load), dropped 25 % of the time in training so the network also runs without it | 0.524 | ● kept | score 0.524 vs 0.476; dev errors +7% vs the incumbent | 0.546 | 0.298 | 0.895 | 0.265 | +7% | 15.8% |
| r4 | more data: 2 × 24k GridSim states (p1, seeds 1 + 2, 70 % GB in the second), 80 epochs | 0.542 | ● kept | score 0.542 vs 0.524; dev errors -11% vs the incumbent | 0.533 | 0.322 | 0.907 | 0.326 | -11% | 14.7% |
| r5 | wider network 512-512-256 (0.44 M parameters), 120 epochs, same data as r4 | 0.555 | **◆ released** | score 0.555 vs 0.542; dev errors +3% vs the incumbent. Released as v1.0 (r6 was rejected). | 0.516 | 0.336 | 0.918 | 0.365 | +3% | 15.0% |
| r6 | physics p2: 30 % of trips ramp out over 0.5-3 s (unit runbacks, staged trips; the 11 Jul 2019 dev event falls slower than its label implies); trained on p2 + the second p1 bank (15 % ramps), r5's network | 0.589 | ○ rejected | score 0.589 vs 0.555; dev errors +14% vs the incumbent (gate: <= +10%); dev inertia MdAPE 15.0% -> 19.7% | 0.608 | 0.355 | 0.92 | 0.383 | +14% | 19.7% |
| widen | error bars: widen-only factor = max(1, 68th percentile of \|z\|) on the dev events (GB net: MW lost, inertia at 60 s, nadir and settling at 10 s; ERCOT summary heads: 5 ERCOT dev events) | — | ● kept | MW lost ×1.11, settling ×1.28, inertia, nadir and the ERCOT heads ×1.00 (training/checks/widen.json); no effect on the readings |  |  |  |  |  |  |

<sub>Columns after *Why*: Skills on the frozen set p2s, then the dev gate: mean dev error relative to the incumbent, and the dev inertia MdAPE.</sub>

r2 changed the measurement model and got the frozen set p1s; r6 changed the physics (p2) and got p2s, on which every rung was re-scored in order. The values here are the p2s re-scores.

Raw per-rung logs are in [`training/`](training/).

## Physics

The shared GridSim frequency model (shared/grid/gridsim, Phase 1): an aggregated swing equation 2E/f0 · dΔf/dt = ΣΔPm + P_FFR + P_shed − P_lost − D·P_load·Δf/f0 with per-technology governors (deadband, lag, rate limit, headroom clip, reheat / CCGT / hydro lead-lags), fast frequency response, load damping, under-frequency load shedding and a secondary ramp. A wrapper (training/gpsim.py, version p1) rescales GridSim's generated fleets to real GB and ERCOT operating points and adds each grid's services.

**Simulator**

- GB: demand, outturn inertia and wind/solar share drawn jointly from NESO half-hours of the training-input years 2018/2020/2022 (70 %) or wide ranges (30 %: 16-46 GW, 90-420 GVA s); losses log-uniform 150-2,000 MW; governor headroom 300-1,500 MW
- GB services in 60 % of states: Dynamic Containment from NESO's service curve (5 % by 0.2 Hz, 100 % at 0.5 Hz, deadband 0.015 Hz), Moderation (0-500 MW), Regulation (0-600 MW); static FFR at 49.7 Hz; damping median 0.75 %/%; secondary 0-30 % of the loss over 30-90 s
- ERCOT: load from EIA-930 training-input years, inertia ≈ 5 s × load (log-sd 0.22); losses 250-2,800 MW; headroom 1,000-3,500 MW; load resources 500-1,500 MW at 59.7 Hz; FFR 0-750 MW at 59.85 Hz; battery 5 % droop in half the states; BAL-001-TRE deadbands; damping median 1.0 %/% (dev-twin refit)
- Dev calibration on 4 GB 2019 events and 5 ERCOT events: generic services put the real GB nadirs at the 2-6th percentile of their twins (sims too shallow); 2019-era response moved them to the 13-20th; ERCOT twins were 8-29 % too shallow at D = 2.5 %/%, so D median 1.0
- 48,000 simulated states → 28,201 events that pass the real catalogues' detection rules (GB 10 s fall <= −0.15 Hz; ERCOT drop >= 60 mHz or loss >= 350 MW)

**Checked against independent references**

- Initial RoCoF of the rescaled states = ΔP·f0/2E: max relative error 0.10 % (100 GB states) and 0.12 % (100 ERCOT); median 0.03 % and 0.05 % (training/checks/physics.json)
- The Dynamic Containment knee reproduces NESO's service curve at 0.1, 0.2, 0.3 and 0.5 Hz (22.97, 50, 366.7, 1,000 MW of 1,000; training/checks/physics.json)
- GridSim Phase 1 (shared/grid/checks): RoCoF = ΔP·f0/2E to 4e-14, closed-form nadir to 0.03 mHz, ANDES Kundur two-area nadir within 24 mHz
- The lab page's JS GridSim port matches Python to 7e-15 Hz on 6 cases; JS network = numpy to 9e-8, numpy = torch to 1.6e-6 (training/checks/js_parity.json)

## Help wanted

- Sub-second GB or ERCOT frequency (PMU, FNET/GridEye): 1 s NESO data blur the first-second slope that carries inertia. Checked 14 Nov 2026: fnetpublic.utk.edu shows live displays (table, gauge, maps) and three 2011 US sample events as plots and videos; it offers no data download and no terms page; event data and alerts go to members of the FNET/GridEye Industry Consortium. None of the 470 GB or 329 ERCOT events (2018-2026) has a public sub-second trace.
- Better GB loss sizes outside 9 Aug 2019: unit-trip MW from a source cleaner than REMIT capacity and interconnector flows (dev event 3 Dec 2019 was labelled 1,788 MW but metered 854 MW: an IFA hour-boundary ramp was counted as a 933 MW trip; GridData corrected it to 854.7 MW on 14 Nov 2026 and lists six exam events with the same FUELINST failure mode for checking).
- The 17 missing ERCOT monthly operations reports (2018-2026), which hold more ERCOT events.

## Next rungs

- Era-aware priors: condition on year (or Dynamic Containment volume / battery fleet) so 2023-25 inertia per MW and ERCOT FFR are represented; weight up the 2022 inertia mix.
- Inertia from the ambient-noise spectrum, with simulations that drive noise through the swing equation; or sub-second (PMU-grade) data where available.
- A better 30-60 s model (secondary response as droop plus delayed reserve) before trusting the 60 s forecast.
- Real MW-lost bars are too narrow (48 % inside 1 sigma): more GB dev events from 2021 if the splits are ever revisited.
- Rung r6 (30 % of trips ramp out over 0.5-3 s) scored 0.589 on the frozen set but failed the dev gate on inertia; retry once more dev events exist.
- Upstream to GridSim behind flags: _as_events with per-state loss lists, per-technology aggregation in generate_states (~40× faster), the Dynamic Containment two-segment droop, ERCOT effective damping ~1.0 %/% for generated states.

## Files

| Path | What |
|---|---|
| `gridpulse.py`, `pipeline.py` | Inference — numpy/scipy only |
| `gridpulse_weights.npz`, `ercot_head.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [NESO system frequency data (1 s), system inertia (outturn), historic generation mix](https://www.neso.energy/data-portal) | noise (quiet 1 s slices), physics (GB operating points), dev, exam (events, inertia truth, demand input) | NESO Open Data Licence |
| [Elexon BMRS REMIT and B1610 metered output (loss labels by GridData)](https://bmrs.elexon.co.uk/remit) | dev (854 MW metered loss for 3 Dec 2019), exam (loss labels) | Elexon BMRS licence (free use with attribution) |
| [National Grid ESO, Technical Report on the events of 9 August 2019](https://www.ofgem.gov.uk) | dev (1,131 MW loss label) | Crown / NESO copyright, cited |
| [ERCOT monthly Operations Reports (frequency events)](https://www.ercot.com) | dev (5 events), exam (292 generation-loss events) | Public ERCOT documents |
| [EIA-930 Hourly Electric Grid Monitor](https://www.eia.gov/electricity/gridmonitor/) | physics (ERCOT load for the simulated operating points) | Public domain (U.S. Government) |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
