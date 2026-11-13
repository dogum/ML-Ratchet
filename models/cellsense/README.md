<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# CellSense <sub>v1.2</sub>

> **Battery health and remaining life from one discharge curve.**

> [!NOTE]
> Released, with part of the exam still waiting on data — see [Help wanted](#help-wanted).

*How worn is this battery cell, why, and how many cycles does it have left?*

CellSense reads one fast discharge curve of a lithium-ion cell and returns its total capacity, resistance, how much lithium and electrode material it has lost, and a forecast of its cycle life. It learned from an electrochemical simulator calibrated on six real cells, and was examined on 118 others. A second, separate network (v1.2) reads slow C/20 capacity tests of an NMC811 cell (LG M50T), including tests stopped early; there it was calibrated on three real cells and examined on seven, and it mostly loses to guessing the median.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/cellsense/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/cellsense/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | LFP: one 4C discharge curve (capacity vs voltage, plus temperature) of an A123 LFP/graphite 18650 cell, optionally beside a reference curve from when it was new. NMC811: one C/20 discharge from full (capacity vs voltage) of an LG INR21700-M50T, complete or stopped early, optionally beside the cell's first test. |
| **Returns** | Total capacity ± error bar, and pulse resistance<br>Lithium lost, graphite lost, cathode lost (degradation modes)<br>Cycle-life forecast with a range<br>A partial-discharge reading for curves that stop before 2.0 V<br>NMC811 (separate network): capacity to 2.5 V at C/20 ± error bar, fade, and lithium / negative / NMC lost, from a C/20 test complete or stopped early |
| **Learned from** | A vectorized single-particle model of each electrode (PyBaMM Prada2013 open-circuit curves, exact Fickian diffusion, Butler-Volmer kinetics, lumped thermal model) with aging: SEI growth (lithium loss), graphite and cathode loss, transport polarisation, and a growing share of slow, poorly connected graphite — the one mechanism that fits early-life change on all six calibration cells. |
| **Noise** | Measurement noise and cell-to-cell scatter in simulation; six real cells calibrate the LFP aging priors, three the NMC811 ones. NMC811 also: cycler gain error, a few mV of voltage offset, per-cell open-circuit-curve jitter. |
| **Network** | Three 1-D residual CNNs along the voltage axis for LFP (health, lifetime, partial discharge) and a fourth, smaller one for NMC811; β-NLL error bars |
| **Dev set** | LFP: six calibration cells. NMC811: W3, W4, W8 (26 tests), split_units(cells, 3, seed=811) |
| **Exam** | LFP: 118 cells, 3,771 health readings; Severson's primary and secondary test splits for lifetime. NMC811: W5, W7, W9, W10, G1, V4, V5 (75 tests) |

## Run it

```bash
cd models/cellsense
pip install numpy
python demo.py        # about 3 s
```

or from anywhere in the repo: `mlratchet demo cellsense`.

```python
import numpy as np, cellsense
cs = cellsense.CellSense(); d = np.load("examples/b3c33.npz")
r = cs.diagnose(d["Q"][3], d["T"][3], d["Q"][0], d["T"][0], cycle=int(d["cycle"][3]), charge_time_min=float(d["charge_time"]))
print(r["q_total"], r["life"], r["lli"])
nmc = cellsense.CellSense(chemistry="nmc811"); m = np.load("examples/m50t_W9.npz")   # python demo_nmc811.py
q = m["Q"][3].copy(); q[q > 2.5] = np.nan                                              # a C/20 test stopped after 2.5 Ah
print(nmc.diagnose(q, m["Q"][0])["q_total"])
```

## The exam

LFP: health on 118 cells; lifetime from cycle 100 on Severson's own test splits. NMC811: capacity from C/20 tests stopped at 1.5, 2.5 and 3.5 Ah on 7 cells (75 tests), with and without the cell's first test as reference. Every number here is on real data that never chose anything.

| Metric | CellSense | Baselines | |
|---|---|---|---|
| Total capacity error (full curves)<br><sub>Fade correlation 0.979.</sub> | **13.2 mAh** | — | ℹ️ |
| Lifetime error at cycle 100, primary test | **22.4%** | Median guess (dumb baseline): 34%<br>Severson's models (trained on 41 real cells) (published): 13–15% | ↔️ mixed |
| Lifetime error at cycle 100, secondary test | **13.1%** | Median guess (dumb baseline): 19%<br>Severson's models (published): 9–11% | ↔️ mixed |
| Partial discharge stopped at 0.90 Ah, capacity error | **17 mAh** | No-model baseline (dumb baseline): 58 mAh | ✅ wins |
| Partial discharge stopped at 0.45 Ah, capacity error | **52 mAh** | Assume no fade (dumb baseline): 46 mAh | ❌ loses |
| NMC811 capacity, test stopped at 2.5 Ah, with reference | **69 mAh** | Dev median (dumb baseline): 98 mAh<br>Assume no fade (dumb baseline): 234 mAh<br>Shift at cut (classical): 152 mAh<br>Voltage fit, same electrode model (classical): 258 mAh | ✅ wins |
| NMC811 capacity, test stopped at 3.5 Ah, no reference | **94 mAh** | Dev median (dumb baseline): 111 mAh<br>Voltage fit, same electrode model (classical): 106 mAh | ✅ wins |
| NMC811 capacity, test stopped at 3.5 Ah, with reference | **115 mAh** | Dev median (dumb baseline): 98 mAh<br>Shift at cut (classical): 84 mAh<br>Voltage fit, same electrode model (classical): 125 mAh | ❌ loses |
| NMC811 capacity, all six cut-short readings<br><sub>Stopped at 1.5 Ah it is worst (169 with reference, 196 without).</sub> | **135 mAh** | Dev median (dumb baseline): 105 mAh | ❌ loses |
| NMC811 capacity, full C/20 curve<br><sub>The shipped system reads a full curve's capacity off its end; the network alone read them to 46 mAh.</sub> | **0 mAh** | Coulomb counting (the curve's end) (classical): 0 mAh<br>Dev median (dumb baseline): 98–111 mAh | ➖ tie |
| NMC811 fade, cut-short tests with reference | **2.47 points** | Dev median fade (dumb baseline): 1.94 points<br>Assume no fade (dumb baseline): 4.8 points | ❌ loses |

Coverage of the 68% error bars on the exam: NMC811 capacity (all readings) **61%**, NMC811 capacity, cut at 1.5 / 2.5 / 3.5 Ah **48%**, NMC811 fade **71%**.

### Where it wins

- Beats the median guess on lifetime from cycle 100, using no real labelled cells for training.
- Partial discharges to 0.75 Ah or more read 2–3× better than the no-model baseline.
- NMC811: a C/20 test stopped at 2.5 Ah, read with the cell's first test, gives capacity to 69 mAh (dev median 98, classical 152–258).

### Where it loses

- Severson's models, trained on 41 real cells, still forecast lifetime better (13–15% vs 22%).
- Last quarter of life — worn cells read about 6% too healthy (36% life error).
- Partial discharges stopped at 0.6 Ah or below are no better than assuming no fade.
- NMC811: over all cut-short C/20 tests it errs by 135 mAh; always guessing the dev median errs by 105. Worst stopped at 1.5 Ah (169–196 mAh).
- NMC811: its 68% error bars cover 36–66% of cut-short readings.

## Limits

- It knows two cells — A123 APR18650M1A at Severson's protocol (4C to 2.0 V, 30 °C), and LG INR21700-M50T NMC811 at a C/20 capacity test (23 °C).
- Other chemistries, sizes, rates and temperatures are outside what it learned.
- The simulator's graphite-staging features are sharper than real cells'.
- NMC811: no lifetime forecast — the Pozzato cells lost only 3–9% and never reached end of life.
- NMC811: its open-circuit curves are corrected on three cells; ±15 mV features near the end of discharge remain, and they set the real error.
- NMC811: lithium lost versus negative lost is poorly separated by a C/20 curve; the degradation modes have no answer key.

## The ratchet

### CellSense

**Objective:** Health half (with reference, without it, and as partial discharges; skill floored at 0) plus lifetime skill, on the frozen simulated sets  
**Rule:** Keep if the objective rises by more than 0.01 AND the six calibration cells' capacity error does not get worse AND their lifetime error (read every 50 cycles) is no more than 2 points worse.

| Rung | Change | Frozen-set score | Decision | Why |
|---|---|---|---|---|
| c1 | Baseline 1-D residual CNN along the voltage axis, 100,000 simulated new/aged pairs | 0.037 | ● kept | First rung. |
| c2 | Transport aging: slow polarisation grows several-fold with age | 0.170 | ○ rejected | Dev gate: calibration cells 13.3 vs 11.4 mAh. |
| c3 | Slow graphite: a growing share sits out the 4C discharge | 0.211 | ○ rejected | Dev gate: calibration cells 21.6 vs 11.4 mAh. |
| c4 | Lifetime network on whole simulated aging trajectories | -0.124 | ○ rejected | Score fell. |
| c5 | Lifetime with aging priors refitted to the six calibration cells | 0.190 | ● kept | 0.190 vs 0.037. |
| c6 | Early-life physics: slow graphite from the first cycles, SEI linear in cycles | 0.316 | ○ rejected | New lifetime gate: dev lifetime error 45% vs 31%. |
| c7 | Early life plus a whole-life check against real curves at 50, 80 and 100% of life | 0.375 | ● kept | 0.375 vs 0.190; dev lifetime 22%. |
| c8 | Partial discharges in one network (half the readings cut short) | 0.625 | ○ rejected | Dev gate: calibration cells 16.6 vs 11.4 mAh. |
| c9 | Partial-discharge specialist reading curves smoothed over 24 mV | 0.466 | **◆ released** | 0.466 vs 0.375; full curves stay with the health network. |

The frozen lifetime set was regenerated twice; all rungs were re-scored.

### v2 — NMC811 (LG M50T)

**Objective:** Mean skill (1 - MAE / MAE of guessing the median, floored at 0) over capacity, lithium lost, negative lost and NMC lost (and fade with a reference) on a frozen set of 3,000 simulated new/aged C/20 pairs, averaged over four readings: full curve with and without the reference, and cut short at a random depth with and without it  
**Rule:** Keep if the objective beats the incumbent by more than 0.01 AND the three dev cells' capacity error on tests cut at 1.5, 2.5 and 3.5 Ah is no more than 5% worse AND their full-curve error is no more than 2 mAh worse.

| Rung | Change | Frozen-set score | Decision | Why | Exam partial (mAh) | Exam full (mAh) |
|---|---|---|---|---|---|---|
| n1 | Textbook Chen2020 cell, wide degradation-mode priors, 1-D CNN on C/20 Q(V), 60,000 simulated pairs | 0.192 | ● kept | First rung. | 585.5 | 252.6 |
| n2 | Simulator calibrated on the dev cells: new cells drawn around their fits, mode ranges from their voltage fits | 0.235 | ● kept | +0.043; dev 255 vs 607 mAh partial, 231 vs 257 full. | 245.0 | 211.3 |
| n3 | Open-circuit-curve correction fitted on the dev cells (one spline per electrode); new cells and mode ranges refitted on it; resistance may fall | 0.423 | ● kept | +0.188; dev 124 vs 255 mAh partial, 59 vs 231 full. | 182.7 | 111.9 |
| n4 | Capacity head predicts the charge still to come below where the curve stops, not the capacity | 0.464 | ○ rejected | +0.041, but dev 181 vs 124 mAh partial and 80 vs 59 full. | 251.1 | 143.6 |
| n5 | n3 reading curves smoothed over 26 mV (the dQ/dV peaks are where the simulator is least right) | 0.413 | ○ rejected | -0.010 on val_ocpj (on val_ocp: +0.024 but dev 167 vs 124 mAh). | 209.7 | 132.3 |
| n6 | n3 reading curves smoothed over 53 mV | 0.422 | ○ rejected | -0.001 on val_ocpj (-0.042 on val_ocp). | 227.2 | 130.2 |
| n7 | n3, and a curve that reaches 2.5 V takes its capacity and fade from its own end (coulomb counting, no network) | 0.484 | ● kept | +0.061 on val_ocpj; dev full 0 vs 59 mAh. (Rejected on val_ocp: +0.006.) | 182.7 | 0.0 |
| n8 | OCP jitter: each simulated cell's correction scaled 0.5-1.5 plus 4 mV spline wiggles per electrode (the size of what the dev fits leave); with n7's rule | 0.773 | **◆ released** | +0.290; dev 126.1 vs 124.4 mAh partial (inside 5%), full 0 vs 0. | 135.4 | 0.0 |
| n9 | Training cells jittered twice as wide (scale 0-2, 8 mV per coefficient); frozen set unchanged; with the rule | 0.742 | ○ rejected | -0.031 on val_ocpj, although dev fell to 84 mAh. The best next rung: set the jitter from dev measurements and rebuild the frozen set. | 91.4 | 0.0 |

<sub>Columns after *Why*: Real exam (7 cells, 75 tests), reported after the fact; never used to choose.</sub>

Raw log: training/nmc_ratchet.json. The frozen set was rebuilt with each physics version (textbook, dev-calibrated, OCP-corrected, OCP-jittered); every rung was re-scored on each and every decision re-taken. On val_ocpj n7 flipped from rejected (+0.006 on val_ocp) to kept (+0.061). n4-n6 were first rejected by the dev gate on val_ocp. n8 judged as a bare network failed the full-curve gate against n7's rule (39 vs 0 mAh); every rung after n7 includes the rule, and n8 was judged again with it.

Raw per-rung logs are in [`training/`](training/).

## Physics

A vectorized single-particle model of each electrode (PyBaMM Prada2013 open-circuit curves, exact Fickian diffusion, Butler-Volmer kinetics, lumped thermal model) with aging: SEI growth (lithium loss), graphite and cathode loss, transport polarisation, and a growing share of slow, poorly connected graphite — the one mechanism that fits early-life change on all six calibration cells.

**Simulator**

- Discharge at 4.4 A to 2.0 V then a hold, as in Severson et al. 2019
- Aging model v2.1 — slow graphite share grows as cycles^0.8 and levels off at 8–14%
- SEI loss linear in cycles (these cells aren't √t)
- Late transport polarisation growth, and a lifetime bank of whole aging trajectories with a knee
- NMC811 (v1.2): single-particle model with PyBaMM's Chen2020 curves and constants, a dev-fitted spline correction to each open-circuit curve, jittered per simulated cell; C/20 discharge from full to 2.5 V; lithium, negative and NMC loss and resistance change drawn up to 2.5x the dev cells' voltage-fit values

**Checked against independent references**

- Matches PyBaMM 26.10 (SPM, Prada2013): capacity within 0.003–0.21%, median voltage error ≈ 0.3 mV
- Severson's own variance feature reproduces on this data (r = −0.93 over 124 cells)
- NMC811 simulator (Chen2020) vs PyBaMM 26.10.1.1 SPM: capacity within 0.0002% and voltage within 0.016 mV at C/20 (0.06% and 0.22 mV at 1C), fresh and with 10% of each degradation mode; vs SPMe at C/20: 0.0005%, 0.06 mV
- NMC811 open-circuit curves corrected on three dev cells: C/20 fit rms 11.9 → 5.1 mV (median per test)

## Help wanted

- More chemistries and rates: Sandia NCA / NMC / LFP cycling files (Preger et al. 2020) from BatteryArchive (download by email request), or Zhu et al. 2022 (Zenodo). NMC811 at C/20 is done (Pozzato et al. 2022 on OSF); it needs aged cells far from new and faster discharges.

## Next rungs

- NMC811: set the open-circuit-curve jitter from per-test dev fits and rebuild the frozen set (rung n9 — twice the jitter — read the dev cells to 84 mAh but was rejected on the old frozen set).
- NMC811: a two-material negative (graphite + SiOx) with its own silicon loss, for the features near the end of discharge.
- NMC811: read the HPPC pulses in the same tests for resistance, and cut-short tests that start below full.
- Multi-chemistry simulator and dev gate once the data is in.
- Fix the late-life lifetime bias — feed the lifetime net the health net's capacity reading.
- Electrode-scale heterogeneity to soften the staging features.

## Files

| Path | What |
|---|---|
| `cellsense.py` | Inference — numpy/scipy only |
| `cellsense_weights.npz`, `cellsense_life_weights.npz`, `cellsense_partial_weights.npz`, `cellsense_nmc811_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [Severson et al. 2019 fast-charging dataset (Toyota Research Institute / Stanford / MIT)](https://data.matr.io/1/) | exam, dev | CC BY 4.0 |
| [Pozzato, Allam & Onori 2022, LG INR21700-M50T aging with EV driving profiles (Data in Brief 41, 107995), processed diagnostic tests](https://osf.io/qsabn/) | exam, dev (NMC811) | CC BY 4.0 |
| [Chen et al. 2020 parameters of the LG M50T, as PyBaMM's Chen2020 set](https://doi.org/10.1149/1945-7111/ab9050) | physics (NMC811) | Journal article; PyBaMM is BSD-3 |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
