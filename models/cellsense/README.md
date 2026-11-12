<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# CellSense <sub>v1.1</sub>

> **Battery health and remaining life from one fast discharge.**

> [!NOTE]
> Released, with part of the exam still waiting on data — see [Help wanted](#help-wanted).

*How worn is this battery cell, why, and how many cycles does it have left?*

CellSense reads one fast discharge curve of a lithium-ion cell and returns its total capacity, resistance, how much lithium and electrode material it has lost, and a forecast of its cycle life. It learned from an electrochemical simulator calibrated on six real cells, and was examined on 118 others.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/cellsense/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/cellsense/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | One 4C discharge curve (capacity vs voltage, plus temperature) of an A123 LFP/graphite 18650 cell; optionally a reference curve from when it was new. |
| **Returns** | Total capacity ± error bar, and pulse resistance<br>Lithium lost, graphite lost, cathode lost (degradation modes)<br>Cycle-life forecast with a range<br>A partial-discharge reading for curves that stop before 2.0 V |
| **Learned from** | A vectorized single-particle model of each electrode (PyBaMM Prada2013 open-circuit curves, exact Fickian diffusion, Butler-Volmer kinetics, lumped thermal model) with aging: SEI growth (lithium loss), graphite and cathode loss, transport polarisation, and a growing share of slow, poorly connected graphite — the one mechanism that fits early-life change on all six calibration cells. |
| **Noise** | Measurement noise and cell-to-cell scatter in simulation; six real cells calibrate the aging priors. |
| **Network** | Three 1-D residual CNNs along the voltage axis (health, lifetime, partial discharge), β-NLL error bars |
| **Dev set** | Six calibration cells |
| **Exam** | 118 cells, 3,771 health readings; Severson's primary and secondary test splits for lifetime |

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
r = cs.diagnose(d["Q"][3], d["T"][3], d["Q"][0], d["T"][0], cycle=int(d["cycle"][3]))
print(r["q_total"], r["life"], r["lli"])
```

## The exam

Health on 118 cells; lifetime from cycle 100 on Severson's own test splits. Every number here is on real data that never chose anything.

| Metric | CellSense | Baselines | |
|---|---|---|---|
| Total capacity error (full curves)<br><sub>Fade correlation 0.979.</sub> | **13.2 mAh** | — | ℹ️ |
| Lifetime error at cycle 100, primary test | **22.4%** | Median guess (dumb baseline): 34%<br>Severson's models (trained on 41 real cells) (published): 13–15% | ↔️ mixed |
| Lifetime error at cycle 100, secondary test | **13.1%** | Median guess (dumb baseline): 19%<br>Severson's models (published): 9–11% | ↔️ mixed |
| Partial discharge stopped at 0.90 Ah, capacity error | **17 mAh** | No-model baseline (dumb baseline): 58 mAh | ✅ wins |
| Partial discharge stopped at 0.45 Ah, capacity error | **52 mAh** | Assume no fade (dumb baseline): 46 mAh | ❌ loses |

### Where it wins

- Beats the median guess on lifetime from cycle 100, using no real labelled cells for training.
- Partial discharges to 0.75 Ah or more read 2–3× better than the no-model baseline.

### Where it loses

- Severson's models, trained on 41 real cells, still forecast lifetime better (13–15% vs 22%).
- Last quarter of life — worn cells read about 6% too healthy (36% life error).
- Partial discharges stopped at 0.6 Ah or below are no better than assuming no fade.

## Limits

- It knows one cell — A123 APR18650M1A at Severson's protocol (4C to 2.0 V, 30 °C).
- Other chemistries, sizes, rates and temperatures are outside what it learned.
- The simulator's graphite-staging features are sharper than real cells'.

## The ratchet

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

Raw per-rung logs are in [`training/`](training/).

## Physics

A vectorized single-particle model of each electrode (PyBaMM Prada2013 open-circuit curves, exact Fickian diffusion, Butler-Volmer kinetics, lumped thermal model) with aging: SEI growth (lithium loss), graphite and cathode loss, transport polarisation, and a growing share of slow, poorly connected graphite — the one mechanism that fits early-life change on all six calibration cells.

**Simulator**

- Discharge at 4.4 A to 2.0 V then a hold, as in Severson et al. 2019
- Aging model v2.1 — slow graphite share grows as cycles^0.8 and levels off at 8–14%
- SEI loss linear in cycles (these cells aren't √t)
- Late transport polarisation growth, and a lifetime bank of whole aging trajectories with a knee

**Checked against independent references**

- Matches PyBaMM 26.10 (SPM, Prada2013): capacity within 0.003–0.21%, median voltage error ≈ 0.3 mV
- Severson's own variance feature reproduces on this data (r = −0.93 over 124 cells)

## Help wanted

- Other chemistries: Sandia NCA / NMC / LFP cycling files (Preger et al. 2020) from BatteryArchive, or another public multi-chemistry dataset with a documented download (Zhu et al. 2022 on Zenodo; Pozzato et al. 2022 on Mendeley Data).

## Next rungs

- Multi-chemistry simulator and dev gate once the data is in.
- Fix the late-life lifetime bias — feed the lifetime net the health net's capacity reading.
- Electrode-scale heterogeneity to soften the staging features.

## Files

| Path | What |
|---|---|
| `cellsense.py` | Inference — numpy/scipy only |
| `cellsense_weights.npz`, `cellsense_life_weights.npz`, `cellsense_partial_weights.npz` | Weights |
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

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
