<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# BearingNet <sub>v1.0</sub>

> **Hearing a bearing fail, from impact physics and a healthy machine's hum.**

*Is this rolling-element bearing healthy, or does it have an outer-race, inner-race or ball defect?*

From 3–20 seconds of one accelerometer on a rotating machine, plus the bearing's geometry and a rough shaft speed, BearingNet says whether the bearing is healthy or has an outer-race, inner-race or ball defect, and refines the shaft speed. It learned from simulated impact trains laid into real healthy-machine vibration and was examined on two real rigs — including one it never heard.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/bearingnet/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/bearingnet/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | 3–20 s of one accelerometer channel, the bearing geometry, and a nominal shaft speed. |
| **Returns** | Condition — healthy (N), outer race (OR), inner race (IR) or ball (B) — with probabilities<br>Refined shaft speed ± error bar<br>Defect size (experimental; did not beat the median guess) |
| **Learned from** | Bearing kinematics (BPFO, BPFI, BSF, FTF from geometry) drive trains of impacts with slip and jitter, load-zone amplitude modulation, entry/exit pulse pairs, random structural resonances, shaft lines, decoy impact trains and random knocks. |
| **Noise** | Healthy CWRU file 98 and Paderborn K001–K003, time-stretched ±15% for speed (never in dev or exam) |
| **Network** | Envelope spectra in 5 bands on fault-aligned log-frequency axes; 1-D residual CNN, 438k parameters, 20k simulations |
| **Dev set** | 10 CWRU files (seed 20261106) and Paderborn K006, KA01, KI03 — file-level split |
| **Exam** | 150 CWRU files on every sensor position, and 26 Paderborn bearings (919 recordings) |

## Run it

```bash
cd models/bearingnet
pip install numpy scipy
python demo.py        # about 3 s
```

or from anywhere in the repo: `mlratchet demo bearingnet`.

```python
import numpy as np, json, bearingnet
net = bearingnet.BearingNet()
z = np.load("examples/03_CWRU_135_OR.npz"); m = json.loads(str(z["meta"]))
r = net.read(z["x"], fs=float(z["fs"]), bearing=m["bearing"], rpm_nominal=m["rpm_nominal"])
print(r["condition"], r["p"], r["rpm"])
```

## The exam

Balanced accuracy over N/OR/IR/B, against classical envelope analysis and the majority class. Every number here is on real data that never chose anything.

| Metric | BearingNet | Baselines | |
|---|---|---|---|
| Paderborn (new machine), balanced accuracy | **0.45** | Envelope analysis (classical): 0.28<br>Majority class (dumb baseline): 0.33 | ✅ wins |
| CWRU far sensor, balanced accuracy | **0.57** | Envelope analysis (classical): 0.51<br>Majority class (dumb baseline): 0.25 | ✅ wins |
| CWRU near sensor, balanced accuracy | **0.39** | Envelope analysis (classical): 0.49<br>Majority class (dumb baseline): 0.25 | ❌ loses |
| CWRU base plate, balanced accuracy | **0.5** | Envelope analysis (classical): 0.67<br>Majority class (dumb baseline): 0.33 | ❌ loses |
| CWRU shaft-speed error | **0.35%** | Nominal speed (dumb baseline): 1.36% | ✅ wins |
| Defect size | **worse** | Median guess (dumb baseline): better | ❌ loses |

Coverage of the 68% error bars on the exam: Shaft speed (CWRU) **80%**.

### Where it wins

- On a machine it never heard (Paderborn), it beats the classical envelope method 0.45 to 0.28.
- Cuts CWRU shaft-speed error from 1.36% to 0.35%.

### Where it loses

- Classical envelope analysis wins at the CWRU near sensor and base plate.
- Ball faults found only 19% of the time on CWRU.
- The size output is useless; speed refinement hurts when the nominal speed is already exact.

## Limits

- "Healthy" is weak — a healthy bearing next to a faulty one is called healthy 10% of the time; Paderborn healthy 35%.
- The dev set was small and optimistic (Paderborn dev 0.78 vs exam 0.45).

## The ratchet

**Objective:** Fault balanced accuracy plus speed and size skill on the frozen simulated set (val2)  
**Rule:** Keep if the objective rises by more than 0.01 AND dev balanced accuracy does not fall.

| Rung | Change | Frozen-set score | Decision | Why | Dev balanced accuracy |
|---|---|---|---|---|---|
| r1 | v1 physics (textbook) | 0.572 | ● kept | First rung. | 0.31 |
| r2 | v2 physics, calibrated on dev (louder faults, longer ringing, shaft modulation) | 0.691 | **◆ released** | +0.119; dev 0.31 → 0.68. | 0.68 |
| r3 | HypNet — one trunk shared across fault hypotheses | 0.692 | ○ rejected | +0.001 and the dev gate failed (0.64). | 0.64 |
| r4 | Rung 2 + band dropout | 0.690 | ○ rejected | Score fell (−0.001). | 0.69 |

<sub>Columns after *Why*: Dev balanced accuracy (real dev files).</sub>

The big win came from a twin test — rung 1 got 100% of simulated dev twins right but only 22% of real CWRU dev files. Real faults were 10–25 dB above background and rang longer; physics v2 fixed that.

Raw per-rung logs are in [`training/`](training/).

## Physics

Bearing kinematics (BPFO, BPFI, BSF, FTF from geometry) drive trains of impacts with slip and jitter, load-zone amplitude modulation, entry/exit pulse pairs, random structural resonances, shaft lines, decoy impact trains and random knocks.

**Simulator**

- Impact trains with 0–2% slip and 0.5–2% jitter
- Stribeck load-zone modulation (IR at shaft rate, B at cage rate with alternating race strikes)
- 1–3 random kHz resonances, shaft lines, decoy trains, random knocks
- v2 physics, calibrated on dev: faults 10–25 dB above background, longer ringing (3.5 ms), shaft-rate modulation on healthy machines

**Checked against independent references**

- Fault frequencies match CWRU's published multiples for SKF 6205/6203 to 1.6e-4
- Simulated envelope lines within 0.16% of theory

## Next rungs

- Healthy-vs-wrong-bearing — simulate the other bearing's fault comb as a decoy; calibrate the healthy prior per machine.
- Bigger, more varied banks and more background machines; score the ratchet on SNR > −6 dB only.
- A hybrid with the envelope method — the two fail on different rigs.

## Files

| Path | What |
|---|---|
| `bearingnet.py`, `pipeline.py` | Inference — numpy/scipy only |
| `bearingnet_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter) | noise, dev, exam | Public download; acknowledge CWRU |
| [Paderborn University KAt bearing data](https://groups.uni-paderborn.de/kat/BearingDataCenter/) | noise, dev, exam | Use under the terms on that page; cite Lessmeier et al. 2016 |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
