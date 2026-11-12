<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# BeatNet <sub>v1.0</sub>

> **Heartbeat classes from a simulated voxel heart. Research and teaching only.**

> [!WARNING]
> Research and teaching only. Not a medical device. Do not use it to make any clinical decision.

*Is each heartbeat in this ECG normal, supraventricular, ventricular or a fusion beat?*

For every heartbeat in a single-lead ECG, BeatNet gives the AAMI class (normal, supraventricular ectopic, ventricular ectopic, fusion), a QRS width with an error bar, and the chance the beat sits inside atrial fibrillation. It was trained only on a simulated heart; no real labelled beat was ever a training sample. It is not a medical device.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/beatnet/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/beatnet/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | A single-lead ECG (modified lead II), any sampling rate; beat positions optional (a QRS detector is included). |
| **Returns** | AAMI class per beat (N, S, V, F) with probabilities<br>QRS width (ms) ± error bar<br>Probability that the beat is inside atrial fibrillation |
| **Learned from** | A 2 mm voxel heart with eikonal activation through a Purkinje-lined endocardium; membrane potentials with action-potential gradients and restitution; the heart dipole projected through Dower lead vectors; and a rhythm model with ectopy patterns, pauses, AF and flutter. |
| **Noise** | Real MIT-BIH Noise Stress Test recordings plus quiet real background cut from the dev records, mains, amplifier noise and 11-bit quantisation |
| **Network** | 1-D residual CNN, 172k weights, 2.5 s window at 180 Hz plus 6 RR features |
| **Dev set** | 6 DS1 records (101, 112, 119, 201, 203, 220; seed 2026, stratified) |
| **Exam** | DS2 — 22 records, 49,705 beats (de Chazal 2004 inter-patient split); also SVDB and QT DB |

## Run it

```bash
cd models/beatnet
pip install numpy scipy
python demo.py        # about 3 s
```

or from anywhere in the repo: `mlratchet demo beatnet`.

```python
import numpy as np, json
from beatnet import BeatNet
net = BeatNet(); z = np.load("examples/221.npz"); m = json.loads(str(z["meta"]))
out = net.classify(z["ecg"], m["fs"])           # runs its own QRS detector
print(out["label"][:20], out["heart_rate_bpm"])
```

## The exam

The standard inter-patient DS2 exam, 49,705 beats. Every number here is on real data that never chose anything.

| Metric | BeatNet | Baselines | |
|---|---|---|---|
| Macro-F1 over N/S/V/F | **0.547** | Textbook rules (prematurity, QRS width, shape) (classical): 0.431<br>Linear discriminant on sims (de Chazal features) (classical): 0.447<br>Always N (dumb baseline): 0.236 | ✅ wins |
| V beats found (sensitivity) | **97.3%** | Textbook rules (classical): 78.8% | ✅ wins |
| Atrial fibrillation, AUC | **0.971** | RR-irregularity rule (classical): 0.844 | ✅ wins |
| Fusion beats found | **3 / 388** | — | ❌ loses |
| QRS width error (QT DB) | **18 ms** | Classical delineation (classical): 15 ms<br>Median guess (dumb baseline): 20 ms | ❌ loses |

Coverage of the 68% error bars on the exam: QRS width **72.5%**.

### Where it wins

- Beats textbook rules and a linear discriminant on the inter-patient exam, with no real labelled beats.
- Atrial fibrillation AUC 0.971 against 0.844 for the RR-irregularity rule.
- Class probabilities are calibrated (ECE 1.0% on DS2).

### Where it loses

- LBBB beats read as V (record 214 — about half of all false V calls).
- Fusion almost never found (3 of 388).
- Classical delineation measures QRS width better (15 vs 18 ms).

## Limits

- Not a medical device. Research and teaching only.
- Models trained on real DS1 (de Chazal 2004, Zhang 2014) still do better on S and F beats.
- Non-premature S beats are missed; aberrated APCs are called V.

## The ratchet

**Objective:** Macro-F1 (N/S/V/F) on a frozen simulated set of 400 patients, rebuilt for each physics version  
**Rule:** Keep if macro-F1 rises by more than 0.01 AND the dev mean S/V F1 drops by no more than 0.02.

| Rung | Change | Frozen-set score | Decision | Why | Dev S/V F1 | DS2 macro-F1 |
|---|---|---|---|---|---|---|
| r1 | Textbook simulator (physics v1) | 0.829 | ● kept | First rung. | 0.467 | 0.605 |
| r2 | Physics v2: dev-calibrated T wave (APD +47 ms), PVC gain 1.35, less noise; + AF output | 0.873 | ● kept | +0.044; dev gate passed. | 0.476 | 0.541 |
| r3 | Template channel (median neighbour beat) | 0.860 | ○ rejected | Score fell (−0.013). | 0.465 | 0.53 |
| r4 | Physics v3: dev noise floor and a fusion visibility rule | 0.892 | ○ rejected | Dev gate failed (−0.035). | 0.441 | 0.528 |
| r5 | Physics v4: dev-calibrated premature-beat coupling (S pre-RR ratio 0.78 → 0.59; dev 0.56) | 0.897 | **◆ released** | +0.024 over rung 2; dev S/V F1 up. | 0.509 | 0.549 |

<sub>Columns after *Why*: Dev S/V F1 (gate input) and DS2 exam macro-F1 (reported after, never used).</sub>

DS2 macro-F1 fell from rung 1 to rung 2; the exam never decided anything, so it is reported as is.

Raw per-rung logs are in [`training/`](training/).

## Physics

A 2 mm voxel heart with eikonal activation through a Purkinje-lined endocardium; membrane potentials with action-potential gradients and restitution; the heart dipole projected through Dower lead vectors; and a rhythm model with ectopy patterns, pauses, AF and flutter.

**Simulator**

- Beat type set by where activation starts — sinus, ectopic atrial focus, ventricular focus, bundle-branch block, fusion
- Dipole from −∇Vm, Dower lead vectors with random orientation, electrode placement and breathing
- ECGSYN RR spectrum, compensatory pauses, bigeminy/trigeminy/couplets/runs, AF and flutter

**Checked against independent references**

- Dipole volume vs surface form r = 0.993; RR LF/HF asked 0.5/1/2 → 0.50/1.05/1.98
- QRS medians: normal 72 ms, LBBB 158, RBBB 115, PVC 142

## Next rungs

- Wider bundle-branch-block morphology in the simulator, calibrated on a DS1 LBBB record kept out of the exam.
- A twin test per dev record — calibration so far is population-level only.
- Refit the fusion visibility window to how annotators used F.

## Files

| Path | What |
|---|---|
| `beatnet.py`, `pipeline.py` | Inference — numpy/scipy only |
| `beatnet_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [MIT-BIH Arrhythmia Database (PhysioNet)](https://physionet.org/content/mitdb/) | dev, exam | ODC-By 1.0 |
| [MIT-BIH Noise Stress Test Database](https://physionet.org/content/nstdb/) | noise | ODC-By 1.0 |
| [MIT-BIH Supraventricular Arrhythmia Database](https://physionet.org/content/svdb/) | exam | ODC-By 1.0 |
| [QT Database](https://physionet.org/content/qtdb/) | exam | ODC-By 1.0 |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
