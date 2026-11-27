<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# RoomSense <sub>v1.0</sub>

> **How long does a room ring after a clap?**

> [!NOTE]
> Released, with part of the exam still waiting on data — see [Help wanted](#help-wanted).

*How much reverberation can one mono clap reveal about a room?*

RoomSense estimates midband reverberation time from a single mono clap and gives an experimental volume estimate. It learned on simulated rooms. The first exam passed synthetic claps through measured responses from three held-out rooms. The accuracy exam on recorded handclaps is pending.

[Project page](https://dogum.github.io/ML-Ratchet/models/roomsense/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | One mono waveform in full-scale units and its sampling rate in Hz. Include a quiet lead-in and the decay. |
| **Returns** | Reverberation time RT60 in seconds, with a nominal 68% interval.<br>Experimental volume in cubic metres, with a nominal 68% interval.<br>Classical band-decay estimate when measurable, along with onset time and prominence. |
| **Learned from** | Early sound follows image-source reflection paths. Eyring absorption sets a stochastic late decay. |
| **Noise** | Twenty recorded ESC-10 clock and rain backgrounds, split by original recording before use. |
| **Network** | A 10,040-parameter 1D CNN on early band energy and band decay curves with a classical RT60 anchor. |
| **Physics also enters** | inputs: Three octave-band early-energy envelopes, backward integrated energy decays, and T20 fits.<br>output: The model predicts a correction to the classical RT60 fit when two or more bands support it. |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | Surrey C and D and Kitamura open lounge, 38 noisy clips from 19 swept responses. |
| **Exam** | Surrey A and B and Kitamura music room, 38 noisy clips from 19 held-out swept responses. |

## Run it

```bash
cd models/roomsense
pip install numpy scipy
python demo.py        # about 1 s
```

or from anywhere in the repo: `mlratchet demo roomsense`.

```python
from roomsense import RoomSense
print(RoomSense().read_file("my-clap.wav"))
```

## The exam

Equal weight per room. A dev median supplies a fallback when an estimator abstains. The T20-only comparison on the four clips with a valid fit is also shown. Volume was scored in two rooms with documented dimensions. Actual-clap accuracy is pending. Every number here is on real data that never chose anything.

| Metric | RoomSense | Baselines | |
|---|---|---|---|
| RT60 MAE across all exam clips<br><sub>Point estimate over three rooms. The best classical pipeline read 15 of 38 clips before fallback.</sub> | **0.1171 s** | T20 + tail fit + dev-median fallback (classical): 0.3153 s<br>T20 + dev-median fallback (classical): 0.4387 s<br>Dev median (dumb baseline): 0.4832 s | ✅ wins |
| RT60 MAE where T20 could read the clip<br><sub>Four clips in three rooms, all at 44 dB peak-to-noise ratio.</sub> | **0.0743 s** | T20 (classical): 0.0225 s | ❌ loses |
| Experimental volume MAE on a log scale<br><sub>Two exam rooms. The decay-only fit selected a zero slope and matched the dev median.</sub> | **0.4237 ln(m³)** | Dev-fitted decay-only estimate (classical): 0.9872 ln(m³)<br>Dev median (dumb baseline): 0.9872 ln(m³) | ℹ️ |

Coverage of the 68% error bars on the exam: RT60 **72.8%**, Experimental volume **88.5%**.

### Where it wins

- Smaller all-clip RT60 error than both classical fallback chains and the dev median in the measured-response exam.

### Where it loses

- Classical T20 was about three times more accurate on its four readable clips.
- On the three dev rooms the network averaged 0.213 s error, versus 0.120 s for the dev median.

## Limits

- The accuracy test uses measured room responses with synthesized excitations. Real phone and microphone transfer awaits paired references.
- The volume check contains only two exam rooms. Reported uncertainty is calibrated on two other rooms.
- The input needs a quiet lead-in and a distinct clap. Six published phone demo clips were too tightly cropped for the input window.
- Reverberation depends on geometry and absorption. The simulator assumes rectangular rooms with diffuse late decay.

## The ratchet

**Objective:** RT60 MAE skill against the dev-median guess on frozen simulations  
**Rule:** Keep when simulated RT60 skill gains more than 0.01 and the measured dev gate passes.

| Rung | Change | Frozen-set score | Decision | Why | Dev RT60 MAE (s) | Dev volume MAE (ln m³) |
|---|---|---|---|---|---|---|
| r1 | Image-source reflections and Eyring tails. Learn a correction to the classical estimate. | 0.758 | **◆ released** | First rung. | 0.21268 | 0.616 |
| r2 | Reduce diffuse pressure from 0.390 to 0.316 using the measured dev twin gap. | 0.758 | ○ rejected | -0.000 on val-1.1, needs more than +0.01. | 0.22817 | 0.55599 |
| r3 | Train on equal numbers of textbook and dev-calibrated simulations. | 0.756 | ○ rejected | -0.002 on val-1.1, needs more than +0.01. | 0.24323 | 0.61733 |
| r4 | Add absolute RT60 error in seconds to the training objective on the mixed bank. | 0.762 | ○ rejected | +0.004 on val-1.1, needs more than +0.01. | 0.23522 | 0.69967 |

<sub>Columns after *Why*: Measured dev experiments. Exam never used..</sub>

Raw per-rung logs are in [`training/`](training/).

## Physics

Early sound follows image-source reflection paths. Eyring absorption sets a stochastic late decay.

**Simulator**

- Rectangular rooms, fractional direct and image-path delays, six frequency-dependent reflecting faces
- Three reverberant bands, generated clap pulses, microphone coloration, and independent real backgrounds

**Checked against independent references**

- First-order path arrival times and pressure reflection match analytic geometry to floating-point precision.
- Eyring reverberation time agrees with the closed-form reference within 0.009%.
- Classical fitting recovered three known exponential decay times to within 2.0%.

## Help wanted

- ARPEGE's 28 annotated clap and sweep pairs across six rooms. The official audio endpoint returned HTTP 403. The matching protocol and locked scorer are ready in training/.

## Next rungs

- Run the frozen actual-clap exam when matched room recordings and reuse terms are available.
- Test a front end for tightly cropped inputs against a separately frozen set of paired clap and sweep recordings.

## Files

| Path | What |
|---|---|
| `roomsense.py`, `pipeline.py` | Inference — numpy/scipy only |
| `roomsense_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [University of Surrey RealRoomBRIRs](https://github.com/IoSR-Surrey/RealRoomBRIRs) | Measured dev and exam swept responses, geometry and published midband cross-check | MIT notice shipped by the provider, reproduced beside the excerpts. |
| [Kitamura Laboratory impulse response dataset](https://github.com/kitamura-laboratory/impResDataset) | Measured dev and exam swept responses and dev geometry | Public academic download. Audio redistribution terms unspecified. Metrics only are distributed. |
| [ESC-10, the attribution-licensed subset of ESC-50](https://github.com/karolpiczak/ESC-50) | Disjoint training, simulated validation, dev and exam background clips | CC BY or CC0 as attributed in examples/ATTRIBUTION.md and the source credits. |
| [Lai et al. ClapRIR phone-clap examples](https://github.com/Akinesia112/ClapRIR) | Post-lock input compatibility check, with no paired reverberation references | Public listening examples. The source lists no separate audio reuse terms. Metrics only are distributed. |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
