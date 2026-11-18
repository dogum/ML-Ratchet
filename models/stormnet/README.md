<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# StormNet <sub>v1.0.1</sub>

> **Three hours ahead, from the solar wind to the ring current.**

*How low will the geomagnetic Dst index fall in the next three hours?*

StormNet reads 36 completed hours of solar wind and the last hourly Dst and forecasts the lowest Dst of the next three hours. On 76,507 real forecasts across solar minimum and maximum, including the May 2024 superstorm, it beats persistence by a wide margin (storm-strength targets 10.7 against 14.2 nT at solar maximum) but only ties ring-current physics tuned on the same dev storms (10.7 against 10.6 nT), and loses on the most extreme hours and on quiet false alarms. Its 68% bars, now widened by how deep the storm already is, cover 60% of storm-strength targets.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/stormnet/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/stormnet/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | 36 hourly values of GSM Bz (nT), bulk speed (km/s) and dynamic pressure (nPa), plus the last completed hourly Dst (nT). |
| **Returns** | Lowest hourly Dst in the following three complete hours (nT) and its nominal 68% half-width (nT) |
| **Learned from** | Pressure-corrected current injection from southward solar wind with fast and slow exponential recovery. |
| **Noise** | Actual unlabelled training-era wind supplies joint Bz, speed, pressure and measurement gaps. Synthetic autoregressive disturbance and parameter variation supply index scatter. Real Dst is used for dev calibration, gates and evaluation, and as the last observed scalar at inference. |
| **Network** | Two recurrent physical states plus a simulation-trained nonlinear residual and log-error scale. 15 causal features, 48 fixed random tanh units, 128 fitted output coefficients, 926 stored scalars including normalization. Standalone NumPy inference. |
| **Physics also enters** | inputs: Southward motional electric field, square-root pressure, rolling wind means and fast/slow current estimates<br>output: Dev-tuned two-state forecast conditioned on the last completed Dst |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | 34 whole storm or quiet blocks, 2,408 issues in 2008, 2011 and 2014. Frozen seed 1729, each year gated separately. |
| **Exam** | 2016–2020, all 43,439 eligible issue hours in 629 blocks. The later 2022–2025 reserve is sealed. |

## Run it

```bash
cd models/stormnet
pip install numpy
python demo.py        # about 1 s
```

or from anywhere in the repo: `mlratchet demo stormnet`.

```python
import numpy as np
from stormnet import StormNet
with np.load("examples/storm.npz") as event:
    print(StormNet().predict(event["wind"][-1], event["last_dst"][-1]))
```

## The exam

Three-hour minimum MAE in nT and quiet false alarms against four competitors, on two exams: the primary 2016–2020 exam (solar minimum, 51 storm blocks) and the sealed 2022–2025 reserve (solar maximum, 85 storm blocks including the May 2024 superstorm at −406 nT), opened once after the v1.0.1 decisions. Paired 95% intervals use 2,000 whole-block resamples. Exam 1 rows score v1.0; v1.0.1 changes only the error bars. Every number here is on real data that never chose anything.

| Metric | StormNet | Baselines | |
|---|---|---|---|
| Exam 1, 2016–2020 · All eligible hours · minimum Dst MAE<br><sub>43,439 issue hours in 629 whole blocks. Model minus best baseline +0.019, paired 95% interval [-0.011, +0.048].</sub> | **3.466 nT** | Persistence (dumb baseline): 3.789 nT<br>Published O'Brien–McPherron (classical): 3.447 nT<br>Dev-tuned O'Brien–McPherron (classical): 3.478 nT<br>Dev-tuned two-state (classical): 3.516 nT | ➖ tie |
| Exam 1, 2016–2020 · Storm-strength (≤ −50 nT) · MAE<br><sub>811 issue hours in 51 whole blocks. Model minus best baseline +0.149, paired 95% interval [-0.221, +0.519].</sub> | **9.636 nT** | Persistence (dumb baseline): 12.587 nT<br>Published O'Brien–McPherron (classical): 10.479 nT<br>Dev-tuned O'Brien–McPherron (classical): 9.534 nT<br>Dev-tuned two-state (classical): 9.487 nT | ➖ tie |
| Exam 1, 2016–2020 · Intense (≤ −100 nT) · MAE<br><sub>57 issue hours in 5 whole blocks. Model minus best baseline -0.136, paired 95% interval [-2.077, +2.045].</sub> | **17.763 nT** | Persistence (dumb baseline): 23.053 nT<br>Published O'Brien–McPherron (classical): 18.368 nT<br>Dev-tuned O'Brien–McPherron (classical): 17.899 nT<br>Dev-tuned two-state (classical): 18.152 nT | ➖ tie |
| Exam 1, 2016–2020 · Extreme (≤ −150 nT) · MAE<br><sub>7 issue hours from one independent storm. A between-storm interval is unidentifiable.</sub> | **26.078 nT** | Persistence (dumb baseline): 33.286 nT<br>Published O'Brien–McPherron (classical): 33.096 nT<br>Dev-tuned O'Brien–McPherron (classical): 27.379 nT<br>Dev-tuned two-state (classical): 27.988 nT | ℹ️ |
| Exam 1, 2016–2020 · Quiet targets (> −50 nT) · false alarms<br><sub>42,628 issue hours in 629 whole blocks. Model minus best baseline +0.127, paired 95% interval [+0.078, +0.180].</sub> | **0.296%** | Persistence (dumb baseline): 0.232%<br>Published O'Brien–McPherron (classical): 0.169%<br>Dev-tuned O'Brien–McPherron (classical): 0.378%<br>Dev-tuned two-state (classical): 0.373% | ❌ loses |
| Exam 2, 2022–2025 · All eligible hours · minimum Dst MAE<br><sub>33,068 issue hours in 515 whole blocks (85 storm). Model minus best competitor +0.013, paired 95% interval [-0.029, +0.055]. Sealed reserve, opened once after the v1.0.1 decisions.</sub> | **4.356 nT** | Persistence (dumb baseline): 5.03 nT<br>Published O'Brien–McPherron (classical): 4.343 nT<br>Dev-tuned O'Brien–McPherron (classical): 4.582 nT<br>Dev-tuned two-state (classical): 4.654 nT | ➖ tie |
| Exam 2, 2022–2025 · Storm-strength (≤ −50 nT) · MAE<br><sub>1,992 issue hours in 85 whole blocks (85 storm). Model minus best competitor +0.074, paired 95% interval [-0.206, +0.341]. Sealed reserve, opened once after the v1.0.1 decisions.</sub> | **10.69 nT** | Persistence (dumb baseline): 14.198 nT<br>Published O'Brien–McPherron (classical): 11.676 nT<br>Dev-tuned O'Brien–McPherron (classical): 10.616 nT<br>Dev-tuned two-state (classical): 10.65 nT | ➖ tie |
| Exam 2, 2022–2025 · Intense (≤ −100 nT) · MAE<br><sub>310 issue hours in 21 whole blocks (21 storm). Model minus best competitor +0.564, paired 95% interval [-0.151, +1.238]. Sealed reserve, opened once after the v1.0.1 decisions.</sub> | **23.65 nT** | Persistence (dumb baseline): 30.206 nT<br>Published O'Brien–McPherron (classical): 26.745 nT<br>Dev-tuned O'Brien–McPherron (classical): 23.086 nT<br>Dev-tuned two-state (classical): 23.468 nT | ➖ tie |
| Exam 2, 2022–2025 · Extreme (≤ −150 nT) · MAE<br><sub>108 issue hours in 8 whole blocks (8 storm). Model minus best competitor +1.167, paired 95% interval [+0.070, +2.161]. Sealed reserve, opened once after the v1.0.1 decisions.</sub> | **36.158 nT** | Persistence (dumb baseline): 44.074 nT<br>Published O'Brien–McPherron (classical): 41.014 nT<br>Dev-tuned O'Brien–McPherron (classical): 34.991 nT<br>Dev-tuned two-state (classical): 35.882 nT | ❌ loses |
| Exam 2, 2022–2025 · Quiet targets (> −50 nT) · false alarms<br><sub>31,076 issue hours in 515 whole blocks (85 storm). Model minus best competitor +0.238, paired 95% interval [+0.178, +0.304]. Sealed reserve, opened once after the v1.0.1 decisions.</sub> | **0.615%** | Persistence (dumb baseline): 0.46%<br>Published O'Brien–McPherron (classical): 0.376%<br>Dev-tuned O'Brien–McPherron (classical): 0.875%<br>Dev-tuned two-state (classical): 0.917% | ❌ loses |

Coverage of the 68% error bars on the exam: Exam 1 all hours (v1.0 single factor) **71.7%**, Exam 1 storm-strength (v1.0) **49%**, Exam 2 all hours (v1.0.1) **63.2%**, Exam 2 storm-strength (v1.0.1; v1.0 51.5) **59.9%**, Exam 2 intense (v1.0.1; v1.0 41.9) **52.9%**, Exam 2 extreme (v1.0.1; v1.0 41.7) **50.9%**.

### Where it wins

- Beats persistence on every storm row of both exams: at solar maximum 10.69 against 14.20 nT on storm-strength targets and 23.65 against 30.21 nT on intense ones.
- Beats published O'Brien–McPherron physics on every storm row of exam 2 (storm-strength 10.69 against 11.68 nT, intense 23.65 against 26.75 nT).

### Where it loses

- Ties dev-tuned ring-current physics everywhere it can be measured; the network adds nothing over its own physics anchor on real storms. Two v1.1 physics candidates and a larger readout reading the window's observed Dst (r5) were tried on dev and rejected.
- Loses on the extreme (≤ −150 nT) hours of exam 2 (36.16 against 34.99 nT, 8 storm blocks, interval [+0.07, +2.16]) and on quiet false alarms in both exams (0.62% against 0.38% at solar maximum).

## Limits

- 68% bars still under-cover storm hours: 59.9% of storm-strength, 52.9% of intense and 50.9% of extreme targets in exam 2 with v1.0.1 (v1.0: 51.5, 41.9 and 41.7%).
- Retrospective Earth-arrival-aligned wind and the last completed historical Dst. Real telemetry latency and index revisions matter in operational use; 2022–2025 Dst in OMNI is Kyoto's provisional or quicklook index.
- The two-state fit reaches the lower pressure-coefficient bound. Simulated dev minima are typically 8–14 nT deeper than their real twins. Re-running the physics with the actual next three hours of wind barely helps (dev 4.36 → 4.21 nT), so the remaining error is the physics and the index's hourly noise, not the unknown wind.

## The ratchet

**Objective:** Mean MAE skill against persistence on all issues and active targets (Dst ≤ −50 nT)  
**Rule:** Gain > 0.01 on frozen skill and no dev year's mean error ratio more than 5% worse.

| Rung | Change | Frozen-set score | Decision | Why | 2008 | 2011 | 2014 |
|---|---|---|---|---|---|---|---|
| r1 | Instantaneous wind and last Dst, single-state teacher | 0.109 | ● kept | First rung. | 0.8188 | 0.8029 | 0.7826 |
| r2 | 36-hour causal state estimates, same single-state teacher | 0.120 | ● kept | +0.011 on the frozen set; dev gate passed. | 0.8342 | 0.7965 | 0.7825 |
| r3 | Dev-calibrated fast and slow teacher with a matched physics anchor | 0.134 | **◆ released** | +0.014 on the frozen set; dev gate passed. | 0.7883 | 0.7933 | 0.7396 |
| r4 | Equal mixture of single and two-state simulations | 0.129 | ○ rejected | -0.005 on val2, needs more than +0.01. | 0.8192 | 0.8003 | 0.7651 |
| r5 | The window's 36 hourly Dst values as inputs (trend, mean, minimum, assimilated fast share) and 256 random units instead of 48, two-state teacher; seeds 103/104/105 score 0.1283 / 0.1410 / 0.1283 | 0.133 | ○ rejected | -0.0015 on val2 against r3 (seed mean); needs more than +0.01. Every dev year is worse than r3. | 0.848 | 0.8343 | 0.7852 |

<sub>Columns after *Why*: Dev error ratios to persistence (lower is better).</sub>

17 Nov 2026: two ring-3 physics candidates were fitted on dev and rejected by the rule fixed beforehand (lower dev MAE on all issues and storm targets in every year): assimilating the fast/slow split from the window's observed Dst (storm MAE 11.77 -> 11.69 / 8.63 -> 8.62 / 6.61 -> 6.61 nT, all-issue MAE worse in 2011 and 2014) and pressure-dependent injection, gamma = 0.106 (storm 11.77 -> 11.33 / 8.63 -> 8.57 nT, 2014 worse: 6.61 -> 6.79). training/calibration_v3.json.

Raw per-rung logs are in [`training/`](training/).

## Physics

Pressure-corrected current injection from southward solar wind with fast and slow exponential recovery.

**Simulator**

- O’Brien–McPherron convection threshold and electric-field-dependent decay time
- Two effective current reservoirs, variable injection and lifetimes, pressure response and correlated unresolved disturbances
- Real historical OMNI wind trajectories with instrument gaps and a 48-hour synthetic warm-up

**Checked against independent references**

- 96-hour drive, shutoff and pressure jump versus independent DOP853 integration: maximum absolute error 8.53e-13 nT (training/physics_check.json)
- Closed-form 32-hour step within 8.53e-14 nT. Quiet 2 to 8 nPa pressure step adds 10.267 nT.
- 744 final hourly Kyoto Dst values in March 2011 match NASA OMNI word 41 exactly (training/index_check.json)

## Next rungs

- Teach the simulator something the anchor lacks, so the network has real structure to learn: a substorm or By-dependent injection term, or a ring-current asymmetry term fitted on SYM-H, each shown failing on dev first.
- Extreme-storm bars: widen by the observed rate of fall as well as the level, validated on whole dev storms.
- Check forecasts on archived as-received L1 telemetry and quicklook Dst with explicit ingestion timestamps.

## Files

| Path | What |
|---|---|
| `stormnet.py`, `stormphysics.py` | Inference — numpy/scipy only |
| `stormnet_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [NASA SPDF OMNIWeb hourly solar-wind magnetic field and plasma](https://omniweb.gsfc.nasa.gov/html/ow_data.html) | Training and frozen wind, dev and primary forecast inputs | NASA SPDF public scientific data. Credit OMNIWeb, the contributing teams and King & Papitashvili (2005). |
| [World Data Center for Geomagnetism, Kyoto and contributing observatories, hourly Dst](https://doi.org/10.17593/14515-74000) | Dev simulator tuning, dev gates, uncertainty and primary exam. Historical index at inference. | Scientific use with acknowledgement and DOI. Kyoto prohibits commercial application of the indices. |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
