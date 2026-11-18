<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# StormNet <sub>v1.0</sub>

> **Three hours ahead, from the solar wind to the ring current.**

*How low will the geomagnetic Dst index fall in the next three hours?*

StormNet reads 36 completed hours of solar wind and the last hourly Dst. On 43,439 real forecasts it estimates the next three-hour minimum to 3.47 nT mean absolute error against 3.45 nT for published ring-current physics and 3.79 nT for holding the last index fixed. The difference to the strongest physics baseline spans zero. During storm-strength intervals its nominal 68% bars cover 49% of outcomes.

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

Three-hour minimum MAE in nT and quiet false alarms against four competitors. Paired 95% intervals use 2,000 whole-block resamples. Every number here is on real data that never chose anything.

| Metric | StormNet | Baselines | |
|---|---|---|---|
| All eligible hours · minimum Dst MAE<br><sub>43,439 issue hours in 629 whole blocks. Model minus best baseline +0.019, paired 95% interval [-0.011, +0.048].</sub> | **3.466 nT** | Persistence (dumb baseline): 3.789 nT<br>Published O'Brien–McPherron (classical): 3.447 nT<br>Dev-tuned O'Brien–McPherron (classical): 3.478 nT<br>Dev-tuned two-state (classical): 3.516 nT | ➖ tie |
| Storm-strength (≤ −50 nT) · MAE<br><sub>811 issue hours in 51 whole blocks. Model minus best baseline +0.149, paired 95% interval [-0.221, +0.519].</sub> | **9.636 nT** | Persistence (dumb baseline): 12.587 nT<br>Published O'Brien–McPherron (classical): 10.479 nT<br>Dev-tuned O'Brien–McPherron (classical): 9.534 nT<br>Dev-tuned two-state (classical): 9.487 nT | ➖ tie |
| Intense (≤ −100 nT) · MAE<br><sub>57 issue hours in 5 whole blocks. Model minus best baseline -0.136, paired 95% interval [-2.077, +2.045].</sub> | **17.763 nT** | Persistence (dumb baseline): 23.053 nT<br>Published O'Brien–McPherron (classical): 18.368 nT<br>Dev-tuned O'Brien–McPherron (classical): 17.899 nT<br>Dev-tuned two-state (classical): 18.152 nT | ➖ tie |
| Extreme (≤ −150 nT) · MAE<br><sub>7 issue hours from one independent storm. A between-storm interval is unidentifiable.</sub> | **26.078 nT** | Persistence (dumb baseline): 33.286 nT<br>Published O'Brien–McPherron (classical): 33.096 nT<br>Dev-tuned O'Brien–McPherron (classical): 27.379 nT<br>Dev-tuned two-state (classical): 27.988 nT | ℹ️ |
| Quiet targets (> −50 nT) · false alarms<br><sub>42,628 issue hours in 629 whole blocks. Model minus best baseline +0.127, paired 95% interval [+0.078, +0.180].</sub> | **0.296%** | Persistence (dumb baseline): 0.232%<br>Published O'Brien–McPherron (classical): 0.169%<br>Dev-tuned O'Brien–McPherron (classical): 0.378%<br>Dev-tuned two-state (classical): 0.373% | ❌ loses |

Coverage of the 68% error bars on the exam: All eligible hours **71.7%**, Storm-strength minimum **49%**, Intense minimum **40.4%**, Extreme minimum **42.9%**, Quiet targets (> −50 nT) **72.1%**.

### Where it wins

- Lower absolute error than persistence over all hours (3.47 vs 3.79 nT) and storm-strength targets (9.64 vs 12.59 nT). The best physical competitors tie on both rows.

### Where it loses

- Quiet false alarms: 0.296% against published physics at 0.169%, a difference of +0.127 percentage points [0.078, 0.180].
- The network has no decisive primary-exam win against the strongest baseline. Seven forecasts at or below −150 nT belong to a single storm.

## Limits

- Nominal 68% intervals cover 71.7% overall, 49.0% on storm-strength targets and 40.4% on intense targets.
- Retrospective Earth-arrival-aligned wind and the last completed historical Dst. Real telemetry latency and index revisions matter in operational use.
- The two-state fit reaches the lower pressure-coefficient bound. Simulated dev minima are typically 8–14 nT deeper than their real twins.

## The ratchet

**Objective:** Mean MAE skill against persistence on all issues and active targets (Dst ≤ −50 nT)  
**Rule:** Gain > 0.01 on frozen skill and no dev year's mean error ratio more than 5% worse.

| Rung | Change | Frozen-set score | Decision | Why | 2008 | 2011 | 2014 |
|---|---|---|---|---|---|---|---|
| r1 | Instantaneous wind and last Dst, single-state teacher | 0.109 | ● kept | First rung. | 0.8188 | 0.8029 | 0.7826 |
| r2 | 36-hour causal state estimates, same single-state teacher | 0.120 | ● kept | +0.011 on the frozen set; dev gate passed. | 0.8342 | 0.7965 | 0.7825 |
| r3 | Dev-calibrated fast and slow teacher with a matched physics anchor | 0.134 | **◆ released** | +0.014 on the frozen set; dev gate passed. | 0.7883 | 0.7933 | 0.7396 |
| r4 | Equal mixture of single and two-state simulations | 0.129 | ○ rejected | -0.005 on val2, needs more than +0.01. | 0.8192 | 0.8003 | 0.7651 |

<sub>Columns after *Why*: Dev error ratios to persistence (lower is better).</sub>

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

- Train on an ensemble of causal future wind paths and examine missed abrupt southward rotations on dev.
- Fit interval widening by issue-time electric-field strength and validate on whole dev storms.
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
