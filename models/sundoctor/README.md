<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# SunDoctor <sub>v2.0</sub>

> **A day-by-day doctor for solar farms, trained on sunlight physics alone.**

*Why did this solar array make less power today, and how much did each cause cost?*

SunDoctor reads two weeks of a PV system's power (with or without an irradiance sensor) and diagnoses it day by day: dead strings with their failure and repair days, shading, clipping, outages, soiling, snow and stuck trackers, plus the kWh lost to each. It learned only from a physics simulator and was examined on two real utility-scale farms.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/sundoctor/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/sundoctor/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | 14 days of 15-minute AC power, optionally plane-of-array irradiance and temperature, plus the array's location and layout. |
| **Returns** | Dead-string count per day, with failure and repair days<br>Daily loss share for outage, clipping, shading, soiling, snow, stuck trackers and dead strings (± error bars)<br>kWh lost to each cause |
| **Learned from** | Sun position and clear-sky irradiance, plane-of-array transposition for fixed and single-axis tracker mounts (with and without backtracking), row-to-row shading, module temperature, inverter clipping, and a library of faults injected on top. |
| **Noise** | Simulated weather and sensor faults (no instrument archive noise injection; this predates the real-noise recipe) |
| **Network** | 2-D CNN over 14 days × 96 solar-time steps, 12 input channels, Gaussian error bars on every output (591k weights) |
| **Dev set** | None separate — the frozen simulated set decided every rung; Colorado was partly seen (see limits) |
| **Exam** | PVDAQ 2107 (Arbuckle, California; 24 sibling inverters as the answer key) and PVDAQ 9068 (Kersey, Colorado; 4.7 MW, 224 trackers) |

## Run it

```bash
cd models/sundoctor
pip install numpy pandas
python demo.py        # about 15 s
```

or from anywhere in the repo: `mlratchet demo sundoctor`.

```python
import sundoctor
doc = sundoctor.SunDoctor()
r = doc.diagnose_csv("examples/pvdaq2107_inverter11_2019H2.csv", lat=38.996, lon=-122.134,
                     tilt=25, azimuth=180, dc_kw=37.2, tz="US/Pacific", n_strings=6)
print(r["summary"]["loss_pct_of_possible"])
```

## The exam

Two real farms, graded against independent answer keys (sibling inverters, the plant's own instruments). Every number here is on real data that never chose anything.

| Metric | SunDoctor | Baselines | |
|---|---|---|---|
| String failures caught within a day (California) | **55 / 56** | SunDoctor v1, power only (classical): 1 / 56 | ✅ wins |
| Dead strings counted right, all inverter-days | **95.9%** | — | ℹ️ |
| Inverters offline caught | **44 / 49** | SunDoctor v1 (classical): 0 / 49 | ✅ wins |
| Snow AUC (Colorado) | **0.98** | — | ✅ wins |
| Big tracker-loss days (> 10%) flagged | **77%** | SunDoctor v1 (classical): 0% | ✅ wins |
| Small tracker losses (2–5%) ranked, AUC | **0.5** | Coin flip (dumb baseline): 0.5 | ➖ tie |

### Where it wins

- Catches 55 of 56 real string failures within a day, with the irradiance sensor; 47 of 56 from power alone.
- Snow and outage AUC 0.98 on a snowy tracker farm it never trained on.

### Where it loses

- Small tracker losses (2–5%) can't be ranked (AUC 0.50).
- Power-only mode over-reports tracker losses (4.7% vs 1.1%) and dust.

## Limits

- A panel buried in snow all day looks like a tripped inverter; it sometimes calls that an outage.
- Colorado is not fully blind for rung 8 — the fronts change was prompted by rung 6's snow misses there.
- Two farms tested, both utility scale. Rooftops, microinverters, bifacial and satellite irradiance are untested.

## The ratchet

### v1 — fixed mounts

**Objective:** Sum of skills over dead strings, soiling, shading, clipping, outage and step detection on the frozen simulated set  
**Rule:** Keep if the objective rises by more than 0.01.

| Rung | Change | Frozen-set score | Decision | Why |
|---|---|---|---|---|
| s1 | Baseline 2-D CNN over 14 days × 96 solar-time steps, Gaussian error bars | 1.263 | ● kept | First rung. |
| s2 | β-NLL: each point's loss weighted by its own error bar, so rare large losses can't be waved away | 1.678 | ● kept | 1.678 vs 1.263. |
| s3 | Polish: 4,000 more steps at a lower learning rate | 1.748 | ● kept | 1.748 vs 1.678. Released as v1. |
| s4 | Longer memory: 28-day windows plus the system's own healthy level | 1.753 | ○ rejected | +0.005, short of the margin. |
| s5 | Realism: arrays drift from the physics prior; long dust seasons | 1.718 | ○ rejected | Score fell. |

### v2 — trackers, snow, no sensor

**Objective:** Mean of the with-sensor and no-sensor scores on a new frozen set of 2,000 fortnights (seed 778), half with no sensor  
**Rule:** Keep if the objective rises by more than 0.01.

| Rung | Change | Frozen-set score | Decision | Why |
|---|---|---|---|---|
| s3 | Start: the v1 release seen through v2 inputs | 0.341 | ● kept | Starting point. |
| s6 | v2 simulator: trackers, snow, stuck rows, thin film, sensor faults, no-sensor fortnights; v1 weights transplanted | 1.579 | ● kept | 1.579 vs 0.341. |
| s7 | Polish: 4,000 more steps at a third of the learning rate | 1.615 | ● kept | 1.615 vs 1.579. |
| s8 | Weather fronts: day-to-day temperature swings, cold fronts on overcast days | 1.628 | **◆ released** | 1.628 vs 1.615. |

The answer keys were fixed twice (outages not graded on cold dim days; snow graded per module). Every rung was re-scored; no decision changed.

Raw per-rung logs are in [`training/`](training/).

## Physics

Sun position and clear-sky irradiance, plane-of-array transposition for fixed and single-axis tracker mounts (with and without backtracking), row-to-row shading, module temperature, inverter clipping, and a library of faults injected on top.

**Simulator**

- Fixed and single-axis tracker mounts; true-tracking rows shade each other
- Snow that falls on cold overcast days and slides off; stuck and stowed tracker rows
- Soiling, shading, clipping, outages, dead strings, CdTe thin film
- Sensor misalignment, dew dropouts, and 40% of fortnights with no sensor at all
- Weather fronts (rung 8) — cold fronts on overcast days

**Checked against independent references**

- Every block checked against pvlib 0.16 — transposition within 0.003%, tracker rotation to 3e-13°
- Row shading matches pvlib shaded_fraction1d to 1e-15

## Next rungs

- Weather context (snowfall, precipitation) to separate buried days from trips.
- A better soiling signal from longer windows plus rain data.
- A real rooftop dataset.
- A money view — dollars lost per cause.

## Files

| Path | What |
|---|---|
| `sundoctor.py` | Inference — numpy/scipy only |
| `sundoctor_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [NREL PVDAQ via OEDI (systems 2107 and 9068)](https://data.openei.org/submissions/4568) | exam | CC BY 4.0 |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
