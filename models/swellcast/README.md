<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# SwellCast <sub>v1.0</sub>

> **One buoy hears a storm thousands of kilometres away.**

*Is swell from a distant storm arriving at this buoy — and how far away, when, and from which direction was the storm?*

From ten days of NOAA buoy wave spectra, SwellCast says whether storm swell is arriving, how far away the storm was, when it peaked and which way it lies. It is Munk's 1963 idea — long waves outrun short ones, so the arriving frequency rises linearly in time at a rate set by the distance — learned by a network from simulated storms laid onto real buoy records.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/swellcast/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/swellcast/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | 240 hours of NOAA NDBC spectral wave data (energy and first directional moments) from one buoy. |
| **Returns** | Swell detected (yes/no with a score)<br>Storm distance (km) with a 68% range<br>Time of the storm's peak ± hours<br>Direction to the storm ± degrees, and the storm's wave height |
| **Learned from** | Deep-water dispersion (group speed g/4πf) from a moving, growing and decaying JONSWAP storm patch, spherical spreading on the great circle, Ardhuin et al. 2009 swell dissipation, directional moments, and the buoy's own estimation noise and quantisation. |
| **Noise** | Quiet 240-hour windows from 27 NDBC buoys (2008–2025); 25% of each batch is a real null window |
| **Network** | 2-D CNN over frequency × time (log energy, anomaly, two directional moments, data mask), 4 residual stages, 296k parameters; β-NLL and von Mises heads |
| **Dev set** | 18 storms (seed 2026), 29 confirmed arrivals |
| **Exam** | 288 confirmed arrivals (632 visible, 2,081 storm–buoy pairs in all) |

## Run it

```bash
cd models/swellcast
pip install numpy scipy
python demo.py        # about 2 s
```

or from anywhere in the repo: `mlratchet demo swellcast`.

```python
import pipeline as PL, swellcast as SC
rec = PL.read_ndbc("51004w2018.txt.gz", "51004d2018.txt.gz", "51004j2018.txt.gz")
win, centre = PL.window(rec, hour)
r = SC.SwellCast().read(win)
print(r["detected"], r["D_km"], r["t0"], r["dir"])
```

## The exam

288 confirmed storm-swell arrivals. Every number here is on real data that never chose anything.

| Metric | SwellCast | Baselines | |
|---|---|---|---|
| Distance, median error | **38%** | Classical ridge fit f = a(t − t0) (classical): 59%<br>Dev median guess (dumb baseline): 28% | ❌ loses |
| Storm time, median error | **33 h** | Classical ridge fit (classical): 48 h<br>Dev median guess (dumb baseline): 24 h | ❌ loses |
| Direction, median error | **19 deg** | Classical ridge fit (classical): 20 deg<br>Dev median guess (dumb baseline): 56 deg | ✅ wins |
| Arrivals detected | **63%** | Classical ridge fit (classical): 43% | ✅ wins |

Coverage of the 68% error bars on the exam: Distance **67%**, Time **74%**, Direction **70%**.

### Where it wins

- Beats the classical ridge fit on distance, time and detection; ties it on direction; calibrated error bars.

### Where it loses

- Doesn't beat the dumb median guess on distance or time — the exam geography is narrow and the window is centred within ±36 h of the arrival.

## Limits

- Truth is tropical cyclones only; Southern Ocean and North Pacific winter storms are not graded.
- One-point truth (peak wind) for a storm that makes swell for days along a track.
- The calibrated v2/v3 simulators match dev statistics better but alone make real dev readings worse; the cause is not understood.

## The ratchet

**Objective:** Mean skill over log distance, storm time, direction, log Hs and detection on the frozen simulated set  
**Rule:** Keep if the objective rises by more than 0.01 AND dev errors get no more than 5% worse.

| Rung | Change | Frozen-set score | Decision | Why | Distance | Time | Direction |
|---|---|---|---|---|---|---|---|
| r1 | 296k CNN, 30k v1 simulations (textbook priors) | 0.386 | ● kept | First rung. | 40% | 35 h | 16° |
| r2 | v2 simulations calibrated on dev: weaker, longer-lived storms; shorter periods; crossing swells | 0.442 | ○ rejected | Dev gate: errors +30%. | 58% | 50 h | 31° |
| r3 | v3: 65% of storms follow real IBTrACS 1985–2007 tracks | 0.582 | ○ rejected | Dev gate: errors +21%. | 46% | 56 h | 24° |
| r4 | Mix of v1 + v2 + v3 banks (54k), fine-tuned from rung 1 | 0.536 | **◆ released** | +0.150 over rung 1; dev +2%. | 37% | 37 h | 22° |
| r5 | Rung 4 + 3,000 steps at half the learning rate | 0.545 | ○ rejected | +0.009, short of the margin. | 43% | 36 h | 19° |

<sub>Columns after *Why*: Real dev set (18 storms), median errors — the gate's input.</sub>

Calibrated simulators (v2, v3) matched dev statistics better but alone made real dev readings worse. Mixing all three banks fixed it.

Raw per-rung logs are in [`training/`](training/).

## Physics

Deep-water dispersion (group speed g/4πf) from a moving, growing and decaying JONSWAP storm patch, spherical spreading on the great circle, Ardhuin et al. 2009 swell dissipation, directional moments, and the buoy's own estimation noise and quantisation.

**Simulator**

- Moving, growing, decaying JONSWAP storm patch (7 sub-areas), partly beamed along the storm heading
- Great-circle spherical spreading and Ardhuin 2009 dissipation
- χ²(24) estimation noise and NDBC 0.01 m²/Hz quantisation
- v3: 65% of storms follow real IBTrACS 1985–2007 hurricane tracks at their real place relative to the buoy

**Checked against independent references**

- Ridge slope vs g/(4πD) within 1.4% at 1,500 km and 0.1% at 9,000 km; energy received/emitted 1.0003
- Hurricane Marie 2014 (not in dev or exam): the real ridge at 4 buoys runs parallel to the predicted one

## Next rungs

- Sliding-window search without the ±36 h centring hint — the honest monitoring task.
- Track-level truth and loss (the energy-weighted source), or a hybrid swap test like QuakeNet's.
- Two-buoy triangulation, and extratropical truth from reanalysis.

## Files

| Path | What |
|---|---|
| `swellcast.py`, `pipeline.py` | Inference — numpy/scipy only |
| `swellcast_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [NOAA National Data Buoy Center spectral wave data](https://www.ndbc.noaa.gov/) | noise, dev, exam | Public domain (U.S. Government) |
| [IBTrACS v04r01 (NOAA NCEI)](https://www.ncei.noaa.gov/products/international-best-track-archive) | truth, physics | Public domain; cite Knapp et al. 2010 |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
