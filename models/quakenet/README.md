<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# QuakeNet <sub>v1.2</sub>

> **One seismometer, one earthquake.**

*How big was that earthquake, how far, how deep, and which way?*

From 41 minutes of three-component ground velocity at a single seismic station, QuakeNet estimates the earthquake's magnitude, distance, depth and direction, each with an error bar. It learned only from simulated earthquakes laid into real quiet-time station noise, and was examined on 3,105 real records.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/quakenet/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/quakenet/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | 41 minutes of three-component ground velocity (1 sample/s, response removed) from one GSN station, plus a rough P-wave time. |
| **Returns** | Moment magnitude Mw ± 68% bar<br>Epicentral distance (degrees) ± bar<br>Depth (km) with a bar on log depth<br>Back-azimuth (compass direction to the quake) ± bar |
| **Learned from** | Green's functions for a 1-D Earth (ak135f) on a 27-depth × 71-distance grid, combined locally into any moment tensor, plus measured corrections for what a 1-D Earth leaves out: slower surface waves, a scattering coda, and per-component surface-wave gains. |
| **Noise** | 1,284 real quiet-time windows from 127 GSN stations |
| **Network** | 1-D residual CNN, 7 stride-2 blocks, 1.44 M parameters; β-NLL on Mw, distance, log depth; von Mises NLL on back-azimuth |
| **Dev set** | 74 records from 40 events (fixed seed) |
| **Exam** | 3,105 records from 1,734 events, M ≥ 5.5, 2012–2025, distance 25–95° |

## Run it

```bash
cd models/quakenet
pip install numpy scipy
python demo.py        # about 3 s
```

or from anywhere in the repo: `mlratchet demo quakenet`.

```python
import numpy as np, json
from quakenet import QuakeNet
net = QuakeNet()
z = np.load("examples/us6000m0xl_IU_POHA.npz"); meta = json.loads(str(z["meta"]))
r = net.read(z["zne"], meta["p_guess"])
print(r["mw"], r["dist"], r["depth"], r["baz"])
```

## The exam

3,105 real one-station records. Every number here is on real data that never chose anything.

| Metric | QuakeNet | Baselines | |
|---|---|---|---|
| Magnitude, median \|error\| | **0.13 Mw** | Ms, dev-calibrated (classical): 0.57 Mw | ✅ wins |
| Distance, median \|error\| | **5.1 deg** | S−P time (classical): 13 deg | ✅ wins |
| Back-azimuth, median \|error\| | **6 deg** | P-wave polarisation (classical): 43.7 deg | ✅ wins |
| Depth within a factor of 2 | **46%** | Always guess 31 km (dumb baseline): 22% | ✅ wins |
| Shallow quakes (≤ 70 km), median depth \|error\| | **33 km** | Always guess 31 km (dumb baseline): 21 km | ❌ loses |

Coverage of the 68% error bars on the exam: Mw **63%**, Distance **66%**, Depth **51%**, Back-azimuth **68%**.

### Where it wins

- Magnitude, distance and direction from one station, far ahead of the classical one-station methods.
- Deep quakes (> 70 km): median depth error 52 km against 136 km for the guess; 82% within a factor of 2.

### Where it loses

- Shallow quakes are read too deep (median ratio 3.7 on the exam); always guessing 31 km beats it there.
- Very deep quakes (500–700 km) read too shallow (ratio 0.83).

## Limits

- Distances 25–95° only, magnitudes about 5.5–8, long-period (LH) data.
- One-station epicentre misses by a median 764 km.
- Depth error bars are too narrow (51% coverage for a 68% bar).
- Magnitude bias rose to +0.05 at rung 7; watch it.

## The ratchet

**Objective:** Mean skill over Mw, distance, log depth and back-azimuth on the frozen simulated set  
**Rule:** Keep if the simulated objective beats the incumbent by more than 0.01 AND the dev records get no more than 5% worse.

| Rung | Change | Frozen-set score | Decision | Why | Mw | Distance (°) | Depth (km, median) | Back-azimuth (°) |
|---|---|---|---|---|---|---|---|---|
| r1 | Textbook ak135 Earth | 0.556 | ● kept | First rung. | 0.21 | 11.0 | 110 | 8.7 |
| r2 | Slower surface waves — warp after S, median 1.065, measured on dev | 0.690 | ● kept | +0.134 on the frozen set; dev gate passed. | 0.15 | 7.4 | 72 | 8.0 |
| r3 | Scattering coda above 0.04 Hz, fitted on dev coda/P energy ratios | 0.808 | ● kept | +0.118; dev gate passed. | 0.14 | 5.7 | 52 | 6.6 |
| r4 | World mix: 70% of depth/Mw drawn from the 1990–2011 catalogue | 0.827 | ● kept | +0.019; dev gate passed. | 0.13 | 5.3 | 44 | 6.4 |
| r5 | Layered Earth — water-column reverberation for offshore sources, Moho Ps and sediment reverberation at the station | 0.838 | ● kept | Fell 0.002 short on val5 and was rejected at first; passed when every decision was re-taken on val7. | 0.13 | 5.2 | 42 | 6.2 |
| r6 | Weaker short-period surface waves on all components (calib_swq.py) | 0.839 | ○ rejected | +0.001 over rung 5 on val8, short of the 0.01 margin. | 0.13 | 5.2 | 43 | 6.1 |
| r7 | Surface waves per component (calib_love.py) plus short-period Rayleigh delay of 17 s per 1000 km | 0.869 | **◆ released** | +0.031 over rung 5; dev gate passed. | 0.13 | 5.1 | 39 | 6.0 |

<sub>Columns after *Why*: Real exam (3,105 records), reported after the fact; never used to choose.</sub>

Every rung was re-scored on each new frozen set and every decision re-taken in order.

Raw per-rung logs are in [`training/`](training/).

## Physics

Green's functions for a 1-D Earth (ak135f) on a 27-depth × 71-distance grid, combined locally into any moment tensor, plus measured corrections for what a 1-D Earth leaves out: slower surface waves, a scattering coda, and per-component surface-wave gains.

**Simulator**

- Syngine ak135f_2s Green's functions, 27 depths (2–700 km) × 71 distances (25–95°)
- Random double couple plus CLVD; source duration scaled to size
- Travel-time stretch, amplitude scatter, sensor misorientation
- Surface-wave slow-down, scattering coda and per-component gains, all measured on dev records
- 70% of sources drawn from the 1990–2011 catalogue (before every exam event)

**Checked against independent references**

- Local synthesis vs Syngine full synthetics: correlation 0.9999 on equatorial receivers

## Next rungs

- Shallow depth, continued — source-region structure, overtones, or a saliency map on real vs twin records.
- Multi-station reading, which turns the one-station locator into a real one.
- Regional distances (< 25°) and broadband channels for M4.5+.

## Files

| Path | What |
|---|---|
| `quakenet.py`, `pipeline.py` | Inference — numpy/scipy only |
| `quakenet_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [USGS ComCat catalogue and moment tensors](https://earthquake.usgs.gov/earthquakes/search/) | truth | Public domain (U.S. Government) |
| [EarthScope / IRIS DMC waveforms (IU and II networks)](https://ds.iris.edu/ds/nodes/dmc/) | exam, dev, noise | Open; cite the networks (IU, II) |
| [EarthScope Syngine Green's functions (ak135f)](https://ds.iris.edu/ds/products/syngine/) | physics | Open; cite Syngine |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
