<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# TransitNet <sub>v1.0</sub>

> **Planet or impostor? Trained on zero real planets.**

*Is this periodic dip in a TESS light curve a planet, and if so how big is it?*

Given a TESS light curve and a candidate period, TransitNet decides whether the dip is a planet, an eclipsing binary, a background blend or nothing, and measures the planet's size, impact parameter and orbit scale with error bars. It was trained on simulated transits injected into real TESS light curves of ordinary stars, and examined on 1,196 confirmed planets and 832 known false positives.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/transitnet/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/transitnet/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | TESS SPOC 2-minute light curves (one or more sectors), a candidate period, epoch and duration, and the star's catalogue values. |
| **Returns** | Probability of planet, eclipsing binary, background binary, nothing<br>Radius ratio Rp/R* with a 68% range (and Earth radii when the star's size is known)<br>Impact parameter b and a/R*, with ranges |
| **Learned from** | Exact quadratic limb-darkened occultation, Keplerian (eccentric) orbits, eclipsing binaries, background binaries diluted by third light, and centroid-shift physics — all injected into real TESS light curves. |
| **Noise** | 2,200 real 2-minute TESS light curves of ordinary stars, 220 from each of 10 sectors across the mission |
| **Network** | Multi-view 1-D CNN (local, secondary, global, odd/even views + scalars) predicting corrections to model-free estimates |
| **Dev set** | 20 planets and 20 false positives (seed 2026); their host stars are excluded from the exam |
| **Exam** | 1,625 confirmed planets, 992 follow-up false positives, 77 false alarms |

## Run it

```bash
cd models/transitnet
pip install numpy scipy
python demo.py        # about 3 s
```

or from anywhere in the repo: `mlratchet demo transitnet`.

```python
import transitnet as tn
net = tn.TransitNet()
sectors = [tn.load_spoc(p) for p in my_spoc_fits_files]
r = net.read(sectors, period=5.66, epoch=1386.27, duration_hours=1.7)
print(r["p_planet"], r["k"], r["b"], r["aR"])
```

## The exam

1,196 planets and 832 false positives with SNR ≥ 7. Every number here is on real data that never chose anything.

| Metric | TransitNet | Baselines | |
|---|---|---|---|
| Planet vs false positive, AUC | **0.835** | — | ℹ️ |
| Rp/R* median error, SNR 100+ | **1.6%** | Least-squares transit fit (classical): 2.3%<br>√depth (dumb baseline): 4.4% | ✅ wins |
| Rp/R* median error, SNR 20–100 | **3.5%** | Least-squares transit fit (classical): 4.7%<br>√depth (dumb baseline): 5.7% | ✅ wins |
| Rp/R* median error, SNR 7–20 | **6.7%** | Least-squares transit fit (classical): 10.7%<br>√depth (dumb baseline): 6.8% | ✅ wins |
| a/R* median error, SNR 100+ | **4.2%** | Least-squares transit fit (classical): 3.7% | ❌ loses |
| False positives also called planet | **66%** | — | ❌ loses |

Coverage of the 68% error bars on the exam: Rp/R* **70%**, b **79%**, a/R* **71%**.

### Where it wins

- Measures planet size better than a least-squares fit at every SNR.
- Calls 98% of real planets planets, with zero real planets in training.

### Where it loses

- 66% of follow-up false positives are also called planet — photometry alone can't rule most of them out.
- Least-squares wins on a/R* at high SNR (3.7% vs 4.2%).

## Limits

- Some real planets score low (TOI-2109 b at 0.00, TOI-1518 b at 0.30).
- The candidates already passed SPOC vetting, so the false positives are the hard ones.
- Claret limb-darkening tables were unreachable; the prior comes from dev fits.

## The ratchet

**Objective:** 0.5 × (2·AUC − 1) + 0.5 × mean parameter skill, on a frozen 5,000-reading set (val2, v2 physics)  
**Rule:** Keep if the objective rises by more than 0.01 AND dev log-loss is no more than 0.05 worse AND the dev radius-ratio error no more than 1 point worse.

| Rung | Change | Frozen-set score | Decision | Why |
|---|---|---|---|---|
| r1 | Baseline multi-view CNN | 0.838 | ● kept | First rung. |
| r2 | Limb-darkening prior from dev planets, plus centroids in the data | 0.735 | ○ rejected | The absolute-k head failed to learn (k error 34%). |
| r3 | Predict corrections to model-free estimates (k from √depth, a/R* from P and T14), on rung 2 data | 0.859 | **◆ released** | 0.859 vs 0.838; removed a 3–5% k bias seen even on simulations. |
| r4 | Centroid branch | 0.861 | ○ rejected | +0.002 only, needed +0.01. |

Raw per-rung logs are in [`training/`](training/).

## Physics

Exact quadratic limb-darkened occultation, Keplerian (eccentric) orbits, eclipsing binaries, background binaries diluted by third light, and centroid-shift physics — all injected into real TESS light curves.

**Simulator**

- Exact quadratic limb-darkened transit (Gauss–Legendre, any radius ratio)
- Eclipsing binaries, background binaries (third light 3–3,000×), eccentric orbits, centroid shifts
- Limb-darkening prior recalibrated on free fits to the dev planets (q1 0.15–0.55)

**Checked against independent references**

- Matches batman to ≤ 0.94 ppm

## Next rungs

- A physical planet-size cap, low-mass stellar companions and R* as an input, against SB1-type false positives.
- Difference-image centroids instead of MOM_CENTR.
- Single transits, and an in-browser BLS "find it" step.

## Files

| Path | What |
|---|---|
| `transitnet.py`, `pipeline.py`, `transitsim.py` | Inference — numpy/scipy only |
| `transitnet_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [TESS SPOC light curves (MAST)](https://archive.stsci.edu/missions-and-data/tess) | noise, exam, dev | Public (NASA); cite TESS and MAST |
| [NASA Exoplanet Archive (pscomppars)](https://exoplanetarchive.ipac.caltech.edu/) | truth | Public; acknowledge the archive |
| [ExoFOP TOI list](https://exofop.ipac.caltech.edu/tess/) | truth | Public; acknowledge ExoFOP |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
