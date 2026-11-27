<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# AirfoilNet <sub>v1.0</sub>

> **The wind tunnel: an airfoil polar from its shape, graded on real tunnel data.**

*What lift, drag and moment curves will this airfoil have, and where does it stall?*

From an airfoil's coordinates and a Reynolds number (4e4-6e5), AirfoilNet predicts Cl, Cd and Cm against angle of attack, with 68% error bars, and from them Cl max and the stall angle. It learned only from 151 alpha sweeps of mfoil, an XFOIL-class viscous-inviscid solver, on airfoil shapes from the UIUC database with every tested airfoil removed. It was examined once on the clean wind-tunnel polars of 75 airfoils from 34 families in the UIUC Low-Speed Airfoil Tests. It beats NeuralFoil on maximum lift and stall angle and loses to it on lift before stall, drag and moment.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/airfoilnet/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/airfoilnet/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Airfoil coordinates (any .dat, Selig or Lednicer; refitted with an 18-parameter CST), Reynolds number 4e4-6e5, angles of attack (deg). Incompressible, clean (free transition). |
| **Returns** | Cl(alpha) +- 68% bar<br>Cd(alpha) with a bar on ln Cd<br>Cm about c/4 (alpha) +- 68% bar<br>Cl max and stall angle from the predicted polar |
| **Learned from** | XFOIL-class viscous-inviscid analysis: linear-vorticity panel method with transpiration, two-equation integral boundary layer (Drela closures), e^N transition (Ncrit 9), lag-entrainment turbulent shear, wake, all coupled in one global Newton solve (mfoil by K. Fidkowski). |
| **Noise** | None injected: no public record of tunnel noise for these polars. Scatter comes from the solver itself (loose-convergence points, different panelings). |
| **Network** | MLP on polars: 50 shape/Re features + 3 alpha features -> 3 hidden layers of 128 (SiLU) -> Cl, ln Cd, Cm and their log-sigmas; beta-NLL; 3 seeds averaged (3 x 40,710 parameters) |
| **Physics also enters** | inputs: CST thickness and camber at 24 stations (a shape fit, not a solver output) |
| **Real data's jobs** | calibration, gate, error bars, exam (never training labels) |
| **Dev set** | 7 airfoils from 4 families drawn by seed 20261122 from families with <= 3 airfoils: A18; FX 63-137, FX 74-CL5-140 MOD; M06-13-128, M6 65%, M6 85%; Trainer 60 (splits.json) |
| **Exam** | All other 75 tested airfoils (34 families), every clean run in vols 1-5, Re 3e4-5e5 |

## Run it

```bash
cd models/airfoilnet
pip install numpy scipy
python demo.py        # about 1 s
```

or from anywhere in the repo: `mlratchet demo airfoilnet`.

```python
from airfoilnet import AirfoilNet, read_dat
net = AirfoilNet()
pol = net.polar(read_dat("my_airfoil.dat"), Re=2e5)
print(pol["clmax"], pol["alpha_stall"], pol["cl"], pol["cl_sd"], pol["cd"], pol["cm"])
```

## The exam

Mean absolute errors per run, then per airfoil, then over airfoils, on all clean UIUC runs of 75 exam airfoils; the raw solver on a paired one-Re-per-airfoil subset. Every number here is on real data that never chose anything.

| Metric | AirfoilNet | Baselines | |
|---|---|---|---|
| Cl before stall, mean \|error\| | **0.091 Cl** | NeuralFoil 0.3.3 xxxlarge (trained on XFOIL) (published): 0.082 Cl<br>Thin-airfoil theory (classical): 0.195 Cl<br>Mean dev polar (dumb baseline): 0.196 Cl | ❌ loses |
| Cl after stall (up to +10 deg), mean \|error\| | **0.208 Cl** | NeuralFoil 0.3.3 xxxlarge (trained on XFOIL) (published): 0.165 Cl<br>Mean dev polar (dumb baseline): 0.292 Cl | ❌ loses |
| Cl max, mean \|error\| | **0.082 Cl** | NeuralFoil 0.3.3 xxxlarge (trained on XFOIL) (published): 0.096 Cl<br>Mean dev polar (dumb baseline): 0.258 Cl | ✅ wins |
| Stall angle, mean \|error\| | **1.76 deg** | NeuralFoil 0.3.3 xxxlarge (trained on XFOIL) (published): 2.19 deg<br>Mean dev polar (dumb baseline): 7.85 deg | ✅ wins |
| Drag before stall, mean \|ln(Cd_pred/Cd)\| | **0.177 ln ratio** | NeuralFoil 0.3.3 xxxlarge (trained on XFOIL) (published): 0.126 ln ratio<br>Mean dev polar (dumb baseline): 0.377 ln ratio | ❌ loses |
| Cd at Cl = 0.5, mean \|ln ratio\| | **0.175 ln ratio** | NeuralFoil 0.3.3 xxxlarge (trained on XFOIL) (published): 0.108 ln ratio<br>Mean dev polar (dumb baseline): 0.406 ln ratio | ❌ loses |
| Cm about c/4 before stall, mean \|error\| | **0.0173 Cm** | NeuralFoil 0.3.3 xxxlarge (trained on XFOIL) (published): 0.0153 Cm<br>Thin-airfoil theory (classical): 0.0256 Cm<br>Mean dev polar (dumb baseline): 0.0432 Cm | ❌ loses |
| Cl max, mean \|error\| (runs the raw solver was run on)<br><sub>Paired subset: one Reynolds number per exam airfoil, 74 airfoils. The raw solver answered 61% of measured angles.</sub> | **0.084 Cl** | NeuralFoil 0.3.3 xxxlarge (trained on XFOIL) (published): 0.102 Cl<br>Raw solver (mfoil, Ncrit 9) (classical): 0.223 Cl<br>Mean dev polar (dumb baseline): 0.248 Cl | ✅ wins |
| Cl before stall, mean \|error\| (runs the raw solver was run on)<br><sub>Paired subset. Where it converges the raw solver is the most accurate method before stall.</sub> | **0.084 Cl** | NeuralFoil 0.3.3 xxxlarge (trained on XFOIL) (published): 0.077 Cl<br>Raw solver (mfoil, Ncrit 9) (classical): 0.07 Cl<br>Thin-airfoil theory (classical): 0.186 Cl<br>Mean dev polar (dumb baseline): 0.194 Cl | ❌ loses |

Coverage of the 68% error bars on the exam: Cl (before stall) **69%**, ln Cd **74%**, Cm **83%**.

### Where it wins

- Cl max: 0.082 mean error against 0.096 for NeuralFoil and 0.22 for the raw solver.
- Stall angle: 1.76 deg against 2.19 for NeuralFoil and 4.0 for the raw solver, which stops converging before stall.

### Where it loses

- Lift before stall (0.091 vs 0.082), drag (|ln ratio| 0.177 vs 0.126) and moment (0.0173 vs 0.0153) all lose to NeuralFoil, trained on far more XFOIL runs.
- Where the raw solver converges it beats both networks before stall (Cl 0.071).

## Limits

- Low Reynolds numbers (4e4-6e5), incompressible, clean airfoils only: no trips, flaps or roughness.
- Trained on 151 sweeps (1,305 points); more simulations improved the simulated score but not the dev airfoils.
- The solver gives no answer near stall on many airfoils, so stall was learned from incomplete polars; predicted stall is 1.3 deg early on average.
- Error bars: Cm bars widened x4.75 on dev cover 83% (too wide); Cl and Cd cover 69% and 74%.

## The ratchet

**Objective:** Mean over Cl, ln Cd, Cm of (1 - MAE / MAE of a constant guess) on the frozen simulated set  
**Rule:** Keep if the frozen-set skill beats the incumbent by more than 0.01 AND the dev error D = mean(Cl_pre/0.10, lnCd/0.20, Clmax/0.20, stall/3 deg) is at most 1.05 x the incumbent.

| Rung | Change | Frozen-set score | Decision | Why | val1 | val2 | dev D | dev Cl pre | dev lnCd | dev Clmax | dev stall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 | MLP on polars: shape sensors + log Re + alpha -> Cl, ln Cd, Cm with log-sigmas; 151 sweeps (snapshot A) | 0.751 | **◆ released** | First rung; never beaten. | 0.7646 | 0.7506 | 1.0329 | 0.142 | 0.2714 | 0.1189 | 2.2808 |
| r2 | DeepONet (branch: shape + Re, trunk: alpha), same data | 0.714 | ○ rejected | val2: sim -0.037 not > +0.01; dev D 1.214 > 1.05 x 1.033. val1: sim -0.038 not > +0.01; dev D 1.214 > 1.05 x 1.033 | 0.7266 | 0.714 | 1.2137 | 0.1568 | 0.2849 | 0.1873 | 2.7791 |
| r3 | DeepONet predicting corrections to thin-airfoil Cl and Cm | 0.691 | ○ rejected | val2: sim -0.059 not > +0.01; dev D 1.553 > 1.05 x 1.033. val1: sim -0.063 not > +0.01; dev D 1.553 > 1.05 x 1.033 | 0.702 | 0.6911 | 1.553 | 0.1912 | 0.2806 | 0.3539 | 3.384 |
| r4 | r3 trained on strictly converged points only (drop loose Newton limit cycles) | 0.685 | ○ rejected | val2: sim -0.065 not > +0.01; dev D 1.602 > 1.05 x 1.033. val1: sim -0.067 not > +0.01; dev D 1.602 > 1.05 x 1.033 | 0.6975 | 0.6853 | 1.6016 | 0.1857 | 0.2931 | 0.426 | 2.8634 |
| r3b | MLP predicting corrections to thin-airfoil Cl and Cm | 0.735 | ○ rejected | val2: sim -0.016 not > +0.01; dev D 1.205 > 1.05 x 1.033. val1: sim -0.015 not > +0.01; dev D 1.205 > 1.05 x 1.033 | 0.7496 | 0.7345 | 1.2048 | 0.1561 | 0.3207 | 0.1509 | 2.7013 |
| r5 | Physics p1c: dev-measured drag mismatch (ln Cd += 0.015 + 0.21 (log10 Re - 5)) built into the labels | 0.757 | ○ rejected | val2: sim +0.006 not > +0.01. val1: sim -0.009 not > +0.01 | 0.7556 | 0.7569 | 1.0139 | 0.1445 | 0.2662 | 0.1097 | 2.1944 |
| r6 | MLP on 256 sweeps (snapshot B) | 0.758 | ○ rejected | val2: sim +0.008 not > +0.01; dev D 1.223 > 1.05 x 1.033. val1: dev D 1.223 > 1.05 x 1.033 | 0.7746 | 0.7583 | 1.2232 | 0.1562 | 0.3312 | 0.1509 | 2.7621 |
| r7 | Physics p1c and 256 sweeps | 0.776 | ○ rejected | val2: dev D 1.120 > 1.05 x 1.033. val1: sim +0.009 not > +0.01; dev D 1.120 > 1.05 x 1.033 | 0.7735 | 0.776 | 1.1197 | 0.1605 | 0.2865 | 0.1286 | 2.3941 |

<sub>Columns after *Why*: metrics: val1 skill, val2 skill, dev D, then dev Cl_pre, ln Cd, Cl max, stall angle (deg) — real dev airfoils, the gate input.</sub>

val1 = 50 sweeps of held-out database shapes with physics p1; val2 = the same sweeps with physics p1c labels. Every decision was re-taken on val2 in order; none changed. sim = val2 skill.

Raw per-rung logs are in [`training/`](training/).

## Physics

XFOIL-class viscous-inviscid analysis: linear-vorticity panel method with transpiration, two-equation integral boundary layer (Drela closures), e^N transition (Ncrit 9), lag-entrainment turbulent shear, wake, all coupled in one global Newton solve (mfoil by K. Fidkowski).

**Simulator**

- mfoil v2023-06-28 (MIT) with numba kernels (same arithmetic, 1e-13 agreement), sharp-trailing-edge and Newton limit-cycle fixes
- Shapes: 1,283 UIUC database airfoils (tested airfoils and near-duplicates removed) plus blends, CST-smoothed, 120 panels
- Re log-uniform 4e4-6e5; alpha sweeps in 2-deg steps with warm starts; points flagged when only loosely converged
- Physics p1c (dev-fitted drag-vs-Re correction) was tried and rejected by the ratchet

**Checked against independent references**

- Panel method vs exact Joukowski airfoil: Cl within 0.10-0.22%, Cp mean abs error 0.0012 (training/verify_physics.json)
- Thin-airfoil theory, NACA 2404: zero-lift angle -2.096 vs -2.077 deg; lift slope 1.031 x 2pi (thickness rule 1.031)
- Published XFOIL 6.99 polars (airfoiltools.com, Ncrit 9), NACA 0012/2412/4412, 42 points: Cl 0.0037, Cd median 0.5%, Cm 0.0007
- NACA 2412 at Re 3e6: aL0 -2.21 deg, slope 0.111/deg, Clmax 1.75, Cdmin 0.0051 vs book values ~-2.0, ~0.105, ~1.6, ~0.006
- Numba port vs original mfoil: Cl, Cd, Cm within 4e-13 (training/verify_fast.json)

## Help wanted

- XFOIL itself (needs gfortran, not installable on the build machine); websites.umich.edu (mfoil home) and github.com answered 403 from the build machine.
- Coordinates (audit, 25 Nov 2026; fixed 27 Nov 2026): polar(coords=...) assumed Selig order, so Lednicer blocks passed directly moved lift by up to 5.44; they are now reordered (the four packaged airfoils give identical lift either way). Reflecting a shape and the angle should reverse lift; the network's sum stays 0.20-1.13 within -6 to +6°, a symmetry it was not trained to keep.

## Next rungs

- Stall: a solver that converges past separation (XFOIL proper, or mfoil with a better Newton globalisation) so the bank has post-stall polars.
- A bigger bank on a free machine (the 2-core build ran 30-50 s per sweep).
- Ncrit and tunnel-turbulence nuisance (draw Ncrit per sweep), judged on the dev twins.

## Files

| Path | What |
|---|---|
| `airfoilnet.py` | Inference — numpy/scipy only |
| `airfoilnet_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [UIUC Low-Speed Airfoil Tests, Summary of Low-Speed Airfoil Data vols 1-5 (Selig et al. 1995-2012)](https://m-selig.ae.illinois.edu/uiuc_lsat.html) | dev, exam | GNU GPL v2 (stated in the data volumes) |
| [UIUC Airfoil Coordinates Database](https://m-selig.ae.illinois.edu/ads/coord_database.html) | physics (training shapes) | No explicit licence; compiled from many sources; cited |
| [mfoil (K. J. Fidkowski), via the pymfoil 0.1.0 PyPI package](https://pypi.org/project/pymfoil/) | physics | MIT |
| [NeuralFoil 0.3.3 weights (P. Sharpe)](https://pypi.org/project/neuralfoil/) | baseline | MIT |
| [XFOIL polars published on airfoiltools.com](http://airfoiltools.com/) | physics check | Used for verification only, not redistributed |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
