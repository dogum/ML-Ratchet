<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# WakeNet <sub>v1.0</sub>

> **Who is stealing whose wind across a real farm — read as a directed graph.**

*How much power is each wind turbine losing to its neighbours' wakes?*

From a wind farm's turbine coordinates, hub heights, rotor power and thrust curves, mechanical yaw orientations, online flags, and the upstream wind measured on its unwaked front row, WakeNet predicts every downstream turbine's waked power ratio, power loss (kW and %), rotor-effective wind speed and farm-wide wake loss, each with a 68% error bar. It learned only from 24,000 procedural 5-to-16-turbine simulations laid into real unwaked turbulence from 2017–2018, and was examined once on 24,035 real 10-minute SCADA snapshots across two UK wind farms — including an unseen 14-turbine farm in Scotland.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/wakenet/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/wakenet/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Turbine coordinates (East, North, ground elevation in m), hub heights (m), rotor diameter and nominal P(U)/Ct(U) curves, each unit's online flag and mechanical nacelle yaw offset (deg), and upstream freestream wind speed (m/s), wind direction (deg), turbulence intensity, 10-min wind-direction standard deviation (deg), air density (kg/m3) and front-row mean power (kW) measured on the geometrically unwaked front row. Downstream wind speeds, vanes and power meters are never read. |
| **Returns** | Per-turbine waked power ratio P_i / P_front and wake loss (%) ± 68% bar<br>Per-turbine active power (kW) ± 68% bar and rotor-effective wind speed (m/s)<br>Farm-wide wake loss (%) and total waked output (MW) ± 68% bar |
| **Learned from** | Sequential 3-D upwind-to-downwind wake propagation on a 3×3 rotor grid: Bastankhah & Porté-Agel (2014, 2016) near- and far-wake anisotropic Gaussian velocity deficit and yaw skew deflection, Crespo & Hernández (1996) wake-added turbulence weighted by rotor overlap, sum-of-squares deficit superposition, power-law vertical wind shear across mixed tower heights, and two real-farm effects calibrated on 2019 dev twins: meso-scale wind-direction meandering separated from rotor-induced nacelle-vane flutter, and 3-D hill exposure speedup. |
| **Noise** | 47,610 real unwaked front-row rotor readings from Kelmarsh 2017–2018 (10.1% spatial turbulent power spread, 4.7% spatial wind spread, 4.9° nacelle yaw-tracking jitter, and empirical UK inflow tuples; 2016 excluded due to a ~1,680 kW commissioning export cap) |
| **Network** | EdgeWakeGNN — directed 3-layer edge-conditioned graph neural network (hidden 96, 245,767 parameters) with additive sigmoid-gated wake superposition over upstream-to-downstream edges (dx/D, dy/D, dz/D, dimensionless meandering number σ_θ·dx/D, pairwise deficit, yaw), predicting a residual correction to the steady 3-D Gaussian wake anchor; trained with β-NLL (β = 0.5) + Huber loss on 24,000 v3 farms |
| **Physics also enters** | inputs: Wind-rotated pairwise displacements (dx/D, dy/D, dz/D), dimensionless meandering number σ_θ·dx/D, analytical pairwise deficit and overlap, and finite-difference directional sensitivity<br>output: Predicts a residual power-ratio correction on top of a 3-D Bastankhah–Porté-Agel Gaussian wake and Crespo–Hernández turbulence solve |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | Kelmarsh 2019: 5,729 clean 6-turbine 10-minute snapshots (16,212 waked turbine readings), gated separately on H1 Jan–Jun (2,691 snapshots) and H2 Jul–Dec (3,038 snapshots); frozen in training/split.json (sha256 bd1dd01f…, seed 20261116) |
| **Exam** | 24,035 held-out 10-minute snapshots (117,335 waked turbine readings): Kelmarsh 2020–2021 (15,059 snapshots, 39,008 waked readings) plus zero-shot Penmanshiel 2019–2020 (8,976 snapshots, 78,327 waked readings across 14 Senvion MM82 turbines never seen in backdrop, priors or dev) |

## Run it

```bash
cd models/wakenet
pip install numpy
python demo.py        # about 1 s
```

or from anywhere in the repo: `mlratchet demo wakenet`.

```python
import json, numpy as np
from wakenet import WakeNet
net = WakeNet()
z = np.load("examples/02_kelmarsh_nw_3row.npz"); m = json.loads(str(z["meta"]))
r = net.read(z["east_m"], z["north_m"], z["hub_h_m"], m["turbine_model"],
             u_inf=m["u_inf_ms"], wind_dir_deg=m["wind_dir_deg"],
             ti_inf=m["ti_inf"], sigma_wd_deg=m["sigma_wd_deg"],
             yaw_deg=z["yaw_deg"], online=z["online"], elev_m=z["elev_m"],
             air_density=m["air_density"], p_front_kw=m["p_front_kw"])
print(f"Farm wake loss: {100*r['farm_wake_loss']:.1f}% +- {100*r['farm_wake_sig']:.1f}%")
```

## The exam

Mean absolute error in waked turbine power (% of unwaked front-row power) and farm-wide wake loss (percentage points of potential farm power), run once on Kelmarsh 2020–2021 and zero-shot on the unseen 14-turbine Penmanshiel farm (2019–2020). Every number here is on real data that never chose anything.

| Metric | WakeNet | Baselines | |
|---|---|---|---|
| Kelmarsh (2020–21), all waked turbines MAE (39,008 readings)<br><sub>Bias −1.99%.</sub> | **8.85%** | NREL FLORIS 3-D Gaussian (classical): 9.51%<br>Jensen Park + Jiménez (classical): 9.95%<br>Unwaked guess (dumb baseline): 13.63%<br>Dev mean wake loss (dumb baseline): 14% | ✅ wins |
| Kelmarsh (2020–21), direct-wake corridor MAE (dx < 8D, \|dy\| < 0.65D; 11,977 readings) | **10.07%** | NREL FLORIS 3-D Gaussian (classical): 10.75%<br>Jensen Park + Jiménez (classical): 11.63%<br>Dev mean wake loss (dumb baseline): 15.3%<br>Unwaked guess (dumb baseline): 22.17% | ✅ wins |
| Kelmarsh (2020–21), multi-wake overlap (≥ 2 upstream wakes; 2,212 readings)<br><sub>Mostly the aligned NW/SE 3-turbine row KWF3–KWF5–KWF6.</sub> | **8.86%** | NREL FLORIS 3-D Gaussian (classical): 9.9%<br>Jensen Park + Jiménez (classical): 10.12%<br>Dev mean wake loss (dumb baseline): 14.46% | ✅ wins |
| Kelmarsh (2020–21), upstream yaw offset ≥ 8° MAE (5,401 readings) | **9.76%** | NREL FLORIS 3-D Gaussian (classical): 10.46%<br>Jensen Park + Jiménez (classical): 11.25%<br>Unwaked guess (dumb baseline): 13.75% | ✅ wins |
| Kelmarsh (2020–21), partial-parked farm (1–2 units offline; 6,146 readings) | **9.18%** | Jensen Park + Jiménez (classical): 9.81%<br>NREL FLORIS 3-D Gaussian (classical): 9.94%<br>Unwaked guess (dumb baseline): 13.62% | ✅ wins |
| Kelmarsh (2020–21), farm-wide wake loss MAE (15,059 snapshots) | **2.57%** | NREL FLORIS 3-D Gaussian (classical): 2.8%<br>Jensen Park + Jiménez (classical): 2.99%<br>Dev mean wake loss (dumb baseline): 4.44%<br>Unwaked guess (dumb baseline): 4.87% | ✅ wins |
| Penmanshiel (14× MM82, zero-shot), all waked turbines MAE (78,327 readings)<br><sub>Unseen 14-turbine farm, different rotor (82 m vs 92 m) and 59 m towers; beats FLORIS by 1.62 points.</sub> | **15.01%** | Dev mean wake loss (dumb baseline): 15.88%<br>Unwaked guess (dumb baseline): 16%<br>NREL FLORIS 3-D Gaussian (classical): 16.63%<br>Jensen Park + Jiménez (classical): 17.9% | ✅ wins |
| Penmanshiel (zero-shot), upstream yaw offset ≥ 8° MAE (14,813 readings) | **15.08%** | Unwaked guess (dumb baseline): 16.27%<br>Dev mean wake loss (dumb baseline): 16.34%<br>NREL FLORIS 3-D Gaussian (classical): 16.75%<br>Jensen Park + Jiménez (classical): 18.05% | ✅ wins |
| Penmanshiel (zero-shot), multi-wake interior (≥ 2 wakes; 24,308 readings)<br><sub>Beats FLORIS (19.12%) and Jensen (21.94%) by 1.8–4.6 points, but loses to the constant dev-mean guess.</sub> | **17.32%** | Dev mean wake loss (dumb baseline): 16.43%<br>Unwaked guess (dumb baseline): 16.51%<br>NREL FLORIS 3-D Gaussian (classical): 19.12%<br>Jensen Park + Jiménez (classical): 21.94% | ❌ loses |
| Penmanshiel (zero-shot), hilltop ridge turbines (elev ≥ 219 m; 26,756 readings)<br><sub>Ridge-top acceleration on T12–T15 (219–228 m vs 180–205 m front row) offsets wake deficit; WakeNet reads −9.2% low.</sub> | **15.92%** | Unwaked guess (dumb baseline): 13.19%<br>NREL FLORIS 3-D Gaussian (classical): 18.3%<br>Dev mean wake loss (dumb baseline): 18.44%<br>Jensen Park + Jiménez (classical): 21.08% | ❌ loses |
| Penmanshiel (zero-shot), farm-wide wake loss MAE (8,976 snapshots)<br><sub>Beats FLORIS (5.16%) and Jensen (5.82%), but loses by 0.20 points to guessing Kelmarsh's 5.34% dev mean.</sub> | **4.75%** | Dev mean wake loss (dumb baseline): 4.55%<br>NREL FLORIS 3-D Gaussian (classical): 5.16%<br>Jensen Park + Jiménez (classical): 5.82%<br>Unwaked guess (dumb baseline): 5.98% | ❌ loses |

Coverage of the 68% error bars on the exam: Kelmarsh waked turbines **72.2%**, Kelmarsh farm wake loss **77.4%**, Penmanshiel waked turbines (zero-shot) **50.7%**, Penmanshiel farm wake loss (zero-shot) **61.5%**.

### Where it wins

- Beats NREL FLORIS 3-D Gaussian and Jensen Park on every stratum of both wind farms — including 15.01% vs 16.63% (FLORIS) and 17.90% (Jensen) zero-shot on the unseen 14-turbine Penmanshiel farm.
- Sweeps all six Kelmarsh 2020–21 exam rows: waked MAE 8.85% vs 9.51% (FLORIS), multi-wake 8.86% vs 9.90%, yaw ≥ 8° 9.76% vs 10.46%, and farm wake loss 2.57% vs 2.80%.
- Separating meso-scale wind-direction meandering (~0.25× nacelle-vane σ_θ) from rotor blade-passage vane flutter plus 3-D hill speedup cut Kelmarsh dev waked MAE from 10.12% to 9.48% (naive 1.0× vane meandering degraded it to 11.24% and was rejected).

### Where it loses

- On Penmanshiel's 48 m Berwickshire coastal ridge, ridge-top wind acceleration onto hilltop turbines T12–T15 (219–228 m elevation) partially cancels wake losses, so WakeNet loses to the constant dev-mean guess on farm wake loss (4.75% vs 4.55%) and multi-wake interior units (17.32% vs 16.43%), and to the unwaked guess on hilltop turbines (15.92% vs 13.19%), despite beating FLORIS on all three.

## Limits

- Error bars calibrated on Kelmarsh's gentle 21.5 m farmland cover 72.2% (turbines) and 77.4% (farm) on Kelmarsh 2020–21, but drop to 50.7% and 61.5% on Penmanshiel's steep 48 m coastal ridge and adjacent forestry.
- Below-rated active wake regime only (freestream 5.0–12.0 m/s); above rated wind speed pitch control caps all turbines at rated power and wakes cause near-zero power loss.
- 10-minute nacelle SCADA carries no atmospheric stability (Obukhov length / heat-flux) measurement, so convective daytime mixing and stable nocturnal wakes share one turbulence-conditioned curve.
- No digital elevation map or forestry roughness map around the farm perimeter is read — terrain enters only through the elevations at the turbine towers themselves.

## The ratchet

**Objective:** Weighted skill (0.65 waked-turbine power-ratio skill + 0.35 farm wake-loss skill, relative to the unwaked baseline) on the frozen simulated set  
**Rule:** Keep if the frozen-set objective beats the incumbent by more than 0.01 AND each of H1 (Jan-Jun) waked MAE, H2 (Jul-Dec) waked MAE, and farm wake-loss MAE on the Kelmarsh 2019 dev set rises by at most 5%.

| Rung | Change | Frozen-set score | Decision | Why | H1 waked (%) | H2 waked (%) | Farm wake (%) |
|---|---|---|---|---|---|---|---|
| r1 | Textbook 2-D Jensen physics in Gaussian noise; node-only MLP (no wake graph) | 0.125 | ● kept | First rung. | 14.25 | 14.68 | 4.27 |
| r2 | Directed graph neural network over (dx/D, dy/D, dz/D, yaw) wake edges, unanchored | 0.374 | ● kept | +0.249 on the frozen set; dev gate passed. | 10.39 | 10.74 | 3.38 |
| r3 | Physics-anchored EdgeWakeGNN: predicts residual correction to 3-D Gaussian wake anchor | 0.426 | ● kept | +0.052 on the frozen set; dev gate passed. | 10.12 | 10.43 | 3.29 |
| r4 | v2 physics: 3-D Bastankhah Gaussian wakes, yaw deflection, Crespo-Hernandez TI, mixed hub-height shear, and real 2017-18 unwaked backdrop | 0.517 | ● kept | +0.091 on the frozen set; dev gate passed. | 9.97 | 10.25 | 3.14 |
| r5 | Naive meandering: integrate wakes over 100% of raw nacelle wind-vane StdDev (~7.7 deg) | 0.464 | ○ rejected | -0.053 on val3, needs more than +0.01. | 11.09 | 11.37 | 3.3 |
| r6 | v3 physics calibrated on dev twins: separated meso-scale meandering (0.16-0.36x vane StdDev) and 3-D hill exposure speedup | 0.552 | **◆ released** | +0.035 on the frozen set; dev gate passed. | 9.38 | 9.56 | 2.88 |
| r7 | Softmax attention over upstream edges in place of additive sigmoid-gated wake superposition | 0.552 | ○ rejected | +0.000 on val3, needs more than +0.01. | 9.41 | 9.6 | 2.89 |
| r8 | Fine-tune from r6 on mixed v2 + v3 banks (48,000 simulated farms) | 0.543 | ○ rejected | -0.009 on val3, needs more than +0.01. | 9.62 | 9.85 | 2.99 |

<sub>Columns after *Why*: Real Kelmarsh 2019 dev set (5,729 snapshots), MAE — the gate's input.</sub>

Every rung was re-scored when the simulator physics changed (val1 -> val2 -> val3) and every decision re-taken in order. Rung 3 (3-D Gaussian anchor) fell 0.005 short of rung 2 on the 2-D Jensen set val1 (0.740 vs 0.735) and flipped to kept (+0.053) when re-scored on val2.

Raw per-rung logs are in [`training/`](training/).

## Physics

Sequential 3-D upwind-to-downwind wake propagation on a 3×3 rotor grid: Bastankhah & Porté-Agel (2014, 2016) near- and far-wake anisotropic Gaussian velocity deficit and yaw skew deflection, Crespo & Hernández (1996) wake-added turbulence weighted by rotor overlap, sum-of-squares deficit superposition, power-law vertical wind shear across mixed tower heights, and two real-farm effects calibrated on 2019 dev twins: meso-scale wind-direction meandering separated from rotor-induced nacelle-vane flutter, and 3-D hill exposure speedup.

**Simulator**

- Procedural 5-to-16-turbine layouts (staggered rows, dart-thrown clusters, curved ridges, perturbed grids; spacing ≥ 2.7 D), never copying Kelmarsh or Penmanshiel coordinates
- Senvion MM92 (D = 92 m), Senvion MM82 (D = 82 m) manufacturer curves and 40% parametric IEC Class I–III rotors (D = 74–132 m)
- Mixed tower heights (±8–12 m on 35% of farms) and rolling Gaussian hills (4–32 m relief) with partial terrain-following streamlines (0.68–0.95)
- Active upstream yaw-steering offsets (±6–22°) on 30% of farms and 1–2 parked turbines (online = False, Ct = 0) on 22%
- v3 physics, calibrated on Kelmarsh 2019 dev: 3-point Gauss–Hermite meandering over 0.16–0.36× nacelle-vane σ_θ (filtering blade-passage swirl) and hill exposure speedup s_hill ~ U(0.05, 0.24)

**Checked against independent references**

- Bastankhah–Porté-Agel 3-D Gaussian single-wake grid (3D–15D × −1.5D..+1.5D) vs NREL FLORIS 4.7: max relative power diff 1.58e-15 (training/physics_check.json)
- Jensen top-hat + Jiménez yaw deflection (−20°..+20°) vs NREL FLORIS 4.7: max relative power diff 2.91e-16
- 3-turbine Gaussian yaw-steering sweep (−25°..+25°) + Crespo–Hernández TI + SOSFS vs NREL FLORIS 4.7: max relative power diff 1.88e-15 (optimal yaw +25°, +12.36% gain)
- 14-turbine Penmanshiel layout × 72 wind directions (1,008 turbine states) vs NREL FLORIS 4.7: correlation 1.000000, max relative power diff 7.37e-15
- Exact +37.5° compass rotational equivariance and parked-turbine removal equivalence: < 1e-12 kW

## Next rungs

- Read surrounding 30 m DEM terrain slopes and forest-canopy roughness upwind of each sector (Jackson–Hunt / WAsP linearised flow over real topography) to close Penmanshiel's −9.2% hilltop ridge bias.
- Terrain-relief-conditioned or Mondrian conformal σ widening so zero-shot error bars widen automatically on steep (>30 m relief) sites.
- Active yaw-steering controller rung (WakePilot) trained through the differentiable wake graph and checked against dedicated field yaw-offset campaigns.

## Files

| Path | What |
|---|---|
| `wakenet.py`, `physics.py`, `pipeline.py` | Inference — numpy/scipy only |
| `wakenet_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [Kelmarsh Wind Farm Open Data — Charlie Plumley / Cubico Sustainable Investments (Zenodo 16807551)](https://doi.org/10.5281/zenodo.5841834) | noise (2017–2018 unwaked front row), dev (2019), exam (2020–2021) | CC BY 4.0 |
| [Penmanshiel Wind Farm Open Data — Charlie Plumley / Cubico Sustainable Investments (Zenodo 16807304)](https://doi.org/10.5281/zenodo.5946808) | exam (2019–2020, zero-shot unseen 14-turbine MM82 farm) | CC BY 4.0 |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
