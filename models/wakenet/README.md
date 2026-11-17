<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# WakeNet <sub>v1.0</sub>

> **Who is stealing whose wind across a real farm — read as a directed graph.**

*How much power is each wind turbine losing to its neighbours' wakes?*

From a wind farm's turbine coordinates, hub heights, rotor power and thrust curves, mechanical yaw orientations, online flags, and the upstream wind measured on its unwaked front row, WakeNet predicts every downstream turbine's waked power ratio, power loss (kW and %), rotor-effective wind speed and farm-wide wake loss, each with a 68% error bar. It learned only from 24,000 procedural 5-to-16-turbine simulations laid into real unwaked turbulence from Sep 2017 – Dec 2018. It was examined on real 10-minute SCADA from two UK wind farms, one never used for anything else: 24,035 snapshots in 2019–2021, then 35,592 snapshots from 2021–2024 that no version had seen. It beats textbook FLORIS everywhere, but the same physics tuned on one year of the farm's own data (no network) does as well or better on 17 of 18 rows of a second, never-seen exam.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/wakenet/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/wakenet/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Turbine coordinates (East, North, ground elevation in m), hub heights (m), rotor diameter and nominal P(U)/Ct(U) curves, each unit's online flag and mechanical nacelle yaw offset (deg), and upstream freestream wind speed (m/s), wind direction (deg), turbulence intensity, 10-min wind-direction standard deviation (deg), air density (kg/m3) and front-row mean power (kW) measured on the geometrically unwaked front row. Downstream wind speeds, vanes and power meters are never read. |
| **Returns** | Per-turbine waked power ratio P_i / P_front and wake loss (%) ± 68% bar<br>Per-turbine active power (kW) ± 68% bar and rotor-effective wind speed (m/s)<br>Farm-wide wake loss (%) and total waked output (MW) ± 68% bar |
| **Learned from** | Sequential 3-D upwind-to-downwind wake propagation on a 3×3 rotor grid: Bastankhah & Porté-Agel (2014, 2016) near- and far-wake anisotropic Gaussian velocity deficit and yaw skew deflection, Crespo & Hernández (1996) wake-added turbulence weighted by rotor overlap, sum-of-squares deficit superposition, power-law vertical wind shear across mixed tower heights, and two real-farm effects calibrated on 2019 dev twins: meso-scale wind-direction meandering separated from rotor-induced nacelle-vane flutter, and 3-D hill exposure speedup. |
| **Noise** | 47,610 real unwaked front-row rotor readings from Kelmarsh Sep 2017 – Dec 2018 (10.1% spatial turbulent power spread, 4.7% spatial wind spread, 4.9° nacelle yaw-tracking jitter, and empirical UK inflow tuples; 2016 excluded due to a ~1,680 kW commissioning export cap) |
| **Network** | EdgeWakeGNN — directed 3-layer edge-conditioned graph neural network (hidden 96, 245,767 parameters) with additive sigmoid-gated wake superposition over upstream-to-downstream edges (dx/D, dy/D, dz/D, dimensionless meandering number σ_θ·dx/D, pairwise deficit, yaw), predicting a residual correction to the steady 3-D Gaussian wake anchor; trained with β-NLL (β = 0.5) + Huber loss on 24,000 v3 farms |
| **Physics also enters** | inputs: Wind-rotated pairwise displacements (dx/D, dy/D, dz/D), dimensionless meandering number σ_θ·dx/D, analytical pairwise deficit and overlap, and finite-difference directional sensitivity<br>output: Predicts a residual power-ratio correction on top of a 3-D Bastankhah–Porté-Agel Gaussian wake and Crespo–Hernández turbulence solve |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | Kelmarsh 2019: 5,729 clean 6-turbine 10-minute snapshots (16,212 waked turbine readings), gated separately on H1 Jan–Jun (2,691 snapshots) and H2 Jul–Dec (3,038 snapshots); frozen in training/split.json (sha256 bd1dd01f…, seed 20261116) |
| **Exam** | Exam 2 (frozen 17 Nov 2026 in training/split_exam2.json before anything was scored on it, never seen by any version): Kelmarsh 2022–2024 (11,750 snapshots, 29,308 waked readings) and Penmanshiel 2021–2023 (23,842 snapshots, 208,508 waked readings). Exam 1, run once for v1.0 on 16 Nov 2026 and re-scored with more baselines on 17 Nov 2026: Kelmarsh 2020–2021 (15,059 snapshots) and Penmanshiel 2019–2020 (8,976 snapshots). |

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

Mean absolute error in each waked turbine's power (as a share of the unwaked front row's power) and in farm-wide wake loss, on exam 2: years no version has seen. The strongest baseline is the same FLORIS physics with the three settings WakeNet's simulator took from 2019 dev (meandering 0.25 × vane spread, hill speedup 0.18, terrain-follow 0.80). Every row gives WakeNet minus the best baseline with a 95% interval from a bootstrap over days; win or loss only when the interval excludes zero. Each note also gives the re-scored exam 1 result. Every number here is on real data that never chose anything.

| Metric | WakeNet | Baselines | |
|---|---|---|---|
| Kelmarsh 2022-24, all waked turbines MAE (29,308 readings)<br><sub>+0.14 points against FLORIS tuned on 2019 dev, 95% CI [+0.07, +0.21] (bootstrap by day). Exam 1 (Kelmarsh 2020-21, re-scored): 8.85 vs tuned FLORIS 8.64, loss.</sub> | **10.96%** | FLORIS tuned on 2019 dev (classical): 10.82%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 11.77%<br>Jensen Park + Jiménez (k = 0.075) (classical): 11.92%<br>Dev mean wake loss (dumb baseline): 14.78%<br>Unwaked guess (dumb baseline): 14.76% | ❌ loses |
| Kelmarsh 2022-24, direct-wake corridor (dx < 8D, \|dy\| < 0.65D) MAE (8,890 readings)<br><sub>-0.17 points against FLORIS tuned on 2019 dev, 95% CI [-0.26, -0.07] (bootstrap by day). Exam 1 (Kelmarsh 2020-21, re-scored): 10.07 vs tuned FLORIS 10.21, win.</sub> | **13.15%** | FLORIS tuned on 2019 dev (classical): 13.32%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 14.13%<br>Jensen Park + Jiménez (k = 0.075) (classical): 13.85%<br>Dev mean wake loss (dumb baseline): 16.21%<br>Unwaked guess (dumb baseline): 21.76% | ✅ wins |
| Kelmarsh 2022-24, inside 2 or more wakes MAE (2,236 readings)<br><sub>+0.09 points against FLORIS tuned on 2019 dev, 95% CI [-0.10, +0.28] (bootstrap by day). Exam 1 (Kelmarsh 2020-21, re-scored): 8.86 vs tuned FLORIS 8.60, loss.</sub> | **11.28%** | FLORIS tuned on 2019 dev (classical): 11.19%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 12.65%<br>Jensen Park + Jiménez (k = 0.075) (classical): 13.72%<br>Dev mean wake loss (dumb baseline): 15.3%<br>Unwaked guess (dumb baseline): 19.74% | ➖ tie |
| Kelmarsh 2022-24, upstream yaw offset ≥ 8° MAE (7,855 readings)<br><sub>-0.04 points against FLORIS tuned on 2019 dev, 95% CI [-0.21, +0.11] (bootstrap by day). Exam 1 (Kelmarsh 2020-21, re-scored): 9.76 vs tuned FLORIS 9.54, loss.</sub> | **13.73%** | FLORIS tuned on 2019 dev (classical): 13.78%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 14.46%<br>Jensen Park + Jiménez (k = 0.075) (classical): 14.55%<br>Dev mean wake loss (dumb baseline): 15.08%<br>Unwaked guess (dumb baseline): 13.98% | ➖ tie |
| Kelmarsh 2022-24, 1-2 turbines parked MAE (7,352 readings)<br><sub>+0.21 points against FLORIS tuned on 2019 dev, 95% CI [+0.08, +0.34] (bootstrap by day). Exam 1 (Kelmarsh 2020-21, re-scored): 9.18 vs tuned FLORIS 8.82, loss.</sub> | **12.14%** | FLORIS tuned on 2019 dev (classical): 11.93%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 13.04%<br>Jensen Park + Jiménez (k = 0.075) (classical): 12.9%<br>Dev mean wake loss (dumb baseline): 15.36%<br>Unwaked guess (dumb baseline): 15.4% | ❌ loses |
| Kelmarsh 2022-24, wind below 10 m/s MAE (16,401 readings)<br><sub>+0.20 points against FLORIS tuned on 2019 dev, 95% CI [+0.11, +0.28] (bootstrap by day). Exam 1 (Kelmarsh 2020-21, re-scored): 11.73 vs tuned FLORIS 11.45, loss.</sub> | **13.73%** | FLORIS tuned on 2019 dev (classical): 13.52%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 14.83%<br>Jensen Park + Jiménez (k = 0.075) (classical): 15.16%<br>Dev mean wake loss (dumb baseline): 17.63%<br>Unwaked guess (dumb baseline): 18.17% | ❌ loses |
| Kelmarsh 2022-24, wind 10.5-12 m/s MAE (8,220 readings)<br><sub>+0.13 points against FLORIS tuned on 2019 dev, 95% CI [+0.03, +0.23] (bootstrap by day). Exam 1 (Kelmarsh 2020-21, re-scored): 5.30 vs tuned FLORIS 5.07, loss.</sub> | **6.1%** | FLORIS tuned on 2019 dev (classical): 5.97%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 6.32%<br>Jensen Park + Jiménez (k = 0.075) (classical): 6.34%<br>Dev mean wake loss (dumb baseline): 10.06%<br>Unwaked guess (dumb baseline): 8.11% | ❌ loses |
| Kelmarsh 2022-24, highest quarter of turbines MAE (10,607 readings)<br><sub>+0.34 points against FLORIS tuned on 2019 dev, 95% CI [+0.20, +0.47] (bootstrap by day). Exam 1 (Kelmarsh 2020-21, re-scored): 9.33 vs tuned FLORIS 8.88, loss.</sub> | **12.22%** | FLORIS tuned on 2019 dev (classical): 11.87%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 13.33%<br>Jensen Park + Jiménez (k = 0.075) (classical): 12.66%<br>Dev mean wake loss (dumb baseline): 16.21%<br>Unwaked guess (dumb baseline): 14.13% | ❌ loses |
| Kelmarsh 2022-24, farm-wide wake loss MAE (11,750 snapshots)<br><sub>+0.01 points against FLORIS tuned on 2019 dev, 95% CI [-0.02, +0.04] (bootstrap by day). Exam 1 (Kelmarsh 2020-21, re-scored): 2.57 vs tuned FLORIS 2.53, loss.</sub> | **3.39%** | FLORIS tuned on 2019 dev (classical): 3.38%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 3.66%<br>Jensen Park + Jiménez (k = 0.075) (classical): 3.73%<br>Dev mean wake loss (dumb baseline): 4.76%<br>Unwaked guess (dumb baseline): 5.32% | ➖ tie |
| Penmanshiel 2021-23 (unseen farm), all waked turbines MAE (208,508 readings)<br><sub>+0.27 points against FLORIS tuned on 2019 dev, 95% CI [+0.19, +0.36] (bootstrap by day). Exam 1 (Penmanshiel 2019-20, re-scored): 15.01 vs tuned FLORIS 14.43, loss.</sub> | **19.78%** | FLORIS tuned on 2019 dev (classical): 19.5%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 21.75%<br>Jensen Park + Jiménez (k = 0.075) (classical): 21.77%<br>Dev mean wake loss (dumb baseline): 19.92%<br>Unwaked guess (dumb baseline): 21.37% | ❌ loses |
| Penmanshiel 2021-23 (unseen farm), direct-wake corridor (dx < 8D, \|dy\| < 0.65D) MAE (53,130 readings)<br><sub>+4.52 points against Dev mean wake loss, 95% CI [+3.58, +5.46] (bootstrap by day). Exam 1 (Penmanshiel 2019-20, re-scored): 18.58 vs tuned FLORIS 18.20, loss.</sub> | **25.12%** | FLORIS tuned on 2019 dev (classical): 25.3%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 27.37%<br>Jensen Park + Jiménez (k = 0.075) (classical): 26.23%<br>Dev mean wake loss (dumb baseline): 20.59%<br>Unwaked guess (dumb baseline): 23.59% | ❌ loses |
| Penmanshiel 2021-23 (unseen farm), inside 2 or more wakes MAE (67,650 readings)<br><sub>+1.69 points against Dev mean wake loss, 95% CI [+1.16, +2.24] (bootstrap by day). Exam 1 (Penmanshiel 2019-20, re-scored): 17.32 vs tuned FLORIS 16.49, loss.</sub> | **22.8%** | FLORIS tuned on 2019 dev (classical): 22.49%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 24.97%<br>Jensen Park + Jiménez (k = 0.075) (classical): 25.73%<br>Dev mean wake loss (dumb baseline): 21.11%<br>Unwaked guess (dumb baseline): 22.51% | ❌ loses |
| Penmanshiel 2021-23 (unseen farm), upstream yaw offset ≥ 8° MAE (65,614 readings)<br><sub>+0.19 points against FLORIS tuned on 2019 dev, 95% CI [+0.07, +0.31] (bootstrap by day). Exam 1 (Penmanshiel 2019-20, re-scored): 15.08 vs tuned FLORIS 14.41, loss.</sub> | **21.56%** | FLORIS tuned on 2019 dev (classical): 21.36%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 23.62%<br>Jensen Park + Jiménez (k = 0.075) (classical): 23.84%<br>Dev mean wake loss (dumb baseline): 21.75%<br>Unwaked guess (dumb baseline): 23.24% | ❌ loses |
| Penmanshiel 2021-23 (unseen farm), 1-2 turbines parked MAE (53,770 readings)<br><sub>+0.25 points against FLORIS tuned on 2019 dev, 95% CI [+0.07, +0.40] (bootstrap by day). Exam 1 (Penmanshiel 2019-20, re-scored): 15.99 vs tuned FLORIS 15.45, loss.</sub> | **19.17%** | FLORIS tuned on 2019 dev (classical): 18.92%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 21.17%<br>Jensen Park + Jiménez (k = 0.075) (classical): 21.43%<br>Dev mean wake loss (dumb baseline): 19.25%<br>Unwaked guess (dumb baseline): 20.57% | ❌ loses |
| Penmanshiel 2021-23 (unseen farm), wind below 10 m/s MAE (114,194 readings)<br><sub>+0.51 points against Dev mean wake loss, 95% CI [+0.11, +0.94] (bootstrap by day). Exam 1 (Penmanshiel 2019-20, re-scored): 20.85 vs tuned FLORIS 20.29, loss.</sub> | **24.19%** | FLORIS tuned on 2019 dev (classical): 24.1%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 26.57%<br>Jensen Park + Jiménez (k = 0.075) (classical): 26.84%<br>Dev mean wake loss (dumb baseline): 23.68%<br>Unwaked guess (dumb baseline): 25.9% | ❌ loses |
| Penmanshiel 2021-23 (unseen farm), wind 10.5-12 m/s MAE (68,917 readings)<br><sub>+0.54 points against FLORIS tuned on 2019 dev, 95% CI [+0.46, +0.62] (bootstrap by day). Exam 1 (Penmanshiel 2019-20, re-scored): 12.86 vs tuned FLORIS 12.25, loss.</sub> | **13.46%** | FLORIS tuned on 2019 dev (classical): 12.93%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 14.77%<br>Jensen Park + Jiménez (k = 0.075) (classical): 14.49%<br>Dev mean wake loss (dumb baseline): 14.66%<br>Unwaked guess (dumb baseline): 15.03% | ❌ loses |
| Penmanshiel 2021-23 (unseen farm), highest quarter of turbines MAE (66,219 readings)<br><sub>+3.31 points against Unwaked guess, 95% CI [+2.92, +3.71] (bootstrap by day). Exam 1 (Penmanshiel 2019-20, re-scored): 15.92 vs tuned FLORIS 14.23, loss.</sub> | **22.64%** | FLORIS tuned on 2019 dev (classical): 21.52%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 25.49%<br>Jensen Park + Jiménez (k = 0.075) (classical): 26.39%<br>Dev mean wake loss (dumb baseline): 23.1%<br>Unwaked guess (dumb baseline): 19.33% | ❌ loses |
| Penmanshiel 2021-23 (unseen farm), farm-wide wake loss MAE (23,842 snapshots)<br><sub>+0.07 points against Dev mean wake loss, 95% CI [-0.20, +0.36] (bootstrap by day). Exam 1 (Penmanshiel 2019-20, re-scored): 4.75 vs tuned FLORIS 4.82, loss.</sub> | **7.08%** | FLORIS tuned on 2019 dev (classical): 7.2%<br>NREL FLORIS 3-D Gaussian (textbook) (classical): 7.67%<br>Jensen Park + Jiménez (k = 0.075) (classical): 7.88%<br>Dev mean wake loss (dumb baseline): 7.01%<br>Unwaked guess (dumb baseline): 8.9% | ➖ tie |

Coverage of the 68% error bars on the exam: Kelmarsh 2022-24 waked turbines **64.1%**, Kelmarsh 2022-24 farm wake loss **66.8%**, Penmanshiel 2021-23 waked turbines (unseen farm) **41.9%**, Penmanshiel 2021-23 farm wake loss (unseen farm) **48.9%**.

### Where it wins

- Beats textbook NREL FLORIS and Jensen on every row of both exams and both farms: exam 2 Kelmarsh waked turbines 10.96% against 11.77% (FLORIS) and 11.92% (Jensen, k = 0.075); unseen Penmanshiel 19.78% against 21.75% and 21.77%.
- Directly behind another turbine it beats even the dev-tuned physics, on both exams: Kelmarsh 2022-24 13.15% against 13.32%, −0.17 points [−0.26, −0.07]; exam 1 −0.13 [−0.23, −0.04].
- Separating meso-scale wind-direction meandering (~0.25 × nacelle-vane σ_θ) from blade-passage vane flutter, plus 3-D hill speedup, cut Kelmarsh dev waked MAE from 10.12% to 9.48% (smearing wakes by the full vane spread made it 11.24% and was rejected).

### Where it loses

- The network adds nothing over its own physics once that physics is tuned on dev. FLORIS with the three settings from 2019 dev beats WakeNet on 13 of 18 exam-2 rows and ties 4 (exam 1: 17 of 18). The gains over textbook FLORIS came from the dev tuning, not from the graph network. Anchoring the network on the tuned physics (r9) gained only +0.002 in simulation and was rejected.
- On the unseen Penmanshiel ridge it loses to constant guesses: directly behind another turbine (25.12% against 20.60% for the dev-mean wake loss), inside 2 or more wakes, below 10 m/s, and on the highest quarter of turbines (22.64% against 19.33% for assuming no wake). Ridge-top speed-up offsets the wakes there.

## Limits

- Error bars are too narrow on years and farms not seen in dev: 68% bars cover 64% of Kelmarsh 2022-24 turbines (72% in 2020-21) and 42% of Penmanshiel 2021-23 turbines (51% in 2019-20).
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
| r9 | Same EdgeWakeGNN anchored on the physics tuned on 2019 dev (meandering 0.25x vane sigma, hill speedup 0.18, terrain-follow 0.80) instead of textbook FLORIS; seeds 106/107/108 score 0.5543/0.5539/0.5542 | 0.554 | ○ rejected | +0.002 on val3, needs more than +0.01. | 9.3 | 9.56 | 2.86 |

<sub>Columns after *Why*: Real Kelmarsh 2019 dev set (5,729 snapshots), MAE — the gate's input.</sub>

Every rung was re-scored when the simulator physics changed (val1 -> val2 -> val3) and every decision re-taken in order. Rung 3 (3-D Gaussian anchor) fell 0.005 short of rung 2 on the 2-D Jensen set val1 (0.740 vs 0.735) and flipped to kept (+0.053) when re-scored on val2. Reference rows, not candidates (17 Nov 2026, training/v11_reference.json): textbook FLORIS alone scores 0.512 on val3 (dev H1/H2/farm 10.00/10.29/3.19 %); the same physics tuned on 2019 dev scores 0.541 (9.18/9.42/2.86 %). The network adds +0.011 over its anchor in simulation but nothing on real dev, where tuned physics alone beats every rung. r9 anchors the network on the tuned physics: +0.002 on val3 over r6 across three seeds (spread 0.0004), so it is rejected, and on dev it still trails tuned physics alone (9.44 vs 9.31 % waked).

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

- Close the sim-to-real gap the twin test shows: the network gains +0.011 over its anchor in simulation and loses on real dev, so the simulator's noise or inflow model is teaching it corrections that do not hold on real turbines. Start with a dev twin test of the residuals the network adds.
- Read surrounding 30 m DEM terrain slopes and forest-canopy roughness upwind of each sector (Jackson–Hunt / WAsP linearised flow over real topography) to close Penmanshiel's hilltop bias.
- Terrain-relief-conditioned or Mondrian conformal σ widening so error bars widen on steep (>30 m relief) sites and later years.
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
| [Kelmarsh Wind Farm Open Data — Charlie Plumley / Cubico Sustainable Investments (Zenodo 16807551)](https://doi.org/10.5281/zenodo.16807551) | noise (Sep 2017–Dec 2018 unwaked front row; 2016 ran under an export cap), dev (2019), exam (2020–2021), exam 2 (2022–2024) | CC BY 4.0 |
| [Penmanshiel Wind Farm Open Data — Charlie Plumley / Cubico Sustainable Investments (Zenodo 16807304)](https://doi.org/10.5281/zenodo.16807304) | exam (2019–2020, zero-shot unseen 14-turbine MM82 farm), exam 2 (2021–2023) | CC BY 4.0 |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
