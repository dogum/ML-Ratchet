# WakeNet v1.0.1 — model card

**What it does.** Given a wind farm's turbine coordinates, tower hub heights, rotor power and thrust curves, online status, mechanical nacelle yaw angles, and the incoming wind measured on its geometrically unwaked front-row turbines, WakeNet predicts how much power every downstream turbine is losing to its neighbours' wakes — per turbine and across the whole farm — each with a calibrated 68% error bar.

**How it learned.** Only from procedural physics simulations laid into real unwaked turbulence. No real waked power reading was ever a training sample. Real 2019 dev readings tuned three simulator settings (meandering, hill speedup, shear), set the error-bar widening, and gated each rung.

**The headline, plainly (17 Nov 2026).** WakeNet beats textbook FLORIS and Jensen on every row of both exams. But the same FLORIS physics, tuned with the three settings from 2019 dev and no network at all, does as well or better on 17 of 18 rows of exam 2 (years no version has seen). The network itself adds nothing measurable on real data yet; only directly behind another turbine does it beat the tuned physics on both exams.

## Inputs

- `east_m`, `north_m`, `elev_m`: `(N,)` local tangent-plane coordinates and ground elevations in metres.
- `hub_h_m`: `(N,)` or scalar tower hub height(s) in metres (handles mixed-height farms such as Kelmarsh's 68.5 m and 78.5 m towers).
- `turbine`: rotor specification dict (`rotor_diameter`, `rated_power_kw`, `wind_speed`, `power_kw`, `thrust_coefficient`, `cosine_loss_exponent_yaw`), or `"MM92"` / `"MM82"`.
- `u_inf`, `wind_dir_deg`, `ti_inf`, `sigma_wd_deg`, `air_density`, `p_front_kw`: 10-minute freestream wind speed (m/s at reference hub height), meteorological wind direction (° clockwise from North), turbulence intensity $\sigma_U / U$, wind-direction standard deviation (°), air density ($\text{kg/m}^3$), and mean unwaked front-row power (kW). All six are measured solely on the geometrically unwaked front-row turbines (`physics.freestream_mask`).
- `yaw_deg`: `(N,)` causal mechanical yaw offset $\gamma_i = \text{wrap}(\theta_\infty - \phi_{\text{nacelle},i})$ in degrees.
- `online`: `(N,)` boolean operating mask (`False` = parked/feathered, $P_i = 0$, $C_{T,i} = 0$, sheds zero wake).
- Valid range: below-rated active wake regime ($5.0 \le U_\infty \le 12.0\text{ m/s}$), any farm size $N \ge 2$. Downstream wind speeds, wind vanes, and power meters are never read.

## Outputs

| key | meaning |
|---|---|
| `eta`, `eta_sig` | per-turbine power ratio $P_i / \bar{P}_{\text{front}}$ and calibrated 68% error bar |
| `wake_loss`, `wake_loss_sig` | per-turbine fractional wake loss $1 - \eta_i$ and 68% error bar |
| `power_kw`, `power_sig_kw` | per-turbine active power $\eta_i \bar{P}_{\text{front}}$ (kW) and 68% bar |
| `u_rotor_ms` | per-turbine rotor-effective wind speed (m/s) |
| `eta_anchor` | steady 3-D Bastankhah–Porté-Agel Gaussian anchor ratio |
| `is_free`, `online` | geometric front-row mask and online mask |
| `farm_wake_loss`, `farm_wake_sig` | farm-wide fractional wake loss $1 - \frac{1}{N_{\text{on}}}\sum_{i\in\text{on}}\eta_i$ and 68% bar |
| `farm_power_kw`, `farm_free_kw` | total waked farm output and unwaked potential output (kW) |

On the 5,729 Kelmarsh 2019 dev snapshots, the raw $\beta$-NLL simulation bars already covered 68.7% of waked-turbine residuals and 73.6% of farm-wake residuals, so `mlratchet.calibrate.widen_factor` (widen-only, floored at $1.00$) set `waked_eta = 1.00`, `free_eta = 1.00`, `farm_wake = 1.00`. Pass `calibrated=False` to `WakeNet.read` for raw bars.

## Physics and simulation

- **Governing wake physics (`physics.py`):** sequential upwind-to-downwind solver on a $3 \times 3$ rotor grid (`[-0.25 D, 0, +0.25 D]`, cubic-mean rotor velocity):
  - **Velocity deficit:** Bastankhah & Porté-Agel (2014, 2016) 3-D near-wake potential core ($x_0 / D$) and far-wake linear expansion ($\sigma_y, \sigma_z$ with $k^* = k_a TI + k_b$), plus classic Jensen (1983) top-hat ($w_e = 0.05$) for rung `r1` and classical baselines.
  - **Yaw wake deflection & thrust:** Bastankhah & Porté-Agel (2016) skew angle $\theta_{c0}$ and logarithmic far-wake centerline deflection, Jiménez et al. (2010) deflection, and $\cos^{1.88}\gamma$ yaw power loss.
  - **Turbulence & superposition:** Crespo & Hernández (1996) wake-added turbulence weighted by rotor area overlap within $15 D \times \pm 2 D$, combined via sum-of-squares of freestream deficits (`sosfs`).
  - **3-D boundary layer:** power-law wind shear $U(z) = U_\infty (z / z_{\text{ref}})^\alpha$, mixed tower heights, partial terrain-following wake centerlines, linearised hill exposure speedup $s_{\text{hill}} (z_{\text{elev},i} - \bar{z}) / H_i$, and 3-point Gauss–Hermite azimuth quadrature over wind-direction meandering.
- **Independent checks (`training/physics_check.py`, `training/physics_check.json`):**
  - Bastankhah 3-D Gaussian single-wake grid ($3D\text{–}15D \times -1.5D\dots+1.5D$) vs NREL FLORIS 4.7: max velocity diff `3.55e-15 m/s`, max relative power diff `1.58e-15`.
  - Jensen top-hat + Jiménez yaw deflection ($-20^\circ\dots+20^\circ$) vs NREL FLORIS 4.7: max velocity diff `0.0 m/s`, max relative power diff `2.91e-16`.
  - 3-turbine Gaussian yaw-steering sweep ($-25^\circ\dots+25^\circ$) + Crespo–Hernández TI + SOSFS vs NREL FLORIS 4.7: max relative power diff `1.88e-15` (both place optimal upstream yaw at $+25^\circ$ for a $+12.36\%$ array gain).
  - 14-turbine Penmanshiel coordinates $\times$ 72 wind directions (1,008 turbine states) vs NREL FLORIS 4.7: correlation `1.000000`, max relative power diff `7.37e-15`.
  - Analytical $+37.5^\circ$ compass rotation invariance (`9.09e-13 kW`) and parked-turbine `online=False` vs removal exactness (`9.66e-13 kW`).
- **Two simulator discoveries on Kelmarsh 2019 dev (`training/calib_dev.py`, `training/twin.py`):**
  1. **Nacelle wind-vane swirl vs true atmospheric meandering:** raw SCADA 10-minute `Wind direction, StdDev` behind spinning rotor blades has a median of `7.74°`. Feeding 100% of it as atmospheric wind meandering (`r5`) washed out direct wakes (`+5.3%` centerline bias, dev waked MAE degraded from `10.12%` to `11.24%`) and was rejected. Separating meso-scale meandering (`0.16–0.36×` vane $\sigma_\theta$, centred at `0.25×` $\approx 1.9^\circ$) from high-frequency blade-passage vane flutter improved direct-wake accuracy without washing out troughs.
  2. **3-D hill exposure on lower-elevation turbines:** rung `r4` (`v2` flat-terrain 3-D Gaussian) read its own Kelmarsh twins with `0.00%` bias on low-elevation downstream units `KWF5` (`142.9 m`) and `KWF6` (`135.0 m` vs `156.6 m` at `KWF2`), but read real 2019 dev `+2.27%` too high (`10.12%` waked MAE). Adding rolling Gaussian hills with linearised hill speedup (`s_hill ~ U(0.05, 0.24)`) in `v3` (`r6`) closed the `KWF5–KWF6` real dev bias to `-0.47%` and cut dev waked MAE to `9.48%` and farm wake MAE to `2.88%`.
- **Real backdrop (`training/noise.py`):** 47,610 unwaked front-row rotor readings from Kelmarsh 25 Sep 2017 – Dec 2018 supply empirical UK inflow tuples and real spatial turbulent power fluctuations (`10.1%` std across unwaked rotors) and nacelle yaw-tracking offsets (`4.9°` std), split chronologically 85% / 15% between training and frozen validation banks. Kelmarsh's 2016 commissioning year was under a `~1,680 kW` grid export setpoint cap, so 0 of 52,416 rows passed the uncurtailed screen (`setpoint >= 2,000 kW`).

## Network

- **Architecture (`EdgeWakeGNN`):** directed 3-layer edge-conditioned Graph Neural Network (`hidden = 96`, `head_hidden = 64`, `245,767` parameters; Collection's first GNN).
- **Nodes (`17` features):** `online`, `is_free`, normalized $U_\infty$, $P_{\text{free}}/P_{\text{rated}}$, $C_T(U_\infty)$, $TI_\infty$, $\sigma_\theta$, yaw $\gamma_i$, $1 - \cos^{1.88}\gamma_i$, relative tower height $(H_i - \bar{H})/D$, relative ground elevation $(z_i - \bar{z})/D$, array depth, steady 3-D Gaussian anchor $\eta_{\text{anc},i}$, velocity deficit, wake-added $TI$, and finite-difference meandering curvature and directional sensitivity.
- **Directed edges (`12` features on $j \to i$ where $\Delta x_{j\to i} > 0.1 D$):** $(dx/D + 2)^{-1}$, $\ln(1 + dx/D)$, signed $dy/D$, deflected-wake-normalized $|dy - \delta|/\sigma_y$, tower step $\Delta H/D$, terrain step $\Delta z/D$, analytical pairwise deficit, rotor overlap, yaw deflection $\delta/D$, upstream yaw $\gamma_j$, upstream $C_{T,j}$, and the dimensionless Buckingham-$\Pi$ meandering width $\sigma_\theta \Delta x / D$.
- **Superposition symmetry:** incoming wake messages are aggregated with a **sigmoid-gated sum** (zero when unwaked; additive when 2–4 wakes overlap) plus an online-mean farm context vector, and enforced to satisfy $\frac{1}{|F|}\sum_{k\in F}\hat{\eta}_k \equiv 1$ on the front row.
- **Parity:** standalone `numpy` (`wakenet.py`) matches PyTorch to `2.38e-7`; browser TypeScript (`site/src/instruments/wakenet/net.ts`, `float16` weights `0.49 MB`) matches Python across 24 test cases to `1.88e-7`.

## Ratchet

Objective: $0.65\,\text{skill}(\eta_{\text{waked}}) + 0.35\,\text{skill}(W_{\text{farm}})$ relative to the unwaked baseline on the frozen simulated set. Rule: beat the incumbent by more than `+0.01` AND increase none of `H1` (Jan–Jun 2019) waked MAE, `H2` (Jul–Dec 2019) waked MAE, or farm wake-loss MAE on Kelmarsh 2019 dev by more than `5%`. Every rung was re-scored when the physics version changed (`val1 -> val2 -> val3`).

| rung | change | sim (`val3`) | dev H1 waked | dev H2 waked | dev farm wake | decision |
|---|---|---|---|---|---|---|
| `r1` | 2-D Jensen in Gaussian noise; node-only MLP (no wake graph) | 0.125 | 14.25% | 14.68% | 4.27% | kept |
| `r2` | Directed GNN over `(dx/D, dy/D, dz/D, yaw)` wake edges, unanchored | 0.374 | 10.39% | 10.74% | 3.38% | kept |
| `r3` | Physics-anchored `EdgeWakeGNN` (rejected on `val1` by +0.005; flipped to kept on `val2`) | 0.426 | 10.12% | 10.43% | 3.29% | kept |
| `r4` | `v2` physics: 3-D Bastankhah Gauss + mixed hub-height shear + real 2017–18 backdrop | 0.517 | 9.97% | 10.25% | 3.14% | kept |
| `r5` | Naive meandering: integrate wakes over 100% of raw nacelle vane StdDev (~7.7°) | 0.464 | 11.09% | 11.37% | 3.30% | rejected |
| `r6` | `v3` physics (dev twins): meso-scale meandering (`0.16–0.36×` vane $\sigma_\theta$) + 3-D hill speedup | **0.552** | **9.38%** | **9.56%** | **2.88%** | **released** |
| `r7` | Softmax attention over upstream edges in place of additive sigmoid-gated sum | 0.553 | 9.41% | 9.60% | 2.89% | rejected (+0.000) |
| `r8` | Fine-tune from `r6` on mixed `v2 + v3` banks (48,000 simulated farms) | 0.543 | 9.62% | 9.85% | 2.99% | rejected (−0.009) |
| `r9` | Same network anchored on the dev-tuned physics instead of textbook FLORIS (17 Nov 2026; seeds 106/107/108: 0.5543 / 0.5539 / 0.5542) | 0.554 | 9.30% | 9.56% | 2.86% | rejected (+0.002) |
| `r10` | Physics v4 (thrust scale drawn around the dev value 0.90) and the network anchored on v4-tuned physics; scored on the new frozen set `val4` (seeds 0.5132 / 0.5135 / 0.5131) | 0.513 (`val4`) | 9.36% | 9.61% | 2.89% | kept, not released |

Reference rows, not candidates (`training/v11_reference.json`): textbook FLORIS alone scores 0.512 on `val3` (dev 10.00% / 10.29% / 3.19%); the physics tuned on 2019 dev scores 0.541 (dev **9.18% / 9.42% / 2.86%**). In simulation the network adds +0.011 over its anchor; on real dev the tuned physics alone beats every rung. That gap between simulation and reality is the model's main open problem. A dev twin test (`training/twin_residuals.py`) shows why: the network's corrections are uncorrelated with what real turbines needed (r = −0.04 over 16,212 dev readings), while the real residuals follow wind speed (+8 % at 5-9 m/s, −1 % near rated) and turbulence. Physics v4 adds the one dev-tuned setting that helps (thrust ×0.90: dev 9.31 → 9.25 %). On its new frozen set `val4` the incumbent r6 re-scores at 0.5008, and r10 clears the rule (+0.012, all dev gates pass) so it is kept; it is not released because on real dev it is no better than r6 (9.50 % against 9.48 %). The v1.0 build kept no checkpoints for r1-r5, r7 and r8, so only r6 and r9 could be re-scored on `val4` (`training/v4_reference.json`).

**v1.0.1 error bars.** The network's own bar is about 11.7 % wide at every wind speed, while its dev error is 14.3 % at 5-9 m/s and 5.7 % at 10.5-12 m/s (dev coverage 47.5 % and 88.0 %). v1.0.1 keeps the network and its predictions and multiplies the bars by widen-only factors per wind-speed bin fitted on dev (`training/calib_sigma.py`): waked turbines ×1.57 below 9 m/s, ×1.11 at 9-10.5 m/s, ×1.00 above; farm wake loss ×1.50 below 9 m/s.

## Exam

Error is the mean absolute error in a turbine's power as a share of the unwaked front row's power (farm row: in farm-wide wake loss). Verdicts come from a 95% interval on WakeNet minus the best baseline, by bootstrap over calendar days (2,000 resamples): win if the interval is below zero, loss if above, tie if it spans zero. The baselines were fixed in `NOTEBOOK.md` before the data were re-read. "FLORIS tuned on dev" is the same physics with the three settings from 2019 dev (`training/dev_calibration.json`), and is the strongest classical baseline.

### Exam 2: years no version has seen (`training/exam2_results.json`)

Frozen in `training/split_exam2.json` on 17 Nov 2026 before anything was scored on it; scored once, after r9 was rejected.

**Kelmarsh 2022-2024** (11,750 snapshots on 400 days; 1 win, 3 tie, 5 loss):

| row | n | WakeNet | FLORIS tuned on dev | FLORIS textbook | Jensen (k 0.075) | dev mean | no wake | WakeNet − best [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| all waked turbines | 29,308 | 10.96% | **10.82%** | 11.77% | 11.92% | 14.78% | 14.76% | +0.14 [+0.07, +0.21] | loss |
| directly behind another (dx < 8D, |dy| < 0.65D) | 8,890 | 13.15% | **13.32%** | 14.13% | 13.85% | 16.21% | 21.76% | -0.17 [-0.26, -0.07] | win |
| inside ≥ 2 wakes | 2,236 | 11.28% | **11.19%** | 12.65% | 13.72% | 15.30% | 19.74% | +0.09 [-0.10, +0.28] | tie |
| upstream yaw ≥ 8° | 7,855 | 13.73% | **13.78%** | 14.46% | 14.55% | 15.08% | 13.98% | -0.04 [-0.21, +0.11] | tie |
| 1-2 turbines parked | 7,352 | 12.14% | **11.93%** | 13.04% | 12.90% | 15.36% | 15.40% | +0.21 [+0.08, +0.34] | loss |
| wind < 10 m/s | 16,401 | 13.73% | **13.52%** | 14.83% | 15.16% | 17.63% | 18.17% | +0.20 [+0.11, +0.28] | loss |
| wind 10.5-12 m/s | 8,220 | 6.10% | **5.97%** | 6.32% | 6.34% | 10.06% | 8.11% | +0.13 [+0.03, +0.23] | loss |
| highest quarter of turbines | 10,607 | 12.22% | **11.87%** | 13.33% | 12.66% | 16.21% | 14.13% | +0.34 [+0.20, +0.47] | loss |
| farm-wide wake loss | 11,750 | 3.39% | **3.38%** | 3.66% | 3.73% | 4.76% | 5.32% | +0.01 [-0.02, +0.04] | tie |

**Penmanshiel 2021-2023, the farm never used for anything else** (23,842 snapshots on 663 days; WT01-10 has no January 2023; 0 win, 1 tie, 8 loss):

| row | n | WakeNet | FLORIS tuned on dev | FLORIS textbook | Jensen (k 0.075) | dev mean | no wake | WakeNet − best [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| all waked turbines | 208,508 | 19.78% | **19.50%** | 21.75% | 21.77% | 19.92% | 21.37% | +0.27 [+0.19, +0.36] | loss |
| directly behind another (dx < 8D, |dy| < 0.65D) | 53,130 | 25.12% | 25.30% | 27.37% | 26.23% | **20.59%** | 23.59% | +4.52 [+3.58, +5.46] | loss |
| inside ≥ 2 wakes | 67,650 | 22.80% | 22.49% | 24.97% | 25.73% | **21.11%** | 22.51% | +1.69 [+1.16, +2.24] | loss |
| upstream yaw ≥ 8° | 65,614 | 21.56% | **21.36%** | 23.62% | 23.84% | 21.75% | 23.24% | +0.19 [+0.07, +0.31] | loss |
| 1-2 turbines parked | 53,770 | 19.17% | **18.92%** | 21.17% | 21.43% | 19.25% | 20.57% | +0.25 [+0.07, +0.40] | loss |
| wind < 10 m/s | 114,194 | 24.19% | 24.10% | 26.57% | 26.84% | **23.68%** | 25.90% | +0.51 [+0.11, +0.94] | loss |
| wind 10.5-12 m/s | 68,917 | 13.46% | **12.93%** | 14.77% | 14.49% | 14.66% | 15.03% | +0.54 [+0.46, +0.62] | loss |
| highest quarter of turbines | 66,219 | 22.64% | 21.52% | 25.49% | 26.39% | 23.10% | **19.33%** | +3.31 [+2.92, +3.71] | loss |
| farm-wide wake loss | 23,842 | 7.08% | 7.20% | 7.67% | 7.88% | **7.01%** | 8.90% | +0.07 [-0.20, +0.36] | tie |

### Exam 1, re-scored (`training/exam_results.json`)

Run once for v1.0 on 16 Nov 2026 (`training/exam_results_2026-11-16.json`: 11 rows reported, 8 wins and 3 losses, against textbook FLORIS, offshore Jensen and two constant guesses). Re-scored on 17 Nov 2026 with the same weights and data (the processed files reproduce the frozen timestamp hashes); nothing was chosen from it.

**Kelmarsh 2020-2021** (15,059 snapshots; 1 win, 0 tie, 8 loss):

| row | n | WakeNet | FLORIS tuned on dev | FLORIS textbook | Jensen (k 0.075) | dev mean | no wake | WakeNet − best [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| all waked turbines | 39,008 | 8.85% | **8.64%** | 9.51% | 9.97% | 14.00% | 13.63% | +0.22 [+0.17, +0.27] | loss |
| directly behind another (dx < 8D, |dy| < 0.65D) | 11,977 | 10.07% | **10.21%** | 10.75% | 11.30% | 15.30% | 22.17% | -0.13 [-0.23, -0.04] | win |
| inside ≥ 2 wakes | 2,212 | 8.86% | **8.60%** | 9.90% | 11.18% | 14.46% | 18.63% | +0.26 [+0.11, +0.40] | loss |
| upstream yaw ≥ 8° | 5,401 | 9.76% | **9.54%** | 10.46% | 11.15% | 14.71% | 13.75% | +0.22 [+0.16, +0.30] | loss |
| 1-2 turbines parked | 6,146 | 9.18% | **8.82%** | 9.94% | 9.94% | 14.63% | 13.62% | +0.35 [+0.24, +0.45] | loss |
| wind < 10 m/s | 18,897 | 11.73% | **11.45%** | 12.78% | 13.40% | 17.34% | 17.96% | +0.28 [+0.22, +0.34] | loss |
| wind 10.5-12 m/s | 14,043 | 5.30% | **5.07%** | 5.37% | 5.64% | 9.90% | 7.70% | +0.23 [+0.15, +0.30] | loss |
| highest quarter of turbines | 13,718 | 9.33% | **8.88%** | 10.34% | 10.02% | 15.08% | 12.84% | +0.45 [+0.36, +0.53] | loss |
| farm-wide wake loss | 15,059 | 2.57% | **2.53%** | 2.80% | 3.02% | 4.44% | 4.87% | +0.03 [+0.01, +0.06] | loss |

**Penmanshiel 2019-2020** (8,976 snapshots; 0 win, 0 tie, 9 loss):

| row | n | WakeNet | FLORIS tuned on dev | FLORIS textbook | Jensen (k 0.075) | dev mean | no wake | WakeNet − best [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|---|
| all waked turbines | 78,327 | 15.01% | **14.43%** | 16.63% | 16.43% | 15.88% | 16.00% | +0.58 [+0.49, +0.67] | loss |
| directly behind another (dx < 8D, |dy| < 0.65D) | 20,036 | 18.58% | 18.20% | 20.44% | 19.21% | **15.63%** | 18.59% | +2.95 [+2.07, +3.88] | loss |
| inside ≥ 2 wakes | 24,308 | 17.32% | 16.49% | 19.12% | 19.32% | **16.43%** | 16.51% | +0.89 [+0.37, +1.48] | loss |
| upstream yaw ≥ 8° | 14,813 | 15.08% | **14.41%** | 16.75% | 16.62% | 16.34% | 16.27% | +0.67 [+0.56, +0.78] | loss |
| 1-2 turbines parked | 8,047 | 15.99% | **15.45%** | 17.79% | 17.72% | 16.33% | 16.43% | +0.54 [+0.30, +0.75] | loss |
| wind < 10 m/s | 14,210 | 20.85% | 20.29% | 23.09% | 22.91% | 19.87% | **18.51%** | +2.35 [+1.53, +3.16] | loss |
| wind 10.5-12 m/s | 48,391 | 12.86% | **12.25%** | 14.16% | 13.94% | 14.32% | 14.70% | +0.60 [+0.51, +0.69] | loss |
| highest quarter of turbines | 26,756 | 15.92% | 14.23% | 18.30% | 18.58% | 18.44% | **13.19%** | +2.72 [+2.34, +3.11] | loss |
| farm-wide wake loss | 8,976 | 4.75% | 4.82% | 5.16% | 5.16% | **4.55%** | 5.98% | +0.20 [+0.02, +0.40] | loss |

### Error bars

Share of readings inside the nominal 68% bar. v1.0.1 changes only the bars (per-wind-speed factors fitted on dev); both exams had been seen before that change.

| set | waked turbines v1.0 | waked turbines v1.0.1 | farm wake loss v1.0 | farm wake loss v1.0.1 |
|---|---|---|---|---|
| Kelmarsh 2020-21 (exam 1) | 72.2% | 77.6% | 77.4% | 80.2% |
| Kelmarsh 2022-24 (exam 2) | 64.1% | 70.3% | 66.8% | 70.8% |
| Penmanshiel 2019-20 (exam 1) | 50.7% | 52.3% | 61.5% | 61.5% |
| Penmanshiel 2021-23 (exam 2) | 41.9% | 48.9% | 48.9% | 54.3% |

The Jensen baseline uses the onshore wake expansion 0.075. With the offshore value 0.05 that the 16 Nov exam used, Jensen scores 9.95% / 17.90% on exam 1 and 12.29% / 24.10% on exam 2 (all waked turbines, Kelmarsh / Penmanshiel).

## Limits and known failure modes

- **The network adds nothing over tuned physics on real data.** See the headline and the reference rows under Ratchet. On real dev and both exams, FLORIS with three dev-tuned settings matches or beats the network on almost every row.
- **Complex coastal ridges and forestry (Penmanshiel losses to constant guesses).** Penmanshiel climbs 48 m from `T07` (`180.2 m`) to `T15` (`228.2 m`) across forested Berwickshire terrain. Under NW and SW flow, topographic ridge speedup accelerates wind onto hilltop units `T12–T15` so strongly that waked hilltop turbines often generate *more* power than unwaked lower-slope front-row units (`truth = -45%` wake loss on `T15` in demo snapshot 06). WakeNet's tower-elevation feature captures part of this (beating FLORIS by 2.4 points on hilltop units, 15.92% vs 18.30%), but without a surrounding digital elevation map (DEM) and forest-canopy roughness model it still underpredicts hilltop power by `-9.2%` and loses to the dumb baselines on Penmanshiel multi-wake, hilltop, and farm-total rows.
- **Zero-shot error-bar under-coverage on steep terrain.** Because Kelmarsh 2019 dev has gentle 21.5 m farmland relief, dev residuals needed no widening (`factor = 1.00`, giving 72.2% coverage on Kelmarsh 2020–21). On Penmanshiel's 48 m relief, unmodelled terrain variance drops waked-turbine coverage to 50.7% (41.9% in 2021-23); Kelmarsh itself falls to 64.1% in 2022-24.
- **Unobserved atmospheric stability.** Standard nacelle SCADA measures wind speed and vane scatter, not surface heat flux or Obukhov length $L$. Convective afternoon wakes mix out faster and nocturnal stable wakes persist farther than a single $TI$-conditioned curve predicts.
- **Below-rated regime only ($5\text{–}12\text{ m/s}$).** Above $12.5\text{–}14\text{ m/s}$, pitch regulation holds both front-row and waked turbines near 2,050 kW.

## Data and physics credits

- **Kelmarsh Wind Farm Data:** Charlie Plumley / Cubico Sustainable Investments, Zenodo 2025 version `10.5281/zenodo.16807551` (concept DOI `10.5281/zenodo.5841834`), CC BY 4.0.
- **Penmanshiel Wind Farm Data:** Charlie Plumley / Cubico Sustainable Investments, Zenodo 2025 version `10.5281/zenodo.16807304` (concept DOI `10.5281/zenodo.5946808`), CC BY 4.0.
- **Physics references:** Jensen (1983), Risø-M-2411; Katić, Højstrup & Jensen (1986), EWEC; Jiménez, Crespo & Migoya (2010), *Wind Energy* 13; Bastankhah & Porté-Agel (2014), *Renewable Energy* 70; Bastankhah & Porté-Agel (2016), *J. Fluid Mech.* 806; Crespo & Hernández (1996), *J. Wind Eng. Ind. Aerodyn.* 61; Gaumond et al. (2014), *Wind Energy* 17. Verified against NREL `FLORIS` v4.7.
