# WakeNet v1.0 — model card

**What it does.** Given a wind farm's turbine coordinates, tower hub heights, rotor power and thrust curves, online status, mechanical nacelle yaw angles, and the incoming wind measured on its geometrically unwaked front-row turbines, WakeNet predicts how much power every downstream turbine is losing to its neighbours' wakes — per turbine and across the whole farm — each with a calibrated 68% error bar.

**How it learned.** Only from procedural physics simulations laid into real unwaked turbulence. Zero real waked power labels were used in training, architecture selection, or calibration.

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
- **Real backdrop (`training/noise.py`):** 47,610 unwaked front-row rotor readings from Kelmarsh 2017–2018 supply empirical UK inflow tuples and real spatial turbulent power fluctuations (`10.1%` std across unwaked rotors) and nacelle yaw-tracking offsets (`4.9°` std), split chronologically 85% / 15% between training and frozen validation banks. Kelmarsh's 2016 commissioning year was under a `~1,680 kW` grid export setpoint cap, so 0 of 52,416 rows passed the uncurtailed screen (`setpoint >= 2,000 kW`).

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

## Exam

Run once after `r6` was locked (`training/exam_results.json`), on **24,035 held-out 10-minute snapshots (117,335 waked turbine readings)**:
1. **Kelmarsh 2020–2021:** 15,059 snapshots (39,008 waked readings) on the 6-turbine Senvion MM92 farm ($D = 92\text{ m}$, mixed $68.5 / 78.5\text{ m}$ towers, $21.5\text{ m}$ farmland relief).
2. **Penmanshiel 2019–2020 (zero-shot unseen farm):** 8,976 snapshots (78,327 waked readings) on a 14-turbine Senvion MM82 farm in Berwickshire, Scotland ($D = 82\text{ m}$, $59\text{ m}$ towers, $48\text{ m}$ coastal ridge relief), never seen in backdrop, priors, calibration, or dev gating.

| exam metric | WakeNet | NREL FLORIS 3-D Gauss | Jensen + Jiménez | Dev mean (dumb) | Unwaked (dumb) | verdict |
|---|---|---|---|---|---|---|
| Kelmarsh, all waked turbines MAE ($n = 39,008$) | **8.85%** | 9.51% | 9.95% | 14.00% | 13.63% | **win** |
| Kelmarsh, direct-wake corridor MAE ($n = 11,977$) | **10.07%** | 10.75% | 11.63% | 15.30% | 22.17% | **win** |
| Kelmarsh, multi-wake $\ge 2$ overlap MAE ($n = 2,212$) | **8.86%** | 9.90% | 10.12% | 14.46% | 18.63% | **win** |
| Kelmarsh, upstream yaw $\ge 8^\circ$ MAE ($n = 5,401$) | **9.76%** | 10.46% | 11.25% | 14.71% | 13.75% | **win** |
| Kelmarsh, partial-parked farm MAE ($n = 6,146$) | **9.18%** | 9.94% | 9.81% | 14.63% | 13.62% | **win** |
| Kelmarsh, farm-wide wake loss MAE ($n = 15,059$) | **2.57%** | 2.80% | 2.99% | 4.44% | 4.87% | **win** |
| Penmanshiel (zero-shot), all waked MAE ($n = 78,327$) | **15.01%** | 16.63% | 17.90% | 15.88% | 16.00% | **win** |
| Penmanshiel (zero-shot), upstream yaw $\ge 8^\circ$ ($n = 14,813$) | **15.08%** | 16.75% | 18.05% | 16.34% | 16.27% | **win** |
| Penmanshiel (zero-shot), multi-wake $\ge 2$ interior ($n = 24,308$) | 17.32% | 19.12% | 21.94% | **16.43%** | 16.51% | **loss** |
| Penmanshiel (zero-shot), hilltop ridge turbines ($n = 26,756$) | 15.92% | 18.30% | 21.08% | 18.44% | **13.19%** | **loss** |
| Penmanshiel (zero-shot), farm-wide wake loss ($n = 8,976$) | 4.75% | 5.16% | 5.82% | **4.55%** | 5.98% | **loss** |

Additional Wind-Speed Strata (both farms):
- Kelmarsh cubic Region II ($< 10\text{ m/s}$): WakeNet **11.73%** vs FLORIS 12.78%, Jensen 13.49%, Unwaked 17.96%. Near-rated ($10.5\text{–}12\text{ m/s}$): WakeNet **5.30%** vs FLORIS 5.37%, Jensen 5.48%, Unwaked 7.70%.
- Penmanshiel near-rated ($10.5\text{–}12\text{ m/s}$, $n = 48,391$, where MM82 is still below its $14\text{ m/s}$ rated knee): WakeNet **12.86%** vs FLORIS 14.16%, Dev mean 14.32%, Unwaked 14.70%. Below $10\text{ m/s}$ ($n = 14,210$): WakeNet 20.85% (bias $-11.6\%$) beats FLORIS (23.09%) and Jensen (25.36%) but loses to Unwaked (18.51%).

Coverage of the nominal 68% error bars on the exam:

| stratum | 68% bar coverage |
|---|---|
| Kelmarsh waked turbines | 72.2% |
| Kelmarsh farm wake loss | 77.4% |
| Penmanshiel waked turbines (zero-shot) | 50.7% |
| Penmanshiel farm wake loss (zero-shot) | 61.5% |

## Limits and known failure modes

- **Complex coastal ridges and forestry (the three Penmanshiel losses).** Penmanshiel climbs 48 m from `T07` (`180.2 m`) to `T15` (`228.2 m`) across forested Berwickshire terrain. Under NW and SW flow, topographic ridge speedup accelerates wind onto hilltop units `T12–T15` so strongly that waked hilltop turbines often generate *more* power than unwaked lower-slope front-row units (`truth = -45%` wake loss on `T15` in demo snapshot 06). WakeNet's tower-elevation feature captures part of this (beating FLORIS by 2.4 points on hilltop units, 15.92% vs 18.30%), but without a surrounding digital elevation map (DEM) and forest-canopy roughness model it still underpredicts hilltop power by `-9.2%` and loses to the dumb baselines on Penmanshiel multi-wake, hilltop, and farm-total rows.
- **Zero-shot error-bar under-coverage on steep terrain.** Because Kelmarsh 2019 dev has gentle 21.5 m farmland relief, dev residuals needed no widening (`factor = 1.00`, giving 72.2% coverage on Kelmarsh 2020–21). On Penmanshiel's 48 m relief, unmodelled terrain variance drops waked-turbine coverage to 50.7%.
- **Unobserved atmospheric stability.** Standard nacelle SCADA measures wind speed and vane scatter, not surface heat flux or Obukhov length $L$. Convective afternoon wakes mix out faster and nocturnal stable wakes persist farther than a single $TI$-conditioned curve predicts.
- **Below-rated regime only ($5\text{–}12\text{ m/s}$).** Above $12.5\text{–}14\text{ m/s}$, pitch regulation holds both front-row and waked turbines near 2,050 kW.

## Data and physics credits

- **Kelmarsh Wind Farm Data:** Charlie Plumley / Cubico Sustainable Investments, Zenodo (`10.5281/zenodo.5841834`, record `16807551`), CC BY 4.0.
- **Penmanshiel Wind Farm Data:** Charlie Plumley / Cubico Sustainable Investments, Zenodo (`10.5281/zenodo.5946808`, record `16807304`), CC BY 4.0.
- **Physics references:** Jensen (1983), Risø-M-2411; Katić, Højstrup & Jensen (1986), EWEC; Jiménez, Crespo & Migoya (2010), *Wind Energy* 13; Bastankhah & Porté-Agel (2014), *Renewable Energy* 70; Bastankhah & Porté-Agel (2016), *J. Fluid Mech.* 806; Crespo & Hernández (1996), *J. Wind Eng. Ind. Aerodyn.* 61; Gaumond et al. (2014), *Wind Energy* 17. Verified against NREL `FLORIS` v4.7.
