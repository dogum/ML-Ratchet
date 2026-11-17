# WakeNet — lab notebook

Build log for WakeNet (`models/wakenet/`), recorded 16 Nov 2026. All raw downloads, processed SCADA tables, simulation banks, and checkpoints live in git-ignored `models/wakenet/work/` via `training/_paths.py`.

## 2026-11-16 — Physics verification, SCADA quirks, 8-rung ratchet, and zero-shot farm exam

### 1. Problem formulation (causal, zero downstream leakage)
- **Question:** How much power is each wind turbine in a farm losing to its neighbours' wakes?
- **Trap avoided:** Reading a downstream turbine's nacelle wind speed $U_i$ makes power prediction a trivial lookup on $P(U_i)$. Instead, WakeNet identifies the **geometrically unwaked front-row turbines** $F$ purely from coordinates, wind bearing, and online status (`physics.freestream_mask`), reads $(U_\infty, \theta_\infty, TI_\infty, \sigma_\theta, \rho, \bar{P}_{\text{front}})$ *only* from $F$, and reads only mechanical control states (`online_i` and nacelle orientation $\phi_{\text{nac},i} \to \gamma_i = \theta_\infty - \phi_{\text{nac},i}$) on every turbine. Downstream anemometers, wind vanes, and power meters are strictly held out as the ground-truth exam target $\eta_i = P_i / \bar{P}_{\text{front}}$.

### 2. Physics engine & FLORIS 4.7 verification (`physics.py`, `training/physics_check.py`)
- Built sequential upwind-to-downwind 3-D wake solver on a $3 \times 3$ rotor grid (`[-0.25 D, 0, +0.25 D]`, cubic-mean rotor velocity) with both Jensen (1983) + Jiménez (2010) and Bastankhah & Porté-Agel (2014, 2016) Gaussian deficit + skew deflection + Crespo & Hernández (1996) wake-added turbulence + SOSFS superposition.
- **Subtleties caught while matching NREL FLORIS 4.7 to machine precision:**
  1. `CosineLossTurbine` multiplies $C_T$ by $\cos\gamma$ and evaluates axial induction as $a = \frac{1}{2\cos\gamma}(1 - \sqrt{1 - C_{T,\text{raw}}\cos^2\gamma})$.
  2. Positive compass yaw $\gamma = \theta_{\text{wind}} - \phi_{\text{nacelle}} > 0$ deflects wakes toward $-y$ in the right-handed `(downwind, left-spanwise, up)` frame (Newton's third law agrees with FLORIS's `yaw *= -1`).
  3. Crespo–Hernández combines wake-added turbulence on the $3\times 3$ grid by elementwise `max` within a $15 D \times \pm 2 D$ box, and Jensen adds `NUM_EPS = 0.001` m to the cone radius denominator.
- **Verification (`training/physics_check.json`):**
  - Bastankhah 3-D Gaussian grid ($3D\text{–}15D \times -1.5D\dots+1.5D$): `1.58e-15` max relative power diff.
  - Jensen + Jiménez ($-20^\circ\dots+20^\circ$ yaw): `2.91e-16` max relative power diff (`0.0 m/s` velocity diff).
  - 3-turbine Gaussian yaw-steering sweep ($-25^\circ\dots+25^\circ$) + Crespo–Hernández + SOSFS: `1.88e-15` max relative power diff.
  - 14-turbine Penmanshiel layout $\times$ 72 wind directions (1,008 states): `r = 1.000000`, `7.37e-15` max relative power diff.
  - Rotational equivariance ($+37.5^\circ$) and `online=False` removal invariants: `< 1e-12 kW`.

### 3. Real SCADA quirks & frozen split (`training/realdata.py`, `training/split.json`)
- Downloaded Kelmarsh (Zenodo `16807551`, 6 $\times$ Senvion MM92, $D = 92\text{ m}$, mixed $68.5\text{ m}$ [`KWF3, KWF6`] and $78.5\text{ m}$ towers, $135.0\text{–}156.6\text{ m}$ elevation) and Penmanshiel (Zenodo `16807304`, 14 $\times$ Senvion MM82, `T01..T15` with `T03` never built, $D = 82\text{ m}$, $59\text{ m}$ towers, $180.2\text{–}228.2\text{ m}$ elevation).
- **Quirk #1 (CSV header & setpoint NaNs):** Greenbyte prefixes the column header row (line 10) with `# `, and leaves `Turbine Power setpoint (kW)` as `NaN` on ~10% of normal uncurtailed rows. Filtering `~(setpoint < 2000)` instead of `setpoint >= 2000` preserves uncurtailed rows while catching active export caps.
- **Quirk #2 (Kelmarsh 2016 commissioning cap):** Throughout 2016, Kelmarsh operated under a `~1,680 kW` grid export setpoint limit, so `0 / 52,416` rows in 2016 pass the uncurtailed gate, whereas 2017–2018 yield `15,420` clean uncurtailed farm snapshots (`47,610` unwaked rotor readings).
- **Compass check (2018 backdrop only):** Empirical wake troughs for `KWF2 -> KWF3` (`[243°, 249°)`, median $P_3/P_2 = 0.41$) and `KWF2 -> KWF1` (`[321°, 327°)`, median $P_1/P_2 = 0.25$) align within `1.5°` of GPS bearings (`248.5°` and `323.1°`) with zero compass offset needed.
- **Frozen split (`split.json`, `sha256: bd1dd01f…`):**
  - Backdrop (noise & inflow priors, zero wake labels): Kelmarsh 2016–2018 (`15,420` snapshots).
  - Dev (calibration, twin test, per-half-year gate, $\sigma$ widening): Kelmarsh 2019 (`5,729` snapshots: `H1` 2,691, `H2` 3,038).
  - Exam (opened once in `exam.py`): Kelmarsh 2020–2021 (`15,059` snapshots, `39,008` waked readings) + zero-shot Penmanshiel 2019–2020 (`8,976` snapshots, `78,327` waked readings).

### 4. Dev calibration trap & twin test (`training/calib_dev.py`, `training/twin.py`)
- **Trap caught early:** Conditioning "deep wakes" on `eta_true < 0.75` created an apparent `+11%` positive bias due to selection on noisy $y_{\text{true}}$ (`10.1%` turbulent spread). Conditioning instead on *independent geometry* ($\Delta x < 8 D, |\Delta y| < 0.65 D$) showed steady 3-D FLORIS is nearly unbiased on Kelmarsh centerlines (`-1.3%`).
- **Real physical finding #1 (nacelle-vane swirl):** Raw SCADA `Wind direction, StdDev` has median `7.74°` because spinning blades shed swirl directly onto the nacelle wind vane. Feeding 100% of `sigma_wd` as atmospheric wind meandering (`r5`) smeared wakes too wide (`+5.3%` centerline bias; dev waked MAE degraded `10.12% -> 11.24%`). Separating meso-scale meandering at `0.25×` vane $\sigma_\theta$ (`U(0.16, 0.36)` in `v3`) fixed centerline bias without washing out wakes.
- **Real physical finding #2 (twin test on low-elevation rotors):** `r4` (`v2` flat-ground 3-D Gaussian) read `v2` simulated twins with `0.00%` bias on low-elevation downstream units `KWF5–KWF6`, but overpredicted real 2019 dev `KWF5–KWF6` by `+2.27%` (`waked_mae = 10.12%` real vs `6.33%` twin). Adding rolling Gaussian hills with linearised hill speedup (`s_hill ~ U(0.05, 0.24)`, `terrain_follow ~ U(0.68, 0.95)`) in `v3` (`r6`) cut real `KWF5–KWF6` bias to `-0.47%`, dev waked MAE to `9.48%`, and dev farm MAE to `2.88%`.

### 5. Ratchet ladder summary (re-scored on `val3`)
- `r1` (node MLP, `v1` Jensen): `sim = 0.125`, dev H1/H2/farm `14.25 / 14.68 / 4.27%` -> **kept**.
- `r2` (unanchored `EdgeWakeGNN`, `v1`): `sim = 0.374` (`+0.249`), dev `10.39 / 10.74 / 3.38%` -> **kept**.
- `r3` (anchored `EdgeWakeGNN`, `v1`): rejected on 2-D Jensen `val1` by `+0.005` (`0.740` vs `0.735`), **flipped to kept** (`+0.053`, `0.459` vs `0.406`) when re-scored on 3-D Gaussian `val2`; `0.426` on `val3`, dev `10.12 / 10.43 / 3.29%`.
- `r4` (`v2` 3-D Gauss + real 2017–18 backdrop): `sim = 0.517` (`+0.091`), dev `9.97 / 10.25 / 3.14%` -> **kept**.
- `r5` (`v2_raw_vane` 100% raw vane meandering): `sim = 0.464` (`-0.053`), dev `11.09 / 11.37 / 3.30%` -> **rejected**.
- `r6` (`v3` separated meandering + 3-D hill speedup): `sim = 0.552` (`+0.035`), dev `9.38 / 9.56 / 2.88%` -> **released**.
- `r7` (`softmax_attn` instead of `gated_sum`): `sim = 0.553` (`+0.000`, short of `+0.01`), dev `9.41 / 9.60 / 2.89%` -> **rejected**.
- `r8` (mixed `v2 + v3` 48k bank fine-tune): `sim = 0.543` (`-0.009`, diluted terrain physics), dev `9.62 / 9.85 / 2.99%` -> **rejected**.

### 6. One-shot exam & parity
- PyTorch-to-numpy max abs diff: `2.38e-7` (`wakenet.py`). Python-to-TypeScript (`float16` `0.49 MB` pack, 24 cases) max rel diff: `1.88e-7`.
- **Kelmarsh 2020–21 (15,059 snapshots):** 6 of 6 wins (all waked `8.85%` vs FLORIS `9.51%`, farm wake `2.57%` vs `2.80%`; 68% bar coverage `72.2%` and `77.4%`).
- **Penmanshiel 2019–20 zero-shot (8,976 snapshots, 14 $\times$ MM82):** Beats FLORIS and Jensen on every stratum (`15.01%` vs `16.63%` FLORIS and `17.90%` Jensen overall; wins overall waked and yaw $\ge 8^\circ$), but loses to constant/unwaked dumb baselines on multi-wake interior (`17.32%` vs `16.43%` dev mean), hilltop turbines (`15.92%` vs `13.19%` unwaked), and farm wake loss (`4.75%` vs `4.55%` dev mean), with 68% bar coverage dropping to `50.7%` / `61.5%` due to steep 48 m Berwickshire coastal ridge + forestry speedups onto hilltop units `T12–T15`.
