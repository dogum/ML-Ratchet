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

## 2026-11-17 — Review findings and the plan, written before any data was re-read

A review on 17 Nov 2026 (Claude, for the owner) checked the v1.0 branch. Every repo check passed, and the site
instrument works. It found three problems:

1. **The strongest classical baseline was missing.** `calib_dev.py` tuned the physics on 2019 dev (meandering
   0.25 × vane σ_θ, hill speedup 0.18, shear 0.14, terrain-follow 0.80). That tuned physics alone scores 9.31 % waked MAE
   on dev (stride 2) against WakeNet r6's 9.48 %, and 2.875 % against 2.88 % on farm wake loss. The exam compared
   WakeNet only with textbook FLORIS, so its Kelmarsh margins may come from the dev tuning, not the graph network.
2. **The card reported 11 of the 18 rows `exam.py` computed.** All 18: 13 wins, 5 losses. The two left out were
   both losses: Penmanshiel direct-wake corridor (18.58 % against the dev-mean guess's 15.63 %) and Penmanshiel below
   10 m/s (20.85 % against 18.51 % for assuming no wake).
3. **The full-screen lab page crashed** (`man.tensors is not iterable`): it read the pack manifest as a list.

Smaller: no confidence intervals; the Jensen baseline used the offshore wake-expansion value 0.05 (onshore is usually
0.075); citations pointed at the 2022 Zenodo records, not the 2025 versions the data came from; the backdrop starts
25 Sep 2017, not 1 Jan.

### Plan (fixed now, before the data are downloaded again)

**Exam 1 re-scored (already seen; nothing is chosen from it).** Same weights, same processed data; the processed
files must reproduce the `ts_sha256` values in `split.json`, and WakeNet's own numbers must reproduce exactly.
- Add the baseline "FLORIS tuned on 2019 dev": `calibrated_v3` from `dev_calibration.json`, unchanged.
- Jensen with the onshore wake expansion 0.075 (the 0.05 result stays in a note).
- Report all 9 rows on both farms.
- 95 % intervals on WakeNet minus the best baseline by bootstrap over calendar days (2,000 resamples, seed 20261117).
  Verdict: win if the interval is below zero, loss if above, tie if it spans zero.

**v1.1 rungs (dev only, same rule as `ratchet.yaml`: +0.01 on `val3` and each dev gate ≤ 1.05 × incumbent).**
- Reference row, not a candidate: the tuned physics alone, scored on `val3` and dev.
- r9: the same EdgeWakeGNN, anchored on the dev-tuned physics instead of textbook FLORIS; banks rebuilt with the same
  seeds as r6 (train seed 3, `val3` seed 303), 3,000 steps, seed 106. Three seeds if the margin is under twice the seed
  spread. σ factors re-fitted widen-only on dev if it is released.
- Any further rung gets its own dated entry here before it runs.

**Exam 2 (never seen by any version).** Kelmarsh 2022, 2023, 2024; Penmanshiel 2021, 2022 and 2023 (WT01-10 is
published for Feb-Dec 2023 only; Penmanshiel 2024 has only WT11-15 and is left out). Same quality control as
`realdata.build_snapshots`. Frozen in `split_exam2.json` with timestamp hashes before anything is scored. Same 9 rows,
same five baselines (tuned FLORIS, FLORIS, Jensen 0.075, dev mean, unwaked), same interval rule. Scored once, for v1.0
and for v1.1 if it is released, after the v1.1 decision.

## 2026-11-17 — Results: exam 1 re-scored, r9 rejected, exam 2 taken

Everything below followed the plan in the entry above; nothing was changed after a result was seen.

- **Rebuilt data reproduce the frozen split.** All four v1.0 processed sets, rebuilt from the Zenodo 2025 versions,
  match `split.json`'s timestamp hashes. On the rebuilt `val3` and dev cache, the released weights reproduce
  `ratchet.yaml` exactly (sim 0.5522; dev 9.38 / 9.56 / 2.88 %). Textbook FLORIS computed in `exam.py` equals v1.0's
  anchor to 0.0.
- **Exam 1 re-scored.** WakeNet's own numbers reproduce (Kelmarsh all waked 8.85 %, Penmanshiel 15.01 %). FLORIS tuned on
  dev scores 8.64 % and 14.43 %. Against it WakeNet loses 17 of 18 rows; it wins only the Kelmarsh direct-wake corridor
  (10.07 % against 10.21 %, −0.13 [−0.23, −0.04]).
- **Reference rows** (`v11_reference.json`): textbook physics alone 0.512 on `val3`, tuned physics alone 0.541, r6
  0.552. The network beats its anchor in simulation and loses to it on real dev (tuned physics alone: 9.31 % waked MAE,
  2.86 % farm).
- **r9 rejected.** The same network anchored on the tuned physics scores 0.5543 / 0.5539 / 0.5542 for seeds 106 / 107 /
  108: +0.002 over r6, against a margin of 0.01. On dev it still trails the tuned physics alone (9.44 % against 9.31 %).
  No other rung was tried, so v1.0 stays released.
- **Exam 2 taken once for v1.0** (`exam2_results.json`; `split_exam2.json` sha256 `3543a6af…`, committed before r9 was
  judged). Kelmarsh 2022-24: 1 win, 3 ties, 5 losses. Penmanshiel 2021-23: 1 tie, 8 losses. The one win is again
  the direct-wake corridor (13.15 % against 13.32 %, −0.17 [−0.26, −0.07]). WakeNet beats textbook FLORIS and Jensen on
  every row of both exams. Error bars cover 64 % (Kelmarsh) and 42 % (Penmanshiel) of waked turbines.
- **What it means.** The v1.0 gains over FLORIS came from tuning three physics settings on dev, not from the graph
  network. The network learns corrections that help in simulation and not on real turbines, so the next rung is a twin
  test of those corrections, not a bigger network.

## 2026-11-17 — Dev twin test of the network, and the r10 plan (written before r10 runs)

`training/twin_residuals.py r9` compares, on 2019 dev only, what the network adds to its anchor with what reality
needed (truth minus the same anchor). Correlation over 16,212 waked readings: **−0.04**. The network's corrections do
not point where reality's residuals point. Those residuals have structure the inputs carry:

- wind speed: +8 % at 5-9 m/s (truth above the tuned physics), +1.6 % at 9-10.5 m/s, −0.6 to −1 % at 10.5-12 m/s;
- turbulence intensity (7-10.5 m/s): −3.3 % below 0.09, +0.8 %, +3.1 %, +4.8 % above 0.15;
- no day/night difference within a wind-speed bin, so atmospheric stability beyond TI is not visible here.

Real wakes recover faster with turbulence and are shallower at low wind than FLORIS's default settings say. In FLORIS
those are the wake-growth sensitivity to turbulence (`ka`, default 0.38) and the thrust the rotor puts on the flow (the
manufacturer Ct curve). The simulator never varied either, so the network could not learn them.

**r10 (physics v4).**
1. Add `ct_scale` (multiplies the manufacturer thrust curve; 1.0 leaves every earlier result unchanged).
2. Extend the dev grid of `calib_dev.py` with `ka` ∈ {0.38, 0.5, 0.65, 0.8} and `ct_scale` ∈ {1.0, 0.9, 0.8},
   holding the v3 settings, and pick by the same score (0.65 waked MAE + 0.35 farm MAE). Stride 6 for the grid, stride 2
   for the chosen setting, as before.
3. v4 simulator: v3 plus `ka` and `ct_scale` drawn per farm around the dev values (± 25 % and ± 0.08), so the network
   sees the variation. New frozen set `val4` (seed 404) and training bank `train_v4` (seed 4), anchored on the v4-tuned
   physics.
4. Re-score the incumbent r6 (released weights, numpy) and r9 (checkpoint) on `val4`. The v1.0 build did not keep
   checkpoints for r1-r5, r7 and r8, so they cannot be re-scored; that is noted, not hidden.
5. r10: the same network on `train_v4`, 3,000 steps, seeds 106 / 107 / 108; same rule (+0.01 on the frozen set, each dev
   gate ≤ 1.05 × incumbent).
6. Fair baseline: any v1.1 is compared with FLORIS tuned with the same v4 settings, as well as with the v3 settings.
   Both exams are now seen, so v1.1's exam numbers are reported as seen, not as a fresh test.
7. Error bars: widen-only factors per wind-speed bin (5-9, 9-10.5, 10.5-12 m/s) fitted on dev, whatever the rung
   decision.

## 2026-11-17 — r10 result, release choice, and per-wind-speed error bars

- **v4 dev grid** (`dev_calibration_v4.json`): the best setting keeps `ka` at 0.38 and scales the thrust curve by 0.90.
  Dev waked MAE 9.31 → 9.25 %, farm 2.88 → 2.82 %. The wind-speed and turbulence biases are real, but turbine-to-turbine
  turbulence (about 10 % of power) dominates every model's error, so the gain is small.
- **r10 kept, not released.** On the new frozen set `val4` the network scores 0.5132 / 0.5135 / 0.5131 (seeds 106-108)
  against r6's 0.5008 re-scored there: +0.012, and every dev gate passes, so the rule keeps it. But on real dev it is no
  better than r6 (9.50 % against 9.48 % waked MAE) and v4 physics alone does better (9.25 %). Releasing it would change
  what users get without improving it on real turbines, so v1.0's network (r6) stays released. This release choice was
  made on dev; neither exam was looked at for r10.
- **Error bars, v1.0.1.** v1.0's bar is about 11.7 % wide at every wind speed, while its dev error is 14.3 % at 5-9 m/s,
  10.3 % at 9-10.5 m/s and 5.7 % at 10.5-12 m/s; dev coverage was 47.5 %, 63.6 % and 88.0 %. `calib_sigma.py` fits one
  widen-only factor per bin on dev (never below 1). The network and its point predictions are unchanged; only the bars
  move. The exams' coverage for v1.0.1 is reported as seen.
- **v1.0.1 coverage on the (seen) exams** (waked turbines, v1.0 → v1.0.1): Kelmarsh 2020-21 72.2 → 77.6 %, Kelmarsh
  2022-24 64.1 → 70.3 %, Penmanshiel 2019-20 50.7 → 52.3 %, Penmanshiel 2021-23 41.9 → 48.9 %. Point predictions and
  every MAE are unchanged. Penmanshiel stays under-covered: terrain, not wind speed, drives its misses.

## 27 Nov 2026 — audit resolution

The two-reviewer audit (`docs/audits/`). The demo summary now scores all 31 waked readings, as the exam does (MAE 14.9 %, was 14.0 % on the 24 shown).
