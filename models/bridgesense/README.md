<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# BridgeSense <sub>v1.1</sub>

> **A Transformer trained on bridge FEMs separates real stiffness changes from winter freezes.**

> [!WARNING]
> Research only. Not a structural safety inspection tool; bridge assessments require licensed engineering inspection.

*From a bridge's tracked natural frequencies and temperature, has its stiffness actually changed — or is it just the weather — and by how much?*

From a bridge's healthy baseline window and a monitoring window of tracked natural frequencies (1 to 14 modes, with missing-mode dropouts) and air temperature history, a dual-axis Transformer (baseline-to-query thermal cross-attention plus masked mode-set self-attention; v1.1 ships a 3-member ensemble) estimates the probability of a real structural change, the effective global stiffness shift dK/K (%) and the environmental frequency shift (%), each with a 68% error bar. Trained only on 2-D thermo-mechanical modal FEM simulations (girders, bowstring arches, trusses) with 1-D deck heat diffusion, asphalt/ballast freezing and real healthy OMA tracking noise. Graded once across three full-scale bridges: KW51 held-out weeks (steel arch retrofit), Concrete Box-Girder (zero-shot, 2 modes, -7.9 C freeze) and Hell Bridge Test Arena (zero-shot, 8 modes, imposed physical damage DS1-DS8). v1.1 (27 Nov 2026) fixes four feature and export bugs found in review, retrains every rung and re-runs the exam once; v1.0 is kept in the notebook.

[Project page](https://dogum.github.io/ML-Ratchet/models/bridgesense/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Healthy reference window and query monitoring window (32 subsampled snapshots each): air temperature, 24 h and 72 h moving averages (C), and fractional natural frequency shifts (f - f0)/f0 with validity masks for 1-14 tracked modes. |
| **Returns** | P(structural change): probability that stiffness or structural mass changed beyond thermal/freezing variation<br>Effective global stiffness shift dK/K (%) +/- 68% bar (dev-widened x1.00)<br>Environmental frequency shift (%) +/- 68% bar (dev-widened x1.70)<br>Per-mode implied stiffness shift dK_m/K_m (%) for each active tracked mode |
| **Learned from** | 2-D multi-DOF Euler-Bernoulli beam, continuous girder, tied bowstring arch-deck and Warren/Pratt truss finite-element modal solver (K(T, d) phi = w^2 M(t) phi), coupled to 1-D Crank-Nicolson Fourier deck heat conduction (thermal lag), thermoelastic modulus E(T), logistic asphalt/ballast freezing phase transition below ~0 C, axial bearing thermal restraint (geometric stiffness K_G), and localized damage (-dK), retrofit (+dK, optionally mass-loaded) and added operational mass (+dM) perturbations. |
| **Noise** | 436 real 32-step x 14-mode OMA tracking noise and missingness windows extracted strictly from KW51's early healthy autumn baseline (2018-10-01 to 2018-11-11, pre-winter, 6 months pre-retrofit, zero structural labels; residual std 0.326%, mode availability 56.0%). |
| **Network** | Ensemble of 3 dual-axis Transformers (537,432 parameters = 3 x 179,144; d_model=64, 4 heads, LayerNorm + GELU, no BatchNorm): 2 baseline-to-query thermal cross-attention layers + 2 masked mode-set self-attention layers -> attention pooling + global anchor features -> change logit, beta-NLL heads for dK/K and thermal shift, and per-mode dK head. One member (r4_mix) adds its outputs to the linear-OLS anchor; two (r6 seed replicates) predict directly. Each member keeps its own config. Numpy inference matches PyTorch to 1.4e-6; the browser port matches numpy to 1.2e-6. |
| **Physics also enters** | inputs: Per-mode baseline linear-OLS thermal slopes, residuals, z-scores, freezing-hour residuals, 24 h / 72 h lagged air temperatures, and cold extrapolation distance<br>output: One of three ensemble members (r4_mix) predicts a correction to the per-mode linear-OLS thermal anchor (dK/K = anchor + correction); the other two predict dK/K directly |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | 10 non-overlapping 7-day KW51 blocks drawn with seed 20261124 (4 healthy mild, 2 healthy freezing, 4 post-retrofit; splits.json, sha256 0f6990ce…). Common warm reference baseline: 2018-10-01 to 2018-11-11 (also the unlabelled noise pool). |
| **Exam** | Three real full-scale bridge exams, opened once: (A) 49 held-out KW51 weekly blocks (10 healthy mild, 10 healthy freezing, 18 during-retrofit scaffolding, 11 post-retrofit); (B) Concrete Box-Girder Bridge (16 blocks, 2 modes, zero-shot); (C) Hell Bridge Test Arena (45 shaker/ambient recordings across held-out UDS and imposed damage DS1-DS8, 8 modes picked from the 5 UDS *_01 reference recordings only, zero-shot). |

## Run it

```bash
cd models/bridgesense
pip install numpy
python demo.py        # about 1 s
```

or from anywhere in the repo: `mlratchet demo bridgesense`.

```python
from bridgesense import BridgeSense
model = BridgeSense()
reading = model.predict(T_ref, T_q, df_ref, df_q, mask_ref, mask_q)
print(reading["p_change"], reading["delta_k_pct"], reading["delta_k_sigma"])
```

## The exam

Graded once (v1.1, 27 Nov 2026; run record training/logs/v1_1/exam_run.json) across 110 real monitoring blocks/recordings on three full-scale bridges beside per-mode Linear Thermal OLS, a KW51-dev-tuned Bilinear Freezing + RMS z-score baseline, and dumb baselines. Same tally as v1.0 (3 wins, 6 losses, 1 tie); no verdict flipped. Raw results in training/logs/v1_1/exam.json. Every number here is on real data that never chose anything.

| Metric | BridgeSense | Baselines | |
|---|---|---|---|
| KW51 Exam A: Structural-change detection ROC-AUC (49 held-out weeks)<br><sub>95% CI [1.000, 1.000]. All three physics-aware methods separate the 20 healthy weeks (incl. 10 freezing) from the 29 during/post-retrofit weeks cleanly. v1.0: 1.000, tie.</sub> | **1** | Dev-tuned bilinear freezing + RMS z (classical): 1<br>Linear thermal OLS + RMS z (classical): 1<br>Raw frequency shift (dumb) (dumb baseline): 0.966 | ➖ tie |
| KW51 Exam A: Detection Brier score (49 held-out weeks)<br><sub>0/10 freezing false alarms (mean P = 0.003); during- and post-retrofit mean P = 1.000. The margin is small in absolute terms (bilinear 0.0017). v1.0: 0.0000, win.</sub> | **0** | Dev-tuned bilinear freezing + RMS z (classical): 0.0017<br>Linear thermal OLS + RMS z (classical): 0.0019<br>Dev prevalence (0.40) (dumb baseline): 0.2784 | ✅ wins |
| KW51 Exam A: Healthy dK/K MAE, 20 weeks incl. 10 freezing (%)<br><sub>Truth is 0 % by construction, so the 0 % guess is exact and the row is a loss. As a false-shift measure BridgeSense is 3.7x below both classical regressions (largest single-week error 0.08 %). v1.0: 0.019 %, loss.</sub> | **0.051%** | Dev-tuned bilinear freezing (classical): 0.189%<br>Linear thermal OLS (classical): 0.19%<br>Dev median guess (0 % shift) (dumb baseline): 0%<br>Raw uncompensated 2*df/f (dumb baseline): 0.217% | ❌ loses |
| KW51 Exam A: Post-retrofit dK/K MAE vs matched-T truth +1.17% (11 weeks)<br><sub>Mean prediction +1.06 % vs +1.17 % truth (OLS +1.10 %). The truth is itself a linear-temperature OLS fit on the exam blocks, the same method as the OLS baseline. v1.0: 0.169 %, loss.</sub> | **0.199%** | Linear thermal OLS (classical): 0.073%<br>Dev-tuned bilinear freezing (classical): 0.075%<br>Zero shift (dumb) (dumb baseline): 1.17% | ❌ loses |
| Concrete Box-Girder Exam B (0-shot, 2 modes): Healthy winter freeze dK/K MAE (%, Tmin -7.9 C)<br><sub>Truth 0 %, so the 0 % guess is exact: loss. The four freezing blocks read -1.71 %, -0.53 %, -0.96 % and +2.95 % (OLS +9.50 % and +6.31 % on the two coldest). 4.1x below the classical methods, but worse than v1.0 (0.881 %). n = 4.</sub> | **1.538%** | Dev median guess (0 % shift) (dumb baseline): 0%<br>Raw uncompensated 2*df/f (dumb baseline): 4.068%<br>Dev-tuned bilinear freezing (KW51 dev) (classical): 6.264%<br>Linear thermal OLS (classical): 6.369% | ❌ loses |
| Concrete Box-Girder Exam B (0-shot): Cold healthy vs 680-2,160 kg static load AUC (16 blocks)<br><sub>95% CI [0.33, 1.00]; only 4 negatives. Classical baselines invert the ranking because January freezing shifts look larger to OLS than the spring sand-bag loads. v1.0: 0.729, win.</sub> | **0.771** | Linear thermal OLS + RMS z (classical): 0.146<br>Dev-tuned bilinear freezing + RMS z (classical): 0.146<br>Raw frequency shift (dumb) (dumb baseline): 0.063 | ✅ wins |
| Concrete Box-Girder Exam B (0-shot): Healthy freezing false-alarm rate at P >= 0.5 (4 blocks)<br><sub>Mean P = 0.583 across the 4 freezing concrete blocks (2/4 flagged: P = 1.00 at -3.0 C and 0.78 at -3.7 C). v1.0: 0.50 (mean P 0.469), loss.</sub> | **0.5** | Dev-tuned bilinear freezing (classical): 0<br>Linear thermal OLS (classical): 0.25 | ❌ loses |
| HBTA Exam C (0-shot, 8 modes): Global primary-structure damage AUC (DS5-DS8 vs UDS, n=25)<br><sub>95% CI [0.71, 1.00]; 5 undamaged recordings, and the margin over the raw shift (0.03) is far inside the interval. DS5 and DS7 read mean P >= 0.98 with dK/K -2.3 %; DS6 is missed (mean P 0.36). v1.0 (7 bands chosen with the held-out controls): 0.800, win.</sub> | **0.87** | Raw frequency shift (dumb) (dumb baseline): 0.84<br>Linear thermal OLS + RMS z (classical): 0.72<br>Dev-tuned bilinear freezing + RMS z (classical): 0.72 | ✅ wins |
| HBTA Exam C (0-shot, 8 modes): All imposed damage AUC (DS1-DS8 vs UDS, n=45)<br><sub>95% CI [0.55, 0.88]. The raw shift wins: HBTA's 10-min shaker tests have almost no temperature spread, so thermal compensation has nothing to remove. DS1-DS2 stringer/cross-beam cracks barely move global frequencies (mean P 0.23 and 0.40). v1.0: 0.595, loss.</sub> | **0.72** | Raw frequency shift (dumb) (dumb baseline): 0.795<br>Linear thermal OLS + RMS z (classical): 0.655<br>Dev-tuned bilinear freezing + RMS z (classical): 0.655 | ❌ loses |
| HBTA Exam C (0-shot, 8 modes): dK/K MAE across all 45 recordings (%)<br><sub>95% CI [0.57, 0.91]. Beats OLS and zero shift but loses to the raw shift. The truth is itself the measured frequency shift against the UDS reference, so the raw shift is close to the truth by construction. v1.0 listed only OLS and zero shift (0.880 % vs OLS 0.824 %, loss); with that list v1.1 would read as a win, so the raw-shift baseline is added rather than flipping the verdict.</sub> | **0.72%** | Raw uncompensated 2*df/f (dumb baseline): 0.473%<br>Linear thermal OLS (classical): 0.888%<br>Dev-tuned bilinear freezing (classical): 0.888%<br>Zero shift (dumb) (dumb baseline): 1.104% | ❌ loses |

Coverage of the 68% error bars on the exam: dK/K on KW51 Exam A (n=31, dev-widened x1.00, mean sigma 0.24%) **97%**, dK/K on Concrete Box-Girder Exam B healthy (n=4, 0-shot) **50%**, dK/K on HBTA Exam C (n=45, 0-shot, mean sigma 0.74%) **62%**.

### Where it wins

- Zero-shot on a 2-mode prestressed concrete box-girder bridge: cold-healthy vs static-load AUC 0.771 against 0.146 for both classical baselines (16 blocks, 4 negatives).
- On 49 held-out KW51 weeks: Brier 0.0000 against 0.0017 and 0.0019, with 0/10 freezing false alarms. The detection task itself is easy: AUC is 1.000 for both classical baselines too, and every rung scored dev AUC 1.0.
- Zero-shot global primary-structure damage detection on the 1905 Hell Bridge Test Arena truss (DS5-DS8 vs UDS): AUC 0.870 vs 0.840 for the raw shift and 0.720 for classical baselines (5 negatives).

### Where it loses

- Healthy dK/K error rows on KW51 (0.051 %) and the concrete box girder (1.54 %): the dev-median guess of 0 % shift is exact on healthy weeks, so both are losses, though BridgeSense is 3.7-4.1x below both classical methods on them.
- HBTA: the raw uncompensated frequency shift beats BridgeSense on all-damage AUC (0.795 vs 0.720) and on dK/K MAE (0.473 % vs 0.720 %). With no temperature spread to compensate, the thermal machinery adds error.
- Probability calibration does not transfer zero-shot to extreme concrete freezing (2/4 freezing concrete blocks flagged at P >= 0.5).
- Zero-shot 68% error-bar coverage is 50% on Concrete Box-Girder and 62% on HBTA because dev widening (x1.00) was fitted on KW51 alone.
- Post-retrofit dK/K MAE on KW51 (0.199%) trails matched-temperature Linear OLS (0.073%).

## Limits

- Needs a baseline window spanning at least several degrees of real temperature variation; on short constant-temperature shaker campaigns (HBTA) the raw frequency shift does better.
- v1.1 trained on a KW51 noise pool detrended with an uncentred thermal slope (fixed in training/noise.py on 27 Nov 2026 for future rungs). Measured effect: per-mode residual sd changes by at most 0.0001 pp (pooled 0.3261 % both ways) and no pool entry by more than 0.05 pp; v1.1 was not retrained for it.
- The KW51 dev gate is noisy: three r6 seed replicates differ only in seed yet scored dev dK/K MAE 0.10 %, 0.23 % and 0.42 % (10 blocks), so single-run gate decisions near the limit are partly luck.
- Global natural frequencies are insensitive to local secondary-member cracks (HBTA DS1-DS2) until stiffness loss reaches ~1-2% of modal strain energy; mode shapes (MAC) or local strain gauges are needed for early local cracks.
- Does not separate added operational mass (+dM) from uniform stiffness loss (-dK) without mode-shape or traffic-load inputs; both read as negative effective dK/K.
- Cannot replace licensed bridge engineering inspection; trained on 2-D simplified FEMs without 3-D torsion-warping or scour soil-structure interaction.

## The ratchet

### v1.0 — released 2026-11-24, superseded by v1.1 (feature bugs; checkpoints chosen on val3)

**Objective:** Composite skill in [0, 1] on frozen simulated bridge episodes: 0.35*AUC + 0.35*dK_skill + 0.15*env_skill + 0.15*cold_extrap_specificity  
**Rule:** Keep if frozen-set composite skill beats the incumbent by more than +0.01 AND on KW51 dev (10 blocks, seed 20261124) AUC drops <= 0.02, Brier rises <= max(15%, +0.015), and 10-block dK/K MAE rises <= max(15%, +0.05 pp).

| Rung | Change | Frozen-set score | Decision | Why | Sim AUC | Sim dK MAE (%) | Sim cold spec | Dev AUC | Dev Brier | Dev dK MAE (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| r1 | Textbook linear thermoelastic 2-D modal FEM + synthetic Gaussian OMA noise (physics p1, train1) | 0.302 | ● kept | First rung. | 0.767 | 2.64 | 0.056 | 1.0 | 0.2386 | 0.104 |
| r2 | Add 1-D Fourier deck heat diffusion, asphalt/ballast freezing phase transition, axial bearing restraint, warm-to-cold extrapolation (physics p2, train2) | 0.806 | ● kept | +0.504 on the frozen set; dev gate passed. | 0.948 | 0.66 | 0.662 | 1.0 | 0.0 | 0.043 |
| r2_no_xattn | Architecture ablation on p2: remove baseline-to-query thermal cross-attention | 0.802 | ○ rejected | -0.005 on val3, needs more than +0.01. | 0.94 | 0.68 | 0.674 | 1.0 | 0.0603 | 0.086 |
| r3 | Inject real healthy KW51 autumn OMA noise + 44% correlated mode dropouts + dev-calibrated slow seasonal drift + mass-loaded retrofits (physics p3, train3) | 0.865 | ○ rejected | +0.058 on the frozen set; dev gate: dev_dk_mae 0.255% > 0.093%. | 0.969 | 0.56 | 0.893 | 1.0 | 0.0 | 0.255 |
| r4_mix | Mixed bank (p2 + p3, 24k episodes) with linear-OLS physics anchor | 0.873 | ○ rejected | +0.067 on the frozen set; dev gate: dev_dk_mae 0.134% > 0.093%. | 0.97 | 0.56 | 0.946 | 1.0 | 0.0 | 0.134 |
| r5_no_anchor | Anchor ablation on p3: predict dK/K directly without the OLS output anchor | 0.879 | ○ rejected | +0.073 on the frozen set; dev gate: dev_dk_mae 0.465% > 0.093%. | 0.971 | 0.5 | 0.934 | 1.0 | 0.0001 | 0.465 |
| r6_mix_no_anchor | Mixed bank (p2 + p3, 24k episodes) without the OLS output anchor | 0.881 | **◆ released** | +0.074 on the frozen set; dev gate passed. | 0.976 | 0.5 | 0.92 | 1.0 | 0.0 | 0.071 |
| r7_ens3 | ensemble of r4_mix (OLS-anchored) and two r6 seed runs, r6_s2 and r6_s3 (training/ratchet.py; the r6 seed runs themselves are not logged as rungs) | 0.876 | ○ rejected | -0.004 on val3, needs more than +0.01. | 0.973 | 0.52 | 0.912 | 1.0 | 0.0 | 0.095 |

<sub>Columns after *Why*: Frozen set (val3) and KW51 dev (10 blocks); real exams never used to choose.</sub>

### v1.1 — fixed thermal fit, mode eligibility, separate selection sets

**Objective:** Composite skill in [0, 1] on frozen simulated bridge episodes: 0.35*AUC + 0.35*dK_skill + 0.15*env_skill + 0.15*cold_extrap_specificity  
**Rule:** Keep if frozen-set composite skill beats the incumbent by more than +0.01 AND on KW51 dev (10 blocks, seed 20261124) AUC drops <= 0.02, Brier rises <= max(15%, +0.015), and 10-block dK/K MAE rises <= max(15%, +0.05 pp).

| Rung | Change | Frozen-set score | Decision | Why | Sim AUC | Sim dK MAE (%) | Sim cold spec | Dev AUC | Dev Brier | Dev dK MAE (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| r1 | Textbook linear thermoelastic 2-D modal FEM + synthetic Gaussian OMA noise (physics p1, train1) | 0.306 | ● kept | First rung. | 0.784 | 2.93 | 0.073 | 1.0 | 0.1116 | 0.077 |
| r2 | Add 1-D Fourier deck heat diffusion, asphalt/ballast freezing phase transition, axial bearing restraint, warm-to-cold extrapolation (physics p2, train2) | 0.818 | ● kept | +0.512 on the frozen set; dev gate passed. | 0.925 | 0.66 | 0.719 | 1.0 | 0.0854 | 0.056 |
| r2_no_xattn | Architecture ablation on p2: remove baseline-to-query thermal cross-attention | 0.805 | ○ rejected | -0.013 on val3, needs more than +0.01. | 0.916 | 0.69 | 0.692 | 1.0 | 0.0 | 0.066 |
| r3 | Inject real healthy KW51 autumn OMA noise + correlated mode dropouts + dev-calibrated slow seasonal drift + mass-loaded retrofits (physics p3, train3) | 0.872 | ○ rejected | +0.054 on the frozen set; dev gate: dev_dk_mae 0.409% > 0.106%. | 0.963 | 0.61 | 0.957 | 1.0 | 0.0001 | 0.409 |
| r4_mix | Mixed bank (p2 + p3, 24k episodes) with linear-OLS physics anchor | 0.879 | ○ rejected | +0.061 on the frozen set; dev gate: dev_dk_mae 0.149% > 0.106%. | 0.968 | 0.54 | 0.907 | 1.0 | 0.0 | 0.149 |
| r5_no_anchor | Anchor ablation on p3: predict dK/K directly without the OLS output anchor | 0.878 | ● kept | +0.061 on the frozen set; dev gate passed. | 0.965 | 0.51 | 0.902 | 1.0 | 0.0 | 0.099 |
| r6_mix_no_anchor | Mixed bank (p2 + p3, 24k episodes) without the OLS output anchor (seed 42) | 0.890 | ○ rejected | +0.011 on the frozen set; dev gate: dev_dk_mae 0.416% > 0.149%. | 0.973 | 0.47 | 0.907 | 1.0 | 0.0 | 0.416 |
| r6_s2 | Seed replicate of r6_mix_no_anchor (seed 2) | 0.890 | ○ rejected | +0.011 on the frozen set; dev gate: dev_dk_mae 0.234% > 0.149%. | 0.971 | 0.48 | 0.92 | 1.0 | 0.0 | 0.234 |
| r6_s3 | Seed replicate of r6_mix_no_anchor (seed 3) | 0.884 | ○ rejected | +0.005 on val3, needs more than +0.01. | 0.967 | 0.51 | 0.922 | 1.0 | 0.0 | 0.097 |
| r7_ens3 | Ensemble of r4_mix (OLS-anchored) with the r6_s2 and r6_s3 seed replicates; each member keeps its own config | 0.892 | **◆ released** | +0.014 on the frozen set; dev gate passed. | 0.973 | 0.47 | 0.917 | 1.0 | 0.0 | 0.147 |

<sub>Columns after *Why*: Frozen set (val3) and KW51 dev (10 blocks); real exams never used to choose.</sub>

Raw per-rung logs are in [`training/`](training/).

## Physics

2-D multi-DOF Euler-Bernoulli beam, continuous girder, tied bowstring arch-deck and Warren/Pratt truss finite-element modal solver (K(T, d) phi = w^2 M(t) phi), coupled to 1-D Crank-Nicolson Fourier deck heat conduction (thermal lag), thermoelastic modulus E(T), logistic asphalt/ballast freezing phase transition below ~0 C, axial bearing thermal restraint (geometric stiffness K_G), and localized damage (-dK), retrofit (+dK, optionally mass-loaded) and added operational mass (+dM) perturbations.

**Simulator**

- Random 2-D bridge FEMs across three structural families (40% continuous/simple girders, 35% tied bowstring arches, 25% steel trusses) and three materials (steel, concrete, composite), tracking 2-14 modes
- 1-D Fourier heat conduction through deck slab (steel/concrete thermal diffusivities) yielding lagged structural and surface temperatures
- Asphalt/ballast freezing stiffening across T_freeze in [-3.5, +2.0] C, axial bearing thermal compression via K_G, and 35-44% warm-baseline-to-freezing-query extrapolation episodes
- Physics p3 injects 436 real healthy KW51 autumn OMA residual noise and correlated missing-mode dropout windows (44% mean missingness) plus dev-twin-measured seasonal drift (v1.1: global 0.113%, per-mode 0.219%)

**Checked against independent references**

- Simply-supported Euler-Bernoulli beam (bending modes 1-6, 24 elements) vs analytical f_n: max rel error 2.60e-4 (0.026%) (training/verify_physics.json)
- Cantilever Euler-Bernoulli beam (bending modes 1-4, 24 elements) vs analytical roots: max rel error 3.03e-5 (0.003%)
- Uniform stiffness scaling vs sqrt(1 + eps): max err 9.70e-14; first-order Rayleigh quotient perturbation: 8.37e-10; modal projection vs full eigh under 20% local damage + freeze: 3.29e-13
- 1-D Crank-Nicolson deck heat diffusion vs exact periodic slab solution: amplitude 0.6059 vs 0.6056 K (rel err 5.87e-4); phase lag 0.7794 vs 0.7854 rad (err 0.0060 rad)
- Plausibility range only: simulated bowstring-arch median f1 2.87 Hz and f4 7.76 Hz fall inside the check's wide bands (f1 0.3-3.5 Hz, f4 1.5-12 Hz); no KW51 FEM values are parsed, and f4 sits above KW51's published 0.51-6.31 Hz span (verify_physics.json)

## Help wanted

- Z24 Highway Bridge benchmark (EMPA / KU Leuven, bwk.kuleuven.be/bwm/z24/registration): 1 year of concrete freeze-thaw plus controlled progressive pier settlement and tendon rupture. Gated by a registration form, not bypassed; would add a 4th real exam bridge with known progressive concrete damage.
- Long-term monitored highway or suspension bridge with both on-bridge embedded concrete/asphalt temperatures and a documented structural repair (to replace external NASA POWER air temperature on Exam B).

## Next rungs

- Include zero-temperature-range (short shaker-campaign) episodes in the simulator so the network learns to widen uncertainty and avoid false alarms when baseline temperature spread is < 2 C (the exact HBTA UDS failure mode).
- Add mode-shape Modal Assurance Criterion (MAC) vectors as an optional per-mode token channel to catch local secondary-member cracks (HBTA DS1-DS4) that barely move scalar frequencies.
- Fit a soft-gated physics anchor (using the OLS anchor only when T_q stays inside the baseline temperature range, and switching to the Transformer under cold extrapolation) to keep OLS's 0.07% matched-T retrofit precision alongside BridgeSense's freezing robustness.
- When another change calls for a new rung, rebuild the KW51 noise pool with the centred detrending (training/noise.py) and regenerate the p3 banks before training.
- Score the dev gate on several seeds (or a larger dev set) so that seed noise of 0.1-0.4 pp in dev dK/K MAE does not decide rungs.

## Files

| Path | What |
|---|---|
| `bridgesense.py` | Inference — numpy/scipy only |
| `bridgesense_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [KW51 railway bridge tracked modal data (Maes & Lombaert 2020, Zenodo 3745914)](https://zenodo.org/records/3745914) | noise, dev, exam | CC BY-NC-SA 4.0 (fetched into work/ only, no rows redistributed in git; the weights are CC BY-NC-SA 4.0 because KW51 noise windows and dev calibration shaped them, LICENSE-WEIGHTS.md) |
| [KW51 finite element model (Maes & Lombaert 2025, Zenodo 17366772)](https://zenodo.org/records/17366772) | physics | CC BY-NC-SA 4.0 (physics check only) |
| [Concrete box-girder bridge eigenfrequencies (Koehncke et al. 2026, Zenodo 20522191)](https://zenodo.org/records/20522191) | exam | CC BY 4.0 |
| [NASA POWER hourly 2 m air temperature (Hamburg, Jan-Apr 2024)](https://power.larc.nasa.gov/) | exam | NASA POWER open data; cite POWER |
| [Hell Bridge Test Arena imposed-damage dataset (Svendsen et al. 2025, Zenodo 14632942)](https://zenodo.org/records/14632942) | exam | CC BY 4.0 |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
