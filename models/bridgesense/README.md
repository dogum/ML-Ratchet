<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# BridgeSense <sub>v1.0</sub>

> **A Transformer trained on bridge FEMs separates real stiffness changes from winter freezes.**

> [!WARNING]
> Research only. Not a structural safety inspection tool; bridge assessments require licensed engineering inspection.

*From a bridge's tracked natural frequencies and temperature, has its stiffness actually changed — or is it just the weather — and by how much?*

From a bridge's healthy baseline window and a monitoring window of tracked natural frequencies (1 to 14 modes, with missing-mode dropouts) and air temperature history, a dual-axis Transformer (baseline-to-query thermal cross-attention plus masked mode-set self-attention) estimates the probability of a real structural change, the effective global stiffness shift dK/K (%) and the environmental frequency shift (%), each with a 68% error bar. Trained only on 2-D thermo-mechanical modal FEM simulations (girders, bowstring arches, trusses) with 1-D deck heat diffusion, asphalt/ballast freezing and real healthy OMA tracking noise. Graded once across three full-scale bridges: KW51 held-out weeks (steel arch retrofit), Concrete Box-Girder (zero-shot, 2 modes, -7.9 C freeze) and Hell Bridge Test Arena (zero-shot, 7 modes, imposed physical damage DS1-DS8).

[Project page](https://dogum.github.io/ML-Ratchet/models/bridgesense/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Healthy reference window and query monitoring window (32 subsampled snapshots each): air temperature, 24 h and 72 h moving averages (C), and fractional natural frequency shifts (f - f0)/f0 with validity masks for 1-14 tracked modes. |
| **Returns** | P(structural change): probability that stiffness or structural mass changed beyond thermal/freezing variation<br>Effective global stiffness shift dK/K (%) +/- 68% bar (dev-widened x1.00)<br>Environmental frequency shift (%) +/- 68% bar (dev-widened x2.34)<br>Per-mode implied stiffness shift dK_m/K_m (%) for each active tracked mode |
| **Learned from** | 2-D multi-DOF Euler-Bernoulli beam, continuous girder, tied bowstring arch-deck and Warren/Pratt truss finite-element modal solver (K(T, d) phi = w^2 M(t) phi), coupled to 1-D Crank-Nicolson Fourier deck heat conduction (thermal lag), thermoelastic modulus E(T), logistic asphalt/ballast freezing phase transition below ~0 C, axial bearing thermal restraint (geometric stiffness K_G), and localized damage (-dK), retrofit (+dK, optionally mass-loaded) and added operational mass (+dM) perturbations. |
| **Noise** | 436 real 32-step x 14-mode OMA tracking noise and missingness windows extracted strictly from KW51's early healthy autumn baseline (2018-10-01 to 2018-11-11, pre-winter, 6 months pre-retrofit, zero structural labels; residual std 0.326%, mode availability 56.0%). |
| **Network** | Dual-axis Transformer (179,144 parameters, d_model=64, 4 heads, LayerNorm + GELU, no BatchNorm): 2 baseline-to-query thermal cross-attention layers + 2 masked mode-set self-attention layers -> attention pooling + global anchor features -> change logit, beta-NLL heads for dK/K and thermal shift, and per-mode dK head. Numpy inference matches PyTorch to 2.67e-6. |
| **Physics also enters** | inputs: Per-mode baseline linear-OLS thermal slopes, residuals, z-scores, freezing-hour residuals, 24 h / 72 h lagged air temperatures, and cold extrapolation distance |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | 10 non-overlapping 7-day KW51 blocks drawn with seed 20261124 (4 healthy mild, 2 healthy freezing, 4 post-retrofit; splits.json). Common warm reference baseline: 2018-10-01 to 2018-11-11 (also the unlabelled noise pool). |
| **Exam** | Three real full-scale bridge exams, opened once: (A) 49 held-out KW51 weekly blocks (10 healthy mild, 10 healthy freezing, 18 during-retrofit scaffolding, 11 post-retrofit); (B) Concrete Box-Girder Bridge (16 blocks, 2 modes, zero-shot); (C) Hell Bridge Test Arena (45 shaker/ambient recordings across held-out UDS and imposed damage DS1-DS8, 7 modes, zero-shot). |

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

Graded once across 110 real monitoring blocks/recordings on three full-scale bridges beside per-mode Linear Thermal OLS, a KW51-dev-tuned Bilinear Freezing + RMS z-score baseline, and a dumb baseline. Every number here is on real data that never chose anything.

| Metric | BridgeSense | Baselines | |
|---|---|---|---|
| KW51 Exam A: Structural-change detection ROC-AUC (49 held-out weeks)<br><sub>95% CI [1.000, 1.000]. All three physics-aware methods separate the 20 healthy weeks (incl. 10 freezing) from the 29 during/post-retrofit weeks cleanly.</sub> | **1** | Dev-tuned bilinear freezing + RMS z (classical): 1<br>Linear thermal OLS + RMS z (classical): 1<br>Raw frequency shift (dumb) (dumb baseline): 0.966 | ➖ tie |
| KW51 Exam A: Detection Brier score (49 held-out weeks)<br><sub>0/10 freezing false alarms (mean P = 0.004); during-retrofit mean P = 0.999; post-retrofit mean P = 0.997.</sub> | **0** | Dev-tuned bilinear freezing + RMS z (classical): 0.0016<br>Linear thermal OLS + RMS z (classical): 0.0019<br>Dev prevalence (0.40) (dumb baseline): 0.2784 | ✅ wins |
| KW51 Exam A: Healthy dK/K MAE, 20 weeks incl. 10 freezing (%)<br><sub>10x lower healthy stiffness-shift error than both classical thermal regressions.</sub> | **0.019%** | Dev-tuned bilinear freezing (classical): 0.188%<br>Linear thermal OLS (classical): 0.189%<br>Raw uncompensated 2*df/f (dumb baseline): 0.217% | ✅ wins |
| KW51 Exam A: Post-retrofit dK/K MAE vs matched-T truth +1.17% (11 weeks)<br><sub>Mean prediction +1.13% vs +1.17% truth (OLS +1.10%); removing the output anchor traded 0.10 pp of matched-T precision for freeze robustness.</sub> | **0.169%** | Linear thermal OLS (classical): 0.072%<br>Dev-tuned bilinear freezing (classical): 0.075%<br>Zero shift (dumb) (dumb baseline): 1.17% | ❌ loses |
| Concrete Box-Girder Exam B (0-shot, 2 modes): Healthy winter freeze dK/K MAE (%, Tmin -7.9 C)<br><sub>7x lower error zero-shot on an unseen concrete bridge: during the -7.9 C and -5.0 C weeks, OLS hallucinated +9.51% and +6.31% while BridgeSense read -0.01% and +0.01%.</sub> | **0.881%** | Raw uncompensated 2*df/f (dumb baseline): 4.068%<br>Dev-tuned bilinear freezing (KW51 dev) (classical): 6.246%<br>Linear thermal OLS (classical): 6.372% | ✅ wins |
| Concrete Box-Girder Exam B (0-shot): Cold healthy vs 680-2,160 kg static load AUC (16 blocks)<br><sub>Classical baselines invert the ranking (AUC 0.146) because January freezing spikes look larger to OLS than the spring sand-bag loads.</sub> | **0.729** | Linear thermal OLS + RMS z (classical): 0.146<br>Dev-tuned bilinear freezing + RMS z (classical): 0.146 | ✅ wins |
| Concrete Box-Girder Exam B (0-shot): Healthy freezing false-alarm rate at P >= 0.5 (4 blocks)<br><sub>Mean P = 0.469 across the 4 freezing concrete blocks (2/4 flagged). Ranking is strong (AUC 0.729), but the KW51-calibrated probability threshold sits too tight for extreme concrete freezing.</sub> | **0.5** | Dev-tuned bilinear freezing (classical): 0<br>Linear thermal OLS (classical): 0.25 | ❌ loses |
| HBTA Exam C (0-shot, 7 modes): Global primary-structure damage AUC (DS5-DS8 vs UDS, n=25)<br><sub>95% CI [0.500, 1.000]. Reads DS5, DS7, DS8 at P >= 0.998 with negative dK/K (-2.0% to -3.0%).</sub> | **0.8** | Linear thermal OLS + RMS z (classical): 0.69<br>Dev-tuned bilinear freezing + RMS z (classical): 0.69 | ✅ wins |
| HBTA Exam C (0-shot, 7 modes): All imposed damage AUC (DS1-DS8 vs UDS, n=45)<br><sub>95% CI [0.341, 0.849]. Secondary local stringer/cross-beam cracks (DS1-DS2) barely move global truss frequencies (< 0.4%), and HBTA's short 10-min shaker tests lack baseline temperature spread.</sub> | **0.595** | Raw frequency shift (dumb) (dumb baseline): 0.665<br>Linear thermal OLS + RMS z (classical): 0.645<br>Dev-tuned bilinear freezing + RMS z (classical): 0.645 | ❌ loses |
| HBTA Exam C (0-shot, 7 modes): dK/K MAE across all 45 recordings (%)<br><sub>95% CI [0.676, 1.091]. Beats zero shift but trails OLS by 0.06 pp due to high-excitation shaker vs ambient variance.</sub> | **0.88%** | Linear thermal OLS (classical): 0.824%<br>Dev-tuned bilinear freezing (classical): 0.824%<br>Zero shift (dumb) (dumb baseline): 1.045% | ❌ loses |

Coverage of the 68% error bars on the exam: dK/K on KW51 Exam A (n=31, dev-widened x1.00, mean sigma 0.22%) **97%**, dK/K on Concrete Box-Girder Exam B healthy (n=4, 0-shot) **50%**, dK/K on HBTA Exam C (n=45, 0-shot, mean sigma 0.49%) **33%**.

### Where it wins

- Zero-shot transfer from steel-only dev to a 2-mode prestressed concrete box-girder bridge: cuts healthy -7.9 C winter freezing dK/K error 7x vs both Linear OLS and KW51-dev-tuned Bilinear (0.88% vs 6.37% / 6.25%), and lifts cold-vs-load AUC from 0.146 to 0.729.
- On 49 held-out KW51 weeks, cuts healthy dK/K error 10x vs classical thermal regressions (0.019% vs 0.188%) with Brier 0.0000 and 0/10 freezing false alarms.
- Zero-shot global primary-structure damage detection on the 1905 Hell Bridge Test Arena truss (DS5-DS8 vs UDS): AUC 0.800 vs 0.690 for classical baselines.
- Removing baseline-to-query thermal cross-attention (r2_no_xattn) raised KW51 dev winter freezing false-alarm probability 240x (0.002 to 0.456), validating the dual-axis Transformer architecture.

### Where it loses

- Secondary local damage on HBTA (DS1-DS2 stringer/cross-beam cracks) barely shifts global modal frequencies; all-damage HBTA AUC (0.595) loses to OLS (0.645) and raw shift (0.665), and dK/K MAE (0.880%) trails OLS (0.824%).
- Probability calibration does not transfer zero-shot to extreme concrete freezing (2/4 freezing concrete blocks flagged at P >= 0.5) or short constant-temperature shaker campaigns (HBTA UDS replicates flagged at mean P = 0.825).
- Zero-shot 68% error-bar coverage drops to 50% on Concrete Box-Girder and 33% on HBTA because dev widening (x1.00) was fitted on KW51 alone.
- Post-retrofit dK/K MAE on KW51 (0.169%) trails matched-temperature Linear OLS (0.072%) because the winning rung dropped the linear output anchor.

## Limits

- Needs a baseline window spanning at least several degrees of real temperature variation; on short constant-temperature shaker campaigns (HBTA), the thermal cross-attention has no baseline temperature gradient to anchor against.
- Global natural frequencies are insensitive to local secondary-member cracks (HBTA DS1-DS2) until stiffness loss reaches ~1-2% of modal strain energy; mode shapes (MAC) or local strain gauges are needed for early local cracks.
- Does not separate added operational mass (+dM) from uniform stiffness loss (-dK) without mode-shape or traffic-load inputs; both read as negative effective dK/K.
- Cannot replace licensed bridge engineering inspection; trained on 2-D simplified FEMs without 3-D torsion-warping or scour soil-structure interaction.

## The ratchet

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
| r7_ens3 | 3-seed Transformer ensemble on mixed p2 + p3 bank with OLS physics anchor | 0.876 | ○ rejected | -0.004 on val3, needs more than +0.01. | 0.973 | 0.52 | 0.912 | 1.0 | 0.0 | 0.095 |

<sub>Columns after *Why*: Frozen set (val3) and KW51 dev (10 blocks); real exams never used to choose.</sub>

Raw per-rung logs are in [`training/`](training/).

## Physics

2-D multi-DOF Euler-Bernoulli beam, continuous girder, tied bowstring arch-deck and Warren/Pratt truss finite-element modal solver (K(T, d) phi = w^2 M(t) phi), coupled to 1-D Crank-Nicolson Fourier deck heat conduction (thermal lag), thermoelastic modulus E(T), logistic asphalt/ballast freezing phase transition below ~0 C, axial bearing thermal restraint (geometric stiffness K_G), and localized damage (-dK), retrofit (+dK, optionally mass-loaded) and added operational mass (+dM) perturbations.

**Simulator**

- Random 2-D bridge FEMs across three structural families (40% continuous/simple girders, 35% tied bowstring arches, 25% steel trusses) and three materials (steel, concrete, composite), tracking 2-14 modes
- 1-D Fourier heat conduction through deck slab (steel/concrete thermal diffusivities) yielding lagged structural and surface temperatures
- Asphalt/ballast freezing stiffening across T_freeze in [-3.5, +2.0] C, axial bearing thermal compression via K_G, and 35-44% warm-baseline-to-freezing-query extrapolation episodes
- Physics p3 injects 436 real healthy KW51 autumn OMA residual noise and correlated missing-mode dropout windows (44% mean missingness) plus dev-twin-measured seasonal drift (global 0.115%, per-mode 0.222%)

**Checked against independent references**

- Simply-supported Euler-Bernoulli beam (bending modes 1-6, 24 elements) vs analytical f_n: max rel error 2.60e-4 (0.026%) (training/verify_physics.json)
- Cantilever Euler-Bernoulli beam (bending modes 1-4, 24 elements) vs analytical roots: max rel error 3.03e-5 (0.003%)
- Uniform stiffness scaling vs sqrt(1 + eps): max err 9.70e-14; first-order Rayleigh quotient perturbation: 8.37e-10; modal projection vs full eigh under 20% local damage + freeze: 2.04e-13
- 1-D Crank-Nicolson deck heat diffusion vs exact periodic slab solution: amplitude 0.6059 vs 0.6056 K (rel err 5.87e-4); phase lag 0.7794 vs 0.7854 rad (err 0.0060 rad)
- Parametric steel bowstring arch-deck FEM fundamental modal range matches KW51 FEM (Zenodo 17366772) scale (0.5-6.3 Hz)

## Help wanted

- Z24 Highway Bridge benchmark (EMPA / KU Leuven, bwk.kuleuven.be/bwm/z24/registration): 1 year of concrete freeze-thaw plus controlled progressive pier settlement and tendon rupture. Gated by a registration form, not bypassed; would add a 4th real exam bridge with known progressive concrete damage.
- Long-term monitored highway or suspension bridge with both on-bridge embedded concrete/asphalt temperatures and a documented structural repair (to replace external NASA POWER air temperature on Exam B).

## Next rungs

- Include zero-temperature-range (short shaker-campaign) episodes in the simulator so the network learns to widen uncertainty and avoid false alarms when baseline temperature spread is < 2 C (the exact HBTA UDS failure mode).
- Add mode-shape Modal Assurance Criterion (MAC) vectors as an optional per-mode token channel to catch local secondary-member cracks (HBTA DS1-DS4) that barely move scalar frequencies.
- Fit a soft-gated physics anchor (using the OLS anchor only when T_q stays inside the baseline temperature range, and switching to the Transformer under cold extrapolation) to keep OLS's 0.07% matched-T retrofit precision alongside BridgeSense's 7x freezing win.

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
| [KW51 railway bridge tracked modal data (Maes & Lombaert 2020, Zenodo 3745914)](https://zenodo.org/records/3745914) | noise, dev, exam | CC BY-NC-SA 4.0 (fetched into work/ only, not redistributed in git) |
| [KW51 finite element model (Maes & Lombaert 2025, Zenodo 17366772)](https://zenodo.org/records/17366772) | physics | CC BY-NC-SA 4.0 (physics check only) |
| [Concrete box-girder bridge eigenfrequencies (Koehncke et al. 2026, Zenodo 20522191)](https://zenodo.org/records/20522191) | exam | CC BY 4.0 |
| [NASA POWER hourly 2 m air temperature (Hamburg, Jan-Apr 2024)](https://power.larc.nasa.gov/) | exam | NASA POWER open data; cite POWER |
| [Hell Bridge Test Arena imposed-damage dataset (Svendsen et al. 2025, Zenodo 14632942)](https://zenodo.org/records/14632942) | exam | CC BY 4.0 |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
