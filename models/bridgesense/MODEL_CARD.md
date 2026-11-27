# BridgeSense — Model Card

**Trained on 2-D bridge FEMs and 1-D heat conduction. Graded once on three real full-scale bridges.**

> **Research only.** Not a structural safety inspection tool; bridge assessments require licensed engineering inspection.

## What it does

BridgeSense answers a classic Structural Health Monitoring (SHM) problem: when a bridge's tracked natural frequencies shift over weeks or months, **did the structure's stiffness actually change (damage, retrofit, or heavy added load), or is it only temperature, thermal lag, and winter freezing?**

Given a healthy baseline window and a query monitoring window (32 subsampled snapshots each) of air temperature ($T, T_{24\text{h}}, T_{72\text{h}}$) and fractional natural frequency shifts $\delta f_m = (f_m - f_{m,0})/f_{m,0}$ across $M \in [1, 14]$ tracked modes (with arbitrary missing-mode masks), BridgeSense outputs:

- `p_change` — probability of a real structural stiffness/mass change beyond environmental variation
- `delta_k_pct` $\pm$ `delta_k_sigma` — effective global stiffness shift $\Delta K / K$ (%, positive for stiffening/retrofit, negative for damage or added mass) with a 68% error bar
- `thermal_shift_pct` $\pm$ `thermal_shift_sigma` — environmental frequency shift (%) with a 68% error bar
- `mode_dk_pct` — per-mode implied stiffness shift for each active mode

## Physics & independent checks

BridgeSim (`training/simulator.py`) assembles randomized 2-D multi-DOF Euler–Bernoulli bridges across three families (40% simply-supported / multi-span girders, 35% tied bowstring arches, 25% Warren/Pratt trusses) and three materials (steel, concrete, composite):

- **Modal dynamics:** $\mathbf{K}(T, \mathbf{d})\,\boldsymbol{\phi}_m = \omega_m^2\,\mathbf{M}(t)\,\boldsymbol{\phi}_m$ with finite bearing springs, composite surfacing layer $\mathbf{K}_{\text{surf}}$, and axial-load geometric stiffness $\mathbf{K}_G(N)$.
- **1-D Fourier deck heat diffusion:** Crank–Nicolson slab conduction (exact spectral `lfilter` decomposition) mapping air temperature to lagged structural and surfacing temperatures.
- **Thermoelasticity & freezing:** Linear modulus slope $E(T)$ plus logistic asphalt/ballast freezing stiffening across $T_f \in [-3.5, +2.0]\,^\circ\text{C}$ and axial bearing thermal restraint.
- **Structural changes:** Localized element damage ($-\Delta K_e$), connection retrofits ($+\Delta K_e$, optionally mass-loaded in `p3`), and added operational/scaffolding mass ($+\Delta M$).

All five independent checks pass (`training/verify_physics.json`):

| Check | Reference | Result |
|---|---|---|
| Simply-supported Euler–Bernoulli beam (modes 1–6, 24 elems) | Analytical $f_n = \frac{n^2\pi}{2L^2}\sqrt{EI/\rho A}$ | max rel err `2.60e-4` (0.026%) |
| Cantilever Euler–Bernoulli beam (modes 1–4, 24 elems) | Analytical roots $\beta_n L$ | max rel err `3.03e-5` (0.003%) |
| Uniform stiffness scaling & Rayleigh perturbation | $\sqrt{1+\epsilon}$ & $\frac{\phi^T \Delta K \phi}{\phi^T K \phi}$ | `9.70e-14` and `8.37e-10` |
| Modal projection vs full `eigh` (20% damage + freeze) | Direct `scipy.linalg.eigh` | max rel err `3.29e-13` |
| 1-D Crank–Nicolson heat diffusion | Analytical periodic slab integral | amp err `5.87e-4` (0.059%), phase err `0.0060 rad` |

## Real noise backdrop & zero-real-labels discipline

No real labelled example ever entered training. Physics `p3` injects **436 real healthy OMA tracking noise and correlated missing-mode dropout windows** (residual std `0.326%`, mean mode availability `56.0%`) extracted strictly from KW51's early autumn baseline (2018-10-01 to 2018-11-11, pre-winter and 6 months before the retrofit).

## Architecture (`attn`)

v1.1 ships an **ensemble of three** dual-axis Transformers (`537,432` parameters = 3 × `179,144`; `2.0 MB` compressed float32; LayerNorm + GELU, zero BatchNorm). The ratchet chose it (`r7_ens3`); see the ladder below. Each member is:

1. **Baseline-to-Query Thermal Cross-Attention (2 layers, 4 heads, $d = 64$):** Query snapshots attend across the healthy reference window's temperature and thermal-lag trajectory. The v1.0 claim that this block prevents freezing false alarms did not survive the v1.1 retrain: the ablation without it (`r2_no_xattn`) scored dev Brier `0.000`, and `r2` with it fired on one freezing dev week (`P = 0.91`). n = 2 freezing dev weeks either way.
2. **Masked Mode-Set Self-Attention (2 layers, 4 heads, $d = 64$):** Permutation-invariant attention over variable $M \in [1, 14]$ active modes, separating coherent thermal/freezing shifts from localized modal-strain-energy structural shifts.
3. **Fusion & Uncertainty Heads:** Combines pooled temporal/mode tokens with 16 global physics-anchor scalars (baseline OLS slopes, z-scores, cold extrapolation distance) into a change logit, $\beta$-NLL heteroscedastic heads, and a per-mode $\Delta K_m$ head.

Member 1 (`r4_mix`) adds its dK/K and thermal outputs to the linear-OLS anchor; members 2 and 3 (`r6_s2`, `r6_s3`) predict them directly. Each member keeps its own config in the weights file. The ensemble averages logits and means, and adds the spread of member means to the mean variance. Pure-numpy inference (`bridgesense.py`) matches PyTorch to **`1.4e-6`** over the 10 KW51 dev blocks; the TypeScript port matches numpy to `1.2e-6` (24 cases).

### What changed in v1.1 (27 Nov 2026)

Four bugs found in review of PR #14, all fixed with tests (`tests/test_bridgesense.py`; four of its five tests fail on the v1.0 code):

1. The per-mode thermal regression had no intercept on the mode's own valid readings. With dropouts, the slope was biased (a test case: true slope −0.001/K fitted as −0.00025/K, giving a +2.8 % false anchor). This fed the anchor features, both classical baselines, training banks, and the Python and TypeScript inference.
2. HBTA modal bands were chosen using all ten undamaged recordings, including the five held-out controls graded in Exam C. Bands now come from the five `_01` reference recordings only and are frozen in `training/hbta_bands.json`. This gives 8 bands, not 7.
3. Modes with query readings but no reference readings entered the global anchor and z-score features. They now use the mode-mask rule (≥ 3 reference and ≥ 2 query readings), in the network, the baselines and the twin test.
4. A mixed-config ensemble kept only the last member's config on export, and the parity check ran after overwriting the weights. Configs are now stored per member and parity is checked before replacing the file.

Also new: checkpoints are chosen on separate selection banks (`sel1`, `sel2`, `sel3`), never on `val3`, which judges the ladder; seed replicates are logged rungs; the raw logs are committed in `training/logs/v1_1/`.

## The ratchet ladder (v1.1, scored on frozen `val3`)

Objective: `0.35*AUC + 0.35*dK_skill + 0.15*env_skill + 0.15*cold_extrap_specificity`, margin `> +0.01`, gated on 10 frozen KW51 Dev blocks (`splits.json`, seed `20261124`). Every rung trained 4,500 steps; checkpoint chosen on its own `sel` bank.

| Rung | Change | `val3` sim | Dev AUC | Dev Brier | Dev dK MAE | Decision |
|---|---|---:|---:|---:|---:|---|
| `r1` | Textbook linear thermoelastic FEM (`p1`) | 0.3057 | 1.000 | 0.1116 | 0.077% | kept (first) |
| `r2` | Add 1-D heat lag, freezing, axial restraint (`p2`) | 0.8176 | 1.000 | 0.0854 | 0.056% | kept (`+0.512`) |
| `r2_no_xattn` | Ablation: remove thermal cross-attention | 0.8049 | 1.000 | 0.0000 | 0.066% | rejected (`-0.013`) |
| `r3` | Real KW51 OMA noise + seasonal drift (`p3`) | 0.8717 | 1.000 | 0.0001 | 0.409% | rejected (dev gate) |
| `r4_mix` | Mixed `p2 + p3` bank with OLS output anchor | 0.8786 | 1.000 | 0.0000 | 0.149% | rejected (dev gate) |
| `r5_no_anchor` | `p3`, no OLS output anchor | 0.8785 | 1.000 | 0.0000 | 0.099% | kept (`+0.061`) |
| `r6_mix_no_anchor` | Mixed `p2 + p3`, no output anchor (seed 42) | 0.8899 | 1.000 | 0.0000 | 0.416% | rejected (dev gate) |
| `r6_s2` | Seed replicate (seed 2) | 0.8897 | 1.000 | 0.0000 | 0.234% | rejected (dev gate) |
| `r6_s3` | Seed replicate (seed 3) | 0.8838 | 1.000 | 0.0000 | 0.097% | rejected (`+0.005`) |
| **`r7_ens3`** | **Ensemble `r4_mix` + `r6_s2` + `r6_s3`** | **0.8924** | **1.000** | **0.0000** | **0.147%** | **released (`+0.014`)** |

The three r6 runs differ only in seed, yet their dev dK/K MAE spans 0.10–0.42 %. The dev gate is noisy at this size, and `r7_ens3` passed it with 0.002 pp to spare (limit 0.149 %). The v1.0 ladder (released `r6_mix_no_anchor`, checkpoints chosen on `val3`) is kept in `ratchet.yaml` and the notebook.

Widen-only calibration on KW51 Dev: `widen_dk = 1.00`, `widen_env = 1.70`, `prob_temp = 1.00` (`training/logs/v1_1/calibration.json`).

## One-shot real exam across three full-scale bridges

v1.1 was opened once on 27 Nov 2026 after `r7_ens3` was exported (`training/exam.py`; raw results `training/logs/v1_1/exam.json`; run record `training/logs/v1_1/exam_run.json`, which names the commit and the weights' sha256). Both classical baselines were fitted on the same KW51 Dev split. A row is a win only if BridgeSense beats every baseline listed.

| Exam & Metric | BridgeSense v1.1 | v1.0 | Dev-tuned Bilinear | Linear OLS | Dumb | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **A. KW51 (49 held-out wk):** Detection AUC | **1.000** | 1.000 | 1.000 | 1.000 | 0.966 | tie |
| **A. KW51:** Detection Brier | **0.0000** | 0.0000 | 0.0017 | 0.0019 | 0.2784 | **win** |
| **A. KW51:** Healthy dK/K MAE (20 wk incl. 10 freezing) | 0.051% | 0.019% | 0.189% | 0.190% | **0.000%** (0 % guess; raw 0.217%) | loss |
| **A. KW51:** Post-retrofit dK/K MAE (11 wk, truth +1.17%) | 0.199% | 0.169% | 0.075% | **0.073%** | 1.170% | loss |
| **B. Box-Girder (0-shot, 2 modes):** Winter freeze dK/K MAE (4 wk) | 1.538% | 0.881% | 6.264% | 6.369% | **0.000%** (0 % guess; raw 4.068%) | loss |
| **B. Box-Girder (0-shot):** Cold healthy vs static load AUC | **0.771** | 0.729 | 0.146 | 0.146 | 0.063 (raw shift) | **win** |
| **B. Box-Girder (0-shot):** Freezing FAR ($P \ge 0.5$, 4 wk) | 0.50 | 0.50 | **0.00** | 0.25 | — | loss |
| **C. HBTA (0-shot, 8 modes):** Global damage AUC (`DS5–DS8` vs `UDS`, 5 negatives) | **0.870** | 0.800 | 0.720 | 0.720 | 0.840 (raw shift) | **win** |
| **C. HBTA (0-shot):** All damage AUC (`DS1–DS8` vs `UDS`) | 0.720 | 0.595 | 0.655 | 0.655 | **0.795** (raw shift) | loss |
| **C. HBTA (0-shot):** dK/K MAE (all 45 recordings) | 0.720% | 0.880% | 0.888% | 0.888% | **0.473%** (raw shift; zero 1.104%) | loss |

**Tally: 3 wins, 6 losses, 1 tie — the same as v1.0; no verdict flipped.** One row came close: on HBTA dK/K MAE v1.1 beats OLS (0.720 % vs 0.888 %), which with v1.0's baseline list would have read as a win. This card adds the raw 2·df/f shift to that row, as on the other rows, and the raw shift wins (0.473 %); the HBTA truth is itself the measured shift, so the raw shift is close to it by construction.

What moved between v1.0 and v1.1: KW51 healthy false shift rose from 0.019 % to 0.051 % (still 3.7x below the classical methods); the box-girder freezing error rose from 0.88 % to 1.54 % (blocks read −1.71, −0.53, −0.96 and +2.95 %); the box-girder AUC rose 0.729 → 0.771; HBTA improved on every row (global AUC 0.800 → 0.870, all-damage 0.595 → 0.720, dK/K MAE 0.880 → 0.720 %, undamaged-control mean P 0.825 → 0.276). The HBTA numbers are not a clean comparison: the bands changed (bug 2), and v1.0's were partly chosen on the controls.

**Empirical 68% error-bar coverage:** `96.8%` on KW51 (n=31), `50.0%` on Concrete Box-Girder healthy (n=4), `62.2%` on HBTA (n=45; v1.0 `33.3%`).

## Limits & credits

- **Zero-temperature-spread baselines:** HBTA's short shaker tests span ~1–2 °C. There the raw frequency shift beats BridgeSense on all-damage AUC and dK/K MAE.
- **Local secondary cracks:** HBTA `DS1–DS2` (stringer/cross-beam connections) barely perturb global frequencies (mean P 0.23 and 0.40); mode shapes (MAC) or local strains are needed. `DS6` is also missed (mean P 0.36).
- **Training noise detrending:** v1.1 trained on a KW51 noise pool detrended with an uncentred thermal slope. Fixed on 27 Nov 2026 for future rungs; the measured effect on v1.1's pool is at most 0.0001 pp of residual sd per mode (pooled 0.3261 % both ways), so v1.1 was not retrained.
- **Noisy dev gate:** seed alone moves dev dK/K MAE by 0.1–0.4 pp on 10 blocks.
- **Post-retrofit truth:** the +1.17 % truth is a linear-temperature OLS fit on the exam blocks, the same method as the OLS baseline, which favours OLS on that row.
- **Z24 waiting on data:** KU Leuven's Z24 highway bridge requires registration (`help_wanted`).
- **Licences & data credits:** Code is MIT; bundled exam examples (`examples/`) are CC BY 4.0 + NASA POWER with zero KW51 rows; shipped weights (`bridgesense_weights.npz`) carry **CC BY-NC-SA 4.0** under ML Ratchet's cautious reading (`LICENSE-WEIGHTS.md`). Sources: Maes & Lombaert (2020, 2025) KU Leuven KW51 (Zenodo `3745914`, `17366772`, CC BY-NC-SA 4.0 — not redistributed in git); Koehncke et al. (2026) Concrete Box-Girder (Zenodo `20522191`, CC BY 4.0) + NASA POWER hourly `T2M`; Svendsen et al. (2025) NTNU Hell Bridge Test Arena (Zenodo `14632942`, CC BY 4.0).
