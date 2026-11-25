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

BridgeSense is ML Ratchet's first **Attention / Transformer** model (`179,144` parameters, `670 KB` compressed float32, LayerNorm + GELU, zero BatchNorm):

1. **Baseline-to-Query Thermal Cross-Attention (2 layers, 4 heads, $d = 64$):** Query snapshots attend across the healthy reference window's temperature and thermal-lag trajectory. Removing this block (`r2_no_xattn`) raised the mean false-alarm probability on KW51's 2 freezing dev blocks from `0.002` to `0.456` (n = 2).
2. **Masked Mode-Set Self-Attention (2 layers, 4 heads, $d = 64$):** Permutation-invariant attention over variable $M \in [1, 14]$ active modes, separating coherent thermal/freezing shifts from localized modal-strain-energy structural shifts.
3. **Fusion & Uncertainty Heads:** Combines pooled temporal/mode tokens with 16 global physics-anchor scalars (baseline OLS slopes, z-scores, cold extrapolation distance) into a change logit, $\beta$-NLL heteroscedastic heads, and a per-mode $\Delta K_m$ head.

Pure-numpy inference (`bridgesense.py`) matches PyTorch to **`2.67e-6`** max absolute error over all 10 KW51 dev blocks.

## The ratchet ladder (scored on frozen `val3`)

Objective: `0.35*AUC + 0.35*dK_skill + 0.15*env_skill + 0.15*cold_extrap_specificity`, margin `> +0.01`, gated on 10 frozen KW51 Dev blocks (`splits.json`, seed `20261124`).

| Rung | Change | `val3` sim | Dev AUC | Dev Brier | Dev dK MAE | Decision |
|---|---|---:|---:|---:|---:|---|
| `r1` | Textbook linear thermoelastic FEM (`p1`) | 0.3024 | 1.000 | 0.2386 | 0.104% | kept (first) |
| `r2` | Add 1-D heat lag, freezing, axial restraint (`p2`) | 0.8063 | 1.000 | 0.0000 | 0.043% | kept (`+0.504`) |
| `r2_no_xattn` | Ablation: remove thermal cross-attention | 0.8018 | 1.000 | 0.0603 | 0.086% | rejected (`-0.005`) |
| `r3` | Real KW51 OMA noise + seasonal drift (`p3`) | 0.8648 | 1.000 | 0.0000 | 0.255% | rejected (dev gate) |
| `r4_mix` | Mixed `p2 + p3` bank with OLS output anchor | 0.8732 | 1.000 | 0.0000 | 0.134% | rejected (dev gate) |
| `r5_no_anchor` | Ablation on `p3`: drop OLS output anchor | 0.8794 | 1.000 | 0.0001 | 0.465% | rejected (dev gate — wrong sign on retrofit) |
| **`r6_mix_no_anchor`** | **Mixed `p2 + p3` (24k), direct output heads** | **0.8808** | **1.000** | **0.0000** | **0.071%** | **released (`+0.074`)** |
| `r7_ens3` | ensemble of `r4_mix` (anchored) + two unlogged `r6` seed runs (`r6_s2`, `r6_s3`) | 0.8764 | 1.000 | 0.0000 | 0.095% | rejected (`-0.004`) |

Widen-only calibration on KW51 Dev: `widen_dk = 1.00`, `widen_env = 2.34`, `prob_temp = 1.00`.

## One-shot real exam across three full-scale bridges

Opened once after `r6_mix_no_anchor` was exported (`training/exam.py`). Both classical baselines (Linear OLS and Bilinear Freezing) were fitted on the exact same KW51 Dev split.

The dumb baseline is the dev-median guess (`docs/method.md`): 0 % shift for dK/K rows, the dev prevalence for probabilities; rows with no dev-median version list the raw frequency shift.

| Exam & Metric | BridgeSense | Dev-tuned Bilinear | Linear OLS | Dumb | Verdict |
|---|---:|---:|---:|---:|---|
| **A. KW51 (49 held-out wk):** Detection AUC | **1.000** | 1.000 | 1.000 | 0.966 | tie |
| **A. KW51:** Detection Brier | **0.0000** | 0.0016 | 0.0019 | 0.2784 | **win** |
| **A. KW51:** Healthy dK/K MAE (20 wk incl. 10 freezing) | 0.019% | 0.188% | 0.189% | **0.000%** (0 % guess; raw 2·df/f 0.217%) | loss |
| **A. KW51:** Post-retrofit dK/K MAE (11 wk, truth +1.17%) | 0.169% | 0.075% | **0.072%** | 1.170% | loss |
| **B. Concrete Box-Girder (0-shot, 2 modes):** Winter freeze dK/K MAE (4 wk) | 0.881% | 6.246% | 6.372% | **0.000%** (0 % guess; raw 4.068%) | loss |
| **B. Concrete Box-Girder (0-shot):** Cold healthy vs static load AUC | **0.729** | 0.146 | 0.146 | — | **win** |
| **B. Concrete Box-Girder (0-shot):** Freezing FAR ($P \ge 0.5$, 4 wk) | 0.50 | **0.00** | 0.25 | — | loss |
| **C. HBTA (0-shot, 7 modes):** Global damage AUC (`DS5–DS8` vs `UDS`, 5 negatives, CI 0.50–1.00) | **0.800** | 0.690 | 0.690 | 0.760 | **win** |
| **C. HBTA (0-shot, 7 modes):** All damage AUC (`DS1–DS8` vs `UDS`) | 0.595 | 0.645 | 0.645 | **0.665** | loss |
| **C. HBTA (0-shot, 7 modes):** dK/K MAE (all 45 recordings) | 0.880% | **0.824%** | **0.824%** | 1.045% | loss |

**Tally: 3 wins, 6 losses, 1 tie.** On both healthy dK/K rows the truth is 0 % by construction, so the 0 % guess is exact; those rows measure false shift, where BridgeSense is 7-10x below both classical methods. The detection task on KW51 is easy: both classical methods also reach AUC 1.000, and every rung scored dev AUC 1.0. The post-retrofit truth (+1.17 %) is a linear-temperature OLS fit on the exam blocks themselves, the same method as the OLS baseline. The box girder's four freezing blocks read −0.01 %, +0.01 %, −1.57 % and +1.94 %. (Card corrected on 25 Nov 2026 review; it first listed 5 wins · 4 losses · 1 tie, using raw 2·df/f as the dumb baseline on the healthy rows.)

**Empirical 68% error-bar coverage:** `96.8%` on KW51 (n=31), `50.0%` on Concrete Box-Girder healthy (n=4), `33.3%` on HBTA (n=45).

## Limits & credits

- **Zero-temperature-spread baselines:** HBTA's short 10-minute shaker tests span ~1–2 °C, leaving thermal cross-attention without a baseline temperature slope and inflating false alarms on `UDS` replicates (`mean P = 0.825`).
- **Local secondary cracks:** HBTA `DS1–DS2` (stringer/cross-beam connections) barely perturb global frequencies (`< 0.4%`); mode shapes (MAC) or local strains are needed.
- **Z24 waiting on data:** KU Leuven's Z24 highway bridge requires registration (`help_wanted`).
- **Licences & data credits:** Code is MIT; bundled exam examples (`examples/`) are CC BY 4.0 + NASA POWER with zero KW51 rows; shipped weights (`bridgesense_weights.npz`) carry **CC BY-NC-SA 4.0** under ML Ratchet's cautious reading (`LICENSE-WEIGHTS.md`). Sources: Maes & Lombaert (2020, 2025) KU Leuven KW51 (Zenodo `3745914`, `17366772`, CC BY-NC-SA 4.0 — not redistributed in git); Koehncke et al. (2026) Concrete Box-Girder (Zenodo `20522191`, CC BY 4.0) + NASA POWER hourly `T2M`; Svendsen et al. (2025) NTNU Hell Bridge Test Arena (Zenodo `14632942`, CC BY 4.0).
