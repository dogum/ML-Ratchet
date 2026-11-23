# AirfoilNet v1.0 — model card

**The wind tunnel.** From an airfoil's coordinates and a Reynolds number, AirfoilNet predicts its lift, drag and
moment curves against angle of attack, each with a 68% error bar, and from them the maximum lift and the stall
angle. It learned only from an XFOIL-class flow solver. It was graded once on real wind-tunnel polars.

NeuralFoil is graded against XFOIL. AirfoilNet, NeuralFoil and the raw solver are all graded here on the UIUC
3×4 ft tunnel.

## Inputs and outputs

| | |
|---|---|
| Reads | airfoil coordinates (Selig or Lednicer `.dat`, any chord or rotation), Reynolds number 4×10⁴–6×10⁵, angles of attack in degrees |
| Returns | Cl(α) ± 68% bar; Cd(α) with a bar on ln Cd; Cm about c/4 (α) ± bar; Cl max and stall angle (from the predicted polar) |
| Scope | incompressible, clean airfoils with free transition; no trips, flaps or roughness |

The coordinates are normalised (leading edge at 0, trailing edge at 1) and refitted with an 18-parameter Kulfan CST
(8 weights a side, a leading-edge term and the trailing-edge thickness). The network reads the fitted thickness and
camber at 24 cosine-spaced stations, log Re and α.

## Physics and simulator

- **Solver:** mfoil (K. J. Fidkowski, MIT licence), an XFOIL-class code: linear-vorticity panels with transpiration
  sources, a two-equation integral boundary layer with Drela's closures, e^N transition (Ncrit 9), lag-entrainment
  turbulent shear stress and a wake, coupled in one global Newton solve. XFOIL itself could not be compiled on the
  build machine (no gfortran).
- **Our changes** (`training/mfoil_fast.py`): numba kernels for the panel influences, the boundary-layer residuals
  (Mach 0) and the Newton assembly, doing the same arithmetic in the same order; whole solves match the original
  to 4×10⁻¹³. Robustness fixes: sharp trailing edges, a `set_coords` typo, a crossed-TE CST bug, and acceptance of
  Newton limit cycles whose forces vary by less than 0.01 in Cl, 5% in Cd and 0.005 in Cm (flagged "loose").
- **Bank:** 151 alpha sweeps (1,305 converged points, 1.7% loose) on shapes from the UIUC coordinate database
  with every tested airfoil and every shape within 0.002 c rms of one removed; Re log-uniform 4×10⁴–6×10⁵;
  2° steps from about −8° to 20°, warm-started, stopping after two failures.
- **No noise injection.** There is no public record of tunnel noise for these polars to lay simulations into.

### Checks (training/verify_physics.json, training/verify_fast.json)

| Check | Result |
|---|---|
| Panel method vs exact Joukowski airfoil (α = 0, 4, 8°) | Cl within 0.10–0.22%; Cp mean abs error 0.0012–0.0021 |
| Thin-airfoil theory, NACA 2404 | zero-lift angle −2.096° vs −2.077°; slope 1.031 × 2π (thickness rule 1 + 0.77 t/c = 1.031) |
| Published XFOIL 6.99 polars, NACA 0012 / 2412 / 4412, 42 points | Cl 0.0037, Cd median 0.5%, Cm 0.0007 |
| NACA 2412, Re 3×10⁶, vs Abbott & von Doenhoff (approximate book values) | α_L0 −2.21° (~−2.0), slope 0.111/° (~0.105), Cl max 1.75 (~1.6), Cd min 0.0051 (~0.006), Cm −0.052 (~−0.045) |
| Numba port vs original mfoil | max difference 4×10⁻¹³ in Cl, Cd, Cm |
| numpy inference vs torch | 7×10⁻⁷ (float32 weights) |
| JavaScript (lab page) vs numpy | 1.3×10⁻¹⁴ relative |

## Network

An MLP on polars: 50 shape and Reynolds features plus (α/10, sin 2α, cos α) → three hidden layers of 128 (SiLU) →
Cl, ln Cd, Cm and a log-sigma for each. β-NLL loss (β = 0.5), AdamW, one-cycle schedule, 3 seeds averaged as a
mixture (3 × 40,710 parameters). The released rung is the first one: no later rung passed the ratchet.

## Ratchet

Rule: keep a rung if its skill on the frozen simulated set beats the incumbent by more than 0.01 **and** its error
D on the 7 real dev airfoils is at most 5% worse. Skill = mean over Cl, ln Cd, Cm of (1 − MAE / MAE of a constant).
D = mean(Cl pre-stall / 0.10, |ln Cd| / 0.20, Cl max / 0.20, stall angle / 3°). val1 = 50 sweeps of held-out
database shapes (physics p1); val2 = the same sweeps with physics p1c labels. Every decision was re-taken on val2;
none changed.

| Rung | Change | val1 | val2 | dev D | Decision |
|---|---|---|---|---|---|
| r1 | MLP on polars, 151 sweeps | 0.765 | 0.751 | 1.033 | **released** |
| r2 | DeepONet (branch: shape + Re; trunk: α), same data | 0.727 | 0.714 | 1.214 | rejected (sim −0.04, dev) |
| r3 | DeepONet predicting corrections to thin-airfoil Cl, Cm | 0.702 | 0.691 | 1.553 | rejected (sim, dev) |
| r4 | r3 on strictly converged points only | 0.698 | 0.685 | 1.602 | rejected (sim, dev) |
| r3b | MLP predicting corrections to thin-airfoil Cl, Cm | 0.750 | 0.735 | 1.205 | rejected (sim, dev) |
| r5 | physics p1c: dev-measured drag mismatch in the labels | 0.756 | 0.757 | 1.014 | rejected (val2 +0.006 < 0.01) |
| r6 | MLP on 256 sweeps | 0.775 | 0.758 | 1.223 | rejected (dev D +18%) |
| r7 | physics p1c and 256 sweeps | 0.774 | 0.776 | 1.120 | rejected (dev D +8%) |

What the ladder says: the operator network (DeepONet), the family this model was meant to try, lost to a plain MLP
on a bank this small. Thin-airfoil anchors made lift after stall worse. More simulations raised the simulated score
by 0.01 but made the real dev airfoils worse by 8–18%: the solver's stall is wrong, so learning it better does not help.

## Twin test (dev)

The solver's own polar for each dev airfoil at its tunnel Reynolds numbers (training/twin_dev.json):

- It gives no answer at 40–45% of the measured angles, mostly near and after stall (Newton fails when the boundary
  layer separates). FX 63-137 gave no converged point at all.
- Re ≤ 10⁵: the converged polar ends 4.2° before the real stall and 0.11 below the real Cl max; drag is 13% high.
- Re > 10⁵: stall 0.5° early; drag 15% low.

The drag trend (ln Cd += 0.015 + 0.21 (log₁₀ Re − 5)) became physics p1c, which the ratchet rejected. The stall gap
is the next rung.

## Exam (run once, 22 Nov 2026, training/exam.json)

All clean runs of the 75 exam airfoils (34 families) in UIUC LSAT volumes 1–5. Errors are averaged per run, then
per airfoil, then over airfoils. Pre-stall = up to the measured Cl max on the increasing-α sweep; post-stall = up to
10° beyond it. Drag is from wake-rake runs (pre-stall only).

| Metric (mean abs error) | AirfoilNet | NeuralFoil 0.3.3 (xxxlarge) | Thin-airfoil theory | Mean dev polar |
|---|---|---|---|---|
| Cl before stall | 0.091 | **0.082** | 0.195 | 0.195 |
| Cl after stall | 0.208 | **0.165** | — | 0.292 |
| Cl max | **0.082** | 0.096 | — | 0.257 |
| Stall angle (deg) | **1.76** | 2.19 | — | 7.85 |
| Cd, \|ln ratio\| | 0.177 | **0.126** | — | 0.377 |
| Cd at Cl = 0.5, \|ln ratio\| | 0.175 | **0.108** | — | 0.406 |
| Cm before stall | 0.0173 | **0.0153** | 0.0256 | 0.0432 |

Raw solver, on the paired subset it was run on (one Reynolds number per exam airfoil, a fixed draw; 74 airfoils):

| Metric | AirfoilNet | NeuralFoil | Raw solver (mfoil) | Thin airfoil | Mean dev polar |
|---|---|---|---|---|---|
| Cl before stall | 0.084 | 0.077 | **0.071** | 0.186 | 0.194 |
| Cl after stall | 0.203 | 0.179 | **0.120** | — | 0.282 |
| Cl max | **0.084** | 0.102 | 0.223 | — | 0.248 |
| Stall angle (deg) | **1.77** | 2.29 | 4.00 | — | 7.92 |
| Cd, \|ln ratio\| | 0.178 | 0.125 | **0.121** | — | 0.407 |
| Cd at Cl = 0.5 | 0.168 | 0.090 | **0.088** | — | 0.460 |
| Cm before stall | 0.0163 | 0.0147 | **0.0142** | 0.0264 | 0.0429 |

The raw solver scores only the angles where it converged (61% of measured angles); its post-stall error covers the
few post-stall points it reached.

By Reynolds number (all runs): at Re ≤ 10⁵ AirfoilNet's Cl before stall is 0.116 (NeuralFoil 0.101) and drag
0.221 (0.155); above 10⁵ Cl 0.072 (0.070), drag 0.150 (0.106), Cl max 0.075 (0.098).

**Error bars.** Raw bars covered 33% (Cl), 36% (ln Cd) and 17% (Cm) on dev, so one widen-only factor per output was
fitted on dev: ×2.77, ×2.44, ×4.75. Exam coverage of the nominal 68% bars: **Cl 69%, ln Cd 74%, Cm 83%**
(raw: 33%, 41%, 29%).

## Where it fails

- Drag. With 151 sweeps it misses drag by 19% on average where NeuralFoil misses by 13%.
- Low Reynolds numbers (≤ 10⁵), where laminar separation bubbles dominate: Cl error 0.116, drag 25%.
- Stall: the solver does not converge there, so stall was learned from incomplete polars. Predicted stall is 1.3°
  early on average. It still beats NeuralFoil on Cl max and stall angle.
- Moment bars are too wide (83% coverage).

## Limits

Low Re (4×10⁴–6×10⁵), incompressible, clean airfoils. Shapes far from the UIUC database (very thick, very cambered,
blunt) are extrapolation. The UIUC data carry their own tunnel corrections and uncertainty; volume 1 has no valid
moment data.

## Credits and licences

- Wind-tunnel data: UIUC Low-Speed Airfoil Tests, *Summary of Low-Speed Airfoil Data* vols 1–5 (Selig, Guglielmo,
  Broeren, Giguère, Lyon, Ninham, Gopalarathnam, McGranahan, Williamson et al., 1995–2012). Distributed under the
  GNU GPL v2. The excerpts in `examples/` are covered by it (`examples/LICENSE-GPL-DATA.txt`).
- Shapes: UIUC Airfoil Coordinates Database (M. Selig and contributors).
- Solver: mfoil by K. J. Fidkowski (MIT), obtained as the `pymfoil` 0.1.0 PyPI package (its licence file is in
  `training/`).
- Baseline: NeuralFoil 0.3.3 by P. Sharpe (MIT), run from its published weights via a numpy port.
- Verification: XFOIL polars published on airfoiltools.com (not redistributed).
- AirfoilNet code: MIT, like the rest of ML Ratchet.
