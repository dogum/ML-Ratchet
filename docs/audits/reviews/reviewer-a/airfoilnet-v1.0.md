# AirfoilNet v1.0 | Raw Lednicer failure, broken mirror symmetry, and stale docstring

**Reviewer A, first pass.** Model base `61c70bf`, audit checkout `d903cb6`, released rung `r1`. **Verdict: CRITICAL DEFECT (public API input contract & physics symmetry).** AirfoilNet v1.0 ships a 3-member MLP ensemble (`r1`, trained on 151 mfoil sweeps) with clean weights, healthy ensemble diversity, exact scale/rotation/translation and reverse-Selig invariance, tight Torch↔NumPy parity, and clean dev/exam splits. However, three concrete defects appear in the public module and learned physics: (1) `polar(coords=...)` documents support for Lednicer-ordered coordinate arrays, but silently produces catastrophic predictions (`|ΔCl|` up to `5.44`, `|Δ ln Cd|` up to `12.24`) when given raw Lednicer coordinates; (2) the learned MLP violates aerodynamic mirror symmetry strongly on both symmetric and cambered airfoils; and (3) the module docstring still describes the rejected thin-airfoil-anchored DeepONet (`r3`).

## 1. `polar(coords=...)` silently corrupts raw Lednicer coordinate arrays

- **What failed?** [`airfoilnet.py:194`](../../../../models/airfoilnet/airfoilnet.py#L194) documents `AirfoilNet.polar(coords=...)` as accepting `(n,2) airfoil coordinates (any chord/orientation, Selig or Lednicer order)`. In reality, Lednicer-to-Selig conversion only exists inside [`read_dat`](../../../../models/airfoilnet/airfoilnet.py#L35-L38) (triggered by a file header count row), whereas [`normalize`](../../../../models/airfoilnet/airfoilnet.py#L42-L55) assumes `X[0]` and `X[-1]` are both at the trailing edge (`te = 0.5 * (X[0] + X[-1])`). On a Lednicer array (upper LE→TE followed by lower LE→TE), `X[0]` is the leading edge `(0,0)` and `X[-1]` is the trailing edge `(1,0)`, placing `te` at the midchord `(0.5, 0)` and corrupting both rotation and the CST fit without raising an error.
- **What changed when tested?** Across all four shipped UIUC exam airfoils in [`models/airfoilnet/examples/`](../../../../models/airfoilnet/examples):
  - **Scale (`2.5×`) + rotation (`37°`) + translation (`+10, -5`):** invariant to **`<= 6.79e-14`**.
  - **Reverse Selig order (`xy[::-1]`):** invariant to **`0.00e+00`**.
  - **Raw Lednicer order (`LE→TE upper, LE→TE lower`) passed to `polar(coords=...)`:** max `|ΔCl|` = **`4.15–5.44`** and max `|Δ ln Cd|` = **`6.12–12.24`** (completely unphysical polars, no warning or exception).
- **What consequence remains unmeasured?** Shipped `.dat` files loaded through `read_dat()` with a Lednicer count header are unaffected; the UIUC exam pipeline reads pre-fitted `shapes.npz` CST parameters rather than calling `polar()` on raw Lednicer arrays, so the published 75-airfoil exam scores are unaffected. Any downstream Python caller passing an in-memory Lednicer array per the `polar()` docstring is silently affected.

## 2. Large aerodynamic mirror-symmetry violation

- **What failed?** Exact incompressible aerodynamics is invariant under vertical reflection (`y -> -y`, `alpha -> -alpha`): `Cl_mir(-α) = -Cl(α)`, `Cd_mir(-α) = Cd(α)`, `Cm_mir(-α) = -Cm(α)`, and for a symmetric section `Cl(0°) = 0`, `Cm(0°) = 0`. AirfoilNet ships an unconstrained MLP (`arch: "mlp", anchor: false`) trained on 151 UIUC sweeps that are predominantly positively cambered over mostly positive angles of attack, without equivariance or mirror-data augmentation.
- **What changed when tested?**
  - **Symmetric NACA 0012 (`Re = 2e5`):** predicts **`Cl(0°) = +0.0670`**, **`Cm(0°) = -0.0129`** (~`0.6°` of phantom positive camber), `Cl(-6°, +6°) = (-0.6721, +0.7273)` (`+0.055` lift asymmetry), and `Cd(-6°) = 0.01705` vs `Cd(+6°) = 0.01486` (`14.7%` drag asymmetry).
  - **All four shipped UIUC exam airfoils (inverted `y -> -y` at `-alpha`):**

| Shipped exam airfoil | `max \|Cl_mir(-α) + Cl(α)\|` | `max \|ln Cd_mir(-α) − ln Cd(α)\|` | `max \|Cm_mir(-α) + Cm(α)\|` |
|---|---:|---:|---:|
| `bw3_100000.json` | 1.526 | 0.731 | 0.1256 |
| `ultimate_202000.json` | 0.579 | 0.393 | 0.0478 |
| `us1000_102000.json` | 1.477 | 0.542 | 0.0802 |
| `wasp_203000.json` | 1.034 | 0.577 | 0.0854 |

- **What consequence remains unmeasured?** Whether adding mirror augmentation (`y -> -y, α -> -α, Cl -> -Cl, Cm -> -Cm`) or an anti-symmetric/symmetric decomposition improves negative-alpha and low-camber accuracy on the 75-airfoil UIUC exam remains unmeasured without retraining.

## 3. Stale module docstring & minor training hygiene

- **Docstring vs shipped weights:** [`airfoilnet.py:10-12`](../../../../models/airfoilnet/airfoilnet.py#L10-L12) describes a thin-airfoil-anchored DeepONet (`r3`), whereas `airfoilnet_weights.npz` ships `meta = {"arch": "mlp", "anchor": false, "p_latent": 0, "n_members": 3}` (`r1`, correctly documented in `model.yaml` and `MODEL_CARD.md`) because every DeepONet and anchored rung (`r2`, `r3`, `r4`, `r3b`) was rejected by the ratchet.
- **Constant-baseline argument in `train.py`:** [`training/train.py:197-198`](../../../../models/airfoilnet/training/train.py#L197-L198) computes `dumb = D['Y'].mean(0)` (train mean, matching the docstring) and then passes `V['Y'].mean(0)` (validation mean) into `sim_skill`. Since all rungs share the same frozen `V`, this shifts the absolute `sim_skill` denominator uniformly rather than biasing between-rung comparisons.

## Weights, export parity, splits, calibration, and ratchet checks

- **Weights & diversity:** All tensors in `airfoilnet_weights.npz` are finite with **0 dead rows or columns** across all 3 ensemble members; first-layer (`net.0.w`) pairwise cosines are `[0.0451, -0.0137, 0.0076]`.
- **Feature sensitivity & OOD:** Raising `Re` from `5e4` to `5e5` reduces predicted mean `ln Cd` by `0.805–2.093` across the four examples (physically right sign). Out-of-range queries (`Re = 1e7`, `alpha ∈ {-30, 0, +30}°`) remain finite (`Cl = [-2.021, -0.069, 1.237]`).
- **Export parity:** Reconstructing all 3 PyTorch `Net` members from [`training/train.py:87`](../../../../models/airfoilnet/training/train.py#L87) matched `AirfoilNet.predict` across all four shipped examples within **`1.55e-6`** across `(Cl, ln Cd, Cm)`.
- **Seeding, splits, calibration, and ratchet:** [`training/train.py:129`](../../../../models/airfoilnet/training/train.py#L129) seeds both Torch and NumPy (`1000 + m`). [`training/splits.json`](../../../../models/airfoilnet/training/splits.json) has **7 dev** and **75 exam** airfoils with **0 overlap**. Shipped widen factors `[2.774, 2.444, 4.755]` satisfy `>= 1.0` (reflecting narrow raw dev coverages `0.330, 0.357, 0.166`). Both [`ratchet.json`](../../../../models/airfoilnet/training/ratchet.json) (`val1`) and [`ratchet_val2.json`](../../../../models/airfoilnet/training/ratchet_val2.json) (`val2`) replay **8/8 rows** identically under the stated rule, retaining `r1` as incumbent.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/airfoilnet_v1_0_probe.py
```

Requires NumPy and CPU PyTorch. [Probe script](airfoilnet_v1_0_probe.py)
