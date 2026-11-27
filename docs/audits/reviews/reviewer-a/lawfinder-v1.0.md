# LawFinder v1.0 | EV road-law sign inversion, post-2022 building inversion, and low-inertia grid singularity

**First pass (`reviewer-a`). Verdict: CRITICAL DEFECT.** Audited against model base `61c70bf` across all six released symbolic laws (`water`, `fire`, `grid_inertia`, `grid_nadir`, `bldg`, `road`) in [`lawfinder_laws.json`](../../../../models/lawfinder/lawfinder_laws.json), [`lawfinder.py`](../../../../models/lawfinder/lawfinder.py), [`splits.json`](../../../../models/lawfinder/splits.json), archived exam reports, and all six `*_ratchet.json` logs following [`SKILL.md`](../../SKILL.md).

## 1. Intended-Use Contract & Information Boundary
- **Intended use:** Provide compact, interpretable one-line physical laws discovered by symbolic regression purely on each platform's simulator (WaterSim, FireSim, GridSim, BuildingSim, RoadSim), with dev-widened 68% error bars (`sigma >= sigma_sim`).
- **Information boundary:** All four domains recorded in `splits.json` (`water`, `fire`, `grid`, `bldg`) have `0` dev/exam overlap; `road` splits are omitted from `splits.json` (inherited externally from EnergyNet/RoadData). `training/exam_road_errata.py` transparently documents a unit-scaling bug in run 1's EnergyNet VED comparison (`exam_road.json` vs `exam_road_reported.json`).

## 2. Findings

### Finding A — Released `road` law (`p1-c23`) mathematically inverts grade, acceleration, and rolling-resistance physics for every electric vehicle (`e = 1`), confirmed on real exam EVs [`CRITICAL DEFECT`]
- **What failed?** The released road law is:
  `E = M * (sq(exp(((1.48 + v) ^ (-0.5951)) - e))) ^ (((K - ((-0.06983) * (G + R))) - h) + K)`
  For ICE (`e=0`) and HEV (`h=1, e=0`), the exponential base `exp(2 * (1.48 + v)^(-0.5951))` exceeds `1`, so positive acceleration `K`, uphill grade `G`, and rolling resistance `R` increase predicted energy. For **every moving EV (`e = 1`, any `v > 0.45 m/s`)**, `(1.48 + v)^(-0.5951) - 1 < 0`, making the base **strictly less than `1`**. Raising a base `< 1` to `2*K + 0.06983*(G + R)` **inverts the sign of every load term**: uphill climbs and hard acceleration *reduce* predicted EV energy, while downhill descents *increase* it.
- **What changed when tested?**
  - At `v = 20 m/s, K = 0.1, M = 1.6 t, R = 0.01`:
    - **ICE (`e=0, h=0`):** flat `1.7069 MJ/km` → uphill `+6%` `1.9538` (`+14.5%`), downhill `−6%` `1.4913` (`−12.6%`), `K 0.1→0.5` `2.2091` (`+29.4%`).
    - **EV (`e=1, h=0`):** flat `1.1426 MJ/km` → uphill `+6%` **`0.5658` (`−50.5%`)**, downhill `−6%` **`2.3075` (`+102.0%`)**, `K 0.1→0.5` **`0.2986` (`−73.9%`)**, `R 0.01→0.02` **`1.1413` (`−0.1%`)**.
  - Checking archived [`training/results/exam_road_reported.json`](../../../../models/lawfinder/training/results/exam_road_reported.json) confirms this inversion broke real-world EV performance while being masked by pooled VED metrics (where EVs are `167 / 22,877 = 0.73%`):
    - **`VED_EV` (`n = 167`):** **`Spearman rho = −0.383`** (negative rank correlation!), **`R² = −1.209`**, **`MdAPE = 43.27%`**, **`bias = −42.15%`**.
    - **`EPA_EV` (`n = 1,492`):** **`R² = −5.496`**, **`MdAPE = 43.06%`** (`EV_HWY`: **`R² = −11.035`**, **`MdAPE = 54.18%`** vs textbook `roadload_dev` `12.32%`).
- **What consequence remains unmeasured?** Whether enforcing positivity/monotonicity constraints in `lawsearch.py` yields a physically valid cross-powertrain road law without sacrificing ICE/HEV accuracy remains unmeasured.

### Finding B — Released `bldg` law (`b2-c13`) has a negative heating-degree-day slope for all buildings built from `2022` onward and ignores cooling (`C`) [`CRITICAL DEFECT`]
- **What failed?** The released building law is `EUI = 186.5 / (1.926 ^ A) + H * (2021.87 - Y)`. Because the fitted intercept constant is `2021.87`, the slope `d(EUI)/dH = 2021.87 - Y` flips sign in `2022`. Cooling degree days `C` is required in `vars` but absent from the tree.
- **What changed when tested?** For `Y = 2022`, `dE/dH = −0.13`; for `Y = 2025`, `dE/dH = −3.13` (`H = 1 → 47.13`, `H = 5 → 34.61`); for `Y = 2030`, `dE/dH = −8.13` (eventually predicting negative annual EUI in cold climates). Varying `C` tenfold (`0.5 → 5.0`) changes output by `0.0000`.
- **What consequence remains unmeasured?** Historical CBECS/RECS exam buildings predate 2022, so this defect primarily impacts new-construction (`>= 2022`) and warm-climate cooling evaluations.

### Finding C — Released `grid_nadir` law (`g1-c12`) returns `NaN` for `E <= 66.89 GWs`—triggered directly by LawFinder's own `grid_inertia` law [`CRITICAL DEFECT`]
- **What failed?** `grid_nadir` evaluates `(((-66.89 + E) * 5.294e4) ^ (-0.1042) * (L + 2)) ^ 2`. In [`lawfinder.py:30`](../../../../models/lawfinder/lawfinder.py#L30), `^` returns `NaN` whenever the base is `<= 0` (`E <= 66.89 GWs`) and diverges to `+inf` as `E → 66.89+`.
- **What changed when tested?** Composing LawFinder's own `grid_inertia` (`D = 30 GW, s = 0.9 → E = 46.41 GWs`) into `grid_nadir(L=1.0, E=46.41, D=30.0)` returns **`NaN`**; at `E = 66.89 GWs` it spikes to **`3.21 Hz`** (vs. `0.37 Hz` at `150 GWs`).
- **What consequence remains unmeasured?** How many future high-renewable (`s >= 0.8`) grid operating points fall below `66.89 GWs` outside the historical GB exam range remains unmeasured.

### Finding D — Declared-but-unused inputs, `fire` zero-division, and `water` `Q/P > 1` bound violation [`MINOR HYGIENE`]
- **What failed?** Five of six laws require input kwargs in `law.vars` that do not appear in the expression tree: `fire` (`-0.5585 / P`) ignores `W`, `M`, and `R0` completely (and returns `-inf` at `P = 0`), `water` ignores `S` (and predicts physically impossible runoff ratio `Q/P = 1.3834 > 1.0` at `phi = 0.1, fs = 0.5`), `grid_nadir` ignores `D`, `bldg` ignores `C`, and `road` ignores `U`, `D`, `T`.
- **What consequence remains unmeasured?** Whether callers expect required kwargs to actively affect predictions is an API contract risk rather than an exam discrepancy.

## 3. Clean Checks (Items 6, 8, 10)
- **Calibration widen-only compliance (CLEAN):** All `6/6` laws satisfy `sigma >= sigma_sim` (`widen` ratios `1.000×` to `6.734×`).
- **Splits & ratchet arithmetic replay (CLEAN):** `0` dev/exam overlap across `water`, `fire`, `grid`, and `bldg` in `splits.json`, and all **`238 / 238`** logged keep/reject decisions across all 6 `*_ratchet.json` files and 11 physics-version replay passes match the published rule (`sim > inc + 0.01` and `dev <= 1.05 * inc`).

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/lawfinder_v1_0_probe.py
```

Model snapshot `61c70bf`, audit checkout `759c345`. Uses NumPy only. [Reproduction probe](lawfinder_v1_0_probe.py)
