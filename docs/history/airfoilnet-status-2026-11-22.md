# AirfoilNet — status (v1.0, 22 Nov 2026)

> Kept as written on 22 Nov 2026 in the cloud project, with names of people and build roles made neutral and the claude.ai page links replaced by each model's `lab/` in this repository. Paths and status refer to that project.

The wind tunnel: from an airfoil's coordinates and a Reynolds number (4×10⁴–6×10⁵), an MLP predicts Cl, Cd and Cm against angle of attack, each with a 68 % error bar, and from them Cl max and the stall angle. It was trained only on mfoil (an XFOIL-class solver) and graded once on the UIUC low-speed wind-tunnel polars of 75 airfoils. B batch, project 4.

- Page (the network runs live in the browser; draw or pick an airfoil and read its polar; explorer of all 82 tested airfoils against NeuralFoil and the raw solver; exam; twin test; ratchet): `models/airfoilnet/lab/`
- Release: `airfoilnet-v1.0.zip` (4.3 MB), containing:
  - numpy-only inference (`airfoilnet.py`), weights with the dev widen factors, `demo.py` on four exam runs (a fixed draw in two Re strata, about 0.2 s)
  - `model.yaml`, `ratchet.yaml`, `MODEL_CARD.md`, `NOTEBOOK.md`, generated `README.md`, `lab/index.html`
  - `training/`: every script as run, the vendored solver, result JSONs (verification, twin, ratchet, exam)
  - `for_repo/models/airfoilnet/` + `for_repo/tests/golden/airfoilnet.txt` + `MERGE_NOTES.md`. `mlratchet validate airfoilnet` passes on a scratch copy of the repo; the demo was run from the unzipped copy and matches the golden file.
- Build folder: `/home/claude/bb/airfoil` (STATUS.md has the full log).

## Honest headline

- **Exam (75 airfoils, 34 families, every clean run in vols 1–5), mean abs error per airfoil:**

| | AirfoilNet | NeuralFoil 0.3.3 xxxlarge | thin airfoil | mean dev polar |
|---|---|---|---|---|
| Cl before stall | 0.091 | **0.082** | 0.195 | 0.195 |
| Cl after stall | 0.208 | **0.165** | – | 0.292 |
| Cl max | **0.082** | 0.096 | – | 0.257 |
| stall angle (deg) | **1.76** | 2.19 | – | 7.85 |
| Cd, \|ln ratio\| | 0.177 | **0.126** | – | 0.377 |
| Cd at Cl 0.5 | 0.175 | **0.108** | – | 0.406 |
| Cm before stall | 0.0173 | **0.0153** | 0.0256 | 0.0432 |

- AirfoilNet wins on Cl max and stall angle and loses to NeuralFoil on everything else.
- **Raw solver** (paired subset: one Re per exam airfoil, 74 airfoils): where it converges it is the best before stall (Cl 0.071, Cd 0.121), but it answers only 61 % of measured angles. Its Cl max is off by 0.22 and its stall angle 4.0° early, because the Newton solve stops before stall.
- **Coverage** of the 68 % bars on the exam: Cl 69 %, ln Cd 74 %, Cm 83 % (raw 33 / 41 / 29 %; widen factors ×2.77 / ×2.44 / ×4.75, fitted on dev).

## Recipe as built

- **Physics:** mfoil v2023-06-28 (K. Fidkowski, MIT), from PyPI `pymfoil` 0.1.0 (umich.edu and github.com answered 403). XFOIL itself needs gfortran.
  - Numba port of the hot loops (`mfoil_fast.py`): matches the original to 4×10⁻¹³; about 20× faster.
  - Fixes: a scipy shape-compare bug, sharp trailing edges, a `set_coords` typo, crossed-TE CST fits, and acceptance of Newton limit cycles ("loose", 1.7 % of points).
  - Considered: `pyfoil` (XFLR5's XFOIL C++ port, compiled wheel). Running its binary was refused by the permission system, so it was dropped.
- **Checks:**
  - exact Joukowski airfoil: Cl within 0.1–0.2 %, Cp 0.0012
  - thin-airfoil theory NACA 2404: α_L0 −2.096 vs −2.077°, slope 1.031 × 2π (theory 1.031)
  - published XFOIL polars (airfoiltools, NACA 0012/2412/4412, 42 points): Cl 0.0037, Cd 0.5 %, Cm 0.0007
  - NACA 2412 at Re 3×10⁶ close to the Abbott & von Doenhoff values
  - NeuralFoil numpy port vs the same XFOIL polars: Cl 0.0025, Cd 0.6 %
  - numpy == torch to 7×10⁻⁷; JS == numpy to 1.3×10⁻¹⁴
- **Noise:** none injected; no public tunnel-noise record exists for these polars.
- **Bank:** 151 sweeps (1,305 points) for the released rung, 256 for r6/r7. Shapes come from the UIUC coordinate database (1,457 CST fits), with every tested airfoil removed by name and any shape within 0.002 c rms of one also removed. Re is log-uniform from 4×10⁴ to 6×10⁵. On the shared machine one sweep took 30–50 s.
- **Splits (`splits.json`, frozen before training, by family):**
  - dev: 7 airfoils from 4 families (A18; FX 63-137, FX 74-CL5-140 MOD; M06-13-128, M6 65 %, M6 85 %; Trainer 60)
  - exam: the other 75 airfoils
- **Network:** MLP on polars, 3 × 40,710 parameters, β-NLL, 3 seeds.

## Rungs

Rule: keep a rung if its frozen-set skill is more than 0.01 above the incumbent's **and** its dev error D is at most 1.05 × the incumbent's. val1 has 50 sweeps (p1); val2 is the same sweeps with p1c labels. Every decision was re-taken on val2 and none changed.

| Rung | Change | val1 | val2 | dev D | Decision |
|---|---|---|---|---|---|
| r1 | MLP on polars, 151 sweeps | 0.765 | 0.751 | 1.033 | **released** |
| r2 | DeepONet (operator) | 0.727 | 0.714 | 1.214 | rejected |
| r3 | DeepONet + thin-airfoil anchors | 0.702 | 0.691 | 1.553 | rejected |
| r4 | r3, strict-converged points only | 0.698 | 0.685 | 1.602 | rejected |
| r3b | MLP + thin-airfoil anchors | 0.750 | 0.735 | 1.205 | rejected |
| r5 | physics p1c (dev drag mismatch) | 0.756 | 0.757 | 1.014 | rejected (+0.006) |
| r6 | 256 sweeps | 0.775 | 0.758 | 1.223 | rejected (dev) |
| r7 | p1c + 256 sweeps | 0.774 | 0.776 | 1.120 | rejected (dev) |

- **Twin test (dev, before any network):**
  - The solver gives no answer at 40–45 % of the measured angles. FX 63-137 gave none at all.
  - At Re ≤ 10⁵ its polar stops 4.2° before stall and 0.11 below Cl max, and its drag is 13 % high. Above 10⁵ its drag is 15 % low.
  - The drag trend became p1c, which the ratchet rejected.

## Limits

- Drag is weak with a small bank: 19 % mean error against NeuralFoil's 13 %. At Re ≤ 10⁵ it is 25 %.
- The model learned stall from incomplete polars, because the solver fails there. Its stall is 1.3° early on average.
- The operator family (DeepONet) lost on this data volume. The model-families slot stays empty, with a logged try.
- Scope: clean, incompressible flow at low Re. Moment bars are too wide (83 %).

## Next rungs

1. A solver that converges past separation (XFOIL proper on another machine, or better Newton globalisation in mfoil), so the bank has post-stall data.
2. A 5–10× bigger bank on a free machine, then re-try the DeepONet.
3. Draw Ncrit per sweep as tunnel-turbulence nuisance, judged on the dev twins.

## Sandbox notes

- Load average was 7–9 on 2 cores. I ran 3 bank workers at nice 5 for about 1.5 h, and memory stayed under the limit (RSS about 2.8 GB in the cgroup).
- numba recompiles everything when `mfoil_fast.py` changes (about 40 s); its cache is in `numba_cache/`.
- The exam ran once, at 17:28 on 22 Nov (flag `work/EXAM_STARTED`).
