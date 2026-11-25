# Model ML & Physics Honesty Audits

Read-only forensic audits of shipped models in `models/`, following the 10-point protocol in [`SKILL.md`](SKILL.md) (also saved as the reusable `ml-audit` skill).

Audits test **falsifiable evidence** in shipped `.npz` weights, input standardization, counterfactual feature sensitivity, PyTorch-vs-NumPy export parity, operational causality / look-ahead, metadata shortcuts, seed reproducibility, rung arithmetic, and calibration — without assuming intent, and without modifying model code or weights during an audit pass.

## Workspace Layout (Multi-Reviewer)

- [`SKILL.md`](SKILL.md): Canonical 10-point audit protocol and mandatory 3-question finding template (*What failed? / What changed when tested? / What consequence remains unmeasured?*).
- `docs/audits/<model>-v<ver>.md`: Canonical audit report per model (updated when independent checks cross-verify or refine findings).
- `docs/audits/reviews/<reviewer>/`: Optional independent reviewer scratch/first-pass folder so separate auditors can check a model independently before reconciling into the canonical report.

## Severity Rubric

| Verdict | Meaning |
|---|---|
| **CLEAN** | All 10 checks pass; any performance ceiling is a documented capacity, data, or simulator limit. |
| **MINOR HYGIENE** | Real code discrepancy, API guardrail gap, or reproducibility issue with bounded or conditional impact. |
| **CRITICAL DEFECT** | Verified silent bug in normalization, export, causality/look-ahead, exam shortcut, loss, or preprocessing that materially affects training, inference, or what the exam measures while passing existing tests. |

## Audit Tracker

| Model | Version | Base Commit | Status | Verdict | Key Finding | Report |
|---|---|---|---|---|---|---|
| **EnergyNet** | v2.1 (`r15_s0..s4`) | `61c70bf` | Cross-checked (2 reviewers) | **CRITICAL DEFECT** | 4 of 5 physics work-fraction inputs (`aero/Ep`, `rolling/Ep`, `grade_dn/Ep`, `inertia_pos/Ep`) crushed to `std = 4e-6 .. 4.5e-5` by outlier-poisoned `x_mu`/`x_sd` (`sd` up to `53,606`, triggered on moving downhill trips where `ph[0] = 0`); holding all 4 constant moves shipped VED predictions `<= 0.00039%` vs `6.08%` for control `grade_up/Ep`. Torch↔NumPy parity and ensemble diversity clean. | [energynet-v2.1.md](energynet-v2.1.md) |
| **FlowCast** | v1.0 (`r1b`, `n1`) | `61c70bf` | Cross-checked (2 reviewers) | **MINOR HYGIENE** | Forward pass, Torch↔NumPy parity (`2.7e-7`/`3.9e-7`), static scaling (`max \|z\| <= 2.01` on shipped examples), and feature alignment all pass. Two minor issues found: (1) `train.py` seeds NumPy only, not PyTorch weight init/dropout, and resume resets RNG; (2) `flowcast.py` nowcast API silently accepts `lead=0` (same-day leakage; exam scripts cleanly use `1, 3, 7`). | [flowcast-v1.0.md](flowcast-v1.0.md) |
| **QuakeNet v2** | v2 (release 1.0) | `2dd401e` | First pass reproduced on audit branch; second review open | **Exam shortcut** | A station-metadata-only fit recovered all four example catalogue epicentres within 4.24 m. Dev/exam IDs were disjoint and 15/15 promotion decisions matched. | [Independent report and probe](reviews/reviewer-b/quakenet-v2.md) |
| **BatteryPilot** | v1.3 | `61c70bf` | First pass reproduced on audit branch; second review open | **Pre-hour information leakage** | Future temperature moved an offer through wear costs. Last-observation wear forecasts preserved both strategies' rankings on three shipped node-years. First-hour price padding also moved its own offer. All 9 candidate decisions matched. | [Independent report and probe](reviews/reviewer-b/batterypilot-v1.3.md) |
| **BuildingDetective** | v1.0 | `61c70bf` | First pass (`reviewer-a`); second review open | **CRITICAL DEFECT** | `featurize.py:76` drops `quirk_hourly`'s `keep_mask`, collapsing `hourly.npz` record-span `xs` to `1.1e-3`/`1.7e-4` (truncating `HUE_2` to 330 d drives `z=-85`, triples predicted area, flips use); full-year BDG2 district-meter zero-fractions hit `\|z\| = 25.8..117.7` unclipped (`+44.8%`/`+99.6%` shift when clipped to `\|z\|<=8`). Torch↔NumPy parity (`<=2.1e-5`) and rung replay (`29/30`, 1 cap withdrawal) clean. | [Independent report and probe](reviews/reviewer-a/buildingdetective-v1.0.md) |
| **LoadCast** | v1.0 (`r3`) | `61c70bf` | First pass (`reviewer-a`); second review open | **CRITICAL DEFECT** | `has_electricity` (`idx=18`) has `sd_x = 0.001` (`z=-1000` when `0`), triggered by default `meters_present=None` on non-electricity meters (`24–50×` cold blow-up, `43–56%` warm drop on shipped `Bull`/`Cockatoo` with identical `hvac`); `night_ratio` `sd_x = 23.37` attenuates sensitivity (`1.3–2.4%` vs `27–116%` control); `forecast()` allows unsliced `history` overlap and `train.py` never seeds PyTorch. Parity (`<=2.6e-6`) and rung replay (`6/6` + `3/3`) clean. | [Independent report and probe](reviews/reviewer-a/loadcast-v1.0.md) |
| **DemandCast** | v1.1 (`r8_wide` + `r9b`) | `61c70bf` | First pass (`reviewer-a`); second review open | **CRITICAL DEFECT** | Solar LST 20:00 column (`idx=66`, `\|\|W0\|\| = 8.36`) pins shipped `BPAT` exam days at `1.200` CF for 11 h and explodes `sigma` to `3.646` (`hi95 = 8.35` CF); zeroing that single scalar cuts nMAE `43.3%->11.2%` and `52.3%->15.6%` (`sigma 3.65->0.16`). Demand `miss_frac` (`col 328`, `xs=0.001`) quadruples MAPE (`2.26%->9.74%`) on 1 missing day (`z=143`). `CISO` `+2 h` lag exceeds `solar_offset` `±1 h` clamp. Parity (`<=1.75e-6`), causality, seeding, and `val_v3` rungs (`8/8`) clean. | [Independent report and probe](reviews/reviewer-a/demandcast-v1.1.md) |

