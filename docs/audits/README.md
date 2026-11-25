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
