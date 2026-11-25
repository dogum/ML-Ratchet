# Model ML & Physics Honesty Audits

Read-only forensic audits of shipped models in `models/`, following the 7-point protocol in [`SKILL.md`](SKILL.md) (also saved as the reusable `ml-audit` skill).

Audits inspect **falsifiable evidence** in shipped `.npz` weights, PyTorch-vs-NumPy export parity, post-normalization real-input spread, training loops, feature pipelines, and calibration logs — without assuming intent, and without modifying model code or weights during an audit pass.

## Severity Rubric

| Verdict | Meaning |
|---|---|
| **CLEAN** | All 7 checks pass; any performance ceiling is a documented capacity, data, or simulator limit. |
| **MINOR HYGIENE** | Small code discrepancy or edge case (e.g., unmasked auxiliary loss on rare rows) with bounded impact. |
| **CRITICAL DEFECT** | Verified silent bug in normalization, export, loss, or preprocessing that materially degrades training or inference while passing existing tests. |

## Completed Audits

| Model | Version | Audited | Verdict | Key Finding | Report |
|---|---|---|---|---|---|
| **EnergyNet** | v2.1 (`r15_s0..s4`) | 2026-11-25 | **CRITICAL DEFECT** | 4 of 5 physics work-fraction features (`aero/Ep`, `rolling/Ep`, `grade_dn/Ep`, `inertia_pos/Ep`) crushed to `std = 4e-6 .. 4.5e-5` on real VED trips due to outlier-poisoned `x_mu`/`x_sd` (`sd` up to `53,606`); Torch↔NumPy parity (`1.3e-6..2.7e-6`) and ensemble diversity (`cos 0.11–0.14`) are clean. | [energynet-v2.1.md](energynet-v2.1.md) |
| **FlowCast** | v1.0 (`r1b`, `n1`) | 2026-11-25 | **CLEAN** | All 7 checks pass (`Torch↔NumPy` `2.7e-7`/`3.9e-7`, healthy static `st_mu`/`st_sd` and LSTM weight spectrum, exact feature/DOY/lag alignment). Limits are transparent CPU-budget choices (single-seed `H=64` LSTM, `24.7k` params, no ensemble). | [flowcast-v1.0.md](flowcast-v1.0.md) |
