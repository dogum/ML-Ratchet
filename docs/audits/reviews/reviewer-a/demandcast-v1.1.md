# DemandCast v1.1 — Independent First-Pass Audit (`reviewer-a`)

Independent first pass, 2026-11-25, on the released v1.1 (`r8_wide` weights + `r9b` solar alignment) at `61c70bfb2a27f3d592fab317ccd40a048e020b4b`.
**Status:** First pass; second review open.
**Overall Verdict:** **`CRITICAL DEFECT`** (two verified unclipped input-amplification bugs — an `8.36`-norm LST 20:00 solar column that pins real `BPAT` exam forecasts at `1.200` CF for 11 hours and blows up `sigma` to `3.646` CF, and a `d.xs[328] = 0.001` demand `miss_frac` blow-up on missing telemetry — plus a `±1 h` solar-alignment clamp that leaves shipped `CISO` exam days 1 hour off).

## 1. Intended-Use Contract & Information Boundary

DemandCast issues day-ahead 24-hour demand, wind, and solar forecasts (mean and uncertainty) for a US balancing authority at `10:00 LST` on day `D-1`, using the 168 hours of pre-issue meter/physics history (`[i0 - 182, i0 - 14)`), fleet capacity at issue (`i0 - 14`), and weather for day `D`. In [`training/region.py:56-108`](../../../../models/demandcast/training/region.py#L56-L108) and [`training/realbank.py:30-65`](../../../../models/demandcast/training/realbank.py#L30-L65), pre-issue slicing is strictly causal, dev years (`2019, 2021`, cross-year calibrated) are disjoint from exam years (`2023–2025`), and `actual`-weather vs. `forecast`-weather exam tracks are reported separately ([`training/exam.py:93`](../../../../models/demandcast/training/exam.py#L93)).

## 2. Critical Defect #1: Unmasked LST 20:00 Solar Column Detonates Real `BPAT` Exam Forecasts & Intervals

**What failed?** In `demandcast_weights.npz`, solar input features `idx = 66` (`a_last24[10]`, reported CF at `20:00 LST`) and `idx = 90` (`p_last24[10]`, physics CF at `20:00 LST`) were nearly always zero in simulated training (`s.xm[66] = 0.00000`, `s.xs[66] = 0.02000`, the floor from [`training/train.py:50`](../../../../models/demandcast/training/train.py#L50)). On the tiny fraction of simulated northern-summer rows with faint twilight output, small `z` values caused training to grow enormous first-layer column norms:

- `||W0[:, 66 (a_20h)]|| = 8.358`, `||W0[:, 90 (p_20h)]|| = 8.053` (`cos = +0.9632`), **6–7× larger** than daytime solar hours (`1.13–1.59`).

Inference ([`demandcast.py:24, 31`](../../../../models/demandcast/demandcast.py#L24)) applies no input clipping and uses an unbounded `softplus` uncertainty head (`np.logaddexp(0, ls)`). When real EIA-930 telemetry reports modest nonzero generation at `20:00 LST` while physics `p_20h = 0.0` (both shipped real `BPAT` exam days in `examples/exam_days.npz`), `W0[:, 66] * (a_20h / 0.02)` injects a massive impulse into layer 0.

**What changed when tested?** On both shipped `BPAT` exam days (`rows 14–15` of `examples/exam_days.npz`), where the physics prior is already accurate (`13.6%` and `12.8%` nMAE):

| Shipped `BPAT` exam day | `a_20h` / `p_20h` | Variant | Solar nMAE | Raw max `mu` (hrs pinned at `1.2`) | Max `sigma` | Max `hi95` (CF) |
|---|---|---|---:|---|---:|---:|
| `2023-07-22` (phys `13.6%`) | `0.0580` / `0.0000` | Shipped v1.1 | **43.3%** | `1.994` (**11 h** at `1.200`) | 0.252 | 1.695 |
| `2023-07-22` | — | **Zero single scalar `xs[66]`** | **11.2%** | `0.941` (0 h) | 0.206 | 1.255 |
| `2023-07-22` | — | Zero all night `a_last24` (`p==0`) | **10.3%** | `0.925` (0 h) | 0.148 | 1.157 |
| `2025-05-31` (phys `12.8%`) | `0.0894` / `0.0000` | Shipped v1.1 | **52.3%** | `2.702` (**11 h** at `1.200`) | **3.646** | **8.347** |
| `2025-05-31` | — | **Zero single scalar `xs[66]`** | **15.6%** | `0.996` (0 h) | **0.164** | 1.295 |
| `2025-05-31` | — | Zero all night `a_last24` (`p==0`) | **12.8%** | `0.965` (0 h) | **0.131** | 1.195 |

Zeroing that single twilight scalar (`xs[66] = 0.0`) drops `BPAT` solar error by 32–37 percentage points and collapses `2025-05-31` peak `sigma` by **22×** (`3.646 -> 0.164`, eliminating an `835%` capacity-factor upper bound).

**What consequence remains unmeasured?** In `training/exam_r8_wide.json`, `BPAT` full-exam solar nMAE is `51.8%` (`cov68 = 0.59`). How much of that 3-year `BPAT` exam error (and other BAs with occasional nighttime EIA-930 solar reporting) is explained by `xs[66]`—and whether masking nighttime history inputs `a_last24` by `p_last24 == 0` resolves it across all 26,280 `BPAT` exam hours—remains unmeasured without the full exam bank.

## 3. Critical Defect #2: Demand `miss_frac` (`col 328`, `xs = 0.001`) Blow-Up on Missing Telemetry

**What failed?** [`training/train.py:50`](../../../../models/demandcast/training/train.py#L50) applied the `2e-2` standardization floor only to wind and solar (`1e-3 if kind == 'd' else 2e-2`), while simulated demand banks never inject missing history hours (`d.xm[328] = 0.0`, `d.xs[328] = 0.001`, `||W0[:, 328]|| = 0.7487`). Real EIA-930 demand histories regularly contain QA-flagged `NaN` hours ([`training/real.py:32`](../../../../models/demandcast/training/real.py#L32)), which set `miss_frac = mean(~isfinite(rho))` in [`training/features.py:43`](../../../../models/demandcast/training/features.py#L43) and pass unclipped into [`_forward`](../../../../models/demandcast/demandcast.py#L24).

**What changed when tested?** Perturbing only `xd[:, 328]` (`miss_frac`) on the 16 clean shipped exam days (`examples/exam_days.npz`, all 16 have `miss_frac = 0.0`):

| Missing hours in 168 h history | `miss_frac` (`z[328]`) | 16-day demand MAPE | Max hourly MW shift | Mean `log_sigma` | Max `hi95 / mw` |
|---|---|---:|---:|---:|---:|
| 0 / 168 (baseline) | 0.000 (`0.0`) | 2.26% | — | 0.0271 | — |
| 4 / 168 | 0.024 (`23.8`) | 2.53% | 5.64% | 0.0309 | 1.14× |
| 8 / 168 | 0.048 (`47.6`) | 3.61% | 12.40% | 0.0409 | 1.30× |
| 24 / 168 (1 missing day) | 0.143 (`142.9`) | **9.74%** | **35.68%** | 0.0657 | 2.27× |
| 48 / 168 (2 missing days) | 0.286 (`285.7`) | **19.82%** | **74.82%** | 0.2583 | **66.5×** |
| 168 / 168 (outage) | 1.000 (`1000.0`) | **110.70%** | **521.13%** | 3.2709 | **`4.38e+13×`** |

**What consequence remains unmeasured?** How many days in the full 2023–2025 `real_exam.npz` bank had `q_ok == False` hours in their 168 h history (and thus suffered silent MAPE/interval inflation during the exam) remains unmeasured because only 16 zero-missing exam days are shipped in the repo.

## 4. Minor Hygiene: `solar_offset` `{-1, 0, +1}` Clamp Leaves Shipped `CISO` Exam Days 1 Hour Off

**What failed?** [`training/real.py:41-43`](../../../../models/demandcast/training/real.py#L41-L43) applies a permanent dev-calibrated roll (`lag = -1` -> `np.roll(ys, +1)`) to `CISO` solar across all years. When `CISO` EIA-930 timestamps also shifted in 2024–2025, the combined lag against `ps` reached **`+2 h`**, while [`demandcast.py:69`](../../../../models/demandcast/demandcast.py#L69) (`int(np.sign(rh[i]))`) clamps the v1.1 alignment `r` to `{-1, 0, +1}`.

**What changed when tested?** On both shipped `CISO` exam days (`examples/exam_days.npz` rows 2–3):

- `2024-04-16`: `r_hat_abs = +1.612`, `r_hat_rel = +1.150 -> r = +1`; `corr(roll(ps, 0/+1/+2), ys) = 0.7612 / 0.9221 / 0.9974`; nMAE drops `53.3% -> 26.8%` but remains 1 h early.
- `2025-04-19`: `r_hat_abs = +2.018`, `r_hat_rel = +1.556 -> r = +1`; `corr(roll(ps, 0/+1/+2), ys) = 0.7546 / 0.9146 / 0.9887`; nMAE drops `82.3% -> 51.2%` but remains 1 h early.

**What consequence remains unmeasured?** The full-exam `CISO` solar nMAE improvement from allowing `|r| = 2` (or removing the hardcoded dev roll in favor of direct `solar_offset`) remains unmeasured.

## 5. Clean Checks & 10-Point Summary

| # | Protocol Check | Result | Measured Evidence |
|---|---|---|---|
| 1 | Intended-use boundary | **CLEAN** | Strict pre-issue `[i0-182, i0-14)` history and issue-time capacity (`region.py:56-108`); dev `2019/2021` disjoint from exam `2023–2025`; actual vs. forecast weather tracked separately. |
| 2 | Weights, norm & sensitivity | **CRITICAL DEFECT** | Zero NaN/Inf; `s.mu` 7 dead rows are strictly always-night hours `[0,1,2,3,21,22,23]`; `s.body.0[:, 66]` (`||W0|| = 8.36`) detonates `BPAT`; `d.xs[328] = 0.001` detonates on missing demand history. |
| 3 | Torch ↔ NumPy parity | **CLEAN** | Exact `model.Net` reconstruction matches `demandcast._forward` (`d: 1.14e-7`, `w: 7.65e-8`, `s: 3.23e-7`). |
| 4 | Causal / cold-start leakage | **CLEAN** | History ends 14 h before target day (`10:00 LST D-1`); `solar_offset` reads only pre-issue history. |
| 5 | Metadata shortcuts | **CLEAN** | No target-leaking identifiers; BA name enters only `solar_offset_ref` (dev-calibrated scalar). |
| 6 | Train-vs-real feature alignment | **DEFECT** (missingness & twilight) | Shared `features.py` code path, but simulated banks omit missing demand hours (`miss_frac = 0`) and real nighttime solar quirks. |
| 7 | Loss, gradient & seed hygiene | **CLEAN** | `train.py:40` seeds both `torch.manual_seed(seed)` and `np.random.default_rng(seed)`; masked Gaussian NLL (`model.py:39-42`) applies masks cleanly. |
| 8 | Rung arithmetic replay | **CLEAN** | All 8 `val_v3` rungs match the strict `>+0.01` & `dev_rel <= 1.02` rule (`8/8`); on `val_v4`, `r9a` (`0.3752`, `dev_rel 1.011`) passed arithmetically and is transparently documented as superseded by dev-calibrated `r9b` (`dev_rel 1.002`). |
| 9 | Twin-test triage | Logged | `training/checks/twin_r4_more.json` and `twin_r8_wide.json` logged and drove simulator v3 noise repair. |
| 10 | Calibration honesty | **MINOR HYGIENE** | Widen-only `sigma_scale` (`d: 1.141, w: 1.000, s: 1.369`) fitted on dev; unbounded `softplus` sigma allows `BPAT` `hi95 = 8.35` CF, and `CISO` exam `cov68` drops to `0.46` (demand) / `0.50` (solar). |

## Reproduction

Run from the repository root with NumPy and PyTorch:

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/demandcast_v1_1_probe.py
```
