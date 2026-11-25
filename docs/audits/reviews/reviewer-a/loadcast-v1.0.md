# LoadCast v1.0 — Independent First-Pass Audit (`reviewer-a`)

Independent first pass, 2026-11-25, on the released v1.0 (`r3`) weights at `61c70bfb2a27f3d592fab317ccd40a048e020b4b`.
**Status:** First pass; second review open.
**Overall Verdict:** **`CRITICAL DEFECT`** (`has_electricity` standardization blow-up triggered by the default API signature `meters_present=None` on non-electricity meters, plus outlier-poisoned `night_ratio` scaling, unclipped `T_below` spikes, an unsliced-history API guardrail gap, and unseeded PyTorch training).

## 1. Intended-Use Contract & Information Boundary

LoadCast predicts hourly building meter load (`electricity`, `chilledwater`, `steam`, `hotwater`, `gas`) and 90% prediction intervals over a target period `[start, end)` from hourly weather, calendar, building metadata, and optional prior meter history (`cold` vs. `warm` mode). At forecast time `t`, an operator knows past meter history (`< start`) and building metadata; if predicting prospectively in real time, future weather beyond the forecast horizon (especially months later in the calendar year) is not yet observed.

## 2. Critical Defect #1: `has_electricity = 0` Blow-Up (`z[18] = -1000.0`) Triggered by Default `meters_present=None`

**What failed?** Every building in the simulated training banks included an electricity meter ([`training/bank.py:21-24`](../../../../models/loadcast/training/bank.py#L21-L24) and all 52 BDG2 dev buildings), so feature `idx = 18` (`has_electricity`) was identically `1.0` in training. Adding the `1e-3` floor in [`training/train.py:70`](../../../../models/loadcast/training/train.py#L70) stored `mu_x[18] = 1.0`, `sd_x[18] = 0.001` with non-trivial first-layer column norms (`||W0[:, 18]|| = 0.9877` in `cold`, `0.8710` in `warm`). Meanwhile:

- [`loadcast.py:318`](../../../../models/loadcast/loadcast.py#L318) defaults `meters_present=None` to `[meter]`, setting `has_electricity = 0` whenever a caller requests a non-electricity meter (`chilledwater`, `steam`, `hotwater`, `gas`) without explicitly listing `'electricity'` in `meters_present` — or whenever a real building's metadata omits an electricity meter.
- [`loadcast.py:272`](../../../../models/loadcast/loadcast.py#L272) (`np_forward`) computes `h = (x - W['mu_x']) / W['sd_x']` with no input clipping, injecting `z[18] = -1000.0` directly into layer 0.

**What changed when tested?** On the two shipped non-electricity BDG2 examples (`models/loadcast/examples/bdg2/`), isolating `has_electricity = 0` alone (holding `hvac='district_all'` and all other features identical) and calling the default `meters_present=None`:

| Example (true 2017 mean) | Setting | Cold mean (CV%) | Warm 1-yr mean (CV%) |
|---|---|---:|---:|
| `Bull_education_Reina` (`chilledwater`, 937.3 kW) | Baseline (`meters_present` incl. elec) | 45.1 kW (100.5%) | 905.1 kW (22.2%) |
| `Bull_education_Reina` (`chilledwater`, 937.3 kW) | **Isolated `has_electricity=0` (same `hvac`)** | **2,255.3 kW — 50.0×** (144.8%) | **398.1 kW — 0.44×** (61.4%) |
| `Bull_education_Reina` (`chilledwater`, 937.3 kW) | Default `meters_present=None` | 2,303.8 kW (149.8%) | 401.5 kW (61.1%) |
| `Cockatoo_education_Minh` (`steam`, 201.9 kW) | Baseline (`meters_present` incl. elec) | 43.9 kW (82.0%) | 180.1 kW (21.4%) |
| `Cockatoo_education_Minh` (`steam`, 201.9 kW) | **Isolated `has_electricity=0` (same `hvac`)** | **1,071.1 kW — 24.4×** (431.9%) | **103.1 kW — 0.57×** (52.4%) |
| `Cockatoo_education_Minh` (`steam`, 201.9 kW) | Default `meters_present=None` | 1,107.1 kW (449.7%) | 108.0 kW (50.1%) |

**What consequence remains unmeasured?** All three shipped BDG2 examples and the iBLEND electricity-only exam (`training/exam2_iblend.py:304`) include `'electricity'` in `meters_present`. How many buildings in the full BDG2 exam (if any) lack an electricity meter in `meters_present`—and how often downstream users omit `meters_present` when calling `LoadCast.forecast()`—remains unmeasured.

## 3. Outlier-Poisoned `night_ratio` Scaling & Unclipped `T_below` Spikes (`warm`)

**What failed?** In [`loadcast.py:238`](../../../../models/loadcast/loadcast.py#L238), `night_ratio = nanmean(yn[night & wk]) / max(nanmean(yn[dayh & wk]), 1e-3)` can spike to `~10^3` on simulated seasonal meters whose daytime history is near zero. That inflated `warm/sd_x[77]` to **`23.3744`** (`mu = 1.337`, `||W0[:, 77]|| = 6.9775`), compressing normal real buildings (`night_ratio` in `0.84..1.19`) into `z ∈ [-0.021, -0.006]`. Conversely, `warm/sd_x[82]` (`T_below`, cold-excursion below the 2nd percentile of history temperature) is **`0.02235`** (`mu = 0.00131`) and enters `np_forward` unclipped.

**What changed when tested?** On the three shipped real BDG2 examples (warm 1-year mode):

- **Counterfactual sweep (`0.3 -> 1.5`):** Sweeping `night_ratio` across its entire typical physical range (`0.3` to `1.5`) moves warm predictions by at most **`1.33%` (Peacock)**, **`2.40%` (Bull)**, and **`1.77%` (Cockatoo)** of mean load — compared to **`26.73%`**, **`98.59%`**, and **`115.82%`** for control feature `we_ratio` (`idx = 76`, healthy `sd_x`) over the exact same `0.3 -> 1.5` sweep.
- **Unclipped `T_below` spikes:** Cold snaps in 2017 drive `T_below` to `|z| = 15.90` (Peacock), `29.77` (Bull), and `17.86` (Cockatoo). At Bull's peak excursion (hour `154`, true load `392.4 kW`), unclipped `T_below` pushes the prediction from `823.9 kW` (`|z| <= 8`) to `974.1 kW` (`+150.2 kW`), though the full-year CV impact is small (`22.16% -> 22.04%`).

**What consequence remains unmeasured?** Whether capping `night_ratio` and clipping `|z|` during retraining improves warm-mode accuracy across the full BDG2 and iBLEND exams remains unmeasured.

## 4. Operational Boundary, API Guardrail & Training Reproducibility (`MINOR HYGIENE`)

- **Unsliced `history` target leakage in `LoadCast.forecast()`:** [`loadcast.py:327-329`](../../../../models/loadcast/loadcast.py#L327-L329) indexes history over `(idx >= history.index.min()) & (idx <= history.index.max())` without truncating to `< t0`. Passing an unsliced multi-year meter series silently fits TOWT and `history_encode` on the target window `[start, end)` (artificially lowering CV on `Bull` from `22.2% -> 19.4%` and `Cockatoo` from `21.4% -> 20.0%`). Training and exam code ([`training/dataset.py:134-140`](../../../../models/loadcast/training/dataset.py#L134-L140)) cleanly isolates `year == hist_year`, so this is an inference API guardrail gap rather than an exam leak.
- **Future-weather coupling (`annual_climate`, `Tdmax`, `Tday`):** [`loadcast.py:317`](../../../../models/loadcast/loadcast.py#L317) computes `annual_climate` over the entire forecast calendar year(s) (noted in the docstring `loadcast.py:12-13`), and [`loadcast.py:111, 302`](../../../../models/loadcast/loadcast.py#L111) compute full-calendar-day `Tdmax` and `Tday`. Perturbing Aug–Dec 2017 temperatures by `+10 °C` changes the **Jan 15 12:00** forecast by **`-40.07%` (Peacock cold)**, **`+572.11%` (Bull cold)**, **`-16.83%` (Cockatoo cold)**, and `-1.08% / -1.93% / -2.93%` (warm). Perturbing July 1 `07:00–23:00` by `+10 °C` changes **July 1 `06:00`** by `-10.79%` (Bull cold) and `+1.31% / +1.60%` (Bull/Cockatoo warm). For retrospective weather-scenario simulation this is expected; for prospective forecasting it requires substituting climatology/forecast weather and breaks within-day causality.
- **Unseeded PyTorch training & capped calibration factors:** [`training/train.py:60-74`](../../../../models/loadcast/training/train.py#L60-L74) never calls `torch.manual_seed` (weight initialization and dropout `0.1` are unseeded, and checkpoint resume at lines `63-66` omits RNG state). Consistent with this, [`ratchet.yaml:22-26, 49-50`](../../../../models/loadcast/ratchet.yaml#L22-L26) documents a `0.019` objective spread across seeds vs. a `0.01` margin and notes that released R3 (`dev 49.2`) sits ~3–4 points below retrains of its own recipe (`52.0, 53.8`). In `widen_r3.json`, `warm_w2_chilledwater` and `warm_w2_gas` both hit the hard cap `30.0`, with `gas` fitted on only `n = 2` dev building-meters (`training/twin_r3.json`).

**What consequence remains unmeasured?** Whether seeding PyTorch and evaluating multi-seed ensembles changes the v1.0/round-2 keep/reject ordering, and how `warm_w2` intervals behave on unseen gas/chilled-water buildings, remain unmeasured.

## 5. Clean Checks & 10-Point Summary

| # | Protocol Check | Result | Measured Evidence |
|---|---|---|---|
| 1 | Intended-use boundary | **MINOR HYGIENE** | Full-year `annual_climate` is documented; same-day `Tdmax`/`Tday` and unsliced `history` in `loadcast.py:328` are unguarded. |
| 2 | Weights, norm & sensitivity | **CRITICAL DEFECT** | Zero NaN/Inf, zero dead rows; `has_electricity` `sd = 0.001` causes `24–50×` cold / `43–56%` warm blow-up when `0`; `night_ratio` `sd = 23.37` attenuates sensitivity to `1.3–2.4%` vs `27–116%` control; `T_below` hits `|z| = 29.8`. |
| 3 | Torch ↔ NumPy parity | **CLEAN** | Exact `lcnet.MLP` reconstruction matches `loadcast.np_forward` (`cold`: `mu=2.24e-6, ls=8.79e-7`; `warm`: `mu=2.57e-6, ls=1.17e-6`). |
| 4 | Causal / cold-start leakage | **MINOR HYGIENE** | Exam splits cleanly slice `2016 -> 2017`; `LoadCast.forecast()` accepts overlapping `history` and couples `06:00` to same-day afternoon `T_out`. |
| 5 | Metadata shortcuts | Not applicable | No target-derived metadata in inputs; cold mode legitimately predicts from building metadata + weather. |
| 6 | Train-vs-real feature alignment | **DEFECT** (`has_electricity`) | Feature formulas match between `lc_common.py` and `loadcast.py`; training banks never generated `has_electricity = 0`. |
| 7 | Loss, gradient & seed hygiene | **MINOR HYGIENE** | No `torch.manual_seed` anywhere in `models/loadcast/`; resume omits RNG state. |
| 8 | Rung arithmetic replay | **CLEAN** | All 6 v1 rungs (`ratchet.json`) and all 3 round-2 3-seed rungs (`r2_ladder.json`) match the stated `> +0.01` & `<= +2%` dev gate rule (`6/6` and `3/3`). |
| 9 | Twin-test triage | Logged | `twin_r3.json` and `optimism_r3.json` are present and candidly document seed variance and cold-mode `1/ybar^2` bias. |
| 10 | Calibration honesty | **MINOR HYGIENE** | All 20 `widen` factors are strictly `>= 1.0`, but `warm_w2_chilledwater` and `warm_w2_gas` hit the `30.0` cap (`gas` `n = 2` dev units). |

## Reproduction

Run from the repository root with NumPy, Pandas, SciPy, and PyTorch:

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/loadcast_v1_0_probe.py
```
