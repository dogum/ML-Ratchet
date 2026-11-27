# BuildingDetective v1.0 — Independent First-Pass Audit (`reviewer-a`)

Independent first pass, 2026-11-25, on the released v1.0 weights at `61c70bfb2a27f3d592fab317ccd40a048e020b4b`.
**Status:** First pass; second review open.
**Overall Verdict:** **`CRITICAL DEFECT`** (two verified unclipped standardization blow-ups in `hourly.npz` — one breaking partial-year records and one distorting full-year real BDG2 inputs — plus minor calibration/reproducibility hygiene items).

## 1. Intended-Use Contract & Information Boundary

BuildingDetective infers 12 physical building parameters (floor area, year built, envelope UA, thermal time constant, heating/cooling balance points and sensitivities, base share, operating hours, weekend ratio) and 3 categorical classes (use, heating fuel, cooling type) from an existing hourly, monthly, or annual energy meter record plus co-located weather. Because it is a retrospective inverse-physics reader over a completed meter record, there is no real-time causal cutoff within the input series; the operational boundary is that real inputs may be partial-year or contain zero-reading periods, and real exam buildings must be disjoint from dev buildings used for calibration.

## 2. Critical Defect #1: Partial-Year Input Blow-Up in `hourly.npz`

**What failed?** In [`training/featurize.py:76`](../../../../models/buildingdetective/training/featurize.py#L76), hourly quirk augmentation calls `mq = F.quirk_hourly(met, rng, n)[0]`, discarding the second return value (`keep_mask` for partial-year records) and passing the full 365–366 day `T, G, loc` arrays to `F.feat_hourly(mq, T, G, loc, w['lat'])`. Every simulated training building therefore had `nd = 365` or `366` days, collapsing the training standard deviation of the two record-length features in `weights/hourly.npz` to pure leap-year noise:

- `col 6` (`span = nd / 365`): `xm = 1.000525`, `xs = 1.1036e-3`, `||W0[:, 6]|| = 1.8576`
- `col 7` (`log10(nd) / 3`): `xm = 0.854233`, `xs = 1.6854e-4`, `||W0[:, 7]|| = 2.1289`

Inference in [`buildingdetective.py:45`](../../../../models/buildingdetective/buildingdetective.py#L45) (`Net.raw`) computes `h = (X - self.xm) / self.xs` with no input clipping.

**What changed when tested?** Truncating the shipped real residential example `HUE_2` (`models/buildingdetective/examples/examples.npz`, `h0_*`) from a full year to shorter real-world meter spans drives `z[6]` and `z[7]` to hundreds of standard deviations and corrupts point estimates, classes, and error bars:

| Meter span | `z[6]` (`span`) | `z[7]` (`log10(nd)/3`) | Predicted floor area (68% CI) | Predicted year built | Top use class |
|---|---:|---:|---|---:|---|
| 365 d (full) | +2.0 | +1.5 | 99.5 m² `[39.6, 250.0]` | 1984.8 | `residential_single` (0.29) |
| 330 d (-35 d) | -84.9 | -84.8 | 295.6 m² `[165.2, 528.9]` | 1975.2 | `healthcare` (0.29) |
| 180 d | -457.2 | -603.2 | 58.6 m² `[57.9, 59.4]` | 1827.5 | `healthcare` (0.47) |
| 90 d | -680.7 | -1193.9 | 10.0 m² `[9.9, 10.1]` | 1776.1 | `other` (0.59) |

Shortening a record by just 35 days triples predicted floor area (`99.5 -> 295.6 m²`) and flips `use` from `residential_single` to `healthcare`; at 180 days the 68% interval collapses to `±0.8 m²` around an 1828 vintage.

**What consequence remains unmeasured?** The shipped BDG2 exam examples are full 365-day records, so this defect does not alter full-year benchmark rows; its impact across partial-year real deployments (or whether fixing `keep_mask` and retraining changes full-year exam metrics) remains unmeasured.

## 3. Critical Defect #2: Unclipped Rare-Channel Zero-Fraction `z`-Scores on Full-Year BDG2 Buildings

**What failed?** In `weights/hourly.npz`, rare-channel zero-fraction features have near-zero training standard deviations (`col 904` `dheat` zero-fraction: `xm = 2.41e-4, xs = 2.268e-3, ||W0[:, 904]|| = 0.983`; `col 1197` `dcool` zero-fraction: `xm = 2.01e-4, xs = 1.940e-3, ||W0[:, 1197]|| = 0.888`) and enter [`Net.raw`](../../../../models/buildingdetective/buildingdetective.py#L45) unclipped. Real district-heated/cooled BDG2 buildings routinely have extended seasonal zero-flow periods, producing massive normalized inputs even on full 365-day records.

**What changed when tested?** Passing the three shipped full-year BDG2 exam buildings (`models/buildingdetective/examples/bdg2/bdg2_hourly.npz`) through `hourly.npz` unclipped vs. clipping normalized inputs to `|z| <= 8`:

| Shipped BDG2 building (365 d) | True area | Max `\|z\|` (col, `xs`) | Unclipped area (err) | Clipped `\|z\| <= 8` area (err) | Shift |
|---|---:|---|---:|---:|---:|
| `b0` `Lamb_education_Bert` | 2,403.0 m² | 25.77 (`col 399`, `7.54e-1`) | 1,691.8 m² (-29.6%) | 1,181.5 m² (-50.8%) | -30.2% |
| `b1` `Eagle_office_Remedios` | 4,781.8 m² | 32.17 (`col 1197`, `1.94e-3`) | 3,445.6 m² (-27.9%) | 4,988.8 m² (+4.3%) | +44.8% |
| `b2` `Peacock_lodging_Sergio` | 10,498.0 m² | 117.74 (`col 904`, `2.27e-3`) | 2,576.2 m² (-75.5%) | 5,141.3 m² (-51.0%) | +99.6% |

**What consequence remains unmeasured?** Only the three shipped BDG2 hourly examples were tested. The aggregate effect across the full BDG2 exam set—and whether retraining with input clipping or realistic seasonal district-meter shutoffs improves full-exam floor-area accuracy—remains unmeasured.

## 4. Minor Hygiene: Calibration Cap & Training Reproducibility

- **Softmax temperature ceiling & uncalibrated cooling head:** In [`training/calibrate.py:30`](../../../../models/buildingdetective/training/calibrate.py#L30), `nll_temp` searches `Ts = np.arange(1.0, 10.01, 0.05)`. Both `hourly.npz` and `monthly.npz` hit the hard cap `temp[0] = 10.0` for `use` (`annual.npz` reaches `7.4`), indicating severe real-data classifier overconfidence beyond the search grid. `temp[2]` (cooling class) is hardcoded at `1.0` across all three modes (`hourly: [10.0, 1.0, 1.0]`, `monthly: [10.0, 1.0, 1.0]`, `annual: [7.4, 1.9, 1.0]`). Regression `widen` factors are cleanly widen-only (`>= 1.0`, max `3.015` on `hourly`).
- **Training/featurization reproducibility:** [`training/featurize.py:70`](../../../../models/buildingdetective/training/featurize.py#L70) seeds training-bank quirks (`sid < 900`) with `int(time.time() * 1e3) % 7919` (val shards `sid >= 900` are deterministic). [`training/train.py:124-129`](../../../../models/buildingdetective/training/train.py#L124-L129) restores `net`, `opt`, and `sched` on checkpoint resume without saving or restoring PyTorch RNG / `torch.Generator` state.

**What consequence remains unmeasured?** Whether extending `Ts` above `10.0` materially improves dev/exam `use` log-loss or whether resume RNG drift affects rung selection margins remains unmeasured.

## 5. Clean Checks & 10-Point Summary

| # | Protocol Check | Result | Measured Evidence |
|---|---|---|---|
| 1 | Intended-use boundary | Pass (retrospective) | Whole-record inverse reader; partial-year contract violated by Defect #1. |
| 2 | Weights, norm & sensitivity | **CRITICAL DEFECT** | Zero NaN/Inf, zero dead rows across all 3 `.npz` files; `hourly.npz` `xs` collapses to `1.685e-4` (`col 7`), `1.104e-3` (`col 6`), `1.94e-3` (`col 1197`), `2.27e-3` (`col 904`). |
| 3 | Torch ↔ NumPy parity | **CLEAN** | Exact `Net` reconstruction matches `buildingdetective.py` (`hourly`: `mu=1.88e-5, sg=2.07e-5, lg=7.40e-6`; `monthly`: `<= 1.74e-5`; `annual`: `<= 1.28e-5`). |
| 4 | Causal / cold-start leakage | N/A (retrospective) | Whole-record input; partial-year boundary tested under Check 2. |
| 5 | Metadata shortcuts | Not re-tested | Inputs are meter + weather + latitude only (no building IDs/timestamps leaked into targets). |
| 6 | Train-vs-real feature alignment | **CLEAN** (code) / **DEFECT** (quirks) | Release vs training `feat_hourly` diff on `b0` is `0.000e+00`; quirk mask drop in `featurize.py:76` causes train-vs-real span mismatch. |
| 7 | Loss, gradient & seed hygiene | **MINOR HYGIENE** | `train.py:56-63` masks NaN regression targets properly and seeds Torch (`train.py:117`), but `featurize.py:70` uses wall-clock time and `train.py:124-129` omits RNG state on resume. |
| 8 | Rung arithmetic replay | **CLEAN** | 29/30 entries in `training/ratchet.json` (24 candidate rungs + 6 rebaselines) match the strict `>+0.01` val skill & `dev >= inc - 0.02` rule; the 1 exception (`R3c-hourly`, `+0.0112`) is explicitly logged as withdrawn for exceeding the 500k parameter cap (1.15M params). |
| 9 | Twin-test triage | Logged | Twin tests documented in `MODEL_CARD.md` / `ratchet.json` drove W3 base-load physics v2. |
| 10 | Calibration honesty | **MINOR HYGIENE** | Regression `widen >= 1.0` strictly widen-only; `temp[0]` hits `10.0` cap on `hourly`/`monthly` and `temp[2]` is uncalibrated (`1.0`). |

## Reproduction

Run from the repository root with NumPy, Pandas, SciPy, and PyTorch:

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/buildingdetective_v1_0_probe.py
```
