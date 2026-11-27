# StarQuake v1.1 | Amplitude-driven white-noise false detections and quality-flagged cadence padding

**Cross-checked broadband-amplitude false-detection mechanism and cadence-boundary sensitivity.** Reviewer A independently reran reviewer B's `69873b7` probes against model base `61c70bf`, ran a pure Gaussian white-noise amplitude sweep on the exact Kepler cadence mask to test the role of amplitude in the `18/30` time-scrambled detections, and reproduced the boundary-padding mass shift through the `pipeline.load_kepler()` entry point using quality-flagged (`q = 16`) boundary rows. [Original report and probe](reviews/reviewer-b/starquake-v1.1.md) · [Released reader](../../models/starquake/starquake.py) · [Spectral preprocessing](../../models/starquake/pipeline.py)

**What failed and reproduced.**
- **Time-scrambled detections (`18/30`):** Permuting finite flux values across timestamps on 90-day Kepler excerpts of three giants still triggered oscillation detections in **`10/10`** (`KIC 2695975`, `p_osc >= 0.943`), **`7/10`** (`KIC 7353997`), and **`1/10`** (`KIC 7618730`) runs. Constant `0.0` and all-`NaN` inputs cleanly returned `detected = False` (`p_osc < 1e-7`).
- **Empty cadence padding shifts inferred mass:** Appending or prepending 90 days of `NaN`s left observing span `days` fixed (`89.99 d`) while halving `duty` and changing FFT grid resolution (`n`), shifting `KIC 2695975` `Δν` (`1.625 → 1.576 µHz`) and hybrid stellar mass (**`1.733 → 1.906 M_sun`**, `+10.0%` at `Teff = 4,800 K`).
- **Controls:** `0/1,361` near-zero kernel rows; folded PyTorch↔NumPy parity **`2.05e-6`** (real) / **`1.04e-6`** (random); all `5/5` logged network promotions matched; widen factors `1.00` and `1.71`.

**Second-pass white-noise amplitude sweep & `load_kepler` quality-flagged boundary test.**
- **A fixed Gaussian noise draw triggers as amplitude increases:** Feeding pure unstructured Gaussian white noise (`seed = 42`, zero oscillation comb) on `KIC 2695975`'s exact 90-day cadence mask (`3,986 / 4,404` finite cadences, `Teff = 5,800 K`) transitions sharply from `p_osc = 0.00001` at `200 ppm` to **`p_osc = 0.89319` (`detected = True`) at `500 ppm`**, in the range of the three giants' real RMS flux (`KIC 2695975 = 1,063 ppm → 10/10`, `KIC 7353997 = 480 ppm → 7/10`, `KIC 7618730 = 362 ppm → 1/10`):

| Pure white-noise RMS (`ppm`) on `KIC 2695975` mask | `detected` | `p_osc` | Inferred `νmax` / `Δν` (`µHz`) |
|---:|---|---:|---:|
| `10` / `50` / `100` | `False` | `0.00000` | `64.6–128.0` / `7.44–9.86` |
| `200` | `False` | `0.00001` | `187.11` / `11.591` |
| **`500`** | **`True`** | **`0.89319`** | `197.35` / `12.330` |
| **`1,000`** | **`True`** | **`0.98724`** | `163.42` / `11.519` |
| **`3,000`** | **`True`** | **`0.94740`** | `112.74` / `9.617` |

**Closing control across random draws.** On the same cadence mask at 5,800 K, reviewer B centred each of ten Gaussian draws and normalized its measured RMS to the target. The detector triggered on 0/10 draws at 200 ppm, **5/10 at 500 ppm** (probability 0.00017–0.99970), and 10/10 at 1,000 ppm. Amplitude can drive detections on unstructured inputs. The spread at identical RMS rules out RMS alone as a sufficient explanation for individual decisions. The real-sky false-alarm rate remains open. [Fixed-RMS probe](reviews/reviewer-b/starquake_v1_1_probe.py)

- **Full-record `load_kepler` entry point with quality-flagged (`q = 16`) boundary rows:** Appending 90 days of nonzero quality-flag (`q = 16`) cadences to full 4-year `2695975.npz` expanded `load_kepler()`'s output array from **`71,964 → 76,368`** while keeping `days = 1,470.48` unchanged, shifting **`Δν` from `1.5766 → 1.5562 µHz`** and inferred mass from **`1.677 → 1.777 M_sun` (`+6.0%`)** on identical valid flux measurements.

**What remains open.** Measuring real-sky false-alarm rates on high-RMS non-oscillating stars (spotted dwarfs, eclipsing/rotational variables above `~400 ppm`) and trimming leading/trailing quality-flagged cadences across the archived Kepler exam set remain open.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/starquake_v1_1_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/wave7_export_probe.py starquake
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_69873b7_probe.py starquake
```

Uses NumPy and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_69873b7_probe.py)

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| Broadband noise triggers detections at real-giant RMS | Yes | **Documented, queued**: high-RMS non-oscillating stars in the noise bank. No inference guard can tell such noise from oscillations without that training. | Card limits and next. |
| Flagged or empty cadences at the ends change duty and FFT grid | Yes | **Warned, opt-in fix, re-score queued**: `grid_lc` / `read_npz` / `load_kepler` take `trim=True`; by default they warn. | Counted 27 Nov 2026: 36 of the 858 archived light curves (exam, dev and noise) start or end on a gap (25 leading, 11 trailing). |
| Follow-up (27 Nov 2026): high-RMS stars in the noise bank | Run | **Rung r6, rejected.** Measured first: the released model calls 16.9% of held-out high-RMS dwarfs (the ones the noise screen dropped) oscillating, against 6.1% of quiet ones. r6 adds 5,000 examples on such stars and retrains from scratch: false alarms 11.1%, but val3 +0.007 (needs +0.01) and dev errors +22% (gate +10%). v1.1's network stays; the exam was not opened. | `training/r6_judgement.json`; NOTEBOOK 27 Nov 2026. |
| Follow-up (27 Nov 2026): re-score with trimmed ends | Run | **Dated re-score, then default (v1.1.1).** 22 of 420 exam stars have edge gaps; 92 of 2,098 readings change. Whole light curves: νmax 1.01 → 0.99%, Δν 0.65 → 0.62%; 27 d Δν 4.29 → 4.37%; Gaia radius scatter 8.04 → 8.12%. No verdict changes. `grid_lc` / `read_npz` / `load_kepler` now trim by default. | `training/exam_trim_result.json`; NOTEBOOK 27 Nov 2026. |
