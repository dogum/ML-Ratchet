# BeatNet v1.0 | Shared MIT-BIH dev/exam subject (201/202) and single-NaN silent 'N' outputs

**Cross-checked inter-patient boundary crossing and public-API NaN defect.** Reviewer A independently reran reviewer B's `69873b7` probes against model base `61c70bf`, verified the MIT-BIH 201/202 subject overlap across dev calibration, background extraction, and DS2 exam lists, and recomputed exact DS2 confusion matrices and F1 metrics from `training/exam.json['ds2']['per_record']` with record `202` held out. [Original report and probe](reviews/reviewer-b/beatnet-v1.0.md) · [Dev IDs](../../models/beatnet/training/calib_v1.json) · [Background extractor](../../models/beatnet/training/quietnoise.py) · [Public classifier](../../models/beatnet/beatnet.py)

**What failed and reproduced.**
- **Shared human subject crossing dev/background and DS2 exam (`201` / `202`):** Although the 6 dev IDs (`[101, 112, 119, 201, 203, 220]`) and 22 DS2 exam IDs have `0` direct record-number overlap, PhysioNet documents that MIT-BIH records `201` and `202` come from the **same patient** (the sole duplicate subject among the 48 MIT-BIH records). Record `201` appears in both dev calibration (`calib_v1.json`) and the training background-noise pool (`quietnoise.py`), while record `202` appears in DS2 and the packaged examples.
- **Single-NaN → silent `'N'` (Normal) defect:** Setting one measured ECG sample to `NaN` at the midpoint of each of the 4 packaged strips propagated through full-strip filtering into **326/326** nonfinite probability vectors, yet every beat returned label **`'N'`** (including **35** beats labelled `S`, `V`, or `F`). Linearly interpolating that single sample restored **326/326** labels; a constant trace raised `ValueError` in automatic mode.
- **Controls:** `0/751` near-zero kernel rows; folded PyTorch↔NumPy parity **`3.34e-6`** (real) / **`7.63e-6`** (random); all `5/5` network rungs matched; QRS widen factor `5.06×`.

**Second-pass DS2 sensitivity check (excluding record `202` from archived `exam.json['ds2']['per_record']`).**
Recomputing the pooled DS2 confusion matrix with record `202` excluded (`2,136` beats: `2,061 N`, `55 S`, `19 V`, `1 F`, on which BeatNet had `216` `N→V` false positives and `4/55` `S` recall) **improves** every headline metric on the remaining 21 DS2 records:

| DS2 evaluation subset | Records / beats | Accuracy | Macro-F1 | SV-F1 (`mean(S, V)`) | `S` F1 | `V` F1 |
|---|---:|---:|---:|---:|---:|---:|
| Published DS2 (all) | `22` / `49,705` | `0.89709` | `0.54655` | `0.61760` | `0.5778` | `0.6574` |
| DS2 excluding record `202` | `21` / `47,569` | **`0.89823`** | **`0.55306`** | **`0.63050`** | **`0.5873`** | **`0.6737`** |
| Delta (`excl 202 − all`) | `−1` / `−2,136` | `+0.00114` | **`+0.00652`** | **`+0.01290`** | `+0.0095` | `+0.0163` |

This scoring sensitivity check shows record `202` did not inflate the archived DS2 headline numbers; however, it does not replace a full subject-grouped retrain and recalibration with record `201` removed from `quiet_dev.npz` and `calib_v1.json`.

**What remains open.** Retraining and recalibrating BeatNet with subject `201/202` wholly excluded from dev calibration and training backgrounds, plus measuring NaN prevalence in live Holter feeds, remain open.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/beatnet_v1_0_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/wave7_export_probe.py beatnet
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_69873b7_probe.py beatnet
```

Uses NumPy, SciPy, and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_69873b7_probe.py)
