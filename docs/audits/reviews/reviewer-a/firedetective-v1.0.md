# FireDetective v1.0 (`R6`) | `wind_hidden` 1,000× normalization spike, empty-input `exp` overflow, and 5 dead rows

**First pass (`reviewer-a`). Verdict: CRITICAL DEFECT (normalization & empty-input overflow) & MINOR HYGIENE.** Audited against model base `61c70bf` across `weights/firedetective_v1.npz`, `weights/calib.json`, [`firedetective.py`](../../../../models/firedetective/firedetective.py), [`feats.py`](../../../../models/firedetective/feats.py), all 4 shipped exam fires, [`training/train.py`](../../../../models/firedetective/training/train.py), and [`training/ratchet.json`](../../../../models/firedetective/training/ratchet.json) following [`SKILL.md`](../../SKILL.md).

## 1. Intended-Use Contract & Information Boundary
- **Intended use:** Given up to 12 half-day FEDS burned-area masks, VIIRS FRP detections, LANDFIRE fuels/topography, and gridMET weather, infer wildfire drivers including MTBS high-severity fraction (`hs`), blind 10 m wind (`wind`, inferred with `wind_hidden = True`), live fuel moisture (`lfmc`), ROS multiplier, and spotting/IA probabilities.
- **Information boundary:** Trained exclusively on ~4,100 FireSim simulations (`v3` bank); `40` dev fires calibrate FEDS emulation, `hs_map`, and widen-only uncertainty factors (`calib.json`). All 4 shipped examples are exam fires.

## 2. Findings

### Finding A — `norm_stats()` computes `seq_sd[24] = 0.001` on unhidden `wind_hidden = 0`, injecting a `+1,000` input spike (`+135..+193` pre-ReLU) on every blind-wind call [`CRITICAL DEFECT`]
- **What failed?** In [`training/train.py:39-46,127,132-133`](../../../../models/firedetective/training/train.py#L39), `norm_stats(S, St)` computes `seq_sd = v.std(0) + 1e-3` strictly on `S`, where `seq[24]` (`wind_hidden`) is identically `0.0`, producing **`seq_sd[24] = 0.001`**. Every blind-wind forward pass (`Sh`, where `seq[24] = 1.0`, used both during `--hidden-p 0.5` training and on every blind-wind inference call) therefore divides `1.0 / 0.001 = +1,000.0`.
- **What changed when tested?** Across all 4 shipped exam fires, `Z_s` (wind shown) spans `[-3.941, 8.279]`, whereas `Z_sh` (wind hidden) spikes to **`+1,000.0`** on channel 24 (`||m{0..2}.emb.weight[:, 24]||_2 = 0.347–0.491`), driving embedding pre-ReLU activations from **`[-5.69, +4.35]`** (wind shown) to **`[-49.06, +193.43]`** (wind hidden)—a **`~40×`** DC offset dwarfing all 23 physical spread/FRP/weather channels inside the first ReLU layer. (Note `seq[9] = elong` was explicitly masked via `--drop-feats elong`, setting `seq_sd[9] = 1e9`.)
- **What consequence remains unmeasured?** Because training saw the same `+1,000` spike, the model did not crash, but whether fixing `seq_sd[24] = 1.0` recovers blind-wind and blind-driver skill (`R6` `wind_hidden` sim `R² = 0.307`, `dev_rho.wind = 0.108`, `lfmc_hidden R² = 0.034`) remains unmeasured without retraining.

### Finding B — Empty / all-zero input overflows `exp()` to `wind = inf` and `lfmc = 425,555%`; all-NaN returns `NaN` silently [`CRITICAL DEFECT` / `MINOR HYGIENE`]
- **What failed?** [`firedetective.py:74,86`](../../../../models/firedetective/firedetective.py#L74) performs no check that at least one valid step (`mask.sum() > 0`) or finite feature exists.
- **What changed when tested?** Passing all-zero `(S, Sh, ST)` triggered `RuntimeWarning: overflow encountered in exp` and silently returned **`wind = inf`**, **`lfmc = 425,555.4%`**, `hs = 0.0118 ± 0.0000`. Passing all-`NaN` silently returned `NaN` across all outputs.
- **What consequence remains unmeasured?** Exam fires have `>= 1` burned mask step, so this impacts unvalidated public API calls.

### Finding C — Five dead hidden rows (`norm = 1.35e-20 .. 2.66e-7`) and unpreserved resume RNG state [`MINOR HYGIENE`]
- **What failed?** Five hidden rows across the 3-member MLP head have near-zero norm (`m0.f2[40]=1.35e-20`, `m0.f2[106]=2.80e-13`, `m1.f2[12]=1.27e-10`, `m2.f1[71]=1.15e-8`, `m2.f2[54]=2.66e-7`), and [`training/train.py:140-142`](../../../../models/firedetective/training/train.py#L140) resumes without saving/restoring PyTorch or NumPy RNG state.

## 3. Clean Checks (Items 2, 3, 8, 10)
- **Ensemble diversity & calibration (CLEAN):** All 3 members have genuinely independent `emb` (`[-0.0684, 0.0311]`) and `st` (`[-0.0056, 0.0205]`) cosines; all widen factors are strictly `>= 1.0` (`hs = 2.535`, `wind = 1.983`, `lfmc = 1.091`).
- **3-member PyTorch ↔ NumPy export parity (CLEAN):** Reconstructing `training/model.py:Net` across all 3 members on both `S` and `Sh` matched `_raw1` to **`mu = 4.74e-7`**, **`sd = 2.38e-7`**, **`p = 5.43e-8`**.
- **Ratchet arithmetic replay (CLEAN):** All **`7/7`** logged rungs (`R0..R6`) in `training/ratchet.json` match the documented promotion rule (`sim > inc + 0.01`, `dev >= inc - 0.05`).

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/firedetective_v1_0_probe.py
```

Model snapshot `61c70bf`, audit checkout `69873b7`. Uses NumPy, SciPy, and CPU PyTorch. [Reproduction probe](firedetective_v1_0_probe.py)
