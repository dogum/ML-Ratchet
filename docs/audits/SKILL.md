---
name: ml-audit
description: Forensic ML and physics honesty audit for ML Ratchet models. Inspects shipped weights, normalization stats, PyTorch-to-NumPy/TS export parity, train-vs-real feature alignment, loss/gradient integrity, calibration, and twin-test triage to surface silent defects and undocumented limits without assuming intent.
---

# ML & Physics Honesty Audit (`ml-audit`)

Use this skill to perform a **read-only forensic audit** of one or more models in `models/<id>/` (or an experiment in `experiments/<id>/`).

## Operating Principles

1. **Audit mode is read-only first.** Never modify model code, weights, or splits while auditing. Document every finding with exact file paths, line numbers, measured numbers, and reproducible verification commands.
2. **Test falsifiable symptoms, never speculate on intent.** Whether a model is degraded by an accidental engineering bug, an undocumented shortcut, or silent hobbling, the evidence lives in the weights, training code, feature pipeline, and logs. Separate every finding into one of four categories:
   - **`CRITICAL DEFECT`**: Verified bug in code, normalization, export, loss, or preprocessing that materially degrades inference or training while passing existing tests.
   - **`MINOR HYGIENE`**: Real code discrepancy or unmasked edge case with small or bounded impact.
   - **`UNDOCUMENTED LIMIT`**: Deliberate or resource-driven design choice (small capacity, single seed, short step budget, last-step checkpointing) that caps performance and should be stated plainly rather than mistaken for a bug.
   - **`CLEAN`**: Verified healthy with measured numbers.
3. **Never trust surface tests alone.** Passing `mlratchet validate` and `site/tests/parity.test.ts` (NumPy vs. TypeScript) does **not** prove that PyTorch training matches exported NumPy inference, that input normalization is unpoisoned, or that the training loop converged properly.

---

## The 7-Point Forensic Audit Checklist

Run all seven checks for each target model (`models/<id>/`):

### 1. Weight & Normalization Health (Shipped `.npz` Files)
Load every `.npz` weight file shipped in `models/<id>/` (and `weights/` if present) and verify:
- **Finiteness & dead units:** Zero `NaN`/`Inf` values; check row norms of every 2-D weight matrix for dead/collapsed rows (`norm < 1e-4`), collapsed biases, or saturated normalization layers.
- **Input normalization poison check (high priority):** Inspect all stored input scaling arrays (`x_mu`, `x_sd`, `st_mu`, `st_sd`, `mean`, `std`). Flag any channel with extreme `sd` (e.g., `sd > 1e2` or `sd <= 1.001e-3`).
- **Real-input post-normalization spread:** Pass real dev or shipped example inputs through the exact normalization formula `(X - mu) / sd` (with any clipping). Flag any non-constant physical feature whose post-normalization `std` collapses (`< 1e-2`), saturates at a clip boundary, or explodes (`> 10`).
  - *Precedent:* Audit #1 (`EnergyNet v2.1`) caught four core physics work-fraction features (`aero/Ep`, `rolling/Ep`, `grade_dn/Ep`, `inertia_pos/Ep`) crushed to `std = 4e-6 .. 4.5e-5` on real VED trips because `Ep = max(ph[0], 1e-6)` exploded on standstill/downhill bank rows before raw `X0.mean(0)` / `X0.std(0)`.
- **Ensemble diversity:** For multi-member ensembles, compute pairwise cosine similarity of first-layer weights across members (and prediction spread on real inputs) to confirm seeds are genuinely independent and not duplicated or collapsed.

### 2. Training-Framework ↔ Inference-Runtime Forward Parity
`site/tests/parity.test.ts` checks NumPy ↔ TypeScript, not PyTorch ↔ NumPy.
- Reconstruct the exact PyTorch `nn.Module` from `training/` using the shipped `.npz` weights, set `model.eval()`, and compare forward outputs against the shipped NumPy module (`models/<id>/<id>.py`) on random non-trivial inputs and shipped examples.
- Verify:
  - `max_abs_diff` `< 1e-5` (float32) or explain any documented precision gap.
  - Activation exactness (e.g., `GELU(approximate='tanh')` vs exact `erf`), BatchNorm/LayerNorm `eps` and `running_mean`/`running_var` export in `eval()` mode, Conv padding modes, and output clamps.

### 3. Train-vs-Real Feature, Unit & Split Alignment
Read the bank builder (`simbank.py` / `bank.py`), dataset loader (`dataset.py`), real featurizer (`realfeat.py` / `evalreal.py`), and release module (`<id>.py`) side by side:
- Confirm identical feature ordering, units, scaling constants, temporal alignment (DOY, warm-up length, lag sign, off-by-one shifts), and `NaN` fallback/interpolation behavior.
- Confirm frozen dev/exam splits cannot leak into training banks, scaling statistics, or donor tables.

### 4. Loss, Gradient & Optimization Integrity
Inspect `train.py` and the training configs for the released rung:
- Verify loss formulas (signs, factors of `0.5`, variance/log-sigma parameterization, quantile monotonicity, pinball signs).
- Confirm invalid-target masks (`okm` / `mask`) are applied consistently to **both** primary and auxiliary loss terms.
- Check gradient flow (no accidental `.detach()`, wrong `no_grad()`, or frozen parameters that should train).
- Check optimizer, weight decay, gradient clipping, LR schedule completion, and checkpoint selection (`best-on-sim-val` vs `last-step`).
- Check whether training logs / ratchet rungs show clear convergence or early termination while validation loss was still falling.

### 5. Twin-Test Attribution & Bottleneck Triage
Inspect `ratchet.yaml`, `MODEL_CARD.md`, `NOTEBOOK.md`, and any `twin_*.json` logs:
- Does the released model perform cleanly on **its own simulated twins**?
- If the model fails on its own simulated twins (not just real data), treat that as a **learner, loss, or pipeline defect** and trace why.
- Check whether rejected rungs or single-seed wins masked noise inside the ratchet margin.

### 6. Uncertainty Calibration & Interval Honesty
Inspect calibration scripts (`calib*.py`, `sigma*.py`) and shipped widening factors:
- Verify widen-only compliance (`k >= 1.0`) unless explicitly documented as a simulation-side pre-scaling factor.
- Check whether a single pooled factor hides severe under-coverage in specific physical strata (e.g., cold weather, low-Re high-alpha, rare powertrains, short observing lengths).
- Check whether calibration factors were fitted strictly on dev (never exam) and with enough units per stratum.

### 7. Reproducibility & Audit Record
Write or update `docs/audits/<model>-v<ver>.md` and add a row to `docs/audits/README.md` containing:
- Exact commit/date audited and files inspected.
- Summary verdict (`CLEAN`, `MINOR HYGIENE`, `CRITICAL DEFECT`) + table of all 7 checks.
- Exact measured numbers (weight stats, Torch-vs-NumPy parity, normalized real-input spread, ensemble cosine similarity).
- Clear separation of **verified defects** vs. **documented/undocumented design limits** vs. **recommended follow-up rungs** (without applying fixes during an audit-only pass).

---

## Quick Verification Probe Pattern

When auditing a model, run a short inline Python probe (no repo edits) that:
1. Iterates over every `.npz` file in `models/<id>/` and checks finiteness, 2-D row norms, and `min`/`max` of every normalization vector (`mu`/`sd`).
2. Loads the shipped `.npz` weights back into the PyTorch `nn.Module` from `models/<id>/training/` and computes `max |torch_out - numpy_out|`.
3. Passes real dev or `examples/` inputs through the exact input transform and prints `(raw p5..p95, mu, sd, normalized p5..p95, normalized std)` for the most extreme channels.
4. For ensembles, prints pairwise cosine similarity of first-layer weights across members.
