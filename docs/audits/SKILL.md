---
name: ml-audit
description: Forensic ML and physics honesty audit for ML Ratchet models. Inspects intended-use information boundaries, shipped weights, normalization stats, counterfactual feature sensitivity, PyTorch-to-NumPy/TS export parity, causal/look-ahead leakage, metadata shortcuts, seed reproducibility, rung arithmetic, calibration, and twin-test triage without assuming intent.
---

# ML & Physics Honesty Audit (`ml-audit`)

Use this skill to perform a **read-only forensic audit** of one or more models in `models/<id>/` (or an experiment in `experiments/<id>/`).

## Operating Principles

1. **Audit mode is read-only first.** Never modify model code, weights, or splits during an audit pass. Record exact commit hashes, file paths, line numbers, measured numbers, and reproducible verification commands.
2. **Test falsifiable symptoms, never speculate on intent.** Accidental bugs, undocumented shortcuts, and silent degradation all leave objective fingerprints in code, weights, and replay behavior. Classify every finding strictly by evidence:
   - **`CRITICAL DEFECT`**: Verified bug in normalization, export, causality/look-ahead, loss, or preprocessing that materially affects training, inference, or what the exam measures while passing existing tests.
   - **`MINOR HYGIENE`**: Real code discrepancy, API guardrail gap, or reproducibility defect with bounded or conditional impact.
   - **`UNDOCUMENTED LIMIT`**: Deliberate or resource-driven design choice (small capacity, single seed, short step budget, last-step checkpointing) that caps performance and should be stated plainly rather than mistaken for a bug.
   - **`CLEAN`**: Verified healthy with measured numbers.
3. **Every finding must answer three questions:**
   - **What failed?** Exact `file:line`, mechanism, and reproduction probe.
   - **What changed when we tested it?** Measured output shift, perturbation delta, ablation, or replay impact.
   - **What consequence remains unmeasured?** Honest scope boundary (e.g., whether a repair rung improves the full exam, or whether a historical keep/reject decision would flip).
4. **Never trust surface tests alone.** Passing `mlratchet validate` and `site/tests/parity.test.ts` (NumPy vs. TypeScript) does **not** verify PyTorch-vs-NumPy export parity, unpoisoned input scaling, operational causality, or seed reproducibility.

---

## The 10-Point Forensic Audit Checklist

Run all ten checks for each target model (`models/<id>/`):

### 1. Intended-Use Contract & Operational Information Boundary
- Write down the model's intended real-world use in one sentence.
- List everything an operator would actually know at prediction/decision time `t`, then compare it against everything the exam and example pipelines knew when selecting windows, aligning timestamps, computing normalization stats, or filtering inputs.
- Reconstruct the eligible scoring population from information available at issue time. For maps, count candidate cells outside the archived grid. For demos, count eligible observations before and after any display filter and recompute the summary over both groups.

### 2. Weight, Normalization & Feature-Sensitivity Health (Shipped `.npz` Files)
Load every `.npz` weight file in `models/<id>/` and verify:
- **Finiteness & dead units:** Zero `NaN`/`Inf`; inspect 2-D weight row norms (`norm < 1e-4`), collapsed biases, and saturated norms.
- **Outlier-poisoned standardization:** Inspect all stored scaling vectors (`x_mu`, `x_sd`, `st_mu`, `st_sd`). Flag extreme `sd` (`> 1e2` or `<= 1.001e-3`).
- **Real-input post-normalization spread:** Pass shipped `examples/` or real dev inputs through the exact normalization transform and flag non-constant physical features whose normalized `std` collapses (`< 1e-2`) or saturates at a clip bound.
- **Counterfactual sensitivity ablation:** On real shipped examples, replace suspect feature columns with their median across examples (holding all other inputs fixed) and compare the max prediction shift against a healthy control feature.
  - *Precedent:* `EnergyNet v2.1` — 4 work-fraction inputs (`aero/Ep`, `rolling/Ep`, `grade_dn/Ep`, `inertia_pos/Ep`) had `sd` up to `53,606` from moving-downhill trips where `ph[0] = 0`; holding all 4 constant moved shipped VED predictions by at most `0.00039%` vs `6.08%` for control `grade_up/Ep`.
- **Ensemble diversity:** Compute pairwise cosine similarity of first-layer weights across ensemble seeds.
- Test feature attribution in both directions. Change only the suspected field, then regenerate every dependent feature from a consistent input change. On an otherwise perturbed input, restore only the suspected field to measure how much of the output shift it explains.

### 3. Training-Framework ↔ Inference-Runtime Forward Parity
- Reconstruct the PyTorch `nn.Module` from `models/<id>/training/` using the shipped `.npz` weights (`model.eval()`), and compare outputs against the shipped NumPy module (`models/<id>/<id>.py`) on random non-trivial inputs and shipped examples (`max_abs_diff < 1e-5` float32).
- Verify activation variants (`GELU(approximate='tanh')` vs exact `erf`), BatchNorm/LayerNorm `eps` and `running_mean`/`running_var` export, padding, and clamps.

### 4. Causal / Future-Perturbation & Cold-Start Leakage Test
For any time-series, streaming, nowcast, or sequential-decision model:
- Freeze everything an operator knew up to decision time `t`, perturb **only** future observations `> t` (including secondary paths like weather/wear planners or global normalization), and verify the submitted decision/prediction at `t` changes by `0.0`.
- Test **cold-start / boundary padding** at `t = 0` (ensure missing history is never back-filled with the clearing/target value at `t = 0`).
- Repeat the future-perturbation test with a gap that straddles the forecast cutoff. Hold onset fixed to isolate interpolation, then test automatic alignment separately. Include a complete-prefix control.
- Trace how the episode builds static inputs such as a building card. Perturb observations strictly beyond the planning horizon, rebuild the static input, and replay the same decision. Compare warm-cache and fresh-cache results when preprocessing is cached.
- For sensor networks, exercise the supported input shape with every sensor unavailable. Record whether the public API abstains, raises an actionable error or returns an estimate. For held-out forecasts, compare the end of the input window with the target arrival and state how preprocessing latency was handled.
- Pass a true prefix to the public API. Count rows before and after the last observation, including any availability jump at minimum-length thresholds. For date selectors, compare the same target day alone and embedded in a longer file.
- In closed-loop simulations, compare outputs strictly before the first changed sample and keep a complete-input control. State whether preprocessing also changed physical demand and whether the exam traces contain the tested gap.
- Test **API guardrails** (e.g., verify lagged/nowcast APIs reject `lead <= 0` rather than silently leaking same-day truth).
  - *Precedent:* `BatteryPilot v1.3` — perturbing future temperatures moved an already-submitted 2 p.m. offer via the day-ahead wear planner (`$1.19/MWh`), and cold-start history back-filled hour 0's clearing real-time price (`$3.02/MWh`); `FlowCast v1.0` nowcast API silently accepted `lead=0`.

### 5. Metadata-Only & Signal-Scrambled Shortcut Test
- Discard or scramble the primary physical signal (waveform, time series, image) and test whether a simple baseline using only metadata, timestamps, window alignment, or helper tables can recover the target.
  - *Precedent:* `QuakeNet v2` — discarding waveforms entirely and fitting only station coordinates + catalogue-derived `t_start + p_guess` timestamps recovered all 4 shipped 10-station epicentres to `< 5 m`.

### 6. Train-vs-Real Feature, Unit & Split Alignment
- Compare `simbank.py`/`bank.py`, `dataset.py`, `realfeat.py`/`evalreal.py`, and `<id>.py` side by side for identical feature ordering, units, scaling constants, DOY/window alignment, and `NaN` handling.
- Verify zero unit/event ID overlap between dev and exam in `splits.json`, and zero exam leakage into donor tables, priors, or scalers.

### 7. Loss, Gradient, Seed & Optimization Reproducibility
- Verify loss formulas (signs, `0.5` factors, log-sigma parameterization, quantile monotonicity) and check that invalid-target masks apply consistently to **both** primary and auxiliary loss terms.
- Check gradient flow (no accidental `.detach()`, frozen trainable parameters, or missing `eval()`/`no_grad()`).
- **Seed & resume reproducibility:** Verify whether the trainer seeds **both** NumPy and PyTorch (`torch.manual_seed`), and whether resume checkpoints save and restore RNG state — especially critical when ratchet margins are small (e.g., `0.01`).
  - *Precedent:* `FlowCast v1.0` — `train.py` seeded NumPy's batch sampler but never seeded PyTorch weight init or dropout, and resume created a fresh RNG stream.

### 8. Rung Promotion-Rule Arithmetic Replay
- Independently reapply the published keep/reject promotion rule from `ratchet.yaml` to the raw logged component scores in `training/ratchet*.json` across all rungs (including rejected rungs). Confirm every logged decision matches the stated rule, and flag any margins sitting inside bootstrap noise.

### 9. Twin-Test Attribution & Bottleneck Triage
- Check whether the released model passes cleanly on **its own simulated twins** (`twin_*.json`, `ratchet.yaml`, `MODEL_CARD.md`).
- Missing its own twins points to a learner, loss, or feature-pipeline defect rather than a simulator defect.

### 10. Calibration & Interval Honesty
- Verify widen-only compliance (`k >= 1.0`) unless explicitly documented as a simulation-side pre-scaling factor.
- Check per-stratum coverage (not just pooled coverage) and verify calibration factors were fitted strictly on dev with sufficient units per stratum.

---

## Multi-Reviewer Collaboration Structure (`docs/audits/`)

To keep independent reviews unbiased while converging on a single source of truth:
- **`docs/audits/SKILL.md`**: Canonical audit protocol (this file).
- **`docs/audits/README.md`**: Master audit index, severity rubric, and review status table.
- **`docs/audits/<model>-v<ver>.md`**: Canonical cross-checked audit report for a model.
- **`docs/audits/reviews/<reviewer>/`**: Optional workspace for independent first-pass notes before cross-checking and reconciling into the canonical report. When two agents audit the same model, keep independent passes separate until each has run its own probes, then merge verified findings and explicitly note any disagreements or corrections.
