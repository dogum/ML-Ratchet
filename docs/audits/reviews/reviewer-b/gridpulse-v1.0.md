# GridPulse v1.0 | A future sample changes a 10-second forecast

**First pass, reviewer B. Critical defect for live use with a missing reading at the cutoff.** Audited the r5 waveform model and p1 ERCOT heads at `61c70bf` from audit checkout `4820cf3`. A grid operator estimating the size of a generation loss ten seconds after onset can use the frequency and demand known by that instant. The raw trace API can incorporate a later frequency sample even when the caller supplies the correct onset.

## Fixed-onset experiment

The release interpolates missing values over the entire 81-sample window. It then masks samples after the requested horizon. If a gap crosses that boundary, interpolation has already changed the visible prefix. In the shipped trace, fix onset at second 80, request ten seconds, remove the observations at +9 and +10, then raise **only** the reading at +11 by 0.050 Hz. All inputs available by +10 agree, including the two missing values. [Window, masking and public API](../../../../models/gridpulse/gridpulse.py#L25)

| Output at +10 s | Original suffix | Changed suffix |
|---|---:|---:|
| Estimated loss, MW | 1,245.82 | 1,346.42 |
| Estimated inertia, GW s | 175.78 | 183.54 |
| Forecast settling deviation, Hz | −0.17068 | −0.13087 |
| Forecast nadir deviation, Hz | −0.23473 | −0.23528 |

With complete readings through +10 and the same fixed onset, the future change moved **zero outputs**. With automatic onset, a separate later 0.25 Hz step moved the selected onset from second 80 to 103. Passing only observations through +10 raises the API’s 60-second post-onset length error. These are distinct mechanisms. Full-record alignment is reproducible for retrospective event analysis.

The GB evaluator uses an archived event timestamp and interpolates across its input record before windowing. The six shipped example windows are already processed. The prevalence of gaps across the cutoff in the original records and their effect on the published exam remain unknown. A second reviewer can resolve that by replaying the original events with timestamps and missingness preserved, imputing from each available prefix. [GB evaluator](../../../../models/gridpulse/training/evaluate.py#L55)

## Signal, export and score checks

- At +10 s, replacing the waveform with zeros while preserving demand and system context raised nadir MAE from 0.0436 to 0.0936 Hz and inertia MAE from 35.91 to 150.27 GW s across six events. For the two events with loss labels, loss MAE rose from 277 to 815 MW. The shipped model uses waveform information here. A separately trained context-only estimator could perform differently.
- Both weight archives were finite. Across six GB windows at +10 and +60 s the largest standardized feature magnitude was 3.77 or less. The waveform training module and release agreed within `6.7e-7` on target-space means and `1.4e-7` on widened standard deviations. An independent Torch reconstruction of all three ERCOT heads matched both outputs within `2.3e-7` in log space on two input tuples.
- All five candidate decisions on p2s and all three post-processing decisions on p2r replayed, as did the p2s baseline. The p2r comparison used its published r5 rescore of 0.5593. The archived ladder preserves the p2s rescores for the original rungs. Recovering historical p1s decisions requires the earlier records. [Ratchet and reference scores](../../../../models/gridpulse/ratchet.yaml)
- Four GB dev events informed the widening factors. Published twins and dev calibration summarize different samples from the six examples above. Training seeds both generators initially. At resume, it reinitializes the NumPy stream and reloads a checkpoint containing model, optimizer and epoch, while the batch permutation uses the Torch RNG. Exact resumed training and coverage on a larger live sample remain unverified. [Trainer](../../../../models/gridpulse/training/train.py#L24) · [Calibration evidence](../../../../models/gridpulse/training/checks/widen.json) · [Twin readouts](../../../../models/gridpulse/training/checks/twin_read_r5.json)

Run from the repository root with NumPy, SciPy, PyYAML and CPU PyTorch.

```sh
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/gridpulse_v1_0_probe.py
```
