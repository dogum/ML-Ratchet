# FirePilot v1.0 · Crew-clock sensitivity

**Cross-checked critical inference defect for schedules outside the afternoon training regime.** On three packaged fire scenarios, the value network predicts more area saved when given a morning crew clock. Part of that increase persists when the simulated operating conditions are identical and only one standardized input differs. Both reviews used model snapshot `61c70bf`. Reviewer B worked from `ea2bfde`. [First pass and field-only experiment](reviews/reviewer-a/firepilot-v1.0.md)

## What the two experiments establish

Training used afternoon scenarios. The shipped standard deviation of the normalized work-hours feature is `1e-6`. Relative to a 34-work-hour afternoon clock, a 36-work-hour clock moves this input by about 55,556 standard deviations. Reviewer A varied the input while holding every operating condition fixed. Reviewer B regenerated the full 72-hour work clock for a 06:00 local-solar start, passed it into `Context.from_store`, and let the planner recompute build times, other features and candidate scores. Forecast fields, budget (300 m per working hour) and planner seed (0) were held fixed. The last column restores only feature 6 while the planner continues operating on the morning schedule. [Clock and context](../../models/firepilot/firepilot.py#L196) · [Training normalization](../../models/firepilot/training/train_value.py#L79)

| Packaged fire | Archived afternoon | Morning clock | Morning clock, feature 6 restored |
|---|---:|---:|---:|
| AZ · River | .7134 / .7376 | .9136 / .7662 | .7637 / .8042 |
| MT | .7782 / .7208 | .8981 / .9060 | .8447 / .9060 |
| OK | .7859 / .9595 | .8980 / .9020 | .7938 / .8885 |

Each cell gives the predicted saved fraction and the chosen plan’s forecast proxy saved fraction. The isolated input raises the selected plan’s prediction by .0534–.1499 on the morning schedule. The proxy falls on AZ, stays equal on MT and rises on OK. Actual area saved would require a separate simulation against the hidden fire trajectory. Reviewer A’s 48-hour result is a one-feature sensitivity test. The shipped context fixes the horizon at 72 hours.

## Exam exposure

All 181 local start times at one-minute intervals from noon through 15:00 produced 34 working hours over 72 hours. All three archived clocks also end at exactly 34. The real scenario builder selects PM detections, converts a nominal 13:30 local overpass to UTC, floors to the hour, and evaluates the clock at the weather location. Using the nominal longitude encoded in each fire ID for both the overpass and weather locations gave 34 working hours for all 48 exam IDs and 40 dev fidelity cases. This approximates the timestamp conversion. The stored pipeline uses separate locations for the two calculations. The other 45 per-fire clocks and the dev/gate clocks are unavailable here, so their individual totals and any change in exam score remain open. [Time conversion](../../shared/fire/firedata/__init__.py#L55) · [Scenario construction](../../models/firepilot/training/fp/scenarios.py#L53) · [Exam record](../../models/firepilot/training/results/exam_summary_p2.json)

Reviewer A’s original probe also passed its weight-health, example and five-rung checks. The largest Torch-to-NumPy difference on this CPU was `1.02e-7` on sigma. Run from the repository root with NumPy, pandas, SciPy and CPU PyTorch.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/firepilot_v1_0_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/peer_ea2bfde_probe.py firepilot
```
