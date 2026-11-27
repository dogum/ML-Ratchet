# BuildingDetective v1.0 | Partial records and district-meter features

**Cross-checked critical defect in hourly inference.** Reviewer B reviewed commit `4820cf3`, reran reviewer A's probe, then varied only the suspected columns on the same model base `61c70bf`. Partial-record augmentation masks the meters while the featurizer retains the full date axis. The hourly release consequently sees a record-length feature far outside its training scale when a user supplies a shorter date axis. [Original report and probe](reviews/reviewer-a/buildingdetective-v1.0.md) · [Training call site](../../models/buildingdetective/training/featurize.py#L76)

**What changed.** On HUE_2, a 365-day record gave 99.5 m² and residential-single. Truncation to 330 days gave 295.6 m² and healthcare. Holding every truncated-record feature fixed and restoring just the two span inputs to their 365-day values gave **94.4 m² and residential-single**. The corresponding 180- and 90-day estimates moved from 58.6 to 75.1 m² and from 10.0 to 86.6 m². Both recovered the original use class. Restoring the span is a diagnostic intervention.

The earlier full-year BDG2 test clipped all standardized features to ±8. Selectively clipping only the two district-meter zero-fraction columns (904 and 1197) gives a more precise attribution.

| BDG2 example | Shipped area, m² | Clip two fields, m² | Clip all fields, m² |
|---|---:|---:|---:|
| Lamb education | 1,691.8 | 1,691.8 | 1,181.5 |
| Eagle office | 3,445.6 | 4,988.8 | 4,988.8 |
| Peacock lodging | 2,576.2 | 5,131.9 | 5,141.3 |

**What remains open.** The two zero-fraction inputs account for essentially the entire effect on Eagle and Peacock. Other inputs account for the Lamb result. Clipping all fields worsened Lamb's area error. The experiments do not validate a global clip, quantify fleet-wide error, or show a repair would pass the ratchet. In this rerun an hourly Torch/NumPy sigma difference reached `2.25e-5`, above the audit skill's generic `1e-5` tolerance. The span experiments use the shipped runtime directly. The original arithmetic replay also identifies one documented parameter-cap withdrawal.

```sh
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/peer_4820cf3_probe.py buildingdetective
```

The follow-up uses NumPy, SciPy and pandas. [Selective probe](reviews/reviewer-b/peer_4820cf3_probe.py)

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| Short date axis puts the two span inputs at about 90 sd | Yes | **Fixed**: `read_hourly` holds days/365 and log10(days)/3 at their trained one-year values and warns (span sd 0.0011 and 0.00017 in training). | HUE_2 through the public reader: 330 d 95.1 m² residential (was 295.6 healthcare), 180 d 76.0 (58.6), 90 d 88.3 (10.0); full year 99.5. Exams unaffected: BDG2 reads single-year 2017 records; HEAPO uses 365-day windows with bad hours as NaN. `tests/test_audit_building.py`. |
| District-meter zero-fraction inputs (columns 904, 1197) drive Eagle and Peacock | Yes (out of distribution) | **Warned, not clipped.** `_run` warns on any input beyond 8 sd. The audit notes a clip is not validated (it worsened Lamb), so the repair is queued as a training change (simulated district meters with zero runs). | The demo warns on the three BDG2 examples. |
| Hourly Torch/NumPy sigma difference 2.25e-5 | Minor | Not changed (above a generic 1e-5 tolerance, far below the sigmas reported). | — |

**Follow-up run (27 Nov 2026, owner-approved).** Rung r11 (`models/buildingdetective/training/r11.py`, `r11.json`):
r10's recipe with simulated district meters that have seasonal shutdowns and outages. It brought columns 904 and
1197 from 387 / 347 sd (v1.0) to 6.1 / 6.1 sd on the 52 dev buildings, but scored 0.8287 on the frozen set against
r10-hourly's 0.8301 (needs +0.01), so it was **rejected**; dev composite 0.2998 (floor 0.2881), BDG2 dev area 60.7 %
(v1.0 50.8 %). The bounded-input variant (v1.0 with the two columns clipped at mean + 8 sd) gained only +0.007 on the
frozen set but improved dev (area 43.5 %, composite 0.3157); it is logged, not released. The frozen set has no district
zero runs, so neither change can pass the rule until it does. The exam was not opened. v1.0 is unchanged.
