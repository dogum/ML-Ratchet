# DemandCast v1.1 | Twilight solar input and missing demand history

**Cross-checked critical feature-sensitivity defects.** Reviewer B reviewed commit `4820cf3`, reran reviewer A's workflows against base `61c70bf`, and extended the demand test to consistent gaps in the input history. [Original report and probe](reviews/reviewer-a/demandcast-v1.1.md)

**What changed for solar.** On both shipped BPAT examples the published predictions hit the 1.2 capacity-factor cap for 11 hourly outputs. Zeroing only solar column 66 reduced daily absolute error divided by total observed generation from 43.3% to 11.2% and from 52.3% to 15.6%. Those two packaged days establish sensitivity to the twilight observation. A policy of erasing real twilight measurements still requires independent validation.

**What changed for demand.** The original 2.26% → 9.74% MAPE comparison changed only the missing fraction to 24/168. The second probe reconstructs the 168 hourly demand-to-prior residuals from the shipped vectors, checks the encoder reproduces all history-dependent fields used by the release to `1e-6`, removes a contiguous 24 hours, and regenerates every affected field. All 16 examples had complete baseline histories. All runs use the same 384 target hours and observed demand. [Feature encoder](../../models/demandcast/training/features.py#L28)

| Missing hours in the 168-hour history | MAPE with consistent gap | Same gap with just the missing fraction reset to zero |
|---|---:|---:|
| First 24 | 10.03% | 2.43% |
| Middle, hours 72–95 | 9.60% | 2.25% |
| Last 24 | 9.34% | 3.12% |

The complete-history baseline was 2.26%. The model still receives the altered daily means, hourly profiles, and recent residuals in the last column of the table. This isolates a large adverse effect from its missing-fraction input across three gap placements.

**What remains open.** These are imposed gaps. Operational outage frequency, performance across other regions or seasons, and the value of a retrained encoding remain unmeasured. The original CISO lag diagnostic used target observations to assess the best retrospective shift. It does not establish an issue-time offset policy. The eight archived val_v3 decisions reproduced, and the val_v4 log identifies the superseded r9a run.

```sh
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/peer_4820cf3_probe.py demandcast
```

The follow-up uses NumPy and pandas. [Gap reconstruction probe](reviews/reviewer-b/peer_4820cf3_probe.py)
