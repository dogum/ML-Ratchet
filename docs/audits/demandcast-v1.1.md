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

## Resolution (27 Nov 2026)

Measured first, on the local exam bank (`real_exam.npz`, inputs only, no labels): 1,734 of 12,642 solar exam rows report output at an hour the physics calls dark (BPAT 806, CISO 536, SWPP 212, PJM 159); 922 have an input beyond |z| 8 (twilight columns 65–68, max |z| 48). 365 of 14,448 demand rows have missing history hours (up to 48 h, `miss_frac` 0.286); 199 of those are beyond |z| 8 (max 286). So the published exam includes affected days, and their cost is not yet measured: that needs a re-score.

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| Twilight solar input amplified (BPAT pinned at the 1.2 cap) | Yes | **Warned, repair queued.** Note: the 2e-2 floor already applies to solar (z ≈ 3 on the BPAT days); the amplification is in the first-layer weights the network grew on near-constant columns, so no inference-side scale fixes it. `predict_cf` now warns when reported output falls at an hour the physics calls dark. The repair (mask those history hours, retrain) is queued in the card. | The demo now warns on 4 rows. |
| Missing demand history (`miss_frac`, sd floor 0.001) | Yes | **Warned, opt-in clip, repair queued.** `_forward` warns on any input beyond |z| 8 and takes `z_clip` (off by default, so the released numbers stand). Repair queued: simulated missing hours in training and a higher floor on count-like inputs. | — |
| CISO solar offset clamped to ±1 h | Yes (limit) | Queued: allow ±2 h in the next solar rung. | Two CISO exam days are 2 h off. |
| Unbounded softplus sigma | Consequence of the above | Not changed separately. | — |

Golden demo output unchanged (warnings go to stderr). Re-scoring the exam with the repaired model waits for the owner's approval.

### Follow-up (27 Nov 2026, approved queue)

- **Re-score with `z_clip=8`** (`training/rescore_zclip.py`, dated supplement, released weights): demand 2.382 → 2.323 %
  with actual weather, 3.051 → 2.961 % with forecast weather 2024-2025, 2.923 → 2.834 % on the released 2024 row. On the
  201 actual-weather rows with missing history hours 5.96 → 2.58 % (the other 8,559 rows unchanged); solar 16.14 →
  16.12 %, and the 2,350 dark-hour rows 25.24 → 25.20 % (the clip does not reach them, as found). No grid got worse.
  Made the default as **v1.1.1**. The audit's 1,734 rows are those with output at 20:00 LST over the whole bank; any
  dark hour gives 4,029 bank rows (2,350 with actual weather).
- **Repair rung r10** (simulator v5 with twilight output, ±2 h offsets and 1-48 h demand-history gaps; dark-hour mask in
  training and inference; floor 0.05 on count-like inputs; ±2 h offsets read): val_v5 0.4054 vs 0.3908 (v1.1.1), dev
  gate −4.8 % (solar 15.67 → 13.11 %, demand 1.98 → 2.05 %). Promoted, committed, exam run once as **v1.2**: solar
  10.6 % with actual weather (v1.1 16.1 %; persistence 19.1 %) and 12.0 % with forecast weather (18.2 %); CAISO 24.9 →
  11.2 %, BPA 52.0 → 20.8 %. Demand got worse: 2.49 % (v1.1.1 2.32 %), 3.14 % with forecast weather (2.96 %). Details in
  the model's NOTEBOOK and `training/exam_v1_2.json`.
