# CellSense v1.3 · Endpoint and missing-curve behavior

**Cross-checked input guardrail and uncertainty discontinuity.** Both chemistries return finite capacity readings on all-NaN curves. On a real NMC811 curve, changing the final voltage by 0.2 mV across the documented completion tolerance multiplies the reported capacity uncertainty by 10.75. Model snapshot `61c70bf`, second review at `ea2bfde`. [First pass](reviews/reviewer-a/cellsense-v1.3.md) · [Public voltage gridding](../../models/cellsense/cellsense.py#L121)

Reviewer B truncated the same measured discharge (`m50t_W9[5]`), interpolated the last capacity from adjacent measured points, and supplied the same full early-life reference curve. Every row went through the public voltage-grid conversion and NMC811 diagnosis.

| Last voltage | Capacity (Ah) | Reported 1σ (Ah) |
|---|---:|---:|
| 2.5000 V | 4.486428 | .002000 |
| 2.5009 V | 4.486264 | .002000 |
| 2.5011 V | 4.491543 | .021492 |
| 2.51 through 2.60 V, 10 mV steps | 4.488812–4.491239 | .021460–.022095 |

The step occurs where the reader switches from measured terminal capacity to a network estimate. At 2.5009 versus 2.5011 V the reported mean rises 5.28 mAh. Performance stays comparatively stable across the next ten cutoffs on this curve. This measures continuity. Establishing which uncertainty is better calibrated requires independent near-cutoff cycles.

**LFP split.** The calibration code names six cells. The exam selector excludes all six by default. Reviewer B exercised the selector on eligible fixture readings for those six and six comparison IDs and measured zero overlap. None of the six is either side of a batch-continuation alias. The raw Severson batches were unavailable for an independent count of the 118 eligible exam cells. [Calibration IDs](../../models/cellsense/training/calibrate.py#L10) · [Exam selector](../../models/cellsense/training/realexam.py#L30) · [Continuation handling](../../models/cellsense/training/realdata.py)

**Other first-pass evidence.** The original probe reproduced the finite all-NaN estimates (0.8168 Ah LFP, 4.0251 Ah NMC811), folded forwards on all seven archives and all 9 LFP and 13 NMC rung decisions. On this CPU the largest folded raw-head difference was `1.05e-5` on the LFP partial reader, narrowly above the protocol’s absolute target. The four NMC members have strongly aligned first-layer weights and a shared default training seed. Their individual prediction errors would be needed to quantify effective ensemble diversity. [First-pass probe](reviews/reviewer-a/cellsense_v1_3_probe.py)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/peer_ea2bfde_probe.py cellsense
```

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| All-NaN curves return a finite capacity | Yes | **Fixed**: `q_of_v` and `q_of_v_nmc` raise with fewer than 2 finite points (raise only, so finite exam curves are untouched). | Test. |
| Sigma step at the completion tolerance | Yes (design) | Documented in the card; deciding which sigma is calibrated needs near-cutoff cycles. | — |
