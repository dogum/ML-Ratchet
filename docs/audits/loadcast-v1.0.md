# LoadCast v1.0 | Omitted electricity-meter metadata

**Cross-checked critical defect in cold and warm predictions.** Reviewer B reviewed commit `4820cf3`, reran the submitted probe, and checked the affected hours and feature identity on base `61c70bf`. With just a steam or chilled-water meter specified, the public API can construct a zero electricity-presence indicator. The training scaler maps it to −1000. Explicitly fixing HVAC and removing electricity from the known meter list changes exactly one static feature, index 18. [Original report and probe](reviews/reviewer-a/loadcast-v1.0.md) · [Public forecast](../../models/loadcast/loadcast.py#L318)

**What changed.** Cold predictions increased in **all 8,760 hours on each building**. The ratios below compare predictions under two metadata inputs. The observed means give them an accuracy reference.

| Shipped 2017 example | Observed mean kW | Cold mean kW, full → omit electricity | Cold CV(RMSE), full → omit | Cold median kW, full → omit |
|---|---:|---:|---:|---:|
| Bull, chilled water | 937.25 | 45.12 → 2,255.34 (49.98×) | 100.5% → 144.8% | 39.32 → 2,256.95 |
| Cockatoo, steam | 201.90 | 43.94 → 1,071.12 (24.38×) | 82.0% → 431.9% | 32.14 → 1,071.02 |

With strictly prior history and the same fixed HVAC, warm CV(RMSE) rose from 22.2% to 61.4% and from 21.4% to 52.4%. Omitting the entire meter list via the public default also changes inferred HVAC on these inputs and reproduces the same direction of failure. The prior report's history-overlap and weather sensitivities are distinct. A target-period weather scenario can legitimately be supplied at issue. Use of observed future weather changes what prospective performance the exercise measures.

**What remains open.** Both examples describe real buildings with electricity removed from their metadata. Error across truly electricity-absent buildings, prevalence of incomplete metadata, and the benefit of a retrained representation remain unmeasured. The original six v1 decisions and three later candidate decisions reproduced from their archived metrics.

```sh
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/peer_4820cf3_probe.py loadcast
```

The follow-up uses the release's Python dependencies, including NumPy and pandas. [Per-hour probe](reviews/reviewer-b/peer_4820cf3_probe.py)

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| A meter list without electricity puts `has_electricity` at z = −1000 | Yes, and it reached the exam | **Warned, opt-in fix, re-score queued.** `forecast()` warns; `assume_electricity=True` adds the meter. The default stays as released so the card's exam numbers describe v1.0. | Training: `has_electricity` mean 1.0, sd 0.001 in both heads. Counted on BDG2 metadata: **58 of 1,584 exam buildings** list no electricity meter (0 of 52 dev), so their steam, chilled-water, hot-water and gas forecasts were scored at z = −1000. The report's open question ("prevalence of incomplete metadata") is answered; their accuracy cost needs the queued re-score. |

The golden demo is unchanged.

**Follow-up run (27 Nov 2026, owner-approved).** Dated re-score of the seen exam with `assume_electricity=True`
(`models/loadcast/training/rescore_noelec.py`, `rescore_noelec_r3.json`). Of the 58 buildings, 15 have exam
building-meters: 19 units (5 chilled water, 13 steam, 1 gas). On those 19, median cold CV(RMSE) fell from 597 % to
114 % and warm (a year) from 51 % to 35 %; cold is better on 18 of 19 (the gas meter is worse). Exam rows: steam a
year 45.5 → 43.6 %, chilled water a year 42.9 → 42.4 %, steam no history 102.8 → 100.8 %; no verdict changed
(6 wins, 16 ties, 18 losses). Released as **v1.0.1** with `assume_electricity=True` by default; `False` keeps v1.0's
behaviour with the warning. The representation rung (simulated buildings without an electricity meter) was not run:
LoadCast's frozen set has no such building, so the ratchet cannot score it; that needs a frozen-set change first.
