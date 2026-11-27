# SnowSense v1.0 | Open thermometer gaps and partial water years

**Confirmed daily inference defect.** The released GRU is causal once it receives its inputs. The cleaner and window builder interpolate temperature in both directions before inference. At Wyoming station 497, changing readings from January 21 onward by +15 °C changed the January 19 estimate from 226.87 to 224.73 mm across a real thermometer gap. Future perturbations after a complete prefix produced zero change on all three stations. Causal forward-filling on those three records moved station NSE by at most 0.00033, with a maximum daily shift of 5.16 mm. The synthetic two-day gap test reached 69.60 mm. [Cleaner](../../models/snowsense/snowsense.py#L50) · [Window construction](../../models/snowsense/snowsense.py#L110)

**The public daily API also imposes a 200-day availability cutoff.** We passed each station's observed prefix to `predict(..., wys=[2022])`. On April 17, day 199 of the water year, all three returned empty daily and annual tables. On April 18, all three returned daily estimates through September 30, including **165 dates after the final input**. The window builder counts available days, reindexes the whole year, extends temperature and fills missing precipitation with zero. The returned daily table carries no observed-through date. A caller requesting today's SWE thus sees an empty spring result and then a full year of dated estimates from one additional observation. The annual summaries also use that padded window. This test concerns partial-year use. It leaves the completed-year exam scores intact. [Daily and annual output](../../models/snowsense/snowsense.py#L132)

The original training script seeds the NumPy sampler and omits a Torch initialization seed. The size of any resulting training variance remains unmeasured. Quantile parity was within 3.02e−6 on this rerun, all six rung decisions reproduced, and the released interval factors were 1.0. The three packaged stations do not determine the effect of causal filling across the 621-station exam. [Trainer](../../models/snowsense/training/train.py#L64)

Model snapshot `61c70bf`, cross-check at `b53c7c7`. Both probes ran with NumPy, pandas and CPU PyTorch.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-a/snowsense_v1_0_probe.py
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/peer_b53c7c7_probe.py
```

[First-pass evidence](reviews/reviewer-a/snowsense-v1.0.md) · [Independent probe](reviews/reviewer-b/peer_b53c7c7_probe.py)

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| Partial water year: empty before day 200, then 165 dates past the last input | Yes | **Fixed**: a year the record ends inside needs 30 observed days, its daily output stops at the last input, and its annual row gives the peak so far with no melt-out and an `observed_through` date. Completed years follow the old path exactly. | `tests/test_audit_water.py` (cuts on 15 Jan, 17 Apr, 18 Apr 2022). The exam builds windows with `training/common.station_windows`, so it is unchanged; the golden demo is unchanged. |
| Temperature gaps interpolated from both sides | Not changed | In live use the record ends today and `interpolate` holds the last reading, so nothing ahead is read. Across a gap inside a past record, using both sides is smoothing for a retrospective estimate, which is what the completed-year exam scores. | Audit: causal forward-fill moved station NSE by at most 0.00033 on the three packaged stations. |
| Trainer seeds NumPy only | Yes | **Fixed for future runs** (`torch.manual_seed(SEED)`). | — |
