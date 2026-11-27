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
