# FireDetective v1.0 · Blind-wind scaling and empty fire histories

**Cross-checked.** An empty history passed through the public raw-fire pipeline returns infinite wind speed. On valid fires, the blind-wind flag is unusually large and the trained model uses it. The closing tests establish an input-scaling concern and an empty-input defect. The size and direction of any loss of skill from training with that scaling remain open. Model base `61c70bf`, closing review at `abd0b51`. [First pass](reviews/reviewer-a/firedetective-v1.0.md) · [Released reader](../../models/firedetective/firedetective.py#L52)

### What the wind flag changes

The scaler is fitted on wind-visible training rows, where the flag is always zero. With the 0.001 floor, setting the flag to one in blind mode produces a standardized input of 1,000. Training explicitly samples these blind examples. On four packaged fires, setting the raw direct wind fields to zero and rebuilding all features changed the visible sequence and left the blind sequence exactly equal. Other weather fields stayed fixed. Removing one valid blind timestep at a time moved the wind estimate by up to 1.08 m/s with static context fixed. The model still responds to time-varying inputs. [Training normalization and mixed-mode sampling](../../models/firedetective/training/train.py#L127) · [Wind fields](../../models/firedetective/feats.py#L131)

| Fire | Observed wind (m/s) | Released blind estimate | Blind with flag rescaled to z = 1, existing weights |
|---|---:|---:|---:|
| Cooks Peak | 6.79 | 3.41 | 1.91 |
| Powder Mill | 5.05 | 4.61 | 1.90 |
| Black | 4.38 | 4.29 | 1.77 |
| Maple | 1.93 | 3.25 | 1.73 |

Setting the flag to zero yielded 1.73–1.92 m/s. As a separate control, changing the stored flag scale to one **and** multiplying its first-layer weight column by the reciprocal of the original scale preserved every released output to numerical tolerance on all four. Both changes were in memory. A scale of 1,000 can affect optimization, but the large activation alone cannot establish an accuracy penalty. A matched retraining comparison would be needed. The archived full-exam blind-wind rank correlation is 0.30. [Exam results](../../models/firedetective/training/results/exam.json) · [Closing probe](reviews/reviewer-b/firedetective_closing_probe.py)

### Empty observations and split integrity

Passing zero observed timesteps with the original landscape and day of year to `pipeline.run` produced `wind = inf` and live fuel moisture of **425,553–425,564%** on the four sites. Without a valid step, masked max pooling passes −10,000 into the next layer and the wind head overflows on exponentiation. The documented input begins at the first burned-area observation. The real record builder rejects fires without burned area. All 944 archived exam fires had 3–12 observed steps, so the empty-history failure did not occur in that exam. This is a public-API defect with a reproducible trigger. [Raw pipeline](../../models/firedetective/pipeline.py#L21) · [Pooling](../../models/firedetective/firedetective.py#L65) · [Real record builder](../../models/firedetective/training/real.py#L185)

The frozen registry has **40 unique dev IDs and 4,197 unique exam IDs with zero overlap**. All **944 distinct scored fires** belong to the exam partition, including all four packaged examples. The first-pass probes reproduced all seven promotion decisions, near-zero incoming rows in five hidden units, widen-only calibration and member-level agreement within 4.2 × 10⁻⁷ in this replay. Row norms alone do not measure activation, and the trainer's checkpoint omits RNG state. [Frozen split](../../shared/fire/data/proc/splits.json) · [Training ledger](../../models/firedetective/training/ratchet.json)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -B docs/audits/reviews/reviewer-a/firedetective_v1_0_probe.py
python3 -B docs/audits/reviews/reviewer-b/firedetective_closing_probe.py
```
