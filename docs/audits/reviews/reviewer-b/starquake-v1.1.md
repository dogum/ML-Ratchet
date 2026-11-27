# StarQuake v1.1 · Time scrambling and empty cadence boundaries

**First pass · oscillation detections can survive loss of time structure.** On three real 90-day Kepler excerpts, shuffling the finite flux values while retaining the timestamps, gaps and amplitude distribution yielded 18 detections across 30 permutations. Adding only unobserved cadences at either end also changed estimates for identical measured flux. Model snapshot `61c70bf`, audit checkout `73ee7e6`. [Released reader](../../../../models/starquake/starquake.py) · [Spectral preprocessing](../../../../models/starquake/pipeline.py)

| Kepler ID | Original P(oscillation) | Detections in 10 permutations | Original → padded νmax (µHz) | Original → padded Δν (µHz) |
|---|---:|---:|---:|---:|
| 2695975 | .999995 | 10/10 | 11.218 → 11.123 | 1.625 → 1.576 |
| 7353997 | .999998 | 7/10 | 70.971 → 70.235 | 7.131 → 7.146 |
| 7618730 | .999987 | 1/10 | 81.690 → 83.688 | 6.733 → 6.865 |

**Detection control.** The ten permutations per star used seeds 0–9 and the release threshold of 0.5. For KIC 2695975, all ten scrambled probabilities were at least 0.943. Shuffling destroys the original timing and mode pattern while retaining other cues, including flux amplitude. This shows a detection can survive that intervention. The result does not identify which retained cue the network used. These are three selected giants, and permutations of one star are correlated. An operational false-alarm rate requires independent real negative stars. The release separately documents 11–15% false alarms on unscreened quiet dwarfs. Constant zero and all-missing traces both cleared the detection flag in this probe. [Exam limits](../../../../models/starquake/MODEL_CARD.md#L131)

**Representation dependence.** The table appends another 90 days of NaNs. Prepending the same amount gave the same displayed values. Duration comes from the first and last observed cadence, so reported observing spans stayed fixed. Fourier resolution uses the full array length, while duty cycle uses the fraction of that array with finite values. For KIC 2695975 the duty input halved from .95232 to .47616. Changing only the padded spectral features moved network-only Δν from 1.62462 to 1.57546 µHz. Changing only duty moved it to 1.62579. With the released hybrid and a fixed 4,800 K temperature, the inferred mass went from 1.733 to 1.906 solar masses on the same observations. This is a public `read_ppm` sensitivity. Its frequency in historical FITS processing and effect on aggregate exam scores remain open. [Power spectrum and observing scalars](../../../../models/starquake/pipeline.py) · [Hybrid and scaling relation](../../../../models/starquake/starquake.py)

**Controls.** All 1,361 kernel rows cleared the near-zero threshold. The original training graph reproduced the released network outputs to `3.01e-6` on a real segment and `9.57e-7` on random features. All five logged network promotions matched the simulation score and dev gate. The hybrid has a separate published comparison on val3, 0.745 → 0.788. Its final widening factors are 1.00 and 1.71. Independent refitting of the hybrid and a full recount of the star-level splits were outside this replay. [Ratchet](../../../../models/starquake/ratchet.yaml) · [Star and noise selection](../../../../models/starquake/training/catalog.py) · [Hybrid release](../../../../models/starquake/training/release_v11.py)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -B docs/audits/reviews/reviewer-b/starquake_v1_1_probe.py
python3 -B docs/audits/reviews/reviewer-b/wave7_export_probe.py starquake
```
