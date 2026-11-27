# StormNet v1.0.1 | Causal forecast windows and a one-hour Dst latency stress

**First pass. The tested exam pipeline respected the forecast cutoff.** The task forecasts the minimum Dst in the **next three complete hours** from 36 hours of wind and the last completed hourly Dst. We passed each of the six gallery series through the actual exam slicer with a two-hour gap in the magnetic-field input. Changing all later field values by +20 nT and later Dst by +30 nT left both the issued median and half-width identical, while the verification target rose by exactly 30 nT. The issue timestamp was the end of the last input hour. A three-hour field gap excluded that issue on all six series. [Forward fill](../../../../models/stormnet/training/noise.py#L88) · [Exam windows and targets](../../../../models/stormnet/training/split.py#L45)

**Freshness matters even with correct indexing.** On 721 matched hourly issues from five deliberately selected historical storms and one quiet block, we supplied the preceding hourly Dst in place of the most recently completed one. Wind, issue time and three-hour target stayed fixed. The table gives mean absolute error (MAE) against the observed three-hour minimum. The largest individual forecast change was **96.79 nT**.

| Selection by observed future minimum | Issues | MAE with current Dst | MAE with one-hour-stale Dst |
|---|---:|---:|---:|
| All matched gallery issues | 721 | 8.97 nT | 11.21 nT |
| At or below −50 nT | 307 | 15.54 nT | 19.37 nT |
| At or below −150 nT | 67 | 39.12 nT | 46.85 nT |

The gallery is enriched for storms. These numbers quantify the effect of a specified delay on this sample. The exam uses archived OMNI values. The files contain no as-issued publication timestamps, so operational lead and revision error remain open. The reserve archive reports 63.2% coverage overall and 50.9% for extreme targets for nominal 68% intervals. NASA's annual file endpoint returned HTTP 403 on this run. The measured gallery replay used the committed data. [Gallery selection](../../../../models/stormnet/training/storm_gallery.py#L1) · [Reserve results](../../../../models/stormnet/training/exam_reserve.json)

**Calibration edge labels disagree with the implementation.** Fitting, Python inference and browser inference consistently put exactly −50 nT in the middle bin (factor 1.2317) and exactly −30 nT in the upper bin (1.0915). The calibration file describes endpoints that put them in the lower bins (1.5752 and 1.2317). We found 6 and 5 matching issues respectively. Applying the factors described in the prose changed interval half-width by up to **2.43 nT**, with zero changes in point forecasts or covered/missed outcomes in this sample. This is a documentation error at two integer index values. [Fitting code](../../../../models/stormnet/training/calib_sigma.py#L33) · [Calibration description](../../../../models/stormnet/training/calibration_sigma.json) · [Public reader](../../../../models/stormnet/stormnet.py#L74)

**Other checks.** All release arrays were finite. The 15 normalized features had gallery standard deviations from 1.20 to 2.01, with 70 clipped feature values. All 48 hidden units had nonzero input norms. The model fits a NumPy random-feature ridge readout. Reassembling both heads from exported blocks reproduced release outputs to numerical precision. Training uses a fixed seed, a separate simulation residual fold for scale and disjoint training, frozen, dev, primary and reserve years in the protocol. Both original promotion decisions and all five release-ladder entries reproduced. The dev twin gaps and alternate-seed R3 margin are already documented. A complete bank rebuild, full exam rerun and as-issued feed replay remain outside this evidence. [Trainer](../../../../models/stormnet/training/train.py#L25) · [Year roles](../../../../models/stormnet/training/protocol.json) · [Twin and seed diagnostics](../../../../models/stormnet/MODEL_CARD.md#L35)

Model snapshot `61c70bf`, audit checkout `b53c7c7`. Uses NumPy and PyYAML. [Reproduction probe](stormnet_v1_0_1_probe.py)

```sh
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/stormnet_v1_0_1_probe.py
```
