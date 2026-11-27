# TsunamiNet v1.0 | A magnitude estimate with zero reporting buoys

**Reviewer B, first pass.** Model base `61c70bf`, audit checkout `fb19db6`, shipped rung r3. The source estimate is intended for an earthquake with a seismic origin and at least one usable DART buoy record. The public API accepts the state before that first observation. The exam contains 28 events with usable input stations.

## Empty input produces a confident-looking number

I kept the four packaged earthquake locations, passed correctly sized minute records with all telemetry marked unavailable, and called `event` followed by `posterior`. All four events had zero selected buoys. All four produced **Mw −2053.32 with a nominal 68% interval [−2055.30, −2051.31]** under the default seed. The max-pooling path subtracts 10,000 from masked embeddings. With every station masked, that sentinel passes through the context network. The release has no empty-set check at this entry point. The trainer deliberately keeps a nonempty station subset. The demo already checks `ev["inputs"]` before inference. This is a verified public API defect. No archived exam row exercised it. [Encoder and posterior](../../../../models/tsunaminet/tsunaminet.py#L56) · [Training dropout](../../../../models/tsunaminet/training/train.py#L32) · [Demo guard](../../../../models/tsunaminet/demo.py)

The release contract requests 1,441-minute arrays. Passing one-minute arrays directly raises a broadcasting error on all four examples. A streaming caller can pad to the specified size and track whether any buoy passed selection. The masked, full-length test above exercises the documented shape.

## When could the forecasts have been issued?

The network reads a 90-minute window ending at predicted arrival +69 minutes for each selected input buoy. I compared the latest such endpoint with each held-out buoy's predicted arrival across **all 119 archived forecasts**. The median nominal lead was **352 minutes**, with a range of **−66 to +1,082 minutes**. Five predicted arrivals and seven target-window starts were at or before the nominal input endpoint. The counts use predicted arrivals and scheduled input-window endpoints. They do not tell us when the measured peak occurred. Missing observations can end usable input earlier, and telemetry or quality control can delay availability. [Window construction](../../../../models/tsunaminet/tsunaminet.py#L146) · [Archived event and target times](../../../../models/tsunaminet/training/logs/exam_rows.json)

Adding 0.2 m to every processed sample strictly after the latest selected endpoint left the input tensor identical on all four examples. This tests the model-facing data. The upstream detider separately documents centered glitch removal using up to 34 later **observations** and interpolation across gaps. A true real-time lead distribution requires the raw observation and issuance timestamps. [Raw processing](../../../../models/tsunaminet/training/dart.py#L1)

## Does station geometry explain the predictions?

I kept the epicentre, station coordinates, Green’s functions, arrival times and availability masks, then zeroed only the observed waveforms **before** feature construction. The same stations remained selected. Wave-derived amplitude features and the NNLS source estimate were recomputed.

| Shipped event | Reference Mw | Full-input Mw | Zero-waveform Mw |
|---|---:|---:|---:|
| Tonga 2009 | 7.60 | 7.59 | 7.40 |
| British Columbia 2012 | 7.80 | 7.95 | 7.40 |
| Northern Chile 2014 | 8.20 | 7.87 | 7.40 |
| Southern Mexico 2017 | 8.20 | 7.83 | 7.45 |

The shipped estimator loses most between-event separation in this control. These observations establish waveform dependence on the packaged events. A separately trained metadata-only baseline on the wider event population remains untested.

## Scoring, calibration and export

- The exam selects held-out stations whose observed peak exceeds five times their pre-arrival noise estimate. Reconstructing the rule from the four archived events reproduced all 18 target IDs. Nine additional stations passed coverage and history checks. Including them changed mean absolute log-amplitude error from 0.22560 to 0.22562 dex, and nominal 68% coverage from 13/18 to 19/27. The outcome filter changed the denominator substantially, with little effect on this sample's mean error. [Target selection](../../../../models/tsunaminet/training/realdata.py#L35)
- A caller using a fresh `GreensFunctions()` must attach `net.cal` and pass the amplitude widening factor to reproduce the demo forecast. For British Columbia's first target, default versus demo settings gave medians of **1.12 vs 1.76 cm** and 68% intervals of **[0.79, 1.57] vs [0.87, 3.54] cm**. The README's explicit settings reproduce the calibrated path. The module's introductory example also passes arguments inconsistent with the forecast signature. The release documents a previously corrected interval-side error in fitting its shipped widening factors. [Forecast method](../../../../models/tsunaminet/tsunaminet.py#L188) · [Release instructions and calibration caveat](../../../../models/tsunaminet/model.yaml)
- Parameters were finite. With the same latent draws and NNLS features, the training Torch network and release agreed within `6.5e-7` across four examples. The five dev and 28 exam IDs were disjoint. All six archived rung decisions replayed on their respective frozen sets. Training seeds NumPy and Torch at startup and restarts the NumPy stream from the step count on resume. This pass did not retrain or refit the five-event calibration. [Trainer](../../../../models/tsunaminet/training/train.py#L21) · [Ratchet log](../../../../models/tsunaminet/training/logs/ratchet.json)

Run from the repository root with NumPy, SciPy and CPU PyTorch.

```sh
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/tsunaminet_v1_0_probe.py
```
