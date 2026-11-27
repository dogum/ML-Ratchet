# RoomSense 1.0

RoomSense estimates how long a room reverberates from one mono clap. The primary output is the median of the 500 Hz, 1 kHz and 2 kHz octave-band RT60 values, in seconds. RT60 is the time a measured sound decay would take to fall by 60 dB at its fitted rate. RoomSense also returns an experimental volume estimate in cubic metres. Both readings include nominal 68% intervals.

The partial release was graded with synthetic claps passed through measured impulse responses. An accuracy exam with recorded claps and paired references remains pending.

## Reading a clip

Use one distinct clap with about a quarter-second of quiet before it and two to three seconds after it when possible. The input accepts mono audio at 8 to 96 kHz. It requires at least 50 ms before the detected onset and 400 ms after it. Clips with clipping, invalid samples, or a clap less than 15 dB above estimated background are rejected. The decay model reads up to 2.56 s after onset. It expects unprocessed recordings and a mostly stationary room. Automatic gain control, denoising, clipping or a second strong transient can distort the estimate.

```bash
python demo.py
python demo.py my-clap.wav
```

## Physics and learning

The simulator sums direct and first- and second-order image paths with distance spreading, fractional delays, and six walls with frequency-dependent pressure reflection. A diffuse tail follows Eyring decay. The clean response supplies the RT60 label. Simulated geometry supplies volume. A generated band-limited clap, simple microphone coloration, and a real background clip generate the observed input. All model training labels are simulated.

In an independent cuboid case, direct and first-reflection travel times and first-reflection pressure agreed to floating-point precision. Eyring RT60 differed from the rounded physical reference by 0.0086%. The T20 implementation recovered known 0.30, 0.75, and 1.50 s exponential decays to within 2.0%.

The 10,040-parameter convolutional model reads three early envelope traces, three integrated decay traces and classical fit values and validity flags. Its RT60 head predicts a correction in log-seconds to the classical reading, with a fixed anchor when too few octave bands support a fit. The model predicts error scales in log units. Dev widened them by 1.168× for RT60 and 1.516× for volume.

## Exam

Dev was Surrey C and D and Kitamura's open lounge. The held-out rooms were Surrey A, Surrey B and Kitamura's music room. Each of 19 held-out swept responses received separate generated claps and backgrounds at specified 27 and 44 dB peak-to-background ratios. The answer key uses a full-rate octave-band fit on the swept impulse response. Surrey's published interrupted-noise RT60 is based on a two-band average, a nearby cross-check with a different definition. A and B had explicit reference dimensions. The music room's source leaves its ceiling height blank.

| Evaluation | Cases | RoomSense MAE | Classical MAE | Dev median MAE |
|---|---:|---:|---:|---:|
| RT60, all clips | 38 | 0.117 s | 0.315 s | 0.483 s |
| RT60, 27 dB | 19 | 0.154 s | 0.433 s | 0.483 s |
| RT60, 44 dB | 19 | 0.080 s | 0.198 s | 0.483 s |
| RT60, both model and T20 have a reading | 4 | 0.074 s | **0.023 s** | — |
| Volume, experimental | 28 across two rooms | 0.424 ln(m³) | 0.987 ln(m³) | 0.987 ln(m³) |

Each room gets equal weight. On all-clip scores, each estimator uses the dev median when it abstains. The classical chain tries octave-band T20, then a fit to the late band-power envelope, then the median. T20 read 4 of 38 clips. The full classical chain read 15 before fallback, and RoomSense read 37. The detected onset for one Room B clip fell outside the required recording window. On actual predictions, the intervals covered 72.8% of RT60 references and 88.5% of volume references, room-weighted. Three rooms are too few for precise confidence limits on these differences. The dev median RT60 was 0.913 s, considerably longer than the held-out rooms. The dev-fitted decay-only volume baseline had a zero slope and returned the same 191 m³ as the dev median.

| Exam room | Clips | Sweep RT60 median | RoomSense RT60 MAE |
|---|---:|---:|---:|
| Surrey A | 14 | 0.284 s | 0.123 s |
| Surrey B | 14 | 0.445 s | 0.136 s |
| Kitamura music room | 10 | 0.557 s | 0.092 s |

The six public ClapRIR phone examples begin too close to the handclap for the quiet-lead-in requirement. Their source supplies no paired references. This compatibility check measures input acceptance.

See [each locked case and the release checksum](training/measured_exam.json), [the data split](training/split.json), and [data credits](examples/ATTRIBUTION.md).

