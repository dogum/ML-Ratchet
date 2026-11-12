# BeatNet v1.0 model card

**Not a medical device.** BeatNet is a research and teaching model. It has not been validated for any clinical use. Do not
use it to diagnose, monitor or treat anyone.

## What it does
For every heartbeat in a single-lead ECG (modified lead II, any sampling rate), BeatNet gives the AAMI EC57 class:
**N** normal (bundle-branch-block beats included), **S** supraventricular ectopic, **V** ventricular ectopic, **F** fusion of
a normal and a ventricular beat. It also gives class probabilities, the QRS width in ms with a 68% error bar, and the
probability that the beat sits inside atrial fibrillation. Class Q (unclassifiable) is not predicted; Q beats are left
out of every score, as is usual.

It was trained **only on simulated heartbeats**. No real patient's labelled beat was ever a training sample. The question
it answers: how far does a physics simulator that spans patient variation get on the hard inter-patient problem?

## Inputs and outputs
- Input: one ECG lead in mV, its sampling rate, and beat positions (R-peaks). If you have no positions, a simple QRS
  detector in `pipeline.py` finds them.
- Front end (`pipeline.py`, numpy + scipy, identical for simulated and real data): resample to 360 Hz, band-pass 0.5-40 Hz,
  decimate to 180 Hz; per beat a 2.5 s window divided by the local QRS size, the median of its 40 neighbouring beats, and six
  rhythm features (pre-RR, post-RR and local mean RR as ratios, RR irregularity, relative QRS size).
- Outputs per beat: `label`, `probs` (N, S, V, F), `qrs_ms`, `qrs_sig_ms`, `af_prob`. Per record: `heart_rate_bpm`, `af_share`.

## The physics (simulator)
- **Heart**: a 2 mm voxel model with a thick left ventricle, a crescent right ventricle and thin atria; random size,
  orientation (frontal axis, rotation about the long axis, tilt) per patient.
- **Activation**: shortest-path (eikonal) spread. Muscle 0.2-0.4 mm/ms, Purkinje-lined endocardium 1.2-2.2 mm/ms, atria
  0.75-1.1 mm/ms. Normal beats start at His-Purkinje root sites (LV septum, LV anterior and posterior, RV moderator band) after
  the sinus P and an AV delay of 125-205 ms. Bundle-branch block removes one side's roots. A PVC starts from a random
  ventricular voxel with little Purkinje engagement, so the QRS is wide and the T is discordant. Atrial premature beats start
  from an ectopic atrial focus (normal QRS, aberrant right-bundle-like QRS when very early). Fusion: tau = min(tau_normal,
  tau_ectopic + delta).
- **Dipole**: each voxel's membrane potential rises at activation and falls one action-potential duration later (transmural and
  apico-basal APD gradients, electrotonic coupling, restitution with RR). Heart vector = volume sum of -grad Vm.
- **Leads**: Dower lead vectors with random electrode placement and breathing rotation; MIT-BIH records modified lead II.
- **Rhythm**: ECGSYN RR spectrum (McSharry et al. 2003) for HRV; PVCs keep the sinus clock (fully compensatory pause); APBs reset
  it; bigeminy, trigeminy, couplets, runs, interpolated PVCs; atrial fibrillation (irregular RR, 4-9 Hz f-waves) and flutter.
  Atrial premature beats come at 0.28-0.60 of the RR (physics v4; v1-v3 used 0.50-0.88, see Calibration).
- **Noise**: REAL baseline wander, electrode motion and muscle noise from the MIT-BIH Noise Stress Test Database at the
  record's SNR, plus (from physics v3) quiet real background taken from the six dev records, mains hum, amplifier noise, Holter
  filters and 11-bit quantisation.

### Physics checks (`physics_check.json`, `training/physics_check.py`)
| Check | Result | Reference |
|---|---|---|
| Eikonal shortest path vs exact straight-line travel time | +8.3% median, 12.8% max | geometry bound of 26-neighbour paths |
| Dipole: volume form vs surface form | r = 0.993 | Gauss theorem |
| Lead II: Dower vector vs infinite-medium lead field | 16 deg (median of 8 leads 16 deg) | torso boundary ignored, tens of degrees expected |
| RR spectrum LF/HF asked 0.5 / 1 / 2 | 0.50 / 1.05 / 1.98 | ECGSYN method |
| Average beat vs ECGSYN lead II | r = 0.87 | McSharry et al. 2003 |
| QRS ms (p10/p50/p90) normal / LBBB / RBBB / PVC | 60/72/88 · 134/158/195 · 94/114/161 · 109/142/197 | textbook: normal 70-110 (vector definition reads shorter), BBB >= 120, PVC 120-200 |
| Frontal QRS axis (p10/p50/p90) | -13/59/77 deg | normal -30..+90 |
| Pause around PVC / APB (x 2 RR) | 0.99 / 0.88 | full compensatory = 1.0; APB < 1 |

## Calibration on the dev records (physics versions)
Each correction was measured by putting the six real dev records next to simulations of the same measurement (`calib.py`,
`calib_coupling.py`), then built into the simulator as a new physics version with a new frozen validation set.
- v2: T wave too early and too small on sims (APD +47 ms, weaker electrotonic smoothing), PVC T waves too weak (gain 1.35), noise too high.
- v3: noise floor still 1.7x the dev floor at every percentile: NSTDB noise -10 dB plus quiet real background cut from the dev
  records; a fusion beat is labelled F only when each front takes 25-75% of the ventricles.
- v4: prematurity of S beats (pre-RR / local mean RR, median and IQR): dev 0.56 (0.48-0.64), v3 sims 0.78 (0.69-0.88),
  v4 sims 0.59 (0.50-0.68).

## Network
A 1-D residual CNN (5 blocks, 16-64 channels, about 172k weights) over the 2.5 s window (180 Hz), with the six rhythm
features joined before a 3-layer head. Outputs: 4 class logits, log QRS width with log-sigma (Gaussian NLL), AF logit.
Loss: cross-entropy with 0.05 label smoothing + 0.3 x QRS NLL + 0.3 x AF BCE. AdamW, cosine schedule, EMA weights.
Banks: 40,000 simulated beats (2,500 simulated patients) for r1-r4 and 24,000 (1,500 patients, physics v4) for r5, rare classes
over-sampled; rungs after r1 start from the incumbent's weights. One CPU thread.
The release runs in numpy (BatchNorm folded into the convolutions); numpy matches torch to 2.0e-05 (largest raw-output difference on 2,000 DS2 beats; labels agree on 100%).

## Ratchet
Objective: macro-F1 over N/S/V/F on a frozen simulated set (400 simulated patients, versioned with the physics; when the
physics changed every rung was re-scored on the new set and every decision re-taken in order). A rung is kept only if it
beats the incumbent by more than 0.01 AND, on the six real dev records (MIT-BIH DS1 101, 112, 119, 201, 203, 220, fixed seed
2026, never in any exam), the mean S/V F1 does not drop by more than 0.02. The DS2 column was logged but never used.

| # | Rung | frozen-sim macro-F1 | dev S/V F1 | DS2 macro-F1 (not used) | kept | why |
|---|---|---|---|---|---|---|
| 1 | r1: textbook simulator (physics v1), window only | 0.829 | 0.467 | 0.605 | yes | first rung |
| 2 | r2: dev-calibrated simulator (physics v2: APD +47 ms, weaker electrotonic coupling, PVC gain 1.35, less noise) + AF output | 0.873 | 0.476 | 0.541 | yes | sim macro-F1 0.873 vs 0.829; dev S/V F1 0.476 vs 0.467 |
| 3 | r3: + second input channel = the record's median neighbour beat (template), physics v2 | 0.860 | 0.465 | 0.530 | no | sim macro-F1 0.860 vs 0.873 (needs +0.01); dev S/V F1 0.465 vs 0.476 |
| 4 | r4: physics v3 = v2 + dev noise-floor calibration (NSTDB -10 dB + quiet real dev background) + fusion visibility rule | 0.892 | 0.441 | 0.528 | no | sim macro-F1 0.892 vs 0.873; dev S/V F1 0.441 vs 0.476 (dev gate failed) |
| 5 | r5: physics v4 = v3 + dev-calibrated ectopic coupling (S beats at 0.55 of local RR on dev, 0.78 in v3 sims) | 0.897 | 0.509 | 0.549 | yes | sim macro-F1 0.897 vs 0.873; dev S/V F1 0.509 vs 0.476 |

## Final exam: MIT-BIH Arrhythmia Database, inter-patient DS2 (de Chazal et al. 2004)
22 records, 49,705 beats (N 44,259, S 1,837, V 3,221, F 388),
MLII, paced records excluded. Beat positions = reference annotations. Se = sensitivity, +P = positive predictivity, in %.
The only real-data choice is one prior scale lambda = 0.25 picked on the dev records.

| Method | N Se / +P | S Se / +P | V Se / +P | F Se / +P | macro-F1 |
|---|---|---|---|---|---|
| **BeatNet** (simulation only, dev prior) | 91.6 / 97.2 | 48.9 / 70.5 | 97.3 / 49.7 | 0.8 / 0.7 | 0.547 |
| BeatNet without the dev prior | 90.1 / 97.6 | 57.6 / 66.8 | 97.2 / 46.2 | 1.3 / 1.1 | 0.549 |
| Linear discriminant (de Chazal features) trained on our simulations | 74.6 / 96.5 | 37.4 / 33.3 | 70.5 / 48.3 | 24.2 / 1.1 | 0.447 |
| Textbook rules (RR prematurity, QRS width, shape) | 85.0 / 97.7 | 47.5 / 31.5 | 78.8 / 30.2 | 0.0 / 0.0 | 0.431 |
| Majority class (always N) | 100.0 / 89.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.236 |
| de Chazal et al. 2004 (linear discriminant, IEEE TBME 51:1196) (trained on real DS1) | 86.9 / 99.2 | 75.9 / 38.5 | 77.7 / 81.9 | 89.4 / 8.6 | |
| Zhang et al. 2014 (feature selection + SVM, Comput Biol Med 48:1) (trained on real DS1) | 88.9 / 99.0 | 79.1 / 36.0 | 85.5 / 92.8 | 93.8 / 13.8 | |

Published methods were trained on real labelled DS1 beats; they are listed for scale only.

BeatNet confusion matrix on DS2:

| true \ called | N | S | V | F |
|---|---|---|---|---|
| N | 40,555 | 341 | 2,963 | 400 |
| S | 862 | 899 | 75 | 1 |
| V | 53 | 35 | 3,133 | 0 |
| F | 246 | 0 | 139 | 3 |

### Same exam, beats found by our own QRS detector
The detector (Pan-Tompkins style, `pipeline.detect`) found the beats; a detection within 150 ms of a reference beat takes its
class. Missed beats (1,006 of 49,705) count against sensitivity, false detections (44) against +P.

| Method | N Se / +P | S Se / +P | V Se / +P | F Se / +P | macro-F1 |
|---|---|---|---|---|---|
| **BeatNet**, own QRS detector | 89.0 / 97.0 | 47.1 / 60.7 | 92.6 / 47.0 | 0.5 / 0.5 | 0.522 |
| Textbook rules, own QRS detector | 82.7 / 97.5 | 46.1 / 28.7 | 76.2 / 29.8 | 0.0 / 0.0 | 0.419 |
| Always N, own QRS detector | 98.1 / 89.1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.233 |

### Other held-out sets
| Set | beats | S Se / +P | V Se / +P | BeatNet macro-F1 | rules macro-F1 |
|---|---|---|---|---|---|
| MIT-BIH DS2 (exam) | 49,705 | 48.9 / 70.5 | 97.3 / 49.7 | 0.547 | 0.431 |
| MIT-BIH DS1 minus dev (held out) | 37,637 | 19.3 / 86.8 | 95.2 / 33.6 | 0.436 | 0.390 |
| MIT-BIH Supraventricular Arrhythmia DB (held out) | 184,503 | 24.2 / 64.2 | 87.8 / 29.6 | 0.422 | 0.415 |
| dev (6 records; used for choosing) | 13,376 | 22.3 / 49.0 | 98.1 / 59.0 | 0.502 | 0.482 |

## Rhythm output: atrial fibrillation (DS2, beat by beat)

| Method | AF Se | AF +P | non-AF specificity | AUC |
|---|---|---|---|---|
| BeatNet AF output (simulation only) | 78.3 | 80.9 | 96.5 | 0.971 |
| RR-irregularity rule, threshold fitted on simulations | 86.3 | 45.6 | 80.2 | 0.844 |
| Never AF | 0.0 | 0.0 | 100.0 | 0.5 |

7,879 of 48,866 DS2 beats lie inside a reference AF episode. Flutter beats are left out.

## Error bars and coverage
- **QRS width** (QT Database, cardiologists' onset/offset marks, beats from our detector): a linear map from the simulator's
  vector definition to calipers (width = 11.9 + 1.21 x raw) and a widen-only sigma factor (5.06) were fitted
  on 8 dev records only. Exam (MLII records, 432 beats, 11 records): mean error 18.0 ms vs classical slope rule
  15.1 ms vs median guess 20.1 ms; 68% bar covers 72.5% (all leads: 81.2%).
- **Class probabilities**: expected calibration error 1.0% on DS2 (10 bins).

## Where it fails
- Left bundle-branch block read as ventricular: record 214 (LBBB throughout) has 2,003 N beats; BeatNet called 1,522 of them V. A wide, slurred LBBB beat and a septal PVC look alike in one lead; the simulator's LBBB beats are not yet varied enough. This one record makes about half of all false V calls on DS2.
- More false V calls in 233 (579 of 2,230 N beats), 221 (281 of 2,031, during atrial fibrillation) and 202 (216 of 2,061, atrial flutter/fibrillation). V sensitivity is 97% but only half of the beats BeatNet calls V really are V (+P 50%).
- Fusion beats are almost never found: 3 of 388 on DS2 (F1 0.01). In record 213, 238 of 362 F beats are called N and 121 V. In record 231 (2:1 AV block, RBBB), 315 normal beats are called F. The simulator's fusion beats do not match how the annotators used F.
- S beats that are not premature are missed: record 232 (sinus bradycardia with long pauses) has 1,382 S beats; BeatNet found 683. The S class depends on prematurity, and 232's ectopic atrial beats come after long pauses.
- Aberrated atrial premature beats (wide QRS) are called V: on dev record 201, 107 of 128 S beats; this is the main reason dev S sensitivity is only 22%.
- Other held-out sets are harder than DS2: S sensitivity 19% on the rest of DS1 and 24% on the Supraventricular Arrhythmia Database (SVDB, 128 Hz, different leads), where the textbook rules do better on S (F1 0.54 vs 0.35).

## Limits
- Not a medical device. Research and teaching only; never use it for diagnosis, monitoring or treatment.
- One lead (modified lead II). It was examined only on MIT-BIH-style Holter records (1975-79 tape, 360 Hz) and SVDB; other devices, leads and populations are untested.
- It classifies beats you give it. With its own simple QRS detector, 1,006 of 49,705 DS2 beats are missed (837 N, mostly record 228 with its large noise and baseline shifts) and 44 false beats appear.
- No Q class: paced and unclassifiable beats are not predicted (paced records are excluded, as is standard).
- The QRS width output is a rough number: mean error 17 ms against cardiologists' calipers on the QT database, and its error bar had to be widened 5x on dev to cover 68%.
- The simulator is calibrated on only six dev records. Two of its four dev corrections (noise floor, ectopic coupling) are population numbers; per-patient twins were not built.
- Published inter-patient methods were trained on real DS1 labels; BeatNet never saw a real label. The comparison is context, not a like-for-like contest.

## Data and licences
- MIT-BIH Arrhythmia Database: Moody GB, Mark RG. IEEE Eng Med Biol 20(3):45-50 (2001).
- MIT-BIH Noise Stress Test Database: Moody GB, Muldrow WE, Mark RG. Computers in Cardiology 11:381-384 (1984).
- MIT-BIH Supraventricular Arrhythmia Database: Greenwald SD, Patil RS, Mark RG. Computers in Cardiology 461-464 (1990).
- QT Database: Laguna P, Mark RG, Goldberger AL, Moody GB. Computers in Cardiology 24:673-676 (1997).
- PhysioNet: Goldberger AL et al. Circulation 101(23):e215-e220 (2000). All four databases: Open Data Commons Attribution License v1.0.
- Inter-patient split: de Chazal P, O'Dwyer M, Reilly RB. IEEE TBME 51(7):1196-1206 (2004).
- ECGSYN RR model: McSharry PE, Clifford GD, Tarassenko L, Smith LA. IEEE TBME 50(3):289-294 (2003). Lead vectors: Dower GE et al. (1980).
- Code and weights: MIT licence. The example excerpts in `examples/` are MIT-BIH data (ODC-By).
