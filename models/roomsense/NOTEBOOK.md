# RoomSense build notebook

## 21 November 2026

### Data and the frozen split

The ARPEGE archive and the direct ACE single-channel download returned HTTP 403 in this workspace. Surrey RealRoomBRIRs and the Kitamura Laboratory impulse-response repository downloaded from their public GitHub archives. The raw archive versions and SHA-256 hashes are in `training/split.json`. A seeded split within source gives three dev rooms and three exam rooms. Surrey contributes every sixth source angle from one ear. Kitamura contributes one target-response microphone per published setup. The sample positions were chosen from archive metadata before scoring. Dev and exam each have 19 responses and 38 noisy inputs. Use geometry for volume only when the corpus publishes the needed dimensions. The Surrey lecture theatre and Kitamura music room lack usable volume for this protocol.

ESC-10 provides 20 clock and rain recordings with distinct Freesound source IDs. Eight supply training noise, four supply frozen simulated validation, four supply measured dev, and four supply the measured exam. The two exam SNR levels (27 and 44 dB relative to peak short-frame power) use deterministic seeds derived from the case IDs. Browse the attribution for the three ESC-10 sources present in the bundled clips.

### Simulator and the classical comparator

Image paths include the direct arrival and reflections to order two. The diffuse pressure tail starts from the Eyring decay law, with a mixing time proportional to mean free path. The frontend computes noise-corrected Schroeder decay curves and T20 fits. Analytic image-path checks agree exactly in the numeric test, and the fitted known decay error peaks at 1.99%. Measured sweeps provide a high-SNR target via a fourth-order octave-band fit at sample resolution. On noisier clap inputs the baseline can also try a late band-power fit. Dev RT60 MAE was 0.131 s for the combined classical chain and 0.120 s for the dev median.

### Training and dev twins

Every rung used 9,000 synthetic training cases and 2,200 optimization steps with seed 541. Each validation bank has 1,800 cases. The first rung scored 0.7767 on physics 1.0. For dev twins, measured Surrey D and Kitamura open-lounge responses carried 3.05 dB less late-to-early energy than modeled, using 45 to 250 ms over the first 45 ms after the peak. The twins use measured dimensions and the known Surrey source radius. Receiver positions and Kitamura source geometry are approximate. Reducing the diffuse-pressure coefficient from 0.390 to 0.316 was a partial correction. The ladder was replayed on physics 1.1 using the same frozen seed and background split.

| Rung | Frozen 1.1 skill | Dev RT60 MAE | Decision |
|---|---:|---:|---|
| r1, physics 1.0 | 0.7580 | 0.213 s | Released |
| r2, lower diffuse pressure | 0.7579 | 0.228 s | Rejected |
| r3, mixed physics | 0.7555 | 0.243 s | Rejected |
| r4, mixed physics and seconds loss | 0.7622 | 0.235 s | Rejected |

The gains had to exceed 0.01 and dev error could rise by at most 5%. All three candidates missed the simulation margin. All three also exceeded the allowed dev RT60 error. Learned dev predictions read too short in the two longer rooms. The physics 1.1 tail calibration improved the targeted energy statistic by construction. That change failed to improve the inverse estimate.

### Lock, exam and release

The chosen checkpoint passed PyTorch-to-NumPy parity with a maximum absolute difference of 9.54e-7. Export fit one widen-only factor per output on dev and hashed the weights and data manifest before the exam. The model favored coverage, supplying 37 of 38 readings. Four clips supported the stricter T20 fit, on which T20 won. The all-case win depended in part on the dev median falling well above the observed exam reverberation times, so the card shows the matched comparison prominently.

A post-lock check read six public ClapRIR phone WAVs. All were rejected for the quiet-lead-in requirement. They lack matching references, and their audio is kept in the download cache. The browser instrument plays licensed measured-response examples, adjusts the length used, and processes uploaded mono audio locally. The NumPy CLI and browser both use the same input-quality criteria.


### Recorded-clap exam follow-up

The official ARPEGE audio download still returned HTTP 403. The authors' GitHub annotations were reachable and identify 28 matching EM32 clap and swept-response WAV pairs across all six rooms. The complete candidate list and matching rule are recorded in `training/arpege_exam_plan.json`. Use the published RoomSense 1.0 checkpoint and dev baselines. The new freeze command records one amplitude-selected clap window per setup and pins both source WAVs before any scoring. The runner reports missing recordings, missing reference estimates, and estimator abstentions, then produces room-weighted metrics and uncertainty coverage. Its transfer command extracts the selected mono channel without changing sample values. An automated test exercises transfer, freeze, scoring, replay and rejection of modified audio with synthetic fixtures.

Other checked sources include The Spheres Dataset, which records handclaps and sweeps in one studio and whose Zenodo endpoint also returned HTTP 403, and SECL-UMons, whose two-room documentation lists handclaps and source coordinates. Its listed reference labels omit sweep responses and reverberation times. The ARPEGE audio remains the requested input for the six-room exam.
