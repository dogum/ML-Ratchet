# Reproduce RoomSense

Start at the repository root with the editable package, NumPy, SciPy, h5py and CPU PyTorch installed. The build used PyTorch 2.15.0+cpu. Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. The download and bank cache goes under the model's git-ignored `work/` or `$ML_RATCHET_WORK/roomsense/`. Per-rung checkpoints resume the exact next batch.

```bash
cd models/roomsense/training
python data.py noise surrey kitamura
python physics_check.py
python exam.py
python twin.py

python simbank.py train-1.0 9000 27571 --role train
python simbank.py val-1.0 1800 93017 --role validation
python train.py r1 train-1.0 --steps 2200
python ratchet.py --first

python simbank.py train-1.1 9000 27571 --role train --physics 1.1
python simbank.py val-1.1 1800 93017 --role validation --physics 1.1
python train.py r2 train-1.1 --steps 2200
python train.py r3 train-1.0,train-1.1 --steps 2200
python train.py r4 train-1.0,train-1.1 --steps 2200 --seconds-loss 1
python ratchet.py

python export.py
python exam.py --exam ../roomsense_weights.npz
python examples.py
python phone_check.py
python ../web/export_instrument.py
```

`export.py` chooses the rung marked released by the dev ladder and records the weights and split digests. `exam.py --exam` requires this lock. On an existing locked run, `--reproduce` reruns the same weights and checks the original case results. `measured_exam.json`, `release_lock.json`, `rung_scores.json` and `dev_baselines.json` are the published evidence from the original build. The browser exporter runs both original and deterministically perturbed versions of the shipped audio through Python for parity.


To reproduce the published exam without retraining, run `python exam.py --exam ../roomsense_weights.npz --reproduce` from this directory. The script verifies the existing release, fetches its pinned data, and compares scores to the published cases.

## Recorded-clap exam on ARPEGE

The authors' pinned annotations identify 28 EM32 handclap recordings in six rooms. Each has an independently measured swept response at the same room and nominal position. The [case list](arpege_exam_plan.json) selects the neutral swept response when supplied and the response with the loudspeaker facing EM32 otherwise. Take channel zero of both files. All six physical rooms are held out, and the published RoomSense 1.0 weights and baseline parameters supply the scorer.

Unpack the [official ARPEGE audio release](https://zenodo.org/records/20344913) so that the input directory contains `ARPEGE_01.2026/`. Preserve the archive-relative paths. The commands below assume the repository root.

```bash
python models/roomsense/training/arpege_exam.py files
python models/roomsense/training/arpege_exam.py freeze /path/to/unpacked-audio
python models/roomsense/training/arpege_exam.py score /path/to/unpacked-audio
python models/roomsense/training/arpege_exam.py score /path/to/unpacked-audio --reproduce
```

A complete recording pair adds one case. The freeze identifies the strongest isolated clap using the recorded amplitude envelope and stores the selected sample bounds. It preserves up to 180 ms of real lead-in and 2.6 s of decay, trimming before a neighboring clap. It hashes both WAVs before reading any sweep labels or predictions. Missing pairs, unusable sweep references, and input abstentions appear in the report. Cases with an independent reference and no prediction receive the same dev-median fallback as the measured-response exam. The scorer records room-weighted errors, reading availability, and interval coverage. Its room-count target is three. The published rectangular envelope dimensions supply the experimental volume reference where available. Room 4 is L-shaped and has no volume reference in this protocol.

For a smaller transfer, download `ARPEGE_01.2026_claps.zip` and the desired `room1_RIR.zip` through `room6_RIR.zip` into one folder. The v3 record lists a CC BY-SA 4.0 license. Run `python models/roomsense/training/arpege_exam.py pack /path/to/downloads arpege-mono.zip` to read the original ZIPs. The packer selects EM32, matches the numbered clap filenames in the archive, and preserves the native channel-zero samples and sample rate. Add `--room room1` to pack one room or repeat `--room` to select several. Extract the smaller ZIP into the untracked workspace and use its top-level directory for the freeze and score commands.

The workspace can access the authors' annotations. The official audio endpoint returned HTTP 403, so this score is pending access to the WAVs.
