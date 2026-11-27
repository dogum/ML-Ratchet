> **Lab notebook.** The build notes as recorded during the original build in a cloud project (Oct–Nov 2026), kept
> as written. Paths like `/home/claude/...` and `/mnt/user-data/outputs` refer to that build machine: in this repository
> the code is in `training/` and everything it writes goes to `work/` (see `training/_paths.py`). Page links to
> claude.ai are the original private copies; the lab now lives at `lab/index.html`.

# BeatNet — status (v1.0, 6 Nov 2026)

For every heartbeat in a single-lead ECG (modified lead II), BeatNet gives the AAMI EC57 class: N normal, S supraventricular ectopic, V ventricular ectopic, F fusion. It also gives a QRS width with an error bar and the chance that the beat sits inside atrial fibrillation. It was trained only on a simulated heart; no real labelled beat was ever a training sample. **Research and teaching only; not a medical device.**

- Page (network runs live on 22 real exam records; heart lab with movable ectopic focus, heart rate, BBB, real noise): https://claude.ai/artifact/BtCFF3jmGchgCqkbjpAyeH
- Release: `beatnet-v1.0.zip` (numpy/scipy inference, pipeline with QRS detector, weights, demo on 4 real DS2 excerpts that runs in 2 s, model card, all training scripts).
- Work folder: `/home/claude/beat/` (`STATUS.md` has resume notes).

## Recipe as built

- **Physics:**
  - 2 mm voxel heart (LV, crescent RV, atria). Eikonal (shortest-path) activation with Purkinje-lined endocardium. Beat type = where activation starts: sinus → His-Purkinje roots, ectopic atrial focus (aberrant RBBB-like QRS when very early), ventricular focus (low Purkinje engagement → wide QRS, discordant T), BBB = one side's roots removed, fusion = min of two fronts.
  - Membrane potential with APD gradients and restitution → heart dipole (volume sum of −grad Vm) → Dower lead vectors with random orientation, electrode placement, breathing.
  - Rhythm: ECGSYN RR spectrum (LF/HF), compensatory pauses after PVCs, APBs reset the sinus node, bigeminy/trigeminy/couplets/runs, AF and flutter.
- **Physics checks** (`physics_check.json`): eikonal vs straight-line travel time +8.3% median (26-neighbour bound); dipole volume vs surface form r = 0.993; Dower vs infinite-medium lead II 16°; RR LF/HF asked 0.5/1/2 → 0.50/1.05/1.98; mean beat vs ECGSYN r = 0.87; QRS p50 normal 72 ms, LBBB 158, RBBB 115, PVC 142; frontal axis p50 59°; pause ratio PVC 0.99, APB 0.88.
- **Noise:** real MIT-BIH Noise Stress Test DB (baseline wander, electrode motion, muscle) + (v3) quiet real background cut from the dev records, mains, amplifier noise, Holter filters, 11-bit quantisation.
- **Data:** MIT-BIH Arrhythmia DB, de Chazal 2004 inter-patient split, paced records out. Dev = 6 DS1 records (101, 112, 119, 201, 203, 220; seed 2026, stratified). Exam = DS2 (22 records, 49,705 beats). Extra held-out: DS1 minus dev, SVDB (78 records), QT DB for QRS width.
- **Ratchet:** objective = macro-F1 (N/S/V/F) on a frozen simulated set (400 patients, rebuilt for each physics version: v1val–v4val; every rung re-scored and every decision re-taken). Keep if > +0.01 AND dev mean S/V F1 drops ≤ 0.02.
- **Network:** 1-D residual CNN, 172k weights, 2.5 s window at 180 Hz + 6 RR features; outputs 4 logits, log-QRS width + sigma, AF logit. numpy == torch to 2.0e-5; JS == Python to 5e-6.

## Rungs (final scores on v4val)

| Rung | Change | Sim macro-F1 | Dev S/V F1 | DS2 macro-F1 (not used) | Kept |
|---|---|---|---|---|---|
| r1 | Textbook simulator (physics v1) | 0.829 | 0.467 | 0.605 | yes (first) |
| r2 | Physics v2: dev-calibrated T wave (APD +47 ms, weaker coupling), PVC gain 1.35, less noise; + AF output | 0.873 | 0.476 | 0.541 | yes |
| r3 | + template channel (median neighbour beat) | 0.860 | 0.465 | 0.530 | no: sim −0.013 |
| r4 | Physics v3: dev noise floor (NSTDB −10 dB + quiet dev background), fusion visibility rule | 0.892 | 0.441 | 0.528 | no: dev gate (−0.035) |
| r5 | Physics v4: dev-calibrated APB coupling (S pre-RR/local RR: dev 0.56, v3 sims 0.78 → v4 sims 0.59) | 0.897 | 0.509 | 0.549 | **yes, released** |

Note: DS2 macro-F1 fell from r1 to r2; the exam never decided anything, so it is reported as is.

## Exam (DS2, Se / +P in %)

| Method | N | S | V | F | macro-F1 |
|---|---|---|---|---|---|
| **BeatNet** (sim only, dev prior λ = 0.25) | 91.6 / 97.2 | 48.9 / 70.5 | 97.3 / 49.7 | 0.8 / 0.7 | 0.547 |
| Linear discriminant (de Chazal features) trained on sims | 74.6 / 96.5 | 37.4 / 33.3 | 70.5 / 48.3 | 24.2 / 1.1 | 0.447 |
| Textbook rules (prematurity, QRS width, shape) | 85.0 / 97.7 | 47.5 / 31.5 | 78.8 / 30.2 | 0 / 0 | 0.431 |
| Always N | 100 / 89.0 | 0 | 0 | 0 | 0.236 |
| de Chazal 2004 (trained on real DS1; context) | 86.9 / 99.2 | 75.9 / 38.5 | 77.7 / 81.9 | 89.4 / 8.6 | |
| Zhang 2014 (trained on real DS1; context) | 88.9 / 99.0 | 79.1 / 36.0 | 85.5 / 92.8 | 93.8 / 13.8 | |

- With our own QRS detector: macro-F1 0.522 (rules 0.419); 1,006 beats missed, 44 false.
- Other held-out sets, macro-F1 BeatNet / rules: DS1 rest 0.436 / 0.390; SVDB 0.422 / 0.415 (rules better on S in SVDB).
- AF output (DS2): Se 78%, +P 81%, AUC 0.971 vs RR-irregularity rule AUC 0.844.
- Coverage: class probabilities ECE 1.0% on DS2. QRS width (QT DB, MLII): MAE 18 ms vs classical 15 ms vs median guess 20 ms; 68% bar (widen factor 5.06 from dev) covers 72.5% (81% all leads).

## Where it fails

- LBBB read as V: record 214, 1,522 of 2,003 N beats → V (about half of all false V calls). Also 233 (579), 221 in AF (281), 202 (216).
- Fusion almost never found (3 of 388). Record 231 (2:1 block, RBBB): 315 N → F.
- Non-premature S beats missed (record 232: 683 of 1,382 found); aberrated APCs called V (dev record 201: 107 of 128).

## Next rungs worth trying

- **LBBB/RBBB variety:** widen the bundle-branch-block morphology range in the simulator (partial blocks, septal activation variants) and calibrate on a DS1 LBBB record kept out of the exam; this targets the biggest error source (214).
- **Twin test per dev record:** simulate each dev patient's twin (axis, QRS width, ectopy pattern) and compare network readings; the calibration so far is population-level only.
- **Fusion labels:** measure on dev how annotators used F (share of ectopic front) and refit the fusion visibility window; consider S-class context from longer RR history for bradycardic records like 232.

## Sandbox gotchas

- Container restarted 3 times. `train.py` writes `.resume` every 500 steps; `src/run_rest.sh` and `src/run_v4.sh` skip finished steps, so just relaunch them.
- `setsid nohup ... &` from a background job forks: `$!` is not the job PID. Find the PID with `ps` before writing `pids/`.
- Other agents also run files named `train.py`/`simbank.py`: never kill by name.
- wfdb from pip reads format 212 and .atr directly; PhysioNet endpoints worked without issue.

## 15 Nov 2026 — round 2: the two losing rows checked on dev (nothing trained, nothing released)

Targets: fusion beats found 3 / 388, and QRS width on the QT database, lead MLII, 18 ms vs classical delineation 15 ms.
Research and teaching only; not a medical device. The rows are public, so the first question was whether dev shows them.

- Data rebuilt in `work/data/` with `fetch.sh` (mitdb 90 MB, nstdb, svdb 53 MB) and `fetch_qt.sh` (QT database, 105
  records). `fetch.sh`'s own QT step fails on a fresh machine: it writes into `work/data/qtdb/` before creating it;
  `fetch_qt.sh` creates it. `wfdb` 4.3.1 was installed into `.venv-train` for `qtbank.py`.
- QRS width (`qt_devprobe.py`, record `training/qt_devprobe_r2.json`): the 8 dev records (seed 11) are on leads ECG1 (5),
  V5, V4 and D3; **none is MLII**. On them the network already beats classical delineation, 18.3 vs 20.2 ms (median guess
  27.3), matching `exam.json`'s dev block. Classical is better on three of the eight (sel811 11.3 vs 12.2 ms, sele0107 7.9
  vs 8.6, sele0116 19.8 vs 26.8). The exam's loss sits on the 11 MLII exam records; over all 96 exam records the network
  wins (16.7 vs 19.4). Dev cannot show the MLII loss, so nothing was changed for it.
- Fusion: the six dev records hold 3 F beats (201: 2, 203: 1); the network reads one as N and two as V. Three beats
  cannot calibrate a fusion window. The F-rich records (208: 373, 213: 362) are in the held-out DS1 rest and the DS2
  exam, and using them would change the split. Nothing was changed.
- v1.0 stays; the exam was not run.

## Corrections

- 14 Nov 2026: Card: S-beat coupling on dev is 0.56 of the local RR (calib_coupling.json median 0.555), was written 0.55; QRS-width error on QT DB lead MLII is 18 ms (exam.json 17.99), was written 17 ms. The demo now prints 'Research and teaching only. Not a medical device.' first; tests/golden/beatnet.txt was regenerated for that one line. No verdict changed.

## 27 Nov 2026 — audit resolution

The two-reviewer audit (`docs/audits/`). Short gaps (up to 10 samples) are filled and longer ones refused (one NaN made every beat 'N'). Records 201 and 202 are one patient across dev and exam; the card gives DS2 without 202, and a retrain without that patient is queued.

## 27 Nov 2026 — retrain without patient 201/202 (rules fixed before scoring)

Audit queue. MIT-BIH 201 (dev, calibration, quiet background pool) and 202 (DS2) are one patient.

- **Split v2** (`training/common.py`, `BEAT_SPLIT`, default v2): dev = 101, 112, 119, 203, 220 (201 dropped from the seeded
  draw's result, not replaced); DS2 exam = 21 records (202 left out); 201 joins the held-out DS1 rest. `BEAT_SPLIT=v1`
  gives the v1.0 split.
- **Calibration without 201** (dev, 5 records): T peak 303 → 306 ms, noise floor 0.0034 → 0.0036, V QRS width 133 → 136
  ms: kept. S-beat coupling moves: 0.555 → 0.64 of the local mean RR (IQR 0.59–0.73); 201's aberrated APCs (128 of the
  229 dev S beats) set v4's coupling. **Physics v5** (`sim.py`): v4 with the coupling ranges moved 0.24 of the way back
  to v3, and the quiet background rebuilt from the 5 dev records (`data/quiet_dev_v2.npz`, 450 pieces). One dev trial
  (200 patients): simulated S coupling 0.632 (IQR 0.536–0.735) vs dev 0.64 (0.591–0.734); kept. V coupling is still
  later in simulation (0.74 vs dev 0.60), as in v4; not changed.
- **Frozen set v5val** (400 patients, seeds 900000–900399) and a training bank of 4,000 patients (64,000 beats).
- **Ladder** (`ratchet_v2.json`): incumbent = the released r5 (rebuilt from `beatnet_weights.npz` by
  `training/from_release.py`; matches numpy to 1.1e-5) re-scored on v5val and dev v2: macro-F1 0.879. Candidates r6_s0
  and r6_s1: physics v5, from scratch (nothing inherited from 201-exposed rungs), 9,000 steps, lr 2e-3, seeds 0 and 1.
  Rule unchanged: keep if v5val macro-F1 rises by more than 0.01 AND dev mean S/V F1 drops by no more than 0.02.
  DS2 is not scored per rung any more; exam.py opens it once for the rung the ratchet releases.
- If no rung is promoted, v1.0's DS2-without-202 score is recorded instead.

## 27 Nov 2026 — result: retrain without 201/202 not promoted; v1.0's DS2 without 202 recorded

Ladder `training/ratchet_v2.json` (v5val, dev split v2; rules fixed above, committed in e293787 before scoring):

| Rung | v5val macro-F1 | dev S/V F1 | dev F1 N / S / V | decision |
|---|---|---|---|---|
| r5 (released, rebuilt from numpy) | 0.879 | 0.579 | 0.960 / 0.436 / 0.722 | incumbent |
| r6_s0 (seed 0) | 0.896 | 0.409 | 0.923 / 0.247 / 0.572 | rejected: dev gate |
| r6_s1 (seed 1) | 0.903 | 0.536 | 0.921 / 0.513 / 0.560 | rejected: dev gate |

- Training: 9,000 steps each on one thread, about 75 minutes, two seeds in parallel; bank 64,000 beats
  (v5 seeds 0–1499 plus v5x0/v5x1 seeds 1500–3999).
- What failed: both from-scratch networks call far more dev beats V (V +P 0.40 against 0.57; dev N sensitivity 0.86
  against 0.93), and dev S sensitivity depends on the seed (0.28 vs 0.60). r5 came through a warm-start ladder
  (v1 → v2 → v4 banks); a single from-scratch run on v5 does not reproduce its dev behaviour. Not tried: a warm start
  from r5 (it would carry the 201-exposed background), a seed ensemble.
- DS2 was not opened for either rung. v1.0 stays.
- **Dated DS2 score of v1.0 without 202** (`training/ds2_without_202.py`, 27 Nov 2026; record
  `training/ds2_without_202.json`): released weights and their stored dev prior; 21 records: macro-F1 0.5531, SV-F1
  0.6305 (rules 0.4321). All 22 records from the rebuilt bank: 0.5465 / 0.6176, which reproduces the published 0.547.
- Calibration record: `training/calib_coupling.json` gains `v5_trial` (dev split v2 vs the v5 trial bank).
