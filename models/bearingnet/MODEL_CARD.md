# BearingNet v1.0 — model card

## What it does
From a few seconds of vibration (one accelerometer channel) on a rotating machine, plus the bearing geometry (ball count n, ball diameter d, pitch diameter D, contact angle) and a rough shaft speed (within about ±5 %), BearingNet says which part of that rolling-element bearing is damaged:

| output | meaning |
|---|---|
| `condition`, `p` | N (healthy), OR (outer race), IR (inner race), B (rolling element / ball), with probabilities |
| `rpm`, `rpm_sig` | refined shaft speed and its 1σ error bar |
| `size_mm` | defect length estimate — **does not work, do not use** (see exam) |

It never saw a real faulty bearing while learning. It was trained only on simulated defects added into real recordings of healthy machines.

## Inputs and front end (`pipeline.py`, numpy/scipy only)
- Any sample rate; resampled to 12 kHz. Up to eight half-overlapping 2.73 s windows (32768 samples) are read and averaged. Shorter records are mirror-padded.
- Per window: squared-envelope spectra in 5 demodulation bands (0.5–5.7, 0.5–2, 2–3.8, 3.8–5.7 kHz, plus the band of highest spectral kurtosis, a simplified kurtogram), each read on a log2 frequency axis (1/64 octave, −2 … +2.6) aligned to each fault hypothesis (shaft, BPFO, BPFI, BSF) from the geometry and nominal speed; plus the raw spectrum on the shaft axis. 21 strips × 295 cells + 8 scalars (band kurtoses, speed, n, d/D).
- On those axes a fault's comb sits at the same place for any bearing, so the network learns a bearing-agnostic comb detector; a speed error shifts every strip by the same amount.

## Physics and simulator (`training/bearsim.py`)
- Kinematics: BPFO = n/2 (1 − r), BPFI = n/2 (1 + r), BSF = D/(2d)(1 − r²), FTF = ½(1 − r), r = d/D cos φ.
  **Check:** against the CWRU published multiples for SKF 6205-2RS and 6203-2RS, largest relative difference 1.6e-4 (`training/kinematics_check.json`). In a simulated recording the envelope lines land within 0.16 % of theory with the expected cage slip, with IR side-lines at ±1× shaft and B side-lines at ±1× cage (`training/physics_check.json`).
- Fault signal (McFadden & Smith 1984; Randall & Antoni 2011; Sawalhi & Randall 2011): impact train at the defect frequency (OR: BPFO, IR: BPFI, B: 2·BSF) with 0–2 % mean slip and 0.5–2 % jitter of each interval; amplitude modulation by the Stribeck load zone (IR at shaft rate, B at cage rate with alternate inner/outer strikes and a 40 % chance of spin-axis wander; OR fixed, plus in v2 a rotating-unbalance load at shaft rate); each strike is a soft entry and sharp exit pulse separated by defect length / rolling speed; each pulse rings 1–3 damped resonances (0.7–5.6 kHz, damping 0.33–4 %); random spectral tilt; extra 1×, 2×, 3× shaft lines; decoy impact trains at non-fault orders; random knocks.
- Real noise: every simulation is the fault signal added into a real healthy recording: CWRU normal-baseline file 98 and Paderborn K001–K003 (all 4 operating conditions), time-scaled by a random factor 0.85–1.18 so the background's own shaft lines follow the simulated speed. Fault-to-background power ratio in 0.5–5.7 kHz: −18 to +26 dB (v2). v2 also adds shaft-order modulation of the background (50 %) and a random sensor/path gain curve (±8 dB bumps, 70 %).
- Random geometries (7–16 balls, d/D 0.15–0.33, D 20–120 mm), defect sizes 0.1–6 mm, speeds from each background's own.

## Network (`training/model.py`)
1-D residual CNN (stem + 4 residual blocks, 40→128 channels, mean+max pooling, MLP head), 437,728 parameters. Loss: cross-entropy (label smoothing 0.02) + Gaussian β-NLL for log speed and log size. 6000 steps, batch 96, AdamW, cosine LR, EMA weights. Trained on 20,000 v2 simulations. Numpy port with BatchNorm folded: max |numpy − torch| = 8.1e-6 on 64 real dev windows. JavaScript port (web page): max |JS − Python| output difference 3.3e-6 on 6 real examples.

## Data and split (file level, fixed seed 20261106)
- CWRU Bearing Data Center: 161 files (12 kHz drive-end, 48 kHz drive-end, 12 kHz fan-end, normal baseline). Noise: file 98. Dev: 10 files (normal 99 + one per set × class). Exam: the other 150 files, read on every sensor (DE, FE, base plate) and also the healthy bearing at the other end of each faulty machine.
- Paderborn KAt bearing data (6203 bearings, 64 kHz housing accelerometer → 12 kHz, 10 recordings per operating condition × 4): noise K001–K003, dev K006, KA01, KI03, exam the other 26 bearings (23 scored + 3 with combined OR+IR damage reported separately).
- MFPT data: the MFPT site no longer offers downloads; not used.

## Ratchet (frozen simulated set val2 = 4000 v2 sims; objective = 0.8 balanced accuracy + 0.1 speed skill + 0.1 size skill; keep if +0.01 AND dev balanced accuracy no more than 0.03 below the incumbent)

| rung | change | sim objective | sim bal. acc | dev bal. acc | decision |
|---|---|---|---|---|---|
| r1 | v1 physics, 20k sims | 0.572 | 0.614 | 0.310 | kept (first) |
| r2 | v2 physics calibrated on dev (louder faults, longer ringing, loud decoys, shaft-modulated background, sensor EQ, OR rotating-load AM) | 0.691 | 0.754 | 0.680 | **kept (release)** |
| r3 | HypNet: one trunk shared by all fault hypotheses + per-class code | 0.692 | 0.752 | 0.637 | rejected (sim +0.001; dev gate failed) |
| r4 | r2 + band dropout fine-tune | 0.690 | 0.754 | 0.691 | rejected (sim gain < 0.01) |

r1 was re-scored on val2 when the physics changed. The v2 change came from a dev twin test: the r1 network got 100 % of simulated twins of the CWRU dev files right but only 22 % of the real ones; real faults were 10–25 dB above background (v1 allowed −26…0 dB) and rang longer (3.5 ms vs 2.3 ms median); healthy machines showed shaft-rate modulation that v1 lacked.

## Final exam (real data never used for any choice; reading level)
Calibration fitted on dev only: class temperature 1.5, a log-prior of +0.5 on "healthy", speed sigma × 2.58, size sigma × 2.08 (widen-only). Envelope baseline: spectral-kurtosis band, squared envelope spectrum, mean prominence of harmonics 1–3 at BPFO/BPFI/2·BSF (±1.5 %), speed from a comb search; one threshold for "healthy" fitted on dev (τ = 0.33).

| exam set | readings | network bal. acc | envelope bal. acc | majority bal. acc | network acc | envelope acc | majority acc |
|---|---|---|---|---|---|---|---|
| CWRU, sensor at the faulty bearing | 152 | 0.39 | **0.49** | 0.25 | 0.53 | 0.67 | 0.49 |
| CWRU, sensor at the other end | 144 | **0.57** | 0.51 | 0.25 | 0.55 | 0.67 | 0.51 |
| CWRU, base plate | 91 | 0.50 | **0.67** | 0.33 | 0.59 | 0.68 | 0.52 |
| Paderborn (second machine, never heard) | 919 | **0.45** | 0.28 | 0.33 | **0.48** | 0.29 | 0.48 |

Per class, CWRU near sensor (network / envelope): N 0.00 / 0.00 (only 4 readings), OR 0.61 / 0.78, IR 0.76 / 0.92, B 0.19 / 0.27. Paderborn: N 0.35 / 0.25, OR 0.54 / 0.18, IR 0.45 / 0.42.

Confusion, CWRU near (rows true N, OR, IR, B; columns predicted): network [[0,2,2,0],[1,45,5,23],[2,4,28,3],[8,16,6,7]]; envelope [[0,1,3,0],[0,58,13,3],[0,0,34,3],[12,4,11,10]].
Confusion, Paderborn: network [[28,36,16,0],[142,236,35,26],[114,78,180,28]]; envelope [[20,16,18,26],[80,78,112,169],[43,52,168,137]].

By condition (network accuracy, CWRU near): load 0/1/2/3 hp 0.46/0.62/0.59/0.44; size 0.007/0.014/0.021/0.028 in 0.51/0.70/0.52/0.25; OR at 6/3/12 o'clock 0.75/0.63/0.37; sensor near/far/base 0.53/0.55/0.59 (envelope 0.67/0.67/0.68). Paderborn: real damage 0.56, artificial 0.43, healthy 0.35; operating condition N09_M07_F10 0.40, others 0.49–0.53. Healthy bearing at the other end of a faulty CWRU machine: called healthy 10 % (envelope 13 %). Paderborn OR+IR bearings: called OR 73 %, IR 13 %.

Speed: CWRU nominal "1750 rpm" is 1.36 % off on average; network 0.35 %, envelope comb 0.72 %; coverage 80 % at 1σ, 97 % at 2σ. Paderborn nominal is already 0.03 % off; network 0.60 % (worse), coverage 62 % / 92 %.

Size: CWRU MAE 0.35 dex vs 0.19 dex for guessing the median; Paderborn 0.76 vs 0.23. **Failed.**

## Limits
1. On the CWRU rig (the near and base-plate sensors) the classical envelope method is better. Ball faults are mostly missed (19 %) and often called outer race.
2. "Healthy" is the weakest answer: fault vibration travels through the machine and the network often blames the wrong bearing.
3. The size output does not work; the speed refinement hurts when the nominal speed is already exact.
4. The dev set was small (10 CWRU files, 3 Paderborn bearings) and optimistic: Paderborn dev balanced accuracy 0.78 vs exam 0.45.
5. CWRU has only 4 near-sensor healthy readings in the exam. Some CWRU files are known to be hard or ambiguous for any method (Smith & Randall 2015).
6. Constant speed, small deep-groove ball bearings, two test rigs. Untested on variable speed, gearboxes, other bearing types, other mounts.
7. For the record only (never used for choosing): the rejected r4 scored slightly higher on the uncalibrated exam (CWRU near 0.42, Paderborn 0.49 vs r2 0.40, 0.44).

## Data credits and licences
- Case Western Reserve University Bearing Data Center, https://engineering.case.edu/bearingdatacenter — public download; please acknowledge CWRU when using it.
- Paderborn University KAt-DataCenter, https://groups.uni-paderborn.de/kat/BearingDataCenter/ — C. Lessmeier, J. K. Kimotho, D. Zimmer, W. Sextro, PHM Society European Conference 2016. Use under the terms on that page and cite the paper.
- Physics: McFadden & Smith, J. Sound Vib. 96 (1984); Randall & Antoni, MSSP 25 (2011); Sawalhi & Randall, MSSP 25 (2011); Smith & Randall, MSSP 64–65 (2015).
- Code: built by Claude for Greg's physics + ML collection.
