> **Lab notebook.** The build notes as recorded during the original work in a cloud sandbox (Oct–Nov 2026), kept
> as written. Paths like `/home/claude/...` and `/mnt/user-data/outputs` refer to that sandbox: in this repository
> the code is in `training/` and everything it writes goes to `work/` (see `training/_paths.py`). Page links to
> claude.ai are the original private copies; the lab now lives at `lab/index.html`.

# BearingNet — status (v1.0, 2026-11-06)

**What it is.** From ~3–20 s of one accelerometer channel on a rotating machine, plus bearing geometry and a rough shaft speed, BearingNet says whether a rolling-element bearing is healthy (N) or has an outer-race (OR), inner-race (IR) or ball (B) defect, and refines the shaft speed with an error bar. Trained only on physics simulations injected into real healthy-machine vibration. Examined on two real rigs: CWRU (the standard benchmark) and Paderborn (a second machine it never heard).

- Page: https://claude.ai/artifact/Q2TRef2NuSbUcuXpw7gNgu (live in-browser network, real-recording explorer, fault lab with speed / size / loudness / noise sliders)
- Release: `/mnt/user-data/outputs/bearingnet-v1.0.zip` (numpy/scipy module, weights, demo on 14 real exam recordings in ~2 s, MODEL_CARD.md, training/)
- Working dir: `/home/claude/bearing/` (STATUS.md there has the resume notes)

## Recipe as built
1. **Physics.** BPFO/BPFI/BSF/FTF from geometry; matches CWRU's published multiples for SKF 6205/6203 to 1.6e-4 (checks/kinematics_check.json). Simulated envelope lines within 0.16 % of theory, IR shaft side-lines and B cage side-lines present (checks/physics_check.json). Simulator: impact trains with 0–2 % slip and 0.5–2 % jitter, Stribeck load-zone AM (IR at shaft, B at cage with alternating race strikes), entry/exit pulse pairs, 1–3 random kHz resonances, shaft lines, decoy impact trains, random knocks.
2. **Real noise.** Injected into healthy CWRU file 98 and Paderborn K001–K003 (all conditions), time-stretched ±15 % for speed. Never in dev or exam.
3. **Features + net.** Envelope spectra in 5 bands read on log-frequency axes aligned to each fault hypothesis (bearing-agnostic comb); 1-D residual CNN, 438k params, 20k sims.
4. **Dev set.** 10 CWRU files (seed 20261106, one per set×class + 1 normal) and Paderborn K006, KA01, KI03. File-level split (no window leakage).
5. **Calibration on dev (the big win).** Twin test: r1 got 100 % of simulated twins right, 22 % of real CWRU dev. Real faults were 10–25 dB above background (v1 allowed −26…0) and rang longer (3.5 vs 2.3 ms); healthy machines had shaft-rate modulation. Physics v2 → dev balanced accuracy 0.31 → 0.68.

## Rungs (frozen sim set val2; keep if objective +0.01 AND dev gate)
| rung | change | sim obj | dev bal. acc | decision |
|---|---|---|---|---|
| r1 | v1 physics | 0.572 | 0.31 | kept (first) |
| r2 | v2 physics (dev-calibrated) | 0.691 | 0.68 | **kept → release** |
| r3 | HypNet (shared trunk per hypothesis) | 0.692 | 0.64 | rejected (sim +0.001, dev gate) |
| r4 | r2 + band dropout | 0.690 | 0.69 | rejected (sim −0.001) |

## Exam (r2; balanced accuracy, network / envelope method / majority)
| set | n | network | envelope | majority |
|---|---|---|---|---|
| CWRU near sensor | 152 | 0.39 | **0.49** | 0.25 |
| CWRU far sensor | 144 | **0.57** | 0.51 | 0.25 |
| CWRU base plate | 91 | 0.50 | **0.67** | 0.33 |
| Paderborn (new machine) | 919 | **0.45** | 0.28 | 0.33 |

Speed: CWRU nominal error 1.36 % → 0.35 % (coverage 80 % at 1σ, 97 % at 2σ, after ×2.58 dev widening). Paderborn nominal already 0.03 %; net 0.60 % (worse), coverage 62 % / 92 %. Size: failed (worse than median guess).

## Limits
- CWRU: classical envelope analysis beats the net at the near sensor and base plate; ball faults found only 19 %.
- "Healthy" is weak: healthy bearing next to a faulty one called healthy 10 %; Paderborn healthy 35 %.
- Size output useless; speed refinement hurts when the nominal speed is exact.
- Dev was small and optimistic (Paderborn dev 0.78 vs exam 0.45).

## Next ideas
1. Healthy-vs-wrong-bearing: simulate the *other* bearing's real fault comb (with the other geometry) as a frequent decoy, and add Paderborn-style broadband healthy variation; calibrate the N prior per machine with a few healthy minutes (a realistic deployment step).
2. Bigger/more varied banks (tr2 has only 20k of the planned 40k), more background machines (other healthy recordings), then re-run HypNet — the sim objective looks saturated by unrecoverable low-SNR cases; score the ratchet on SNR > −6 dB only.
3. Hybrid: feed the envelope method's line prominences as extra scalars, or ensemble — the two fail on different rigs.

## Sandbox gotchas
- Container restarted 3 times; training resumes from ckpt/*.resume every 500 steps. Another agent pkill'ed processes named `simbank`, so scripts here are named `bn_*`.
- CWRU normal-baseline files are 48 kHz (not 12 kHz). Some CWRU 48 kHz files are 1.3–2.7 s (mirror-padded).
- Paderborn .rar archives read with libarchive-c; MFPT no longer offers downloads (contact-only).
- 2 shared cores: r2 (6000 steps) took ~1 h wall under contention.
