# Training: BridgeSense

Run every script from `models/bridgesense/training/` (or from the repo root with the script path). Large downloads, simulation banks, checkpoints, and scratch logs go to `../work/` (git-ignored) or `$ML_RATCHET_WORK/bridgesense/`.

| Step | Script | Writes |
|---|---|---|
| 1. Fetch public data | `python3 fetch_data.py` (Zenodo KW51, Concrete Box-Girder + NASA POWER Hamburg `T2M`, HBTA 100 Hz) | `work/raw/` |
| 2. Physics | `simulator.py` — 2-D beam/arch/truss modal FEM + 1-D Fourier deck heat diffusion + thermoelastic $E(T)$ + freezing phase transition | — |
| 3. Check physics | `python3 physics_check.py` — 5 independent analytical / exact-eigen / KW51 FEM checks | `verify_physics.json` (committed) |
| 4. Freeze splits | `python3 split.py` — seeded KW51 warm-ref / Dev / Exam blocks; Box-Girder and HBTA held out 100% zero-shot | `../splits.json` (committed) |
| 5. Real OMA noise | `python3 noise.py` — extract healthy KW51 autumn residual windows + availability patterns | `work/noise_pool.npz` |
| 6. HBTA OMA modes | `python3 extract_hbta_modes.py` — Welch-SVD peaks picked on the 5 UDS `*_01` reference recordings only (frozen in `hbta_bands.json`), tracked across 50 recordings × 8 windows | `hbta_bands.json` (committed), `work/hbta_modes.npz` |
| 7. Banks (`p1`, `p2`) | `python3 simbank.py val1 2000 101 --physics p1`, `simbank.py sel1 2000 201 --physics p1`, `simbank.py train1 10000 1 --physics p1`, and the same for `p2` with `val2 102`, `sel2 202`, `train2 12000 2`. `sel*` banks choose checkpoints; `val*` banks judge rungs | `work/banks/` |
| 8. Twin test (`p3`) | `python3 twin_dev.py` — inspect KW51 Dev healthy residuals with `r2`, calibrate `p3` (`work/cal_p3.json`), then `simbank.py val3 2000 103`, `sel3 2000 203`, `train3 12000 3`, each with `--physics p3 --cal ../work/cal_p3.json` | `work/cal_p3.json`, `work/banks/` |
| 9. Train rungs | `python3 train.py --name r6_mix_no_anchor --bank train2,train3 --val sel3 --steps 4500 --no-anchor` and the other rungs in `ratchet.py` (`r1` selects on `sel1`, `r2*` on `sel2`, the rest on `sel3`; seeds `--seed 2`, `--seed 3` for `r6_s2`, `r6_s3`) | `work/checkpoints/` |
| 10. Ratchet replay | `python3 ratchet.py --val val3` — frozen-bank score + KW51 Dev gate; refuses any rung selected on `val3` | `../ratchet.yaml` (committed), `work/ratchet_log.json` |
| 11. Calibrate | `python3 calibrate.py r4_mix r6_s2 r6_s3` — Dev-only widen / temperature fit | `work/calibration.json` |
| 12. Export numpy | `python3 export.py` — pure-numpy weights with one config per member; PyTorch parity checked before the released file is replaced | `../bridgesense_weights.npz` |
| 13. Real exam | `python3 exam.py` — run once across KW51 Exam A, Concrete Box-Girder (zero-shot), and HBTA (zero-shot) vs baselines; refuses to run if `logs/v1_1/exam_run.json` exists | `logs/v1_1/exam.json`, `logs/v1_1/exam_run.json` |
| 14. Examples | `python3 make_examples.py` — bundle 8 `CC BY 4.0` real exam examples (no KW51 `CC BY-NC-SA 4.0` data) | `../examples/` |
| 15. Web export | `python3 ../web/export_instrument.py` — fp16 browser weights for every member, examples, and 24 parity cases | `site/public/instruments/bridgesense/` |

The shipped v1.1 model is the `r7_ens3` ensemble (`r4_mix` + `r6_s2` + `r6_s3`): `537,432` parameters (3 × `179,144`; `2.0 MB` float32 `.npz`; `1.0 MB` float16 browser payload). Zero real labelled bridge examples enter training.

Committed run logs for v1.1 are in `logs/v1_1/`: per-rung training logs and metadata (`train/`), the twin test (`twin_dev.json`, `cal_p3.json`), the ratchet (`ratchet.log`, `ratchet_log.json`), calibration (`calibration.json`), the exam (`exam.json`) and the dated exam-run record (`exam_run.json`). v1.0's logs were never committed.
