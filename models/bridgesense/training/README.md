# Training: BridgeSense

Run every script from `models/bridgesense/training/` (or from the repo root with the script path). Large downloads, simulation banks, checkpoints, and scratch logs go to `../work/` (git-ignored) or `$ML_RATCHET_WORK/bridgesense/`.

| Step | Script | Writes |
|---|---|---|
| 1. Fetch public data | `python3 fetch_data.py` (Zenodo KW51, Concrete Box-Girder + NASA POWER Hamburg `T2M`, HBTA 100 Hz) | `work/raw/` |
| 2. Physics | `simulator.py` — 2-D beam/arch/truss modal FEM + 1-D Fourier deck heat diffusion + thermoelastic $E(T)$ + freezing phase transition | — |
| 3. Check physics | `python3 physics_check.py` — 5 independent analytical / exact-eigen / KW51 FEM checks | `verify_physics.json` (committed) |
| 4. Freeze splits | `python3 split.py` — seeded KW51 warm-ref / Dev / Exam blocks; Box-Girder and HBTA held out 100% zero-shot | `../splits.json` (committed) |
| 5. Real OMA noise | `python3 noise.py` — extract healthy KW51 autumn residual windows + availability patterns | `work/noise_pool.npz` |
| 6. HBTA OMA modes | `python3 extract_hbta_modes.py` — UDS-anchored Welch PSD tracking across 50 recordings × 8 windows | `work/hbta_modes.npz` |
| 7. Banks (`p1`, `p2`) | `python3 simbank.py val1 2000 101 --physics p1`, `simbank.py train1 10000 1 --physics p1`, `simbank.py val2 2000 102 --physics p2`, `simbank.py train2 12000 2 --physics p2` | `work/bank/` |
| 8. Twin test (`p3`) | `python3 twin_dev.py` — inspect KW51 Dev healthy residuals, calibrate `p3` (`work/cal_p3.json`), then generate `val3` (2k) & `train3` (12k) with `--physics p3 --cal work/cal_p3.json` | `work/cal_p3.json`, `work/bank/` |
| 9. Train rungs | `python3 train.py --name r6_mix_no_anchor --bank train2,train3 --val val3 --steps 4500 --no-anchor` (and ablations `r1..r7`) | `work/ckpt/` |
| 10. Ratchet replay | `python3 ratchet.py --val val3 --release r6_mix_no_anchor` — frozen-bank score + KW51 Dev gate | `../ratchet.yaml` (committed), `work/ratchet_log.json` |
| 11. Calibrate | `python3 calibrate.py r6_mix_no_anchor` — Dev-only widen / temperature fit | `work/calibration.json` |
| 12. Export numpy | `python3 export.py r6_mix_no_anchor --verify` — pure-numpy weights + PyTorch parity check | `../bridgesense_weights.npz` |
| 13. Real exam | `python3 exam.py` — run once across KW51 Exam A, Concrete Box-Girder (zero-shot), and HBTA (zero-shot) vs baselines | `work/exam_results.json` |
| 14. Examples | `python3 make_examples.py` — bundle 8 `CC BY 4.0` real exam examples (no KW51 `CC BY-NC-SA 4.0` data) | `../examples/` |
| 15. Web export | `python3 ../web/export_instrument.py` — fp16 browser weights, examples, and 24 parity cases | `site/public/instruments/bridgesense/` |

The shipped dual-axis transformer (`r6_mix_no_anchor`) has `179,144` parameters (`670 KB` float32 `.npz`; `350 KB` float16 browser payload). Zero real labelled bridge examples enter training.
