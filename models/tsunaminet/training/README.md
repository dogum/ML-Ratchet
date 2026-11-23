# TsunamiNet training scripts (as run, 22 Nov 2026)

Order of the build (each writes to the work folder, see `_paths.py`):

| Step | Script | What |
|---|---|---|
| data | `fetch_bathy.py`, `fetch_meta.py`, `fetch_dart.py`, `events_scan.py`, `events.py` | ETOPO 2022 Pacific grid, NGDC/USGS events, NDBC DART files, coverage, the frozen split (`splits.json`) |
| physics | `okada.py`, `swe.py`, `check_okada.py`, `check_swe.py`, `check_gf_resolution.py` | Okada and linear SWE with their checks |
| Green's functions | `gf_bank.py` (`GF_STRIDE=2` = 8', `1` = 4') | one receiver-started run per DART site, projected on 1,317 unit sources |
| noise | `dart.py`, `noise_bank.py` | causal detiding + 3-h high-pass; 955 real quiet DART windows |
| simulator | `tsim.py`, `features.py`, `simbank.py`, `run_banks.sh` | random sources, superposition, real noise, network features; banks p0 and p1 |
| twin test | `twin.py`, `plot_twin.py` | dev events re-simulated from USGS finite faults -> physics p1 (`logs/physics_versions.json`) |
| network | `model.py`, `train.py` | CNN set encoder + RealNVP flow (`--anchor` adds the NNLS features) |
| ratchet | `ratchet.py`, `dev_eval.py`, `evallib.py`, `baselines.py` | frozen-set skill + dev gate; log in `logs/ratchet.json` |
| calibration | `calibrate.py` | widen-only factors on dev -> `logs/calibration.json` |
| exam | `exam.py`, `summarize_exam.py` | run once -> `logs/exam_rows.json`, `logs/exam_summary.json` |
| release | `export.py`, `export_gf.py`, `make_examples.py`, `build_page_data.py`, `build_page.py`, `hero_run.py` | numpy weights, compact Green's functions, examples, lab page |

Rungs as run: `train.py r1 train_p0 6000`; `train.py r2 train_p1 6000`; `train.py r4 train_p0:20000,train_p1:20000 6000`;
`train.py r3 train_p1 6000 - --anchor` (released); `train.py r5 train_p1 3000 r3 --anchor --polish`.
