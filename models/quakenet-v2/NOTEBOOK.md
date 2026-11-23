> **Lab notebook.** Build notes as recorded in the cloud sandbox (22 Nov 2026, B batch, project 1). Paths like
> `/home/claude/bb/quake2/` are that machine's; in the repository the scripts are in `training/` (see `_paths.py`).

# QuakeNet v2 — build notes

## Order of work (all on 2 shared cores, one thread per job)

1. `make_splits.py`: froze `splits.json` before anything else (v1.2's 40 dev / 1,734 exam events from its
   `real_meta.json`; 400-event exam subset, seed 7, depth strata). sha256 fb84d351….
2. `fetch2.py`: extra LH records from EarthScope for the 40 dev events and the 400 exam events, in a seeded station
   order written first to `station_order.json`, until each event had 10 usable records (3,523 new records; 436 of 440
   events reached 10). Same processing as v1.2's `fetch.py`. 51 requests answered "too many requests"; not retried.
3. `tt.py`: ak135 P/S tables (TauP) on v1.2's 27 depths × 0-100° every 0.25°; checks in `physics_check.json`.
4. `sim2.py`: multi-station events through v1.2's per-station chain, read by v1.2's frozen rung-7 network (torch
   checkpoint) → 256-number summary + 8 outputs per station. val1 (2,000 events), tr1 (10,000 events, 60,093 records),
   about 25 ms per station record once the Green's functions are cached (~370 MB in RAM).
5. `realnodes.py`, `prep.py real`: the same reading for the real records; 3 exam records with empty windows (v1.2
   outputs not finite) are dropped. Dev: 80 graphs (2 and all stations); exam: 2,656 graphs.
6. `prep.py`: graph samples with the anchor (`locate2.combine_v12`), `train2.py` (4,000 steps, batch 64, ~15 min per rung
   on a busy core), `ratchet2.py` (scores, dev gate, `rescore` re-takes every decision on a new frozen set).
7. `twin2.py` + `twin_compare.py`: dev twins from USGS moment tensors (40/40 available).
8. `physics_v2.py`, `physics_v3.py`: reading-level mismatches measured on the twins → val2/tr2, val3/tr3.
9. `calib_sigma2.py` (dev widening), `export2.py` (numpy, 6e-6), `js_parity.py` (JS vs Python), `exam2.py` (once,
   223 s), `export_web.py` + `build_page.py` (lab page), `make_examples.py` (seed 5, 2 shallow + 2 deep).

## Findings on dev (before the exam)

- Real P picks vs ak135 from the USGS origin: robust sd 8.9 s, 46% within 5 s. Simulated: 13.3 s (val1), twins 11.8 s.
  Real picks are tighter at every magnitude; the gap is largest for M 5.4-5.8 (11.9 vs 16.6 s).
- Paired real minus twin readings (399 records): distance +1.5° median (robust sd 3.8°); back-azimuth extra scatter
  (8.6° vs 5.5°); log10 depth +0.37 for shallow quakes, +0.04 for deep; Mw +0.057 / +0.019.
- Twin test, v1.2 combined: twins miss by 145 km (median), the real records of the same quakes by 407 km.
  r6: real 192 km, twins 166 km; shallow depth ratio real 1.15, twin 0.44 (the twins no longer carry the real bias the
  network now expects). Deep depth ratio 0.74 on both: an over-correction, seen before the exam.

## Dead ends

- v1.2's 256-number summary per station (r3) and P-wave cross-correlation lags (r4): no gain in simulation (+0.001 and
  +0.006 on val3).
- An AIC refinement of the P pick did not beat v1.2's STA/LTA pick on simulated records (within 5 s: 38% vs 42%).
- The travel-time locator with P and S picks from LH records is hopeless at 2 stations (mirror ambiguity; S picks often
  land on PP or surface waves) and only competitive with 5+.

## Gotchas

- `setsid` returns at once; the PID from `$!` is not the python process. Find your PIDs with `ps` before killing.
- A heredoc after `cmd1 || cmd2 <<EOF` feeds the last command only; an earlier `cat > file` waited on stdin.
- EarthScope rate limits ("Sent too many requests") at three parallel threads.
- scipy `least_squares` refuses a start with non-finite residuals: drop unreadable stations first.

## 2026-11-22 — imported into ML Ratchet

- Copied from `for_repo/models/quakenet-v2/` of `quakenet-v2-v1.0.zip` (22 Nov 2026) with `tests/golden/quakenet-v2.txt`. Every `.npz` loads with
  `allow_pickle=False` and holds no object arrays; inference imports only numpy and scipy.
- `demo.py` run from this folder reproduces the shipped golden output line for line (timing aside), so
  `tests/golden/quakenet-v2.txt` is the zip's own file, unchanged.
- No training script was re-run, and nothing was fetched.
- New model id `quakenet-v2`, beside `models/quakenet/` (v1.2), which is unchanged. The folder keeps its own copies
  of `quakenet.py`, `pipeline.py` and `quakenet_weights.npz` so it runs on its own; all three are byte for byte
  v1.2's files (git stores the 5.3 MB weights once).
- `order: 31`. README generated (the zip shipped none in `for_repo/`).
- Added from the zip's `results/`: `training/results/` (calib_picks, exam_results, export_check, js_parity,
  physics_check, sigma_factors, station_order, twin_compare). `splits.json` and `training/ratchet.json` were already in
  `for_repo/`.
- Not done: no site instrument. `web/` has the page's JS port (`qnet2.js`, JS == Python to 0.14 km) but no
  `export_instrument.py` in the contract of `site/src/instruments/README.md`.
- The last sandbox paths ported to `training/_paths.py` (`build_page.py`, `make_examples.py`, `make_splits.py`,
  `web/export_web.py`, the shell wrappers; `build_release.sh` writes its zip to `WORK/out/`).
- Training still needs v1.2's work folder and its rung-7 torch checkpoint `q7_zrt.pt`, which is not in the
  repository (`training/_paths.py`).

## 2026-11-22 — QuakeNet v1.2 torch checkpoints received (not committed)

- The owner supplied `quakenet-v1.2-torch-checkpoints.zip` (36.7 MB, sha256 `3f4b2517c42f4dde4912d86b92ad6ca5b160097287bf8db629f056f17672ac4b`),
  the seven v1.2 rung checkpoints from the build sandbox. Per the repository rules they are not committed.
- Contents (sha256): `q1_base.pt` 8d912901…, `q2_slowsw.pt` 86bf5af2…, `q3_coda.pt` 6e983a61…, `q4_world.pt` 6e555af0…,
  `q5_layers.pt` e9312210…, `q6_swdamp.pt` 9aa32b3c…, `q7_zrt.pt` d736c0a0acf5e926dec9f8f00902ca50feaffd4f123df5fd005f002a6223b254.
- To train v2, put `q7_zrt.pt` at `models/quakenet-v2/work/ckpt/q7_zrt.pt` (`training/prep.py` reads `H + 'ckpt/q7_zrt.pt'`).
  `prep.py` loads it with `torch.load(..., weights_only=False)`, which runs pickle code: check the sha256 above first.
- Open (owner's decision): publish the zip as a GitHub release asset and point `models/quakenet/training/_paths.py` at
  it, as the cloud project's follow-up list suggests. Not done.
- Nothing was loaded or run from the checkpoints in this import.

## 2026-11-22 — PR #13 review: `err_ellipse` is now the 68 % contour

- `quakenet2.py` returned `err_ellipse` semi-axes as the square roots of the covariance eigenvalues: the 1-sigma
  contour, which holds about 39 % of a bivariate Gaussian. The model card, the coverage scoring (`training/score2.py`,
  Mahalanobis² < 2.2789) and the lab page (axes × √2.2789) all mean the 68 % contour. The axes are now multiplied by
  √2.2789 = 1.51; `cov_km2` and `epi_sig_km` are unchanged.
- The demo prints the ellipse, so `tests/golden/quakenet-v2.txt` changed on those 8 numbers only (e.g. 1,421 × 680 km
  → 2,145 × 1,026 km). The published coverage (ellipse 62 % all stations, 65 % at 2) was always scored on the 68 %
  contour and stands.
- `training/tt.py` looked for its travel-time table in `models/quakenet-v2/data/`, which neither the work layout nor
  the release uses, so a rebuild's `TT.tt()` calls raised FileNotFoundError. It now builds into `WORK/data/` via
  `_paths.H` and reads that copy if present, else the shipped `tt_ak135.npz` (P at 40°, 33 km: 451.56 s, the same as
  `quakenet2.tt`).
- `training/export2.py` and `js_parity.py` now also look for `quakenet2.py` in the model folder (they searched only `WORK/src` and `WORK/release`).
