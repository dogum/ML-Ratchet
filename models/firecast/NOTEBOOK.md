# FireCast lab notebook

Historical record. Entries are dated; earlier entries are kept as written (paths updated only where marked).

## 2026-11-19 — build notes (firecast-status.md, as written in the cloud sandbox)

Next-day wildfire spread: from today's observed fire (FEDS burned area and fire line, VIIRS S-NPP hot spots) and
gridMET weather for today and tomorrow, a small U-Net gives every 90 m cell the probability that it has burned by the
next afternoon overpass. It was trained only on FireSim simulations and graded once on 3,030 real exam fires.
Wildfire Phase 2, project 1.

- Page: the network runs live in the browser; explorer of 12 seeded exam fire-days, weather lab, exam, ratchet, twin
  test (now `lab/index.html`).
- Release `firecast-v1.0.zip`: numpy + scipy inference (`firecast.py`, feature code generated from the training
  source) and weights with the dev calibration; `demo.py` on three exam fire-days (one small, one medium, one large;
  about 1.5 s; reproduces the exam scores exactly); README, model card; all scripts (bank, training, evaluation,
  calibration, twin, exam), configs, physics versions, `ratchet.json`, result JSONs.

### Honest headline
- **Exam (3,030 fires, 25,105 growth fire-days):** AP **0.323** [0.318, 0.328] and F1 **0.239** [0.234, 0.244]. The
  best baseline (yesterday's growth from the fire line) scores 0.252 and 0.184.
- FireCast wins in every period: early 0.318 / 0.232, main 0.327 / 0.249, recent 2022–24 0.318 / 0.222. It also wins
  in all 9 regions and all 5 size bins. Its per-fire-day AP win rate against the best baseline is 71 %.
- **Against the calibrated physics replay** (48 exam replay fires, 627 days): FireCast 0.351 / 0.265, replay ensemble
  0.165 / 0.216, deterministic replay 0.150 / 0.217. The replay over-predicts new area 3.5×.
- **Where it loses:** persistence still has the best total-area Dice (0.964 against 0.922). Probabilities above about
  0.35 are over-confident: a 0.9 forecast burned 61 % of the time.

### Recipe as built
- **Physics:** FireSim (Phase 1), wrapped in `src/simbank.py` without editing it.
  - Multi-day fires (2–8 days, 12 h segments) on real LANDFIRE landscapes from dev and guard-passed non-exam windows
    (30 % parametric), with real HRRR weather.
  - Randomised: ROS factor, per-fuel-group scatter, night slow-down, growing containment, spotting, wind and RH error,
    live fuel moisture.
  - p1 adds a fire-age ROS factor fitted to the p0 twin test.
- **Observation noise:** `firesim.observe` gives S-NPP VIIRS at real overpass times with clouds. A FEDS emulator fitted
  on dev turns that into perimeters and fire lines (dev Dice 0.95, fire-line F1 0.62).
- **Weather:** the network sees gridMET-like daily features. In simulation they come from HRRR through a dev-fitted
  operator with residual noise (gridMET exists for every exam fire; HRRR does not).
- **Network:** U-Net with 374k weights. Inputs: 22 map channels plus 22 scalars, joined by FiLM-style biases. BCE loss
  on the 10 km zone. Training rungs take about 30 min on one shared core.
- **Checks:** numpy == torch to 2.5e-6; JS == numpy to 2.1e-6 with f16 weights; B's dev baselines reproduced exactly
  with this pipeline (fire-line baseline dev AP 0.252).
- **Dev gate:** 40 dev fires, 535 growth fire-days.
- **Calibration:** dev only, widen-only. T = 1.44, b = −1.23, threshold 0.10. Dev F1 0.227, leave-one-fire-out 0.228.

### Rungs
Rule: keep a rung if sim AP on the current frozen set is more than 0.01 above the incumbent **and** dev AP is at least
0.95 × the incumbent's. Frozen set p1: 149 fires, 562 fire-days. r0 was re-scored on it.

| Rung | Change | Sim AP p1 | Dev AP | Dev F1 @0.5 | Kept |
|---|---|---|---|---|---|
| r0 | p0 bank (668 fires), 1,800 steps | 0.573 | 0.303 | 0.158 | base |
| r1 | p1 physics (fire-age ROS from twin p0), init r0, p1 bank (415 fires) | 0.582 | 0.325 | 0.176 | no (+0.009) |
| r2 | init r0, p0+p1 banks (1,083 fires, 4,402 fire-days) | 0.587 | 0.316 | 0.181 | **yes, released** |
| r3 | r2 + 1,000 steps | 0.588 | 0.327 | 0.188 | no (+0.001) |

- **Twin test p0** (470 dev days × 3 members): median log₁₀(sim/real new area) +0.16. By fire-day: k0 −0.44, k1–2
  −0.18, k3–5 +0.04, k6–10 +0.33, k>10 +0.35. 34 % of real days fell below all three members. This led to p1.
- **Twin test p1:** stopped at 126 days because the process needed 1.3–1.45 GB, over the limit. On the same 126 days:
  k0 went from −0.82 to +0.20 and k>10 from +0.44 to +0.01. Days 1–10 are now about −0.34.

### Limits
- Over-confident above p ≈ 0.35. Widen-only calibration (2 numbers) cannot reshape the curve.
- The yes/no threshold (0.10) sits at the edge of the searched grid.
- It forecasts the FEDS/VIIRS picture of the fire (375 m, 12-hourly), not ground truth.
- LANDFIRE mode adds only +0.002–0.004 AP, and only 48 exam fires have packaged landscapes.
- Sim AP is 0.59 against real dev AP 0.32. The simulator has no suppression tactics beyond growing containment, no
  plume-driven runs and no fuel change since the LANDFIRE year.
- Only three trained rungs fit on 2 shared cores. Margins of ±0.01 are at noise level.

### Next rungs worth trying
- **Reliability shape:** a monotone (isotonic or beta) calibration on dev, which would need a rule change from
  widen-only. Or train with a distribution-matching term: simulated growth is more predictable than real growth, so
  the network is too sure in the core of the forecast.
- **Finish twin p1 with lower memory** (stream the dev cache) and fit a p2 age factor for days 1–10. Then re-score r1,
  which was the best on dev.
- **Wind-event nights and suppression:** condition the night slow-down on gridMET night wind and RH, and add a
  suppression-effort prior by region and size. Then run the twin test by wind tercile.
- **Lower threshold grid** (0.03–0.10) and a 12 h product (AM overpass target).
- **Bigger bank:** the p1 bank stopped at 35 of 72 shards.

### Sandbox gotchas
- `twin.py`, `calibrate.py` and the exam hold the 171 MB dev cache plus logits, and peak around 1.2–1.35 GB. The exam
  script peaked transiently at about 1.35 GB on large fires.
- `exam_replay.py` needed a crop-size guard (500k cells). The first attempt was killed at 1.7 GB and restarted from
  scratch (it is deterministic).
- `calibrate.py`'s leave-one-fire-out loop takes about 45 min per mode on a shared core.
- `setsid nohup` returns the wrapper's PID. Record the python PID from `pgrep` and check its cwd before killing
  anything.

## 2026-11-19 — rule notes from the build handoff (section 6, the parts about FireCast)
- Read one exam fire's landscape array shape outside the exam script, to size a memory guard.
- Its first physics-replay baseline attempt was killed after 2 fires for memory and re-run from scratch,
  deterministically. The killed attempt is kept.
- The exam process peaked at about 1.35 GB and was allowed to finish rather than abort a once-only run.
- All four Phase 2 agents were interrupted once (an interface stall) and resumed from their STATUS notes; each checked
  that no exam had started before resuming.
- Dev twins and dev replays fed physics calibration and the gates, so the gates are partly in-sample. The exams were
  untouched.

## 2026-11-19 — upstream candidates for FireSim (status doc and handoff section 4, FireCast's)
1. **A fire-age (days since detection) ROS factor:** dev twins show early under-growth and late over-growth (p1:
   `phys/p1.json` `age_ros`, ×2.6 on day 0 down to ×0.55 after day 10). Unify with FireDetective's persistence knobs
   into one documented persistence module.
2. **A FEDS-like perimeter emulator** (`src/feds_emu.py`, fitted in `calib_feds.py`: buffer 190 m, closing 1,000 m,
   line distance 500 m): merge with FireDetective's into `firesim.observe`.
3. **A memory-bounded `replay.replay_day` ensemble.** Each member copies the prepared fields, so 12 members on crops
   over 500k cells need more than 1.2 GB.

## 2026-11-21 — imported into ML Ratchet

**What changed.**
- Layout kept as shipped (`training/{configs,phys,results,src,ratchet.json}`, every file name unchanged).
  `training/_paths.py` added; `src/common.py` imports it. Outputs (`ROOT`, `OUT`, `BANK`) now go to `WORK`
  (`models/firecast/work/` or `$ML_RATCHET_WORK/firecast`; the old `out/` is `WORK/out`); the shipped configs and
  physics files are read from `training/configs` and `training/phys` (`CONFIGS` in `train.py`, `PHYS` in
  `simbank.py`); FireSim and FireData come from `shared/fire`; the data paths (`fire/data/proc/...` in
  `calib_feds.py`, `realdays.py`, `evaluate.py`, `exam.py`, `exam_replay.py`) use the data root `$ML_RATCHET_FIRE`
  (default `shared/fire`), and `exam_replay.py` reads `shared/fire/checks/replay_calibration.json`. Scripts that read
  a result another script wrote (`calib_feds.json`, `calib_wx.json`, `cal_r2_*.json`, `eval_*.json`,
  `twin_compare_p0_p1.json`, the exam summary, `ratchet.json`) now take a re-run's copy under `WORK` if there is one,
  else the shipped copy in `training/results` (`shipped()` in `_paths.py`). `build_release_module.py` writes to
  `WORK/release/`; `webdata.py` packs the released weights unless a re-run exported new ones. `run_rung.sh` and
  `chain_bank.sh` `cd` to their own folder and put the numba cache and logs under `WORK`. No sandbox absolute path
  remains. Nothing was re-run.
- `firecast.py`: `FireCastNP()` takes the weight path as optional (default: the file next to `firecast.py`) and closes
  the npz. Nothing else; the generated module from `build_release_module.py` would still print the old signature.
- `demo.py`: reads `examples/demo_days.npz` (moved from the release root), puts its own folder on `sys.path` so it runs
  from any directory, closes the npz. Output unchanged; golden output `tests/golden/firecast.txt`.
- The shipped `README.md` is replaced by the generated one; its input description and scoring snippet moved into
  `MODEL_CARD.md` ("Inputs and scoring"). `MODEL_CARD.md`: repository paths, an import line, and new sections at the
  end (inputs, verdicts, rule notes, live instrument); the shipped text and numbers unchanged.
- `lab/index.html`: the published page with the host's wrapper replaced by a plain document head; the page itself is
  byte for byte as published (`lab/README.md`).
- New: `model.yaml`, `ratchet.yaml`, `examples/README.md`, `lab/README.md`, `web/export_instrument.py` and the site
  instrument (`site/src/instruments/firecast/`, `site/public/instruments/firecast/`).

**How to re-run** (from the scripts' usage lines; the order is inferred from what each script reads, not logged):
1. `calib_feds.py` (FEDS emulator on dev), `calib_wx.py` (HRRR → gridMET operator on dev), `cache_dev.py` (dev
   fire-day cache).
2. `chain_bank.sh p0 <n_val> <n_train>` (`simbank.py <ver> val|train <n_fires>`: frozen set and training bank),
   `run_rung.sh r0` (`train.py`, `evaluate.py`), `twin.py p0`; then the p1 bank, `run_rung.sh r1|r2|r3` and
   `ratchet.py decide <rung> <frozen_ver> "<change>"` after each.
3. `twin.py p1` (partial), `twin_compare.py p0 p1`.
4. `calibrate.py r2 F` and `calibrate.py r2 L`, `export.py r2 <out.npz> <cal json>`, `build_release_module.py`.
5. `exam.py run r2`, `exam_replay.py`, `exam.py summary r2`, `exam.py examples r2`, `make_demo.py r2`,
   `webdata.py r2`, `build_page.py r2`.

Every script needs the FireData tables (`mlratchet data fire`) and torch; banks and twins also need numba. Do not
re-run anything that reads exam fires (`exam.py`, `exam_replay.py`) except as a seen re-score.

**Checked against the shipped JSONs.** Every exam number in the status doc, card and handoff reproduces from
`exam_summary_r2.json`: means and fire-bootstrap intervals for all methods and periods, medians of total-area Dice and
log10 area ratio, the win rates (AP 71.5 %, F1 50.1 %), the 9 regions and 5 size bins (FireCast leads in each), the
replay subset (627 days: F 0.351 / 0.265, L 0.355 / 0.271, replay 0.150 / 0.217 and 0.165 / 0.216, fl_prev 0.273 /
0.210; 713 − 627 = 86 = 14 crop-guard skips + 72 without HRRR) and the reliability bins (0.90-0.95: forecast 0.923,
burned 0.611). The rung table reproduces from `ratchet.json` and `eval_r*.json`, the calibration from `cal_r2_F.json`
(T 1.440, b −1.231, threshold 0.10, dev F1 0.227, leave-one-fire-out 0.228, raw 0.181) and `cal_r2_L.json` (T 1.567,
b −1.204), the twin-p1 comparison from `twin_compare_p0_p1.json`, the FEDS emulator from `calib_feds.json` (Dice
0.950, line F1 0.62), and the dev fire-line baseline (0.252) from `eval_baselines.json`. The demo reproduces the exam
run's AP and F1 on all three fire-days exactly (0.296 / 0.185, 0.056 / 0.000, 0.575 / 0.430).

**Did not reproduce or cannot be checked from the shipped files.**
- The twin p0 split by fire-day (k0 −0.44, k1–2 −0.18, k3–5 +0.04, k6–10 +0.33, k>10 +0.35): `twin_p0.json` has the
  overall numbers only (470 days, median +0.16, 34 % below all members). The p0 numbers by fire-day in
  `twin_compare_p0_p1.json` are for the 126-day subset and differ (k0 −0.82 there).
- "LANDFIRE adds +0.002-0.004 AP": L − F is +0.004 on the 627 replay days and +0.001 on all 713 growth days of the
  replay fires; on dev L is below F for every rung (r2: 0.308 vs 0.316). The range is not in any shipped file.
- numpy == torch to 2.5e-6 and JS == numpy to 2.1e-6: no shipped result file records them (`export.py` prints the
  first; the page text says 2×10⁻⁶ for the second).
- Rung sizes: "p0 bank 668 fires" (2,710 fire-days in `ratchet.json`) and "p1 bank 415 fires, 35 of 72 shards" are
  only in the logs' text; the banks were not shipped.
- The weight count "374k": the file holds 373,122 numbers with BatchNorm folded; the torch model (`src/model.py`,
  conv without bias + BatchNorm) has 373,714 trainable weights, which rounds to 374k. Consistent.
- Only r0 has a score on the p0 frozen set; `ratchet.py`'s docstring says both sets are re-scored on every version.

**Found on import (recorded, not fixed).**
- The page template (`web/page_tpl.html`), the JavaScript port (`web/fcnet.js`) and the page text (`web/text.json`) that
  `build_page.py` reads were not shipped; the published page in `lab/` contains all three.
- `firecast_weights.npz` has no JSON `meta` entry (the model contract asks for one); left as shipped, the weights
  are not edited.
- `common.mem_mb()` reads a sandbox-specific cgroup file; elsewhere it returns NaN (memory logging only).
- The exam's baselines score F1 at 0.5 on their distance score while FireCast uses its dev-fitted 0.10; at 0.5
  FireCast's F1 is 0.058, below the growth baselines. Both are in the card; `model.yaml` says so on the F1 row.

**Instrument.** The TypeScript port (`site/src/instruments/firecast/net.ts`) rebuilds the 22 input maps from the raw
inputs of the three demo fire-days: VIIRS FRP rasters, exact Euclidean distance transforms (Felzenszwalb &
Huttenlocher) for the distances to the burn and to the nearest hot spot on the full grid, the fire-line distance and
along/across-wind offsets from scipy's nearest-feature offsets (shipped, since its tie-breaking picks the offset), and
the 22 scalars; then it runs the U-Net on the map window exam.py used for the page (≤ 240 × 240 cells). It reproduces
`firecast.py` (make_inputs with the same crop, forward with float16-roundtripped weights) on 35 reference cases (3
fire-days × recorded, wind speed, wind direction, humidity, fuel moisture, no hot spots, no previous perimeter,
combined) to 3.9e-6 relative, scalars and logits. The f16 weights move the window's probabilities by at most 3.1e-4
against the float32 release (Pole Creek). The window changes the scores slightly against the full-grid exam run
(Pole Creek AP 0.532 on the window, 0.575 on the full grid); the instrument shows both. Public assets 1.52 MB
(weights 0.75 MB, fire-days 0.28 MB, reference 0.48 MB). A blocked 3×3 convolution (four output channels per pass) keeps a run to about 0.7-0.8 s on the two smaller
windows and 1.9 s on Pole Creek's 240 × 240 (Node 24 on a laptop CPU; ops.conv2d took 2.3-6.7 s).

## 2026-11-21 — PR #7 review: off-grid detections were pulled onto the grid edge

`pixel_raster` clipped a footprint's lower bounds before enforcing its one-cell minimum, so a detection lying wholly
north or west of the grid (row 5, col -100, 375 m footprint on the 90 m grid) became a pixel in column 0. Fixed in
firecast.py, training/src/features.py and the site instrument: off-grid footprints are dropped before clipping.
On-grid detections give identical rasters; the golden demo and the 35 instrument parity cases are unchanged. The exam
was scored with the old code; real-day detections come from the fire's own FEDS record inside a crop with a 10 km
margin, so few if any can be affected, but this was not re-scored.

## 2026-11-22 — exam re-run on rebuilt data (reproduction check, nothing chosen)

The sandbox's exam loaded torch checkpoints that were not shipped. `training/src/rescore_np.py` runs `exam.py`
unchanged except that the net is the released numpy forward pass (`FireCastNP` on `firecast_weights.npz`, rung r2),
writing to `work/`. Inputs were the rebuilt FEDS-MTBS, gridMET and replay landscapes (21–22 Nov), and the current
features.py, so the off-grid footprint fix above is included.
- 3,030 fires, 25,105 fire-days, as shipped. about 9,700 s on one CPU (3.2 s per fire).
- All 616 metric numbers in `training/results/exam_summary_r2.json` match within 3.7e-5 (largest: one reliability
  bin's observed rate, 0.60976 vs 0.60980). AP 0.323, F1 0.239, fl_prev AP 0.252; AP win rate over fl_prev 71.45%
  and F1 win rate 50.09%, identical.
- 16 of 45 reliability-bin counts differ, by at most 24 cells in 1.3 billion (relative 1e-4): cells moving across a
  bin edge. These can come from numpy vs torch float order or from the footprint fix; not separated.
- Calibrated physics replay rebuilt for the 48 replay fires: 627 replay days, ensemble AP 0.165, deterministic 0.150,
  FireCast F 0.351 and L 0.355 on the same days, as shipped. Fire-days whose crop exceeds the memory cap are skipped
  inside the replay, as in the sandbox.
- My slip: a second copy of the run was started by mistake and appended two fires twice before it was stopped. The
  rows were byte-identical; the second copies were removed before scoring.

## 2026-11-22 — r2 scored on a FireSim physics p2 frozen set (nothing re-decided)

FireSim's solver is now p2 (U1, U9, U10, U11, U12 fixed; `shared/fire/STATUS.md`). FireCast's own "p1" is its
priors file. `simbank.py` now passes FireSim's solver from the priors file: absent means p1, the solver every
shipped bank used. `phys/p2.json` is `p1.json` plus `"physics": "p2"`.
- **p1 rebuild:** the val split was rebuilt with `simbank.py p1 val`. The shipped attempt count was not recorded;
  300 attempts (12 shards) keep 149 fires, as shipped. r2 (released numpy weights, `rescore_np.py`'s loader) scores
  AP F 0.586571 and L 0.595789 against the shipped 0.586571 and 0.595789 (difference 2e-8). F1 matches exactly.
- **p2 set:** the same 300 seeds keep 147 fires (564 fire-days). r2 scores AP F 0.5923 (+0.006) and L 0.5991
  (+0.003); F1 F 0.3752 (+0.005).
- **Why so little changes:** p2's line fixes do not apply here, because simbank's containment is a non-burnable
  mask. The weather fix does not apply either, because the weather is real HRRR. What remains is U9 and the
  cell-by-cell step.
- Only r2's weights were shipped, so r0, r1 and r3 were not re-scored and no decision was re-taken. r2 was trained
  on the p1 solver.

## 27 Nov 2026 — audit resolution

The two-reviewer audit (`docs/audits/`). The audit's grid-crop concern is an evaluation limit; outward padding barely moved AP on the packaged days. The card now states it; no code change.
