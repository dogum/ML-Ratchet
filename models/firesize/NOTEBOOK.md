# FireSize lab notebook

Historical record. Entries are dated; earlier entries are kept as written (links and sandbox paths removed or marked
where noted).

## 2026-11-19 — build notes (firesize-status.md, as written in the cloud sandbox)

Paths in this entry are the sandbox's: `results/` is now `training/results/`, the scripts are in `training/src/`,
`demo_examples.csv` is in `examples/`.

FireSize starts from a wildfire ignition record: location, date, cause, agency and discovery hour. From that record, a coarse LANDFIRE summary around the point and two days of gridMET weather, a small MLP ensemble predicts P(escape ≥ 100 acres), P(≥ 1,000 acres) and the final-size distribution over 12 FPA-FOD classes. It was trained only on FireSim ignitions with a dev-calibrated initial-attack model, and graded once on 655,869 real FPA-FOD exam ignitions (2012–2020). This is Wildfire Phase 2, project 2.

- **Page:** a private page [link removed on import; the page itself is `lab/index.html`]
  - The network runs in the browser on 8 exam ignitions, with a weather and record lab. The Rothermel ROS inputs are recomputed live.
  - It also has a containment-race toy, an exam map of 1,602 ignitions, the exam tables, regional AUCs and the ratchet.
- **Release:** `firesize-v1.0.zip` [sandbox output path removed on import] (0.47 MB):
  - numpy inference and the feature builder (`firesize.py`), the weights, the ROS lookup table;
  - `demo.py` on 8 exam ignitions (about 1 s; it asserts equality with the exam run);
  - the model card, every script, and the ratchet, calibration, exam and post-exam JSONs.

### Honest headline
- **Exam, all 655,869 ignitions:** escape AUC **0.727**, AP **0.071**, Brier 0.0273; size MAE 0.642 log10, ρ 0.286; large-fire AUC **0.815**.
  - Climatology (region × cause): 0.688 / 0.061 / 0.0256, MAE 0.505.
  - ERC/BI/wind: 0.634 / 0.042.
  - GBM fitted on dev real labels: 0.777 / 0.099 / 0.0255, MAE 0.510, large-fire AUC 0.803.
- **Recent 2020 (a held-out season):** AUC 0.740, AP 0.071. **Main 2012–19:** AUC 0.725.
- **Escape ranking:** better than climatology (+0.039 AUC, 95 % bootstrap +0.035 to +0.044) and ERC (+0.093). Worse than the real-label GBM (−0.050, −0.053 to −0.047).
- **Large fires:** the best of all methods, GBM included.
- **Within regions:** better than the GBM in CA, NW and RM. Far worse in the SE (0.566 vs 0.704) and SP (0.715 vs 0.808).
- **Fails on calibration and size:**
  - Mean P(escape) is 0.055 against a real rate of 0.027, so the Brier score is worse than prevalence (0.0260).
  - The size point estimate is worse than climatology.
  - The 80 % size interval covers the real class 90 % of the time.

### Recipe as built
- **Features (same code for sim and real).**
  - The 250 m national LF2016 grids (`natgrid.py`): fuel class at the point, class fractions in 1.25 km and 3.25 km boxes, CC, CH, slope.
  - gridMET for the discovery day and the day before (11 variables each).
  - Rothermel head ROS from a `behave()` lookup table under the afternoon weather (rung r7).
  - Cause, federal agency, hour, day of year, region. 75 columns in all.
  - 655k landscape lookups take about 2 min, so no exam subsetting was needed.
- **Ignition prior (`prior.py`).**
  - 2,400 guard-checked 18 km windows (dev ignition location + N(0, 20 km)), 8 ignitions each.
  - Each ignition borrows the attributes of a dev ignition within 150 km, with the date ± 7 days.
  - 304 windows are frozen for validation. 911 dev twins come from the calib half of dev, with class weights for the escapes the guard skipped.
- **Sim (`sim.py`).**
  - FireSim free burn on the real 90 m window for 96 h: analytic sub-cell ellipse plus MTT grid spread, crown fire on, no spotting.
  - Nelson dead moisture, GSI live moisture.
  - Draws: pre-discovery time LN(0, 1) h, ROS multiplier logU[0.2, 1.5].
  - About 1 s per simulation. The v1 bank has 8.5k training ignitions plus twins (18 MB).
- **Physics versions.**
  - v0: LANDFIRE as shipped.
  - v1: agriculture (93) burns as GR1, and a non-burnable ignition cell is moved to the nearest burnable cell ≤ 180 m away.
- **Suppression (`supp.py`, `calib.py`).**
  - A Fried & Fried race on A(t): escalating line against a share of the ellipse perimeter.
  - Two attack modes (confine/monitor at 2 % of the line rate), access and slope terms, a federal line-rate term.
  - Reporting floors: 0.1 acre, regional 1-acre shares, a 0.01-acre share.
  - Fitted by Powell on the twin likelihood of the real size class, plus an escape Bernoulli term.
  - v1rep2 twin NLL: 3.60 with literature values, 2.64 fitted. 2-fold held-out: 2.66 / 2.73, against 3.56 / 3.68 for literature values.
- **Network.**
  - 3 × MLP (75 → 128 → 128 → 12), soft-label cross-entropy with importance weights, early stopping on held-out windows. A few minutes per rung on 1 thread.
  - JS forward = numpy (P(escape) identical to 4 decimals on the demo ignitions; ROS recomputation reproduces the exam features).
- **Widen-only:** T = 1.459 fitted on the dev calib half (size-class NLL 1.881 → 1.827; gate half 1.876 → 1.822; 80 % coverage 0.84 → 0.90).
- **Dev gate:** the dev gate half (4,429 ignitions), never used in calibration. Climatology on it scores AUC 0.704, AP 0.106, MAE 0.477.

### Rungs
Rule: frozen-sim RPSS > incumbent + 0.01, dev-gate escape AUC ≥ incumbent − 0.01, and gate MAE ≤ 1.05 × incumbent. Every rung is re-scored on the latest frozen set (v1|v1rep2); the column in brackets is the rung's own frozen version.

| Rung | Change | Sim RPSS latest (own) | Gate AUC | Gate AP | Gate MAE | Kept |
|---|---|---|---|---|---|---|
| r0 | FireSim as shipped, literature suppression | −0.256 (0.220) | 0.697 | 0.076 | 0.725 | base |
| r1 | suppression calibrated on twins (v0) | −0.292 (0.199) | 0.675 | 0.071 | 0.694 | no |
| r2 | physics v1 (agriculture burns, NB ignition relocation) + recalibration | −0.124 (0.267) | 0.757 | 0.081 | 0.744 | **yes** |
| r3 | regional response offsets + ROS centres per fuel group | 0.038 (0.294) | 0.655 | 0.052 | 0.651 | no (gate AUC) |
| r3b | regional reporting floors + escape term | 0.213 (0.308) | 0.634 | 0.047 | 0.633 | no (gate AUC) |
| r4 | r2 + Rothermel ROS features | −0.139 (0.266) | 0.760 | 0.087 | 0.730 | no (margin) |
| r5 | two-mode IA + federal line-rate term + floors (v1rep2) | 0.265 | 0.766 | 0.085 | 0.618 | **yes** |
| r6 | r5 on the full 8.5k bank | 0.278 | 0.746 | 0.079 | 0.626 | no (gate AUC) |
| r7 | r6 + Rothermel ROS features | 0.280 | 0.756 | 0.082 | 0.627 | **yes → v1.0** |

#### Notes on the rungs
- **r2 is the biggest physical fix.** FireSim treats LANDFIRE agriculture (93) as non-burnable, but thousands of FPA-FOD fires start in fields. Ignitions on non-burnable cells never grew at all.
  - Twin likelihood: v1 3.356 vs v0 3.378. Gate AUC: 0.675 → 0.757.
- **r3 and r3b improved the simulator's fit but lost escape ranking.** Twin NLL fell, and the sim RPSS on their own sets rose. The gate refused them: the regional offsets soak up variance that the landscape and weather should explain.
- **Twin test (`twin.py`, dev calib half, 911 twins):**

  | Model | Net AUC vs real | Net AUC vs twin outcome | Twin sim AUC vs real | Escape rate net / twin / real |
  |---|---|---|---|---|
  | r0 | 0.690 | 0.912 | 0.596 | 0.062 / 0.047 / 0.026 |
  | r2 | 0.714 | 0.882 | 0.604 | 0.029 / 0.028 / 0.026 |
  | r5 | 0.722 | 0.892 | 0.671 | 0.023 / 0.024 / 0.026 |
  | v1.0 (r7 + T) | 0.734 | 0.906 | 0.671 | **0.050** / 0.024 / 0.026 |

  - The network reads a real ignition the way the simulator does: AUC about 0.9 against the twin's own simulated outcome. The ceiling is the simulator's own skill on real outcomes, 0.67.
  - The 2× over-prediction comes from the widen step, not from the simulator or the network. Before widening, r5 and r7 run about 12–15 % low on dev (r5 0.023 vs 0.026 on twins; r7 0.025 vs 0.029 on the gate).
- **Rule note.** After the exam, it looks likely (not computed) that T = 1 would have given a better exam Brier. This was visible on dev before the exam (gate Brier 0.0274 → 0.0285), and I kept the release as frozen. It is listed as the first next rung.

### Exam (run once)

| Method | AUC all | AP | Brier | AUC 2020 | Size MAE | Size ρ | Large AUC |
|---|---|---|---|---|---|---|---|
| **FireSize** | **0.727** | 0.071 | 0.0273 | 0.740 | 0.642 | 0.286 | **0.815** |
| GBM (dev real labels) | 0.777 | 0.099 | 0.0255 | | 0.510 | 0.533 | 0.803 |
| Climatology | 0.688 | 0.061 | 0.0256 | | 0.505 | 0.369 | 0.767 |
| ERC/BI/wind | 0.634 | 0.042 | 0.0259 | | 0.645 | 0.201 | 0.748 |
| Prevalence / median | 0.500 | 0.027 | 0.0260 | | 0.699 | – | 0.500 |

- **By region,** escape AUC (FireSize / GBM / climatology):

  | Region | FireSize | GBM | Climatology |
  |---|---|---|---|
  | CA | 0.739 | 0.690 | 0.570 |
  | NW | 0.776 | 0.735 | 0.594 |
  | NR | 0.681 | 0.714 | 0.612 |
  | GB | 0.639 | 0.655 | 0.562 |
  | SW | 0.757 | 0.782 | 0.724 |
  | RM | 0.748 | 0.703 | 0.569 |
  | SP | 0.715 | 0.808 | 0.563 |
  | SE | 0.566 | 0.704 | 0.539 |
  | NE/MW | 0.782 | 0.821 | 0.505 |

  The baselines were re-run with B's code in `post_exam.py` and reproduce B's stored AUCs exactly.
- **By era:** 2012–16 AUC 0.708; 2017–19 0.750. MTBS-matched fires (3,818, 99 % escapes): size bias −3.3 log10, so FireSize badly under-sizes the big MTBS fires.
- **Size:** bias +0.22 log10 overall. NW +0.87 and GB +1.07 (over-sized), SE −0.21.

### Limits
- **Escape probabilities are about 2× too high,** because the widen-only temperature was fitted to the size-class likelihood.
- **Size.** It loses to climatology on FPA-FOD's rounded small sizes, and it under-sizes the largest fires: there is no multi-day weather after 96 h and no spotting.
- **Southeast (0.57).** Debris and agricultural burning, small parcels and dense fast suppression are not in the simulator. LANDFIRE 2016 fuels are stale for 2020.
- **Suppression parameters at their bounds:** response 0.1 h, line-rate multiplier 30×, federal term e^−3, confine share 0.5, ROS sd 0.3. The race is an effective model; widen the bounds or change its form.
- **Coarse landscape:** 250 m features by design (the simulator itself used 90 m windows).

### Next rungs worth trying
1. **Escape-aware calibration.** Fit the temperature on the escape Bernoulli term, or keep T = 1 for P(escape) and widen only the size tail. On dev twins the T = 1 escape rate already matches (0.023 vs 0.026).
2. **SE and SP physics.** Burnable agricultural and debris-burn fuel states by season, and a WUI/parcel suppression term from dev twins in those regions.
3. **Multi-day containment.** Extend the sim past 96 h with real multi-day gridMET for fires that are not contained. This would fix the −3.3 log10 under-sizing of MTBS fires and the large-fire tail.
4. **Rounded reporting.** Model FPA-FOD's reporting granularity (0.1, 1 acre) inside the simulator's reporting floor so the network learns the point masses. A smooth median cannot hit them.
5. **More twins per region.** The guard skipped 45 % of escaped dev twins; LFPS fetches of guard-passing windows could recover some.
6. **Real-data features the GBM uses that FireSize lacks:** owner class and lat/lon. They help the GBM between regions. They could only enter through the simulator, not as learned real-label effects.

### Sandbox gotchas
- The previous session was interrupted at about 10:23. On resume, no jobs were running and the exam had not started. The v1 full train bank had stopped at 85 chunks (8.5k ignitions); r6 and r7 were trained on that and it was not resumed.
- `exam.py` writes an audit line to `results/exam.started` at every start. It has exactly one line.
- The exam process held about 0.7 GB at its start (not profiled to peak); the cgroup stayed under 4.2 GB with four agents running. The 250 m national grids are 216 MB each, loaded one at a time.
- Disk: about 0.5 GB in total (90 m windows 280 MB, national grids 154 MB, bank 18 MB).

## 2026-11-19 — per-model state from the build handoff (section 5, FireSize)

**FireSize v1.0: best at ranking large fires, loses overall to a dev-fitted GBM.** Exam: 655,869 ignitions.

| Method | Escape AUC | Large-fire AUC |
|---|---|---|
| FireSize | 0.727 | **0.815** |
| GBM on dev real labels | **0.777** | 0.803 |
| Climatology | 0.688 | |
| ERC/BI/wind | 0.634 | |

- **By region:** beats the GBM in CA, NW and RM; far worse in SE (0.566 vs 0.704) and SP.
- **Calibration fails:** mean P(escape) is 0.055 against a real rate of 0.027.
- **Size:** biased +0.22 log10; MTBS-size fires under-sized by 3.3 log10.
- **Twin test:** the network matches the simulator at AUC about 0.91, but the simulator matches reality at only 0.67.
  The ceiling is the physics and suppression model, not the network.
- **Next:** (1) keep T = 1 for P(escape) and widen only the size tail; (2) SE/SP physics: agricultural and debris-burn
  fuel states, a WUI suppression term; (3) containment beyond 96 h with multi-day weather; (4) model FPA-FOD's rounded
  reported sizes; (5) recover the escaped dev twins the guard skipped; (6) widen the suppression-parameter bounds.

## 2026-11-19 — rule notes from the build handoff (section 6, the parts about FireSize)

- FireSize read FPA-FOD records straight from parquet, filtered by its split, because firedata has no FOD accessor.
- Several dev-calibrated suppression parameters sit at their bounds.
- **Interruption:** all four Phase 2 agents were interrupted once (an interface stall) and resumed from their STATUS
  notes. Each checked that no exam had started before resuming.
- **Dev data in the gates:** dev twins and dev replays fed physics calibration *and* the gates in several models, so
  the gates are partly in-sample. The exams were untouched. (Import note: for FireSize the suppression fit, the
  importance weights and the temperature used the dev calib half; the ratchet gate is the other half.)

## 2026-11-19 — upstream candidates for FireSim (handoff section 4, FireSize's)

1. **Agriculture fuel.** LANDFIRE 93 should burn, e.g. as GR1 by season. It was FireSize's biggest single gain (r2).
   Also move ignitions that land on non-burnable cells to the nearest burnable cell (≤ 180 m).
2. **A sub-cell analytic ellipse** for fires under 0.81 ha (`sim.py`).
3. **A containment race module**: Fried & Fried style, with two attack modes and reporting floors. It is in
   `training/src/supp.py` and `training/results/calibration/calib_v1rep2.json`.

## 2026-11-21 — imported into ML Ratchet

**What changed.**
- Layout follows the Watershed imports: the release zip's `training/` is now `training/src/` (every file name
  unchanged), its `results/` is `training/results/` (calibration fits in `training/results/calibration/`), and
  `results/ratchet.json` and `results/rungs.json`, which lived in the workspace root in the sandbox, are
  `training/ratchet.json` and `training/rungs.json`. `demo_examples.csv` is in `examples/`; the page
  (`firesize_page.html`) is `lab/index.html`, byte for byte.
- `training/_paths.py` added; `src/common.py` imports it. Outputs (`ROOT`, `DATA`) now go to `WORK`
  (`models/firesize/work/` or `$ML_RATCHET_WORK/firesize`), the platform's processed data (`FP`) is read from
  `$ML_RATCHET_FIRE/data/proc` (default `shared/fire`), and `shared/fire` and `shared/fire/firesim` are put on
  `sys.path` for `firedata` and `firesim`. Three more path edits: `ratchet.py` reads `rungs.json` through `latest()`
  (the shipped copy unless a re-run wrote one), `train.theta_of` falls back to the shipped calibration fit when
  `WORK/data/calib_<name>.json` does not exist, and `post_exam.py` imports `baselines_firesize` from
  `shared/fire/data/scripts`. The five shell scripts `cd` to their own folder and log to `WORK/logs`. No sandbox
  absolute path remains. Nothing was re-run.
- `firesize.py` and the two weight files: unchanged.
- `demo.py`: reads the CSV with the `csv` module instead of pandas (the demo now needs numpy only), prints the class
  probabilities as plain numbers (numpy 2 printed `np.float64(...)`), and adds one line: the real size class is inside
  the 80 % range for 5 of the 8 ignitions (the exam's definition; 90 % on the whole exam). The table and the assert
  are the shipped demo's, digit for digit. Golden output: `tests/golden/firesize.txt`.
- `MODEL_CARD.md`: an import paragraph, a note on the 2020 GBM score, a verdict table, rule notes and the new file
  layout. No number changed.
- New: `model.yaml`, `ratchet.yaml`, `examples/README.md`, `examples/patches.json` (the 13 × 13 fuel patches, copied
  from the page's data block), `lab/README.md`, `web/export_instrument.py` and the site instrument
  (`site/src/instruments/firesize/`, `site/public/instruments/firesize/`).
- Page choice: the published page is the zip's `firesize_page.html` inside a host wrapper (a 531-byte head and a
  closing `</body></html>`); the zip's file is the cleaner of the two and is the one kept.

**Checked against the shipped files.**
- `firesize_weights.npz` has sha256 72289217f20bcb44475ae19f6527c57b427397822d4bbdce234957508f533902, the
  `model_sha256` that `exam.py` recorded: the released file is the one the exam ran.
- The demo reproduces the exam's P(escape) for all 8 ignitions to 1e-4 (its assert), with the repository's numpy.
- Every exam number in the card, status doc and handoff reproduces from `exam.json` and `post_exam.json`: AUC, AP,
  Brier, size MAE, ρ, large-fire AUC, the periods, eras, regions, MTBS subset, bias, coverage, the bootstrap
  differences and the reliability bins. The rung table reproduces from `ratchet.json`, the twin table from
  `twin_*.json`, the widening from `widen.json`, the dev-gate climatology from `dev_gate_baselines.json`, and the
  twin NLLs (3.60 → 2.64; held out 2.66 / 2.73 against 3.56 / 3.68; v1 3.356 vs v0 3.378) from the calibration fits.
- Recomputing the four Rothermel columns of the 8 demo ignitions from `examples/patches.json` and the CSV weather with
  `firesize.py`'s own functions reproduces the exam features to 1.1e-7 (they are stored as float32), and the 13 × 13
  fuel-class fractions match `f13_*` to 3e-8. So the page's patches are the exam's.

**Found on import (recorded, nothing fixed).**
- The card and status doc compare the 2020 AUC (0.740) with the GBM's all-years 0.777 / 0.099. The GBM's own 2020
  scores in `exam.json` are 0.789 / 0.092 (climatology 0.711 / 0.067), so the 2020 gap is −0.048, not −0.037.
  `model.yaml` uses the 2020 numbers.
- The card lists four suppression parameters at their bounds and the status doc five. In `calib_v1rep2.json` 10 of
  the 31 sit within 0.5 % of a bound: `mu_resp` (0.1 h), `mu_resp_fed` (−1.5), `sd_resp` (1.5), `r0_mult` (30×),
  `sd_r0` (1.2), `ros_sd` (0.3), `mu_resp_nat` (−1.0), `rep_SW` (0), `r0_fed` (−3.0) and `p_man` (0.5). `ros_sd` is
  exactly on its bound: its raw value is written as `-Infinity`, which Python's json reads but strict JSON parsers
  reject.
- r7 passed the gate AUC test by 0.0004 (0.7561 against a threshold of 0.7557) and r6 failed it; no interval was
  computed for any rung difference, and the shipped 95 % bootstrap of the exam AUC differences uses only 100
  resamples.
- `widen.json` reports a gate size MAE of 0.648, from the widened median; the card's 0.627 is r7's MAE with the
  T = 1 median, which the release uses. Both are right for what they measure.
- Not shipped, so these steps cannot be re-run from the repository alone: `results/exam_preds.parquet` (needed by
  `post_exam.py` and `build_release.py`), the exam audit file `results/exam.started` (the status doc says it has one
  line), `web/template.html` (needed by `build_page.py`), `web/conus_fuel_q.png`, the per-rung `models/<rung>.npz`
  and `.json`, the simulated banks and the feature parquets. The weights of the released rung are
  `firesize_weights.npz` (r7 + T).
- The page renames two records: "Fy2012-Wilcox-076" is shown as "Wilcox-076" and "151" as "Fire 151"
  (`build_page.clean`). The dev-fitted ERC coupling the page's lab uses (BI +0.821, FM100 −0.175, FM1000 −0.197 per
  ERC unit) is in no results file; it is copied into `web/export_instrument.py` from the page.
- The 8 demo ignitions were drawn after the exam with the exam predictions merged in (`build_release.py`); the draw
  is seeded within size groups and chooses nothing.

**Instrument.** The TypeScript port (`site/src/instruments/firesize/net.ts`) reproduces `firesize.py` on 104
reference cases (8 ignitions × 13 what-ifs: wind, humidity, temperature, ERC with its coupled BI and fuel moistures,
discovery hour, cause, agency, region, and a combination) to 1.1e-14 relative, including the Rothermel inputs
recomputed from the fuel patches. Weights are float16 (P(escape) moves by at most 6.4e-6 against the float64 release);
normalisation and the ROS table rows are float32 (the table is float32 already). Public assets: 346 kB.

## 2026-11-21 — PR #7 review: missing weather crashed make_features()

With a missing or NaN temperature, minimum humidity or wind speed, `make_features()` passed NaN into the ROS lookup
and its integer indexing raised an IndexError before the advertised missing-value fill could run. Training
(training/src/physfeat.py) substitutes 1-h moisture 10 % and 10 m wind 3 m/s before the same lookup; inference now
does the same. Complete inputs are unchanged: the demo's golden output is identical. Test:
tests/test_firesize_inputs.py.
