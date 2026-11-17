> **Lab notebook.** FaultFinder was built in a claude.ai cloud session (16-17 Nov 2026) on the shared BuildingSim +
> BuildingData platform (to land in `shared/building/`). The first entry is that session's status document as
> written; its paths (`bldg/faultfinder/`, `release/`) refer to the cloud sandbox. `MODEL_CARD.md` holds every number.

## 17 Nov 2026 — build notes (faultfinder-status.md, as written)

### FaultFinder (BuildingSim Phase 2) — status

**What it is.** Air-handler / rooftop-unit fault detection and diagnosis from one day of BMS trend log (OA/RA/MA/SA temperatures, setpoint, damper and valve commands, fan speed/power, occupancy). Outputs P(fault), a probability over 18 fault classes (BuildingSim's 20 fault types mapped onto the 27-type exam taxonomy; 13 exam types are not simulable and are graded for detection only), a severity and an energy-waste estimate. Trained only on BuildingSim simulations.

- Page: the original page, now `lab/index.html` (real-day explorer with rule vs FaultFinder verdicts, AHU schematic, fault-injection lab, network running in the browser; JS == numpy to 5e-6 on all 217 exam days)
- Release: faultfinder-v1.0.zip (numpy-only `faultfinder.py`, `pipeline.py`, weights, `demo.py` on 4 real exam days drawn with a fixed seed, MODEL_CARD.md, training/)
- Working folder: the sandbox workspace bldg/faultfinder/ (STATUS.md, ratchet.json, exam/exam.json)

### Recipe as built
- **Physics:** BuildingSim (Phase 1 checks: RC step 0.003 K, 31/36 BESTEST inside range, psychrometrics 0.02 K vs CoolProp, 26/26 fault signatures in the expected direction).
- **Bank:** random RTU / VAV buildings x 17 TMYx cities x any season; 12-day windows simulated healthy and faulted (fault from day 5). Two banks of 38,987 days (v1 generic, v3 dev-calibrated).
- **Real noise:** residual library from real fault-free dev days, static offsets, fan-off sensor drift, gaps, per-source channel patterns; Korean office AHUs (unlabelled) for start/stop irregularity and SAT tracking tail.
- **Dev calibration (20 dev unit-days only):** twin tests found (1) BMS logs commands not positions (FLEXLAB valve stuck at 100 % logs 0 %) -> `log_commands`; (2) real damper at 100 % command mixes only 81-90 % outdoor air, Nesbitt 15 % mixes 14-28 % -> installed damper characteristic; (3) sensors disagree by 1-1.5 K on normal days -> larger static offsets. Plus operator jitter, schedule-shift faults, 24x7 units. Calibration temperature T = 4.3 (widen-only) on dev.
- **Network:** 1-D CNN, 96 x 15-min bins, 35 feature rows, 161k weights.

### Ratchet (decided on frozen sim val v4; keep if skill > incumbent + 0.01 and dev AUC within 0.05)
| Rung | skill v4 | dev AUC | decision |
|---|---|---|---|
| r1 Gaussian noise | 0.415 | 0.60 (dev FPR 0.91) | kept |
| r2 real noise | 0.454 | 0.74 (FPR 0.45) | kept |
| r3 calibrated bank v3 | 0.500 | 0.71 | kept |
| r4 damper characteristic + offsets | 0.511 | 0.73 (FPR 0.36) | kept |
| r5 both banks | 0.518 | 0.74 | rejected (+0.007) |
| r6 wider net, 10 epochs | **0.543** | 0.71 | **kept, final** |

### Exam (217 real days, run once)
| | FaultFinder | APAR rules | majority |
|---|---|---|---|
| AHU AUC (200 days, 91 faulty) | **0.78** [0.72-0.83] | 0.69 | - |
| AHU balanced acc. | 0.70 | 0.65 | 0.50 |
| AHU recall at 2 / 5 / 10 % false alarms | 23 / 27 / 47 % | 32 / 41 / 45 % | - |
| AHU FPR at 0.5 | 33 % | 2 % | 0 |
| RTU AUC (17 days) | 0.65 | 0.72 | bal. acc. 0.50 |
| **Nesbitt Hall (real building, 76 days) AUC** | **0.78** | 0.54 | - |
| FLEXLAB / Iowa ERS AUC | 0.93 / 0.73 | 0.67 / 0.73 | - |
| Diagnosis top-1 / top-3 (66 simulable faulty days) | 20 % / 48 % | - | 21 % top-1 |
| ECE (5 bins) all / AHU | 0.12 / 0.13 | | |

Calibration: 164/217 days in P 0.4-0.6 (temperature from 20 days over-shrinks; Brier 0.216 vs 0.204 uncalibrated). Energy-waste head: Spearman 0.19 on 46 ERS paired days (not useful).

### Limits
1. Diagnosis no better than the majority guess at top-1; cc valve stuck (50 % top-1) and zone sensor (100 %) work, OA damper stuck does not (0 %).
2. Many false alarms at P = 0.5; rules win at strict alarm rates.
3. RTU worse than the one-line rule; one day at a time, single-zone simulator, no exhaust damper / return fan / duct model.

### Next ideas
1. Multi-day context: compare a day with the unit's own recent days (most real false alarms are unit-specific habits such as 24x7 operation with the chiller off at night).
2. Simulate the missing plant faults (chiller off / CHW sensor) and exhaust/return-fan dampers in BuildingSim, so 13 more exam types become diagnosable.
3. Calibrate with a 2-parameter (Platt) map or a larger dev set; the 20-day temperature over-shrinks.

### Sandbox gotchas
- Container restart mid-run: banks are sharded per 100 windows and training checkpoints per epoch; resumed cleanly.
- Machine load 8-10 on 2 cores from four agents: one bank (3000 windows) takes ~1 h, one training rung 25-30 min.
- `data.json` must not contain NaN (node JSON parse); sanitised in webdata.py.

## 17 Nov 2026: imported into ML Ratchet

- **Layout.** The release zip (faultfinder-v1.0.zip) became `models/faultfinder/`: `faultfinder.py`, `pipeline.py`,
  `faultfinder_weights.npz` (+ its `.check.json`, numpy vs torch 9.5e-7), `demo.py`, `examples/` (four exam days and
  `labels.json`), `MODEL_CARD.md` and every training script in `training/` with `configs/`, the raw ratchet log
  (`training/ratchet.json`), the exam results (`training/exam.json`) and the browser port (`training/ffnet.js`), all
  as shipped. The inference files, weights and examples are byte-identical to the zip. The original page was fetched
  from its claude.ai artifact and saved byte for byte as `lab/index.html` (3.1 MB; it embeds all 217 exam days and the
  weights). `build.py` refers to `page.html` and `exam.js`, which were not in the zip; it is kept as shipped.
- **Paths.** New `training/_paths.py` (SRC, MODEL, REPO, WORK, BLDG, LAB). `common.py` now takes `ROOT` from
  `_paths.WORK` (`models/faultfinder/work/` or `$ML_RATCHET_WORK/faultfinder/`) and puts `BLDG` and `BLDG/buildingsim`
  on `sys.path` instead of `/home/claude/bldg`. `exam.py` and `webdata.py` read `faults_summary.json` from
  `BLDG/data/proc/baselines/`; `examples.py` writes to `WORK/release/examples`; `noise.py`'s default output is
  `WORK/data/noise_v1.npz`; `export.py` and `webdata.py` import `faultfinder.py` from the model folder (it used to sit
  in `release/`); `ratchet.py` reads `training/configs/ladder.json` (it used to be `ROOT/configs`). No other change.
- **Waits on the platform zip.** Every training script needs `buildingsim` and `buildingdata` (shared/building/,
  from buildingsim-buildingdata-v1.0), and the banks, noise model and checkpoints are not in the repository, so no
  training step was re-run. Inference, the demo and the instrument exporter need only numpy and pandas.
- **What was re-run.** The demo, from the repository folder: deterministic, under a second; the zip ships no demo
  output to compare with, and the files it runs are byte-identical to the zip's, so this is the shipped demo's
  output (`tests/golden/faultfinder.txt`). The 217 exam days embedded in the lab page, re-scored with the shipped
  weights: P(fault) matches the page's stored Python values to 5e-6 (they were stored to 5 decimals), the detection
  AUC is 0.7747 for all units (exam.json 0.7747), 0.7796 for AHUs (0.7795; the page's inputs are rounded) and 0.65
  for the RTU, and diagnosis top-1/top-3 is 19.7 %/48.5 % on the 66 simulable faulty days, as released.
- **Licences.** Nothing FaultFinder uses or ships is BDG2-derived, so there is no CC BY-SA folder. The handoff called
  the fault subset CC0; the release's own credits say the LBNL synthesis (FLEXLAB, ORNL, MZVAV-2) is CC0 but the
  ASHRAE RP-1312 ERS and Nesbitt Hall data come from Ghalamsiah's figshare package under CC BY 4.0. They are credited
  in `model.yaml`, in a new "Data and licences" section of `MODEL_CARD.md` (with a file-by-file list) and on every
  day of the live instrument.
- **Inconsistencies left as shipped.** `ratchet.json` describes r6 as "179k weights"; the exported network has
  160,871 parameters (160,103 after BatchNorm folding), and the status doc says 161k. The lab page's "Use it" section
  says 101k weights. The model card's r3 sim AUC on v4 is 0.69; `ratchet.json` has 0.6849, so `ratchet.yaml` says 0.68.
- **Exam rows.** Every row of the card's exam table went into `model.yaml`, with the per-source rows. One baseline
  was added at import: always naming the three commonest exam classes gets 36 of 66 days (55 %) at top-3, against
  48 % for FaultFinder, so top-3 is a loss (the release reported it without a baseline). The fixed-false-alarm recall
  rows set their thresholds on the exam's own healthy days and are marked report-only.
- **Rule notes.** No exam result was used to choose anything. The 20 dev unit-days set the noise library, the twin
  tests, the dev gate and the calibration temperature. Each bank or noise change got a new frozen simulated set
  (v1-v4); all rungs were re-scored on all four and the decisions re-taken in order on v4. The fixed-false-alarm
  detection rates are report-only (handoff section 5).
- **Live instrument.** `web/export_instrument.py` writes `site/public/instruments/faultfinder/` (0.51 MB): the
  network in float16, twelve exam days (the four examples, read through `raw_day` as the demo reads them, plus one
  faulty and one healthy day per source drawn with seed 20261117 from the page's exam days) and 108 parity cases.
  `site/src/instruments/faultfinder/net.ts` ports `featurize` and the network from `training/ffnet.js` onto the
  shared ops; it matches Python to 1.4e-6. Controls: a sensor offset on one temperature, the outdoor-air damper stuck
  (mixed air from the mixing balance, the logged command unchanged), extra sensor noise and removing the mixed-air
  sensor. float16 weights move P(fault) by at most 8.5e-5.
