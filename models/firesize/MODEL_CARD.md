# FireSize v1.0.1: model card

FireSize is ML Ratchet Wildfire Phase 2, project 2. It starts from a US wildfire ignition record: location, date, cause, reporting agency and discovery hour. Using the landscape around the ignition and gridMET weather on the discovery day and the day before, it predicts:
- **P(escape)**: the chance the final size reaches 100 acres or more;
- **P(large)**: the chance it reaches 1,000 acres or more;
- **the final-size distribution** over 12 FPA-FOD size classes, with a median and an 80 % range.

**No real fire label was ever a training sample.** The network learned only from simulated ignitions. Real dev ignitions were used to calibrate about 30 physical suppression and reporting parameters, as ratchet gates, and to fit one widen-only temperature. The exam was run once.

Part of ML Ratchet ("Trained on physics. Graded on reality."), fire platform (Wildfire Phase 2, project 2). Imported into the repository on 21 Nov 2026 (see `NOTEBOOK.md`); weights, inference and every number unchanged. The released weights file is byte-identical to the model the exam ran (its sha256 is the one in `training/results/exam.json`). "B" below is FireData, the Phase 1 data agent, whose baseline code (`shared/fire/data/scripts/baselines_firesize.py`) scored the baselines.

## v1.0.1 (27 Nov 2026): inputs clipped at 8 sd

The audit found that no simulated ignition sat on slash-blowdown fuel, so the 91 exam ignitions on it entered the
network at z = 1e6: escape probability exactly 0 and the same 1.73-acre median for all of them. With the owner's
approval:
- **Repair rung r8, rejected.** The v1 bank was rebuilt (the control reproduces r7 to 5e-6 in every weight) and 479
  simulated slash-blowdown ignitions added. Sim RPSS on the frozen set rose 0.2799 → 0.2881, +0.008 against the +0.01
  the rule needs, so the exam was not opened. The frozen set has no slash-blowdown ignition; on 64 held-out simulated
  ones r8 scored RPSS 0.127, r7 with an 8 sd clip 0.254, r7 as released −1.668.
- **Dated re-score of v1.0 with the clip.** Pooled escape AUC 0.7266 → 0.7264, AP 0.0714 → 0.0711, size MAE 0.6416 →
  0.6418. On the 91 slash-blowdown ignitions the size error fell from 1.237 to 0.750 log10 and the 80 % range coverage
  rose from 0.18 to 0.96. The clip touches 19,337 exam ignitions, not 91: every Great Basin ignition has its region
  input at z = 8.1 (escape AUC there 0.6393 → 0.6380).
- **v1.0.1** clips by default (`FireSize.z_clip = 8`; `None` restores v1.0). Weights unchanged. `model.yaml` carries
  the re-score; the exam table below is v1.0's, as run once. Details: `NOTEBOOK.md`, 27 Nov 2026.

## Exam (run once; 655,869 FPA-FOD ignitions, 2012–2020)

| Method | Escape AUC | AP | Brier | Size MAE log10 | Size ρ | Large AUC |
|---|---|---|---|---|---|---|
| **FireSize (sim-trained)** | **0.727** | **0.071** | 0.0273 | 0.642 | 0.286 | **0.815** |
| GBM fitted on dev (real labels) | 0.777 | 0.099 | 0.0255 | 0.510 | 0.533 | 0.803 |
| Climatology, region × cause | 0.688 | 0.061 | 0.0256 | 0.505 | 0.369 | 0.767 |
| ERC + BI + wind (logistic) | 0.634 | 0.042 | 0.0259 | 0.645 | 0.201 | 0.748 |
| Prevalence / global median | 0.500 | 0.027 | 0.0260 | 0.699 | – | 0.500 |

Results by period:
- **Main, 2012–19** (582,986 ignitions): AUC 0.725, AP 0.072.
- **Recent, 2020** (72,883 ignitions, a whole season held out): AUC 0.740, AP 0.071, against GBM 0.777 / 0.099 on all years. (Import note: the GBM's own 2020 scores in `exam.json` are 0.789 / 0.092; climatology 0.711 / 0.067.)

Results by region:
- **Within regions, FireSize beats the GBM in CA (0.739 vs 0.690), NW (0.776 vs 0.735) and RM (0.748 vs 0.703).**
- It loses badly in the Southeast (0.566 vs 0.704) and the Southern Plains (0.715 vs 0.808).
- Details are in `training/results/post_exam.json`. The baselines were re-run with B's exact code, and they reproduce B's stored AUCs to every digit.

**What it does well:**
- It ranks escapes better than climatology and the fire-danger indices.
- On large fires (≥ 1,000 acres) it ranks better than every baseline, including the GBM fitted on real labels.

**What it does badly:**
- Its probabilities are too high: mean P(escape) is 0.055 against a real rate of 0.027, so its Brier score is worse than predicting the prevalence.
- Its size point estimate is worse than climatology: MAE 0.642 vs 0.505 log10, with a +0.22 log10 bias.

The 80 % size interval covers the real class 90 % of the time, so the size intervals are conservative.

## How it works
1. **Ignition prior** (`prior.py`):
   - 2,400 training windows of 18 km, centred on dev ignition locations with a N(0, 20 km) jitter.
   - 8 simulated ignitions per window. Each takes the attributes (cause, agency, hour, date ± 7 days) of a dev ignition within 150 km.
   - Every landscape window passes `fd.guard.check_landscape` and every weather period passes `fd.guard.check_weather`, so no exam footprint or exam fire-day weather is used.
   - 304 windows are frozen as the simulated validation set.
2. **Free-burn simulation** (`sim.py`, FireSim physics):
   - Rothermel/Albini surface fire with Scott & Burgan fuels, Van Wagner crown fire, and minimum-travel-time spread on the real 90 m LANDFIRE window.
   - Hourly weather is disaggregated from real gridMET. Dead fuel moisture follows Nelson; live moisture comes from the GSI. There is no spotting.
   - Fires smaller than a cell are tracked as an analytic sub-cell ellipse, because most FPA-FOD fires are smaller than a 90 m cell (0.81 ha).
   - Output: a free-burn area curve A(t) over 96 h.
   - Per simulation, the pre-discovery burn time ~ LN(0, 1) h and the ROS multiplier ~ logU[0.2, 1.5].
3. **Suppression and reporting** (`supp.py`, `calib.py`):
   - A Fried & Fried (1996) containment race. Line building starts after a response time and escalates; the fire is contained when the line built catches the line needed (a share of the ellipse perimeter).
   - The final size is A at containment. A reporting floor is applied (0.1 acre, a 1-acre floor by region, and a share booked at 0.01 acre).
   - Two attack modes: a share of fires gets only confine/monitor effort (2 % of the line rate).
   - The parameters are fitted on 911 dev "twins": dev ignitions from the calib half, simulated 4× each in their own real landscape and weather. The fit maximises the likelihood of each twin's real size class.
   - The pre-discovery time and ROS distributions are recalibrated by importance weights.
   - Twin NLL fell from 3.60 (literature values) to 2.64. In 2-fold cross-validation it is 2.66 / 2.73, against 3.56 / 3.68 for the literature values.
4. **Labels:** each simulated ignition gets a soft label, the distribution of its reported size class over 24 suppression draws.
5. **Network:**
   - A 3-member ensemble of MLPs (75 → 128 → 128 → 12) trained in torch with cross-entropy on the soft labels, using importance weights, 8,500 simulated ignitions and 1 CPU thread.
   - Inference is numpy only (`firesize.py`).
6. **Widen-only calibration:** one temperature, T = 1.459, fitted on the dev calib half by size-class log-likelihood and checked on the gate half. Point medians use T = 1; this was decided on dev before the exam.

## Inputs (75 columns)
- **Landscape.** The 250 m national LANDFIRE 2016 grids (FBFM40, CC, CH, slope), nearest-neighbour sampled from 30 m. `training/natgrid.py` rebuilds them.
  - The fuel class at the ignition pixel: GR, GS, SH, TU, TL, SB, urban, agriculture, or other non-burnable.
  - Class fractions in 5 × 5 (1.25 km) and 13 × 13 (3.25 km) boxes.
  - Mean canopy cover, canopy height and slope in the 5 × 5 box, and slope at the pixel.
- **Weather.** gridMET daily on the discovery day and the day before: ERC, BI, FM100, FM1000, wind speed, RHmin, RHmax, Tmax, Tmin (°C), precipitation, VPD.
- **Physics.** log1p of the Rothermel head ROS (m/min) under the afternoon weather (Simard EMC at Tmax/RHmin, 10 m wind = 1.45 × the daily mean), at the pixel and as 5 × 5 and 13 × 13 means, plus the pixel value for the day before. It comes from a lookup table built with FireSim's `behave()` (`ros_lut.npz`).
- **Record.** NWCG cause class, federal reporting agency, discovery hour (sin, cos, missing flag), day of year (sin, cos) and GACC-like region (9).

`firesize.make_features(...)` builds all 75 columns from 13 × 13 patches plus the record.

## The ratchet (frozen sim set v1|v1rep2; dev gate = the dev ignitions in the gate half, 4,429)
Keep rule: sim RPSS > incumbent + 0.01, dev-gate escape AUC ≥ incumbent − 0.01, and dev-gate size MAE ≤ 1.05 × incumbent. Every rung is re-scored on the latest frozen set.

| Rung | Change | Sim RPSS | Gate AUC | Gate AP | Gate MAE | Kept |
|---|---|---|---|---|---|---|
| r0 | FireSim as shipped, literature suppression | −0.256 | 0.697 | 0.076 | 0.725 | base |
| r1 | suppression calibrated on dev twins (v0 physics) | −0.292 | 0.675 | 0.071 | 0.694 | no |
| r2 | physics v1: agriculture burns as GR1, non-burnable ignition moved ≤ 180 m | −0.124 | 0.757 | 0.081 | 0.744 | **yes** |
| r3 | regional response offsets + ROS centres per fuel group | 0.038 | 0.655 | 0.052 | 0.651 | no (gate AUC) |
| r3b | regional reporting floors | 0.213 | 0.634 | 0.047 | 0.633 | no (gate AUC) |
| r4 | r2 + Rothermel ROS features | −0.139 | 0.760 | 0.087 | 0.730 | no (margin) |
| r5 | two-mode initial attack + federal line-rate term + floors | 0.265 | 0.766 | 0.085 | 0.618 | **yes** |
| r6 | r5 on the full 8.5k bank | 0.278 | 0.746 | 0.079 | 0.626 | no (gate AUC) |
| r7 | r6 + Rothermel ROS features | 0.280 | 0.756 | 0.082 | 0.627 | **yes → v1.0** |

On the dev gate half, climatology scored AUC 0.704, AP 0.106 and MAE 0.477. FireSize v1.0 scored 0.759, 0.083 and 0.627 (after widening; MAE uses the sharp median).

## Limitations
- **Over-prediction.** Escape probability is about 2× too high. Widening with T = 1.46 raised it further: dev-gate Brier went from 0.0274 to 0.0285. The temperature was fitted to the size-class likelihood, not to escape.
- **Size point estimates.** FPA-FOD sizes are heavily rounded (0.1, 1, 0.01 acres). Climatology medians land on those values; a continuous median does not.
- **Southeast.** AUC is 0.57. Prescribed and agricultural burning, small parcels and dense suppression are not in the simulator. LANDFIRE 2016 fuels are stale for 2020.
- **Suppression model.** Several calibrated parameters sit on their bounds: response time 0.1 h, line-rate multiplier 30× literature, a federal line-rate term of e^−3, a confine share of 0.5. The race model is an effective model, not a measured one.
- **Simulator gaps.** No spotting and no multi-day re-burn after containment. The weather is daily gridMET disaggregated to hours.
- **Landscape resolution.** Landscape features come from 250 m grids (a coarse summary by design). The simulator itself ran on 90 m windows.
- **Labels.** Final size is the FPA-FOD reported size; complexes and merged fires are booked to one record.

## Verdicts (added on import)
A row is a win only if FireSize beats every baseline on it. The GBM is fitted on the real labels of the 8,913 dev ignitions and reads lat/lon and owner; it is a fair, strong baseline. The only shipped interval is the bootstrap of the all-exam escape-AUC differences (100 resamples, `post_exam.py`).

| Row | FireSize | Best baseline | Verdict |
|---|---|---|---|
| Escape AUC, all | 0.727 | GBM 0.777 (difference −0.050, 95 % −0.053 to −0.047) | loss |
| Escape AUC, 2020 | 0.740 | GBM 0.789 | loss |
| Large-fire AUC, all | 0.815 | GBM 0.803 | **win** (no interval) |
| Escape AP | 0.071 | GBM 0.099 | loss |
| Escape Brier | 0.0273 | GBM 0.0255; prevalence 0.0260 | loss |
| Size MAE log10 | 0.642 | climatology 0.505 | loss |
| Size ρ | 0.286 | GBM 0.533 | loss |
| Escape AUC by region | CA 0.739, NW 0.776, RM 0.748 | GBM 0.690, ERC 0.742, GBM 0.703 | **win** in CA, NW, RM (no interval) |
| | NR, GB, SW, SP, SE, NE/MW | the GBM in each | loss |

Against climatology (+0.039, 95 % +0.035 to +0.044) and ERC/BI/wind (+0.093) the escape ranking wins.

## Rule notes (from the build handoff, section 6)
- FireSize read FPA-FOD records straight from the platform's parquet, filtered by its split (`common.fod_records`), because `firedata` has no FPA-FOD accessor. Only `exam.py` and, after the exam, `post_exam.py` and `build_release.py` select the exam records.
- Several dev-calibrated suppression parameters sit at their bounds (10 of 31 in `calib_v1rep2.json`; listed in `ratchet.yaml`).
- The dev calib half fitted the suppression model and the temperature; the ratchet's gate half was never used for calibration.
- The widening raised the dev-gate Brier score (0.0274 → 0.0285) before the exam; the release was kept as frozen.

## Files
- `firesize.py`: inference and feature builder (numpy only).
- `firesize_weights.npz`: the ensemble weights.
- `ros_lut.npz`: the ROS lookup table.
- `demo.py` and `examples/demo_examples.csv`: eight exam ignitions, features as in the exam. `examples/patches.json`: their 13 × 13 fuel patches.
- `training/src/`: every script as run (prior, windows, gridMET, simulator, suppression, calibration, train, ratchet, widen, the exam, post-exam, release and page builders); `training/_paths.py` maps the sandbox paths onto the repository.
- `training/results/`: the exam, post-exam, widening, twin and dev-gate JSONs; `training/results/calibration/`: the suppression fits; `training/ratchet.json` and `training/rungs.json`: the ratchet log.
- `lab/index.html`: the original page; `web/export_instrument.py`: the site's live instrument.
