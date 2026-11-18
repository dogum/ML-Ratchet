# FlowCast lab notebook

Historical record. Entries are dated; earlier entries are kept as written (paths updated only where marked).

## 18 Nov 2026: build notes (flowcast-status.md, as written in the cloud sandbox)

Daily streamflow for a basin with no gauge: from basin-average ERA5-Land weather and 11 static attributes, a small
LSTM predicts daily flow with 5-95 % quantiles. It was trained only on WaterSim simulations and graded once on 619
real CAMELS-US exam basins. Watershed Phase 2, project 1.

### Honest headline
- **Ungauged exam (main, WY1990-99, 619 basins):** median NSE **0.341**, KGE 0.367. Classical ungauged
  regionalisation (U-dev) 0.372, U-k12 0.385, NWM 0.550, gauged GR4J 0.473.
- **Recent exam (WY2016-23, 603 basins):** 0.378 against U-dev 0.433 and NWM 0.687.
- FlowCast **does not beat** the classical ungauged baseline overall. It ties or wins in arid, regulated and large
  basins and loses in snowy and humid ones.
- **Gauged nowcast mode** (optional), lead 1 / 3 / 7 days: main-exam NSE **0.569 / 0.474 / 0.426**, against
  persistence 0.475 / −0.19 / −0.47. Recent: 0.591 / 0.511 / 0.466.

### Recipe as built
- **Physics:** WaterSim (Phase 1): HBV, GR4J, SAC-SMA and VIC-lite; degree-day or energy-balance snow; UH or
  Muskingum-Cunge routing; reservoirs and irrigation. Checks in `water/checks/` (now `shared/water/checks/`; GR4J vs
  airGR max 1e-7 mm; water balance 1e-15). FlowCast wraps it with three versioned physics knobs (`phys/p*.json`)
  and does not edit it.
- **Bank:** 12,000 runs per physics version, plus 800 frozen validation runs. Each run: a random basin (50 %
  jittered dev attributes, 50 % parametric prior), random parameters, one of 4 models × 2 snow routines, 25 %
  Muskingum-Cunge routing. Forcing: 5 years of real ERA5-Land from a random CAMELS basin in a free window.
  Generating a bank takes 7-15 min on one shared core.
- **Noise:** WaterSim's forcing-error model on the network input, and gauge-like rating error on the target.
- **Network:** LSTM, 64 units, 25k weights; 11 daily weather features plus 18 statics (7 of them climate indices
  computed from the forcing). Mean head with an NSE-type loss; 5 monotone quantile heads with a pinball loss.
  Target in units of mean precipitation. 3,000 steps × 32 windows of 730 days, 365 of them scored. About 12-25 min
  per rung.
- **Checks:** numpy == torch to 1.7e-7; JS == numpy to 7.6e-6.
- **Dev gate:** 52 dev basins, WY2001-2014, Caravan Q. The warm-up starts 1999-10, so the gate is entirely
  non-exam. References on this window: U-dev 0.466, NWM 0.573, GR4J 0.586 (partly in-sample).
- **Error bars:** widen-only factors fitted on dev: 50 % band × 1.00, 90 % band × 1.08.

### Rungs
Rule: a rung is kept if it scores more than +0.01 sim median NSE on the current frozen set **and** the dev median
NSE drops no more than 5 %.

| Rung | Change | Sim (vs inc.) | Dev NSE | Dev KGE | Dev FHV | Kept |
|---|---|---|---|---|---|---|
| r0 | WaterSim as shipped (p0) | 0.441 | 0.390 | 0.319 | −51 % | base |
| r1 | p1 water yield, old loss | 0.332 (0.397) | 0.331 | 0.259 | −59 % | no |
| r0b | **Loss fix:** loss scale from inputs (Fu runoff ratio), not from the target's own std | 0.476 (0.441) | 0.436 | 0.442 | −27 % | **yes** |
| r1b | **p1 water yield** (dev-measured ERA5 multiplier) + new loss | 0.425 (0.395) | 0.425 | 0.424 | −32 % | **yes, released** |
| r2 | p2 flashier runoff (full dev prior shift) | 0.463 (0.446) | 0.380 | 0.490 | −16 % | no (dev NSE −11 %) |
| r3 | p3 forcing error cv 0.4 + 30 % timing jitter | 0.326 (0.335) | 0.406 | 0.430 | −38 % | no |
| r4 | p4 = p3 + p2 | 0.404 (0.380) | 0.378 | 0.490 | −23 % | no |
| r5 | r1b + 5,000 more steps | 0.419 (0.425) | 0.434 | 0.441 | −28 % | no |

- **Loss bias (r0b).** Kratzert's NSE* scales each basin by the std of its own flow. In simulation that std depends
  on hidden parameters, so wet and flashy draws were down-weighted and the network ran low: median volume ratio 0.82
  on near-unit-multiplier sims, dev FHV −51 %. A general trap for sim-trained ungauged networks.
- **Water yield (p1, `calib_water.py`).** For each dev basin, the attribute-prior simulator was run with ERA5-Land
  P × m on a grid of m, and m* was taken where it matches the observed mean flow (median of 4 models). Fit:
  log m* = −0.088 + 0.058 elev_km + 0.153 snow + 0.397 log(P/3), robust scatter 0.146. m* runs from 0.53 (Texas,
  North Dakota plains) to 1.71 (Olympic Peninsula). Before p1, the network's volume error on dev correlated with
  log m* at −0.92. Effect on arid dev basins: over-prediction ×1.76 → ×1.24, and the share of dev basins with
  NSE < 0 fell from 15 % to 12 %.
- **Flashiness (p2, `calib_flash.py`).** With WaterSim's half prior correction, the simulated Richards-Baker
  flashiness is 0.62× the observed on dev. The full correction gives 1.00×. It improved dev KGE, FHV and peaks, but
  cost NSE in both tries (r2, r4).
- **Forcing noise (p3, `calib_noise.py`).** Under the default noise, a calibrated GR4J twin matches its clean run at
  NSE 0.88. Reality allows 0.56 for the same model. cv 0.4 with jitter 0.3 gives 0.72. Not kept.
- **Re-scoring.** Every rung was re-scored on every frozen set (p0-p4; matrix in `results/sim_*`). r1b is the best
  or within 0.01 of the best on p1 and p3.
- **Twin tests (`twin.py`).** Median NSE of the same r1b prediction: real record 0.425 (flashiness ratio 0.79),
  calibrated twin 0.595 (0.75), prior twin 0.380. The network reads the real record and the calibrated twin alike.
  Mixing in the next UTC day gave no gain, so there is no time-zone offset to model. For r0, the flashiness ratio
  was 0.54 on real data and 0.51 on the calibrated twin; that led to the loss fix and to p2.

### Nowcast (n1, `configs/n1.json`, `evalnow.py`)
r1b fine-tuned 2,500 steps (lr 1e-3, seed 11) with the gauge-like flow `lead` days earlier, an availability flag
(gaps in 30 % of training windows) and lead/7 as extra inputs. Dev (WY2001-14): 0.633 / 0.567 / 0.520 at 1 / 3 / 7
days against persistence 0.549 / −0.139 / −0.515. No frozen-set comparison and no widening were done for it.

### Exam (run once, medians)
As in `MODEL_CARD.md` (all numbers reproduce from `results/exam_r1b_exam_cal.json`). Win rates: FlowCast beats U-dev
at 34 % of basins and NWM at 21 %. CAMELS-531 subset 0.357. Flood peaks −46 % bias. Coverage: the 90 % band holds
97 % of days in the median basin.

### Limits (as written)
- Loses to regionalisation in snowy and humid basins. Only 10 dev basins are snowy, and no snow rung was tried.
- Peaks are about 46 % low because the mean prediction is smooth. Flashier physics trades peaks against NSE.
- The bands are too wide in most basins; widen-only calibration cannot narrow them.
- The p1 multiplier is fitted on the same dev basins that gate it (in-sample, 4 coefficients). It over-predicts
  some wet Pacific NW basins (demo: South Prairie Creek, ×1.76).
- ERA5-Land / Caravan conventions only.

### Next rungs worth trying
- **Snow.** A twin test on the 10 snowy dev basins, with the full dev snow shift (TT −0.66 °C, SFCF +0.11, CFMAX),
  as in p2 but for snow only, and an elevation-dependent temperature bias for ERA5-Land. Snow is the largest exam
  gap (−0.16 against U-dev).
- **Per-basin water yield.** Learn m from attributes with uncertainty instead of the regional regression, or feed
  ERA5 `SWE_era5` and soil-moisture states as inputs.
- **Peaks without losing NSE:** a separate high-flow head, or report the q75/q90 annual max as the peak estimate;
  train the flashier physics with a larger network, since NSE fell when the sims got flashier.
- **Capacity.** H=128 with longer training, or an ensemble of seeds. Rung decisions near ±0.01 sim are noise-level:
  r5's 300-basin validation said +0.02, the full 800 said −0.006.
- **Narrower bands:** a calibration that can also narrow (would need a rule change) or a log-space quantile head.
- From the handoff (section 5): out-of-domain exams on GB, CL, BR and AUS; hourly flows.

### Sandbox gotchas
- Process matching: `pgrep -f "train.py r1"` also matches other agents' `train.py r1 ...`. Wait on your own PIDs only.
- A training process holds about 1.2 GB with a 12k × 1462-day bank. Drop true Q from memory; keep forcing as float16.
- With four agents on the machine, a training step took 0.23-0.55 s depending on load. Bank generation in parallel
  roughly doubles training time.
- Disk: about 0.4 GB per 12k bank. Rejected banks' train shards were deleted; val shards were kept for re-scoring.

### Reproduce (the shipped README's commands; paths now through `training/_paths.py`)
```
python training/src/calib_water.py                      # dev water-yield multipliers (fit as in phys/p1.json)
python training/src/simbank.py p1 val 800 7 && python training/src/simbank.py p1 train 12000 1
python training/src/train.py r1b                        # ~15 min on one shared core
python training/src/evalreal.py r1b dev && python training/src/calib_sigma.py r1b
python training/src/evalreal.py r1b exam 1.0,1.08 && python training/src/exam_report.py r1b r1b_exam_cal
```
Every script needs the real time series (`mlratchet data water`) and torch; bank generation needs numba.

## 18 Nov 2026: rule notes from the build handoff (section 6, the parts about FlowCast)
- **Dev data used both to fit physics and to gate rungs (partly in-sample):** FlowCast's 4 water-yield
  coefficients. The exam was untouched.
- **Free-window forcing from exam basins** was used in training banks, as the brief allows. No exam Q was used for
  training.
- **Climate statics at exam time:** FlowCast computes them from the forcing of the window being predicted
  (exam-period forcing, no Q).
- **Ratchet noise:** sim-metric differences of about 0.01 are noise-level for FlowCast; a 300-basin subset and the
  full 800 disagreed on rung r5. Use larger frozen sets.

## 18 Nov 2026: upstream candidates for WaterSim (handoff section 4, FlowCast's)
1. **ERA5-Land water-yield multiplier** (`src/calib_water.py`, `simbank.p_multiplier`): log m = −0.088 +
   0.058 elev_km + 0.153 frac_snow + 0.397 log(P/3), σ 0.146, clipped to [0.4, 2.5]. BasinDetective's simpler fit is
   log m ~ N(−0.076 + 0.30 frac_snow, 0.12); reconcile the two into one documented prior.
2. **Forcing error too small:** a calibrated twin scores 0.88 against its clean run under the default noise, while
   reality allows about 0.56 (`results/calib_noise.json`).
3. **Prior shift for flashiness:** the full dev shift (not half) matches flashiness 1.00×, but networks trained on it
   lose NSE (r2, r4). Keep both priors available.
4. **Doc note:** an NSE* loss weighted by the target's std biases simulation-trained ungauged models low (the std
   depends on hidden parameters). Scale by an input-derived quantity instead, e.g. the Fu runoff ratio (r0b).
5. **Bug** (WaterSim `realdata.random_forcing`): `span*365` days is shorter than `DateOffset(years=span)`, so a window
   can end a few days inside an exam period. The guard raises `PermissionError`, so no exam data leaked.

## 18 Nov 2026: imported into ML Ratchet

**What changed.**
- Layout kept as shipped (`training/{configs,phys,results,src,ratchet.json}`, every file name unchanged).
  `training/_paths.py` added; `src/common.py` imports it: outputs (`ROOT`) now go to `WORK` (`models/flowcast/work/`
  or `$ML_RATCHET_WORK/flowcast`), shipped inputs are read from `training/phys` and `training/configs` (`PHYS`,
  `CONFIGS` in `bankdata.py`, `simbank.py`, `train.py`), WaterSim/WaterData come from `shared/water`, and data paths
  (`{WATER}/data/...` in `devref.py`, `evalnow.py`, `exam_report.py`, `webdata.py`) now use the data root
  `$ML_RATCHET_WATER` (default `shared/water`). `export.py` imports `flowcast.py` from the model folder and checks the
  newly exported weights (it used to reload the release's own); `buildpage.py` reads the shipped results unless a
  re-run wrote newer ones, and its template and JS port from `src/`. No sandbox absolute path remains.
  Nothing was re-run.
- `flowcast.py`: `df.index.dayofyear` → `day_of_year` (pandas 3 deprecation warning; identical values). Nothing else.
- `demo.py`: the PNG moved behind `--plot` and goes to `work/`; it now also prints the volume ratio and the gauged
  nowcast at leads 1/3/7 against persistence on the same three basins; files are closed. The ungauged numbers are
  the shipped demo's (NSE 0.31 / 0.39 / 0.24, band 98 / 96 / 100 %). Golden output: `tests/golden/flowcast.txt`.
- `pipeline.py`: the `--camels` default path is `shared/water` (was the sandbox's absolute water/ path); only that optional mode
  imports waterdata.
- The shipped `README.md` is replaced by the generated one; its run and reproduce notes are in this notebook and the
  card. `MODEL_CARD.md`: paths, the weight count, a verdict-and-interval table, the nowcast against GR4J and NWM, the
  ERA5-Land caveat, rule notes. `NOWCAST.md` unchanged.
- New: `model.yaml`, `ratchet.yaml`, `examples/README.md`, `web/export_instrument.py`, `web/build_lab.mjs`,
  `lab/` (offline page) and the site instrument (`site/src/instruments/flowcast/`).

**Checked against the shipped JSONs.** Every exam number in the status doc, card, NOWCAST.md and handoff reproduces
from `exam_r1b_exam_cal.json` and `nowcast_summary.json` (medians, KGE, FHV, NSE < 0, CAMELS-531, strata, peaks,
coverage, win fractions 34 % and 21 %), as do the rung table (`ratchet.json`), the twin numbers, the noise and
flashiness calibrations, the water-yield coefficients and the band factors. The demo's ×1.76 for South Prairie
Creek reproduces (5.55 / 3.15 mm/d).

**Did not reproduce or cannot be checked from the shipped files.**
- The card's "24,966 weights": the weight files hold 24,710 (25,478 for the nowcast copy). 24,966 would be a 30-input
  LSTM; the released one has 29 inputs.
- Not in any shipped file (would need the dev parquets or logs, not shipped): the dev references U-dev 0.466 /
  NWM 0.573 / GR4J 0.586, the arid-dev over-prediction ×1.76 → ×1.24 (the same 1.76 as the demo basin, a
  coincidence or a mix-up that cannot be told apart), the −0.92 correlation, the m* range 0.53-1.71, the volume
  ratio 0.82, r5's 300-basin +0.02, and the shipped page's JS parity 7.6e-6.
- The card rounds r2's dev drop to −11 %; it is −10.5 %.

**Added on import.** Binomial (Clopper-Pearson) intervals on every shipped win fraction (exam rows in `model.yaml`);
a paired bootstrap of rung differences over the 800 frozen basins from the shipped per-basin sim scores
(`ratchet.yaml` note; r1b over r0b +0.030, 95 % CI +0.001 to +0.052). Every rung's p2 and p3 sim files hold one NaN basin
(a flat simulated record, zero target variance); `np.nanmedian` ignores it.

**Instrument.** The TypeScript port reproduces `flowcast.py` on 54 reference cases (3 basins × 13 ungauged what-ifs
+ 5 nowcast cases) to 5.3e-8 relative, using float16 weights. The f16 rounding is the instrument's only change to the
network: it moves the daily mean by at most 0.04 % of each basin's peak (0.01 mm/d) against the float32 release.

## 18 Nov 2026: reproduction re-score of the main exam (seen)

Approved by the owner after the import. The exam was opened at release, so this is a **seen** reproduction:
nothing (weights, widening, decisions) was changed because of it.

- **Data.** Caravan CAMELS-US (all 671 basins) rebuilt by `mlratchet data water` into the git-ignored data root
  `work/water` (`ML_RATCHET_WATER`). `build.py --only verify`: 1,170 of 1,170 shipped files identical; splits.json
  file sha256 33f88ca0…, field ed8109e5… both OK. NWIS, NWM time series and the NWIS peak file were not fetched.
- **Code.** `training/src/rescore_main.py` (new): the released `flowcast.py` on `wd.forcing(b, 1988-10-01,
  1999-10-01)` and `wd.spec(b)`, window 1989-10-01..1999-09-30, Caravan Q, `waterdata.metrics.summary` plus band
  coverage, ≥ 365 observed days per basin, exactly as `evalreal.py exam 1.0,1.08` (the torch checkpoint is not
  shipped; numpy == torch to 1.7e-7). The 619 scored basins were asserted equal to `splits.json` camels_us exam.
  The nowcast weights were scored as `evalnow.py` does (leads 1/3/7, persistence = obs shifted by the lead).
  `.venv-train` Python, about 6 min on one core. Intermediates in `work/results/`; committed:
  `training/results/rescore_main_2026-11-18.json` (per basin: NSE, KGE, FHV, FLV, peak error, timing, coverage,
  nowcast NSE) and `rescore_main_summary_2026-11-18.json` (medians, intervals, verdicts).
- **Reproduction.** Every claimed main-exam number reproduces: median NSE 0.34117 (claimed 0.34117), KGE 0.3666,
  FHV −30.94 %, peak error 52.23 %, timing 0.848 d, 90 % / 50 % coverage 97.3 % / 55.5 %, NSE < 0 14.2 %,
  CAMELS-531 0.3571, the five strata, the win fractions against all six baselines (identical counts) and the nowcast
  medians, KGE and wins at 1/3/7 days. Largest gap: FLV 59.11915 vs 59.11918 % (4e-7 relative). Nothing failed to
  reproduce.
- **Intervals.** Paired bootstrap over basins of median(FlowCast) − median(baseline) on the basins both have,
  10,000 resamples, numpy `default_rng(20261118)`; win or loss only if the 95 % interval excludes 0. NSE: vs U-dev
  −0.031 (−0.056 to −0.014), U-k12 −0.044 (−0.067 to −0.028), GR4J −0.132 (−0.157 to −0.109), NWM −0.205 (−0.235 to
  −0.182, 600 basins), persistence −0.134 (−0.182 to −0.087), climatology +0.276 (+0.251 to +0.294). KGE vs U-dev
  −0.067 (−0.087 to −0.039), U-k12 −0.095 (−0.118 to −0.069). Median |FHV| 39.1 % vs U-dev 34.3 % (+4.8, +2.9 to
  +7.8) and U-k12 31.3 % (+7.7, +5.6 to +11.6): the release-time "tie vs U-dev alone" (from signed medians) is a
  loss. Daily peak error vs U-dev +4.9 (+2.7 to +6.9), U-k12 +6.7 (+4.4 to +9.2). Peak timing: tie. CAMELS-531 (487
  exam basins) vs U-dev −0.045 (−0.063 to −0.017). Strata vs U-dev: snow −0.154 (−0.210 to −0.067), humid −0.124
  (−0.166 to −0.086), arid +0.009, regulated +0.054, large +0.006 (all three intervals span 0). Nowcast vs
  persistence / GR4J / NWM: lead 1 +0.093 (+0.060 to +0.133) / +0.095 (+0.073 to +0.122) / +0.025 (−0.011 to
  +0.058); lead 3 +0.665 / +0.001 (−0.016 to +0.027) / −0.072 (−0.100 to −0.034); lead 7 +0.892 / −0.047 (−0.064 to
  −0.025) / −0.121 (−0.147 to −0.090). Every verdict matches the Clopper-Pearson cross-check made on import; only the
  FHV row's reasoning changed (it was already a loss).
- **Not re-scored.** The recent exam (WY2016-23, NWIS flow) and the instantaneous flood peaks (NWIS peak file):
  their rows stay median-only with the binomial intervals on the shipped win fractions.
