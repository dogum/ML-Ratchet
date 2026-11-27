# BuildingDetective: lab notebook

The build's status document comes first, as written at release (17 Nov 2026), with only the page link, the release
path and the sandbox folder replaced by their names in this repository. The dated entries after it are new.

---

## Status at release (v1.0, 17 Nov 2026)

BuildingSim Phase 2 project for ML Ratchet ("Trained on physics. Graded on reality."). From a building's meters plus weather (hourly, monthly bills or one annual total) it infers, each with an error bar or class probabilities: envelope UA and UA per m², thermal time constant, heating and cooling balance points, heating energy per kelvin (heating and cooling), non-HVAC share, weekday operating hours and weekend ratio, primary use (12 classes), heating type (heat pump / resistance / fuel / district / none), cooling type (none / DX / chiller / district), floor area and year built.

- Page: `lab/index.html` (the original page). It has a live JS explorer of real exam buildings, a lab with insulation, mass, system and noise sliders, the exam, the ratchet and the twin test.
- Release: `buildingdetective-v1.0.zip` (3.2 MB). Inference uses numpy and scipy only. The demo runs in 0.6 s from an unzipped copy.
- Working folder: `bldg/detective/` in the build sandbox (STATUS.md, src/, ratchet.json, exam.json).

## Recipe as built
- **Physics.** BuildingSim, the Phase 1 simulator (5R1C plus HVAC curves), verified in `bldg/checks`. Labels come from `labels()`; UA is `UA_network_W_K`.
- **Bank.** 12,000 generated buildings (30 % homes, 70 % commercial across 10 types) on real 2016–17 weather from 18 BDG2 sites plus TMYx cities, at 1 h. The frozen validation set is shards 900–904 (1,000 buildings).
- **Meter emulation.** Each simulated meter gets real-world problems:
  - gaps, zero runs, spikes, level shifts and clock offsets
  - readings spread evenly across the day
  - shared loads, scale errors and noise
  - for bills: shifted billing windows and missing months
  - for surveys: unreported fuels
- **Calibration on dev only.** These are wrappers in `detective/src/wrappers.py`, all upstream candidates.
  - **W1 intensity.** One factor per use type, so generated site-energy medians match the RECS/CBECS prior and dev splits. Examples: lab 0.42, education 0.55, lodging 1.55.
  - **W2 campus.** District distribution factors fitted on the 52 BDG2 dev twins: heat 0.18, chilled water 0.33. 30 % of commercial buildings are on campus service.
  - **W3 always-on base loads (physics v2).** The dev twin test showed three gaps:
    - Real district heat stays at 0.72× its annual mean in summer; the twins dropped to 0.06×.
    - Real chilled water keeps running at night (night/day 0.69 vs 0.31).
    - Real electricity has a higher night/day ratio than the twins (0.77 vs 0.66).

    W3 moves a share of each district meter's energy into a flat base (heat 0.70, chilled water 0.50) and adds a flat electricity load. After W3 the twin values are: heat slope 1.0 vs real 0.82 (was 4.4), and chilled-water night/day 0.63 vs real 0.69.
- **Network.** One MLP per resolution, 2×256 GELU (hourly has 0.37M parameters). The hourly model takes about 1,200 engineered features:
  - temperature-binned daily load
  - hour-of-week profile
  - PRISM 5P fit
  - correlations with exponentially smoothed temperature (the thermal-mass signal)
  - lag correlations

  Heads: heteroscedastic Gaussian outputs for the 12 regression targets plus 3 softmax classifiers. Training used simulated buildings only.
- **Dev set and error bars.** Dev = 52 BDG2 buildings, 1,848 RECS respondents and 661 CBECS respondents.
  - Error bars use widen-only factors per output. Area and year are widened using real dev data; outputs with no real truth are widened using the simulated dev twins. The hourly area factor is 3.0.
  - Class temperatures are fitted on dev. The use-type temperature reached the cap of 10, meaning use probabilities on real data are close to uninformative.

## Rungs
Rule: keep a rung if it beats the incumbent's frozen-val skill by more than 0.01 and the real dev composite drops by no more than 0.02.

| rung | change | val skill (h / m / a) | decision |
|---|---|---|---|
| R1 | MLP on f1 (W1+W2), 12 shards | 0.703 / 0.573 / 0.490 | kept |
| rebaseline | physics v2 (W3): frozen val rebuilt, incumbents re-scored | 0.663 / 0.501 / 0.483 | — |
| R2 | retrain on f2 (W3) | 0.693 / 0.547 / 0.486 | h, m kept; a rejected (+0.004) |
| R3a–d | wider/deeper hourly nets | R3a diverged at 0.41; R3b 0.684; R3d 0.695 | rejected |
| R3c | 3×512 net | 0.705 | withdrawn: 1.15M parameters exceeds the ~500k cap |
| label hygiene | 3 training buildings had near-zero metered site energy (base_share −8e12); balance points clipped to [−25, 30] | — | — |
| R4 | retrain after label hygiene | 0.746 / 0.597 / 0.517 | kept ×3 |
| R5 | 23 shards | 0.783 / 0.630 / 0.530 | kept ×3 |
| R6 | 3×288 hourly | 0.779 | rejected |
| R7 | 40 shards | 0.801 / 0.641 / 0.539 | h, m kept; a rejected |
| R8 | 3×288 hourly on 40 shards | 0.803 | rejected |
| R9 (final) | full bank, 60 shards | 0.813 / 0.656 / 0.541 | kept ×3 |

The dev gate drifted from 0.331 to 0.306 across kept rungs. Each step stayed inside the 0.02 tolerance. The gate is small and noisy.

## Exam (run once; exam.json)
Real truth, all on the exam split:

| set | task | Detective | baselines |
|---|---|---|---|
| BDG2 (1,428 buildings, 2017) | floor-area MdAPE | 50 % hourly / 49 % monthly / 48 % annual | electricity ÷ dev EUI of the true use: 51.5 % (same buildings); B's quoted 46.4 % (1,133 meters); dev median 68 % |
| BDG2 | primary use | 11 % accuracy, F1 0.10 | majority 40 % / 0.06; B's centroid 34 % / 0.20 |
| BDG2 | year built MAE | 25–28 y | dev median 24.9 y |
| BDG2, electricity only | has chilled-water meter (AUC) | 0.75 | — |
| BDG2, electricity only | has steam/hot-water meter (AUC) | 0.70 | — |
| RECS (7,401 homes, annual) | floor-area MdAPE | 33.6 % | B's rule with true type and climate 33.1 %; dev median by true type 28.8 % |
| RECS | fuel vs electric heat | 87.3 % | rule 85.3 %; majority 64.8 % |
| RECS | heat pump vs resistance | balanced accuracy 0.46, AUC 0.40 (worse than chance) | rule 0.50 |
| RECS | vintage MAE | 22.1 y | 22.3 y |
| CBECS (2,558 buildings) | floor-area MdAPE | 43.0 % | 38.7 % (true type); dumb 75.7 % |
| CBECS | heating type | 74.7 % | rule 68.6 %; majority 55.6 % |
| CBECS | chiller (AUC) | 0.77 | — |
| CBECS | primary use | F1 0.15 | majority 0.03 |
| CBECS | weekly hours MAE | 44 h | dev median 41 h; median of true use 30 h |
| CBECS | vintage MAE | 27.0 y | 26.0 y |
| HUE (24 homes) | electric heat | hourly AUC 0.46, 50 % accuracy (monthly AUC 0.69, 75 %) | majority 75 % |

Simulator-vs-simulator results. These are **not reality**:

- **Held-out BuildingSim buildings, hourly input:**

  | output | Detective MAE | PRISM | median of true use |
  |---|---|---|---|
  | UA (log10) | 0.10 | 0.85 | 0.35 |
  | tau (log10) | 0.14 | — | 0.21 |
  | heating balance point | 2.1 K | 6.5 K | 3.9 K |
  | cooling balance point | 2.2 K | 4.9 K | 3.9 K |
  | weekday hours | 1.0 h | — | 3.2 h |

  Heating type balanced accuracy 0.84 (meter rule 0.35). Heat pump vs resistance AUC 0.94.
- **NREL EnergyPlus meters, with truth from a BuildingSim twin of the same characteristics:**
  - ResStock (58 homes): UA 0.29 log10, biased +0.25. That is worse than the sample median (0.23); PRISM gets 0.73.
  - ComStock (90 buildings): UA 0.19 log10, vs sample median 0.44 and PRISM 0.76.

**Coverage at ±1σ** (68 % is ideal):
- BDG2 area: 72 % hourly, 54 % annual
- RECS: area 69 %, vintage 74 %
- simulated: UA 67 %, tau 72 %

## Top limits
1. Primary use does not transfer to real buildings, and year built has no real skill anywhere.
2. Heat pump vs resistance fails on real data: RECS AUC 0.40, HUE hourly AUC 0.46. The simulator's heat-pump signature is not what real homes show.
3. UA and tau have no real truth. On ResStock, hourly UA is biased and no better than the median. Real BDG2 load shapes still differ from simulated ones, which is why hourly area needed 3× wider error bars.

## Next rungs
- **Physics v3 from the remaining twin gaps.** Real electricity is much less weather-sensitive than the twins (PRISM r² 0.07 vs 0.24). Add a calibrated mix of meters that serve unconditioned space or process load.
- **Heat-pump realism.** Fit the HP signature (backup strips, defrost, sizing, ResStock-like controls) on the RECS and HUE dev splits, or on WPuQ heat-pump homes as a dev set.
- **Domain-adversarial or twin-consistency training** using unlabelled real BDG2 history-year meters (2016), plus a population prior for use classes fitted on dev.

## Upstream candidates (for BuildingSim)
- W1 intensity factors
- W2 campus distribution factors
- W3 base-load shares
- `common.clean_labels`: base_share is undefined when metered site energy is ~0
- The JS feature port (`web/detective.js`), verified against Python to 2e-7

## Rule notes (unsure if allowed)
- ResStock/ComStock meters were used only for the reported sim-vs-sim check, never for choosing.
- Exam features were computed before the exam, with no metrics looked at.
- The R3c withdrawal was done by hand: it was kept by the rule, then reverted for breaking the size cap.
- BDG2-derived example series in the zip are CC BY-SA 4.0.

## Sandbox gotchas
- `nohup ... &` writes the PID of the setsid wrapper, not the job. Use `pgrep` to get the real PID.
- pandas tz-aware `asi8` is not in ns in this pandas version. Use `datetime64[D]` for day keys.
- The ResStock/ComStock area column is binned.
- Four agents shared 2 cores. The bank ran at 2–6 min per 200-building shard depending on load.

---

## 17 Nov 2026: imported into ML Ratchet

**Layout.** The zip had no top-level folder: a `buildingdetective/` package (`__init__.py`, `detective.py`,
`features.py`, `common.py`), `weights/`, `examples/`, `training/`, `demo.py`, `pipeline.py`. The repository's tests
import every inference file as a top-level module of the model folder (`tests/test_contract.py`), so the package was
flattened: `buildingdetective/detective.py` became `buildingdetective.py` (with `__version__` from `__init__.py`), and
`features.py` and `common.py` sit beside it. The only code changes: `from . import common/features` became plain
imports, and `Detective()` now defaults to `weights/` next to its file (`Detective('weights')` still works).
`from buildingdetective import Detective` works as before, so `demo.py` and `pipeline.py` are unchanged apart from
the example split below. `weights/` (three .npz files and calibration.json) is as shipped. The zip's `__pycache__`
was not copied.

**Licence folders.** BDG2 is CC BY-SA 4.0. The three BDG2 exam buildings moved out of `examples/examples.*` into
`examples/bdg2/bdg2_hourly.{npz,json}` (same arrays, same keys) with `examples/bdg2/LICENSE-DATA.md`; the HUE home
and the RECS/CBECS respondents stay in `examples/examples.*`. `demo.py` reads both and prints the same units in the
same order. `lab/index.html` (copied byte for byte from the page, sha256 ad5d9543…3d30, 6,781,024 bytes) embeds 29
BDG2 buildings and 5 HUE homes, so `lab/LICENSE-DATA.md` was added; the instrument exporter writes the same notice
into `site/public/instruments/buildingdetective/`. MODEL_CARD.md's "Data and licences" lists every BDG2-derived file.

**Paths.** `training/_paths.py` (SRC, MODEL, REPO, WORK, BLDG). `common.ROOT` is now WORK (the sandbox's
`bldg/detective`), so banks, features, checkpoints, the web build and a re-run's ratchet.json / exam.json go to
`models/buildingdetective/work/` or `$ML_RATCHET_WORK/buildingdetective`. BLDG (`shared/building/` or
`$ML_RATCHET_BUILDING`) replaces the sandbox's `bldg/` in `common.py`, `calib_campus.py`, `realdata.py` (HUE parquet
files) and `stockcheck.py` (checks, buildingsim/tools, ResStock/ComStock samples); `run_bank.sh` uses relative paths
and `$ML_RATCHET_WORK`. The shipped logs (ratchet.json, exam.json, twintest.json, calibration.json) stay in
`training/` as they were.

**What was re-run.** The demo, from the repository folder with the venv python (numpy 2.5): every printed number is
identical to a run of the shipped copy (the zip's README shows no demo output; the status doc gives 0.6 s, here
0.8 s). Recorded as `tests/golden/buildingdetective.txt`. Nothing in `training/` was re-run: it needs BuildingSim,
BuildingData and their data, which arrive with the platform zip (buildingsim-buildingdata-v1.0.zip). Numpy against
torch could not be re-checked (no checkpoints shipped).

**Ratchet and exam as imported.** `ratchet.yaml` has one ladder per network, each rung exactly as `ratchet.json`
logs it, with the physics-v2 re-baseline and the label-hygiene re-score as rungs, R3c-hourly as rejected (kept by the
rule, withdrawn by hand for its 1.16 M parameters) and R9 released in each. The dev gate is chained across the three
networks (each rung against the last kept rung of any network): it drifted 0.331 → 0.306, 0.025 in all, with every
step inside the 0.02 tolerance. Its area scores are normalised by BuildingData's dumb-baseline errors (64.3 %, 76 %),
which BuildingData computed on the exam split; they are constants, not a model's exam result. Exam verdicts: a margin
smaller than about one sampling error is a tie. That makes BDG2 floor area (50.0 / 49.4 / 48.0 % against 51.5 % for
the like-for-like rule, and 46.4 % for the same rule on BuildingData's quality-checked subset) three ties, and RECS
vintage (22.1 vs 22.3 years; bold in the card) a tie. Real rows: 5 wins, 4 ties, 9 losses; 14 simulator rows are info.

**Live instrument.** `web/export_instrument.py` writes the three networks (float16 layers, float32 normalisation;
float16 moves raw outputs by at most 9.4e-3), the 12 example units with local time fields and a fixed noise series,
and 60 reference cases. `site/src/instruments/buildingdetective/net.ts` ports features.py from the shipped JS port
(`training/detective.js`) and the network; worst relative difference 1.1e-6 on the 60 cases, 2.4 MB in all, about
20-50 ms per run. A wider sweep (1,440 control settings on the 4 hourly units) matched to 1e-6 except 30 settings,
all of them degenerate PRISM fits: when every day lies on the heating side of the best balance point, all higher
balance points fit equally well and numpy and JS break the tie differently (Jan-Mar only, or the HUE home with the
weather shifted down). The release's own reading depends on floating-point noise there; the instrument flags it. The
one-month setting was dropped for that reason.

**Found with the instrument.** With the meters unchanged, a ±4 K weather shift should move the balance points by
±4 K. The hourly network moves Lamb_education_Bert's heating balance point 11.3 / 12.4 / 13.4 °C and the HUE home's
17.2 / 17.1 / 16.3 °C; the monthly network follows more (Lamb 5.8 / 10.1 / 13.0 °C). The hourly network reads the
balance point from the load shape more than from where the load meets the weather.

**Small things.** `features.py` uses `DatetimeIndex.dayofweek`, which pandas 3 deprecates (Pandas4Warning); it still
works with the venv's pandas. `verify_js.py` and `buildpage.py` expect the JS port in WORK/web/.

## 17 Nov 2026: integrity pass, the plan (written before anything is re-run)

The v1.0 exam compared the Detective mostly with rules that are given the true use type, or with chance, and called
three floor-area margins "ties" by eye. This pass re-scores the exam with the released weights and adds fair
baselines and intervals. **The exam is already seen** (v1.0 took it once): every re-score is labelled "seen", and
nothing about the model is chosen from it. No new rung. `splits.json` is not touched.

1. **Re-run.** Rebuild the real feature matrices (`realfeats.py` logic: BDG2 dev + exam hourly / monthly / annual /
   electricity-only, RECS, CBECS, HUE) into `work/feats/real/` from the unpacked platform data, at most 3 processes.
   Run the real-data parts of `exam.py` (`evaluate.eval_bdg2 / eval_recs / eval_cbecs / eval_hue`, `cbecs_hours`,
   `bdg2_baselines`, `survey_rules`) with `weights/`. Compare with `training/exam.json`, including the per-unit z
   lists it kept, and say how close. The simulator rows (frozen validation set, dev twins, ResStock/ComStock) need
   the 60-shard bank and the stock features, which are not here; they are not re-run (step 6 times what they need).
2. **Every row exam.py computes goes into model.yaml and MODEL_CARD.md**, including the ones v1.0 left out (for
   example CBECS heat-pump presence AUC, the electricity-only floor area, the monthly and annual BDG2 rows, HUE monthly).
3. **Fair baselines, all fitted on dev only (52 BDG2 buildings, 1,848 RECS, 661 CBECS respondents; W1 targets use
   prior + dev as the simulator did), scored on the same exam rows:**
   - BDG2 floor area: electricity ÷ dev median EUI of the TRUE use (v1.0's rule; it is given the use, the Detective
     is not); electricity ÷ dev median EUI of all uses (no truth); a dev-fitted log-log line log A = a + b log kWh
     (no truth).
   - RECS / CBECS floor area: energy ÷ dev median EUI by TRUE type × climate (BuildingData's rule, recomputed per
     unit); energy ÷ W1 target EUI of the TRUE use (the weighted prior + dev median the simulator was tuned to); dev
     median by TRUE type; a dev-fitted linear model of log area on the same 12 annual features the network gets
     (no truth).
   - Heating class (RECS fuel / electric / none; CBECS fuel / electric / district / none): the meter rule, and a
     dev-fitted multinomial logistic model on the 12 annual features.
   - RECS heat pump vs resistance: BuildingData's kWh/m²/HDD rule with the TRUE area and its dev threshold
     (recomputed); the same without area (kWh/HDD, dev threshold); a dev-fitted logistic model on the annual features.
     Balanced accuracy and AUC. A change-point inversion is not possible on annual totals.
   - CBECS chiller and heat-pump presence (AUC), primary use, weekly hours, vintage; RECS vintage: dev median / dev
     median of the TRUE use where v1.0 had it, plus dev-fitted logistic / linear models on the annual features.
   - BDG2 use: majority; nearest dev centroid of the 168-hour electricity week profile, on the same rows (v1.0 only
     quoted BuildingData's centroid number on a different set). BDG2 year: dev median; dev median of the same site.
   - BDG2 chilled-water / steam meter from electricity: a dev site prior (share of dev buildings at the same site
     with that meter; the site is known from the latitude and weather the Detective also gets) and a dev-fitted
     logistic model on the electricity PRISM parameters.
   - HUE (24 homes, no dev split exists): majority; a PRISM change-point inversion of daily electricity: AUC of the
     heating share (heating slope × HDD at the fitted balance point ÷ annual energy), and "electric heat if that share
     ≥ 0.20" for accuracy (threshold fixed here, before HUE is read).
4. **Intervals.** For every real row: 95 % bootstrap interval on Detective minus the best baseline, 2,000 resamples,
   seed 20261117, resampling the exam unit (building, respondent, home), both scored on the rows both can score.
   Win if the interval is on the good side of zero, loss if on the bad side, tie if it spans zero. The best baseline
   is the best point value among those listed (some are given truth; the note then also gives the interval against
   the best baseline that needs no truth).
5. **Dev twin test of what the network adds** (dev only): for area, vintage, hours and the heating class, correlate
   the network's correction to the dev-calibrated rule (network − rule) with what reality needed (truth − rule), on
   BDG2, RECS and CBECS dev; report where the residuals have structure (use type, climate, housing type).
6. **Readiness.** One tiny shard (~20 buildings) through `simbank.py` and `featurize.py`, one training epoch into
   `work/`, timed; extrapolate the 60-shard bank and the frozen set; list blockers.
7. Update model.yaml (seen labels, intervals in notes, verdicts by the interval rule), MODEL_CARD.md, the ladder
   note and a results entry here.

## 2026-11-17 — Fresh exam frozen before round-2 rungs: HEAPO (Zurich heat-pump homes)

Written before HEAPO is downloaded. Nothing is scored in this step: no Detective run, no baseline score. Built by
`training/exam2_heapo.py` into `work/exam2/heapo/` (git-ignored); frozen in `training/split_exam2_heapo.json` (units,
hashes, rows, baselines, interval rule). Scored once for v1.0 and once for any round-2 release, after the round-2
release decision.

**Data.** HEAPO, Zenodo 10.5281/zenodo.15056919 v1 (CC BY 4.0, `heapo_data.zip`, 458 MB, md5 2b92b0d5...):
1,408 heat-pump homes in the canton of Zurich, 15-min and daily smart-meter kWh, Nov 2018 - Mar 2024, 8 weather
stations, 410 site-visit protocols. Unit-clean: no Swiss home is in training, dev or exam 1.

**Input contract.** `read_hourly(meters, T, GHI, loc, lat)` with `meters={'elec': hourly kW}`; monthly and annual
modes are derived from the same hourly year (`monthly_from_hourly`; annual kWh with HDD/CDD base 18.33 degC from the
same T). HEAPO has electricity only (no gas, no district heat), so only `elec` is given. elec = the sum of all of a
home's 15-min electricity meters (household and, where separate, heat pump); an hour counts only if every quarter-hour
of every meter is present. T from the home's assigned HEAPO station; GHI from the station if it logs global radiation,
otherwise clear-sky x 0.6 (pvlib), as `realdata.hue_cases`. tz Europe/Zurich; lat of the station.

**Truth** (protocol only, first visit if several): heated floor area (`Building_FloorAreaHeated_Total`), construction
year, building type (-> residential_single / residential_multi), heat-pump type (air / ground), emitters (floor
heating / radiators). The Detective has no air-vs-ground or emitter output: those two are recorded as an output gap
and not scored.

**Quality control** (BuildingData's BDG2 electricity flags, `build_bdg2.flags_for`: missing, zero, zero runs >= 24 h,
negative, spikes, flatlines >= 24 h, linear-fill runs >= 12 h, level-shift days; then HUE's window rule): use the most
complete 365-day window starting on a month boundary; keep the home if it has >= 300 x 24 good hours and < 30
level-shift days. Drop homes with no protocol, with PV (protocol or survey; the meter reads net import), with floor
area outside 30-2,000 m2 (area rows) or construction year outside 1800-2024 (year rows). No filter uses energy per
area, so no rule leans on the truth-to-meter relation the baselines exploit. Every dropped home is counted by reason.

**Split, by home, seed 20261117.** Protocol homes that pass QC: 20 % dev, 80 % exam, stratified by building type x
heat-pump type (at least one dev home per stratum with >= 3 homes). Dev only fits the baselines.

**Rows** (each for hourly, monthly and annual input): (1) heated floor area, median abs. % error; (2) construction
year, mean abs. error (yr); (3) single vs multi-family accuracy (the higher of P(residential_single) and
P(residential_multi)); (4) info, no verdict: share of homes with heat_pump as the top heating class and the median
P(heat_pump) (every home has one, so no fair baseline exists). Model input: the released `weights/` through
`Detective.read_hourly / read_monthly_from_hourly / read_annual`.

**Baselines, fitted on the dev homes.** Area: annual elec / dev median kWh/m2 of the TRUE building type (given truth
the Detective lacks); annual elec / dev median kWh/m2 (no truth); dev-fitted line log A = a + b log kWh; dumb dev median
area. Year: dev median; dev-fitted line year = a + b log kWh. Type: dev majority.

**Intervals.** 95 % bootstrap over exam homes, 2,000 resamples, seed 20261117, paired, on the model's statistic minus
the best baseline's (best = better point value on the row). Win / loss if the interval is entirely on one side of
zero, tie if it spans zero. n homes stated.

## 17 Nov 2026: integrity pass, results

Everything below followed the integrity plan above (written before anything was re-run); nothing was changed after a
result was seen. Scripts: `training/integrity_feats.py` (real features, 3 processes, 66 s for BDG2 exam),
`training/integrity.py` (re-score, baselines, intervals, twin test); log `training/integrity.json`. No rung.

- **Reproduction.** RECS, CBECS, HUE, BDG2 annual and every v1.0 baseline (`bdg2_baselines`, `survey_rules`,
  `cbecs_hours`, majority) match `training/exam.json` to 1e-12, including the per-unit z lists. BDG2 hourly: 24 of
  1,428 buildings read differently (Lamb, Panther, Wolf, Cockatoo), median area error unchanged at 50.0 %. BDG2
  monthly: 157 buildings differ; area error 49.6 % (v1.0 49.4 %), use 14.4 % (14.1 %), year 25.3 yr (25.2). Cause:
  exact ties in the PRISM grid fit broken by floating-point noise; a 1e-12 relative perturbation of the meters moves
  the PRISM features by up to 0.75 (monthly) and 14.6 (hourly) on the affected buildings.
- **Baselines recomputed.** BuildingData's floor-area rules per unit give its quoted numbers (RECS 33.1 %, CBECS
  38.7 %); the TRUE-use BDG2 rule gives 51.5 % on 1,364 buildings as v1.0.
- **Verdicts (28 real rows): 0 wins, 21 losses, 7 ties** (v1.0: 5 wins, 4 ties, 9 losses on 18 rows). Detective
  minus the best baseline, 95 % interval:
  - BDG2 floor area against a dev-fitted line log10 A = 0.73 log10 kWh − 0.49 (40.9 %, no use given): hourly +8.8
    [+5.9, +11.3], monthly +6.2 [+3.2, +9.5], annual +4.8 [+2.2, +7.3], electricity only +9.8 [+7.3, +12.3]. All
    losses. Against v1.0's TRUE-use rule: hourly −1.8 [−6.0, +2.0] (tie), monthly −4.4 [−8.5, −0.4] and annual
    −5.9 [−9.3, −2.5] (wins). So v1.0's ties were a tie and two wins against the weak rule, and losses against a fair one.
  - BDG2 meters from electricity: a dev site prior gives AUC 0.83; chilled water −0.077 [−0.112, −0.043], steam /
    hot water −0.130 [−0.165, −0.094]. A dev logistic model on the electricity PRISM statistics ties the network on
    chilled water (+0.003) and beats it on steam (−0.052).
  - Survey rows against ridge-regularised linear / logistic models on the same 12 annual features, fitted on dev:
    RECS heating 87.3 vs 91.9 % (−4.7 [−5.2, −4.1]); CBECS heating 74.7 vs 79.2 % (−4.5 [−5.9, −3.2]); CBECS use
    16.5 vs 28.4 %; CBECS chiller AUC 0.77 vs 0.80 (−0.036 [−0.052, −0.020]); RECS heat pump vs resistance AUC
    0.40 vs 0.63; RECS area 33.6 vs 29.6 % (+4.0 [+3.3, +4.7]); RECS vintage 22.1 vs 21.8 yr (+0.3 [+0.1, +0.6]);
    CBECS hours 44.2 vs 37.4 h. CBECS area: loss against the TRUE type × climate rule (+4.2 [+1.9, +6.3]), tie
    against the dev linear model (−1.8 [−3.9, +0.1]).
  - Ties: BDG2 year built monthly (+0.3 [−0.4, +1.1]) and annual (−0.1 [−0.8, +0.6]) against the dev median; CBECS
    heat-pump presence (−0.031 [−0.069, +0.007]); the four HUE rows (24 homes, 6 electric): hourly accuracy −25
    [−54, +8], monthly AUC 0.69 vs the PRISM heating share's 0.55, +0.15 [−0.04, +0.36].
- **Dev twin test.** The network's correction to the dev-calibrated rule against truth minus that rule: RECS area
  r = 0.60 (n = 1,848), RECS fuel heat 0.60, CBECS area 0.54, CBECS fuel heat 0.56, CBECS hours 0.43, RECS vintage
  0.22, BDG2 area 0.12 / 0.10 / 0.22 (hourly / monthly / annual, n = 52). The corrections point the right way but are
  biased and too large: CBECS hours +37 h median (offices +56 h, healthcare −42 h), error 27 → 45 h; BDG2 hourly area
  median −0.13 log10, education needed +0.15 and got −0.16; RECS apartments in 5+ unit buildings needed +0.09 and got
  +0.28; CBECS food service needed −0.66 and got −0.17. A dev-fitted model gets the size right, which is why it wins.
- **Readiness smoke test** (Windows, 2 processes): `simbank.py` with 20 buildings per shard (shards 0 and 900) took
  24-25 s per shard, no failures; `featurize.py --w3 --once` 14 s then 3 s; `train.py --epochs 1` ran for all three
  networks (about 20 s each, mostly torch start-up). One full-size epoch: hourly 2.0 s, monthly 0.7 s, annual 1.2 s.
  Extrapolated: 13,000 buildings (60 shards + frozen 900-904) ≈ 4.5 CPU-hours of simulation, about 1.5 h on 3
  processes, plus 0.2-1 h of features; 60 epochs ≈ 4 min. Blockers: ResStock/ComStock samples not unpacked
  (`data/proc/stock`, ~1.2 GB) so the sim-vs-sim rows cannot run; `run_bank.sh` needs bash and `python3`; the
  rebuilt frozen set must re-score the released weights at 0.813 (hourly) before it judges a rung; training-shard
  features use time-based seeds, so a rebuilt training set will not match v1.0's bit for bit; copy
  `training/calibration.json` to `work/models/` first.
- **What it means.** v1.0's real wins were wins over weak rules. With baselines fitted on the same dev data, the
  network adds nothing measurable on any real row; its corrections are informative in direction only. The next rung
  should predict corrections to the dev-fitted baselines rather than raw outputs (model.yaml `next`).

## 2026-11-17 — Fresh exam frozen: HEAPO results (nothing scored)

Built by `training/exam2_heapo.py`; no Detective run and no baseline score. `--check` rebuild reproduces the split.

- **Downloaded.** `heapo_data.zip`, 458 MB, md5 matches Zenodo (sha256 `9979e1e3…`).
- **What the data hold.** 217 of the 410 protocols link to a household with smart-meter data (214 homes; 193
  protocols have no household). Among the 214: 208 single-family, 2 multi-family, 6 "other"; 131 air-source, 85
  ground-source, 1 water-source. The weather stations log temperature and sunshine duration, not irradiance, so GHI is
  clear-sky x 0.6 as planned. Station coordinates are anonymised: lat 47.38 (Zurich) is used.
- **Two changes after reading QC counts, before any scoring.** (1) The BDG2 level-shift rule flags the heating season
  itself (median 145 flagged days per home; 72 homes failed the window because of it). It is not applied; its day
  count is kept per home. (2) The whole-home meter is `kWh_received_Total`; 5 homes with only a heat-pump meter are
  dropped (the plan's "sum of meters" would have double-counted where Total exists).
- **Units.** Dropped 84: PV 49, < 300 good days in any 365-day window 30, heat-pump meter only 5. Kept 130: **25 dev,
  105 exam** (area row 105, year row 100: 5 have no or an implausible year). Exam: 63 air, 41 ground, 1 water source.
  The single vs multi-family row is info only (2 multi-family exam homes).
- `split_exam2_heapo.json` sha256 `fda36a6846ec1aaa3ddac7cb0b018dae0bf4d158f26414d5237d67c1f57a9b16` (units block
  `443961af…`; no readings, floor areas or years stored: truth is re-read from the protocols at scoring).

## 2026-11-17 — Rung r10 plan: adopt BuildingSim physics (written before any bank is simulated)

Physics 1.1-1.4 are measured on dev (`shared/building/NOTEBOOK.md`); none met its own done-when. They count for this
model only if its own ladder says so. This entry fixes the physics, the frozen set, the rung and the rule before any
building is simulated. The fresh HEAPO exam (3d9f949) is not scored here and no HEAPO home (dev or exam) is used for
anything in this rung. `splits.json` is not touched; exam units train and tune nothing.

**Physics chosen: 1.1 + 1.2 + 1.4, without 1.3.** BuildingSim versions are cumulative in `simulate()` (`campus.at_least`:
1.3 = 1.2 + heat pumps), but 1.4 lives only in `generate_buildings(physics='1.4')` (intensity per use type on the
generated spec); `simulate()` at 1.4 is 1.3. So the two public flags select what I want without overriding any fitted
parameter: specs from `generate_buildings(..., physics='1.4')`, simulated with `options={'physics': '1.2', 'seed':
crc32(spec id)}`. Bank tag `p12`. 1.3 is left out because it made the HEAPO dev profile worse (r 0.24 -> 0.21, annual
ratio 1.59 -> 1.71) and was not recommended. Caveat: the 1.4 factors were fitted with 1.3 inside the simulation; that
touches only the heat-pump homes in the residential fit sample, and I use the factors as published (not refitted).
Physics 1.2 changes only non-residential twins; 1.1 only district-served ones; 1.4 every generated building.

**Wrappers (training/wrappers.py).**
- W1 intensity: **dropped.** 1.4 is the same mechanism (`phase2.apply_intensity`), refitted on the prior split.
  Applying both would scale twice.
- W2 campus: **assignment kept, factors and knobs dropped.** The generator rarely makes district-served buildings;
  BDG2 is full of them, and no physics version decides which generated building sits on a campus. So 30 % of
  commercial buildings still get campus district service (kind both / heat / cool, hot water vs steam), drawn with the
  same rng calls as v1.0 (so the same buildings and weather as bank v1 where the generator draws agree). Dropped: the
  distribution factors (heat 0.18, chilled water 0.33, meter = delivered / factor) and the operating knobs (VAV minimum
  0.3-0.6, outdoor air x1-2, economizer, 24 h air handlers). 1.1's loop base was fitted on metadata twins with factor
  1 and no knobs (`check_bdg2.py --physics`), and it brings the annual district ratio to 1.0 on dev by itself; W2's
  factor on top would roughly triple chilled water again. Lost: W2's per-building spread (log-sd 0.8); 1.1's base is
  one value per m2 per loop type, so campus district meters in the bank will be more uniform than BDG2's (real steam
  base 1.4-81 W/m2 across sites). Recorded as a limit, not fixed here.
- W3 flat base loads: **dropped.** 1.1's loop base and lockouts and 1.2's always-on share and day-to-day activity
  replace it, and W3 was wrong for hot-water loops (0.70 of the mean in summer against 0.01 real).
- Labels: unchanged code. 1.1's loop base is booked as its own end uses (`plant_*_district`), not as heating or
  cooling, so it counts in `base_share` as non-HVAC energy, as W3's base did.

**Bank and frozen set.** `simbank.py p12`, the same shard ids and seeds as v1: 60 training shards (0-59, 12,000
buildings) and the new frozen validation set, shards 900-904 (1,000 buildings), built once from fixed seeds and never
regenerated for a rung. Featurized as `f3` (no `--w3`), 2 hourly and 2 annual meter-quirk draws per building, 1
monthly; frozen-set quirks are seeded per building (as v1.0). At most 3 processes; a shard is claimed by a lock file.
If the bank is not done in the time I have, r10 trains on the shards finished, and says how many.

**Re-score of the incumbent.** R9 (the released `weights/`, all three networks) on the f3 frozen set: its skill there
is the number r10 must beat. R1-R8 cannot be re-scored (no checkpoints shipped); the ladder says so.

**Rung r10.** Retrain hourly, monthly and annual on f3, the same architecture and settings as R9 (MLP 2 x 256 GELU,
dropout 0.1, AdamW lr 2e-3, wd 1e-4, 60 epochs, OneCycle, seed 0). One run per network, no other variant.

**Rule (the ladder's, per network, in the order hourly, monthly, annual).** Keep if frozen-val skill > R9's f3 skill +
0.01 AND the real dev composite (`evaluate.dev_gate`, the candidate in place of its network, the others as currently
kept) is no more than 0.02 below the composite of the last kept set. Starting composite: the v1.0 weights' composite
recomputed now with today's features (the logged 0.3061 is reported beside it; temperatures and widen factors do not
move the composite, it reads argmax and means only).

**Also reported (dev only, not part of the rule).**
- Dev twin test of what the network adds, as in the integrity pass (`training/integrity.py` functions): the
  network's correction to the dev-calibrated rule against truth minus that rule, for BDG2 area (3 resolutions), RECS
  and CBECS area, vintage, fuel heat, CBECS hours; Pearson r, median correction vs median needed, MAE of the anchor
  vs after the network, and the breakdowns the integrity pass named (BDG2 area by use, CBECS hours by use, RECS area
  by housing type, CBECS area by type). For v1.0 and r10 side by side. The question: does the new physics shrink the
  bias?
- The dev-fitted baselines' dev numbers next to the network's: the same classical models as the integrity pass
  (BDG2 log-log line; ridge on the 12 annual features for RECS/CBECS area, vintage, hours; logistic for heating
  class, CBECS use, RECS heat pump vs resistance), scored on dev by 5-fold cross-validation (folds by unit, seed
  20261117), so no baseline is scored on the rows it was fitted on.

**If a network is released (v1.1).** Kept networks replace v1.0's, the others stay. The release steps: rebuild the
BDG2 dev twins at p12 (`devtwins.py`, same seeds) for the sim-only widen factors, `calibrate.py` (dev only) for widen
factors and class temperatures; weights CC BY-SA 4.0; numpy inference unchanged in form; demo + golden; model.yaml,
ladder, card, notebook; the seen exam re-scored for v1.1 with `integrity.py`'s baselines and intervals, labelled seen;
instrument re-exported, parity test. If nothing is kept, v1.0 stays and the ladder logs r10 and the re-baseline.

## 2026-11-17 — Rung r10 results: hourly kept and released as v1.1; monthly and annual rejected; physics did not fix the bias

Everything followed the plan entry above. Nothing was changed after a result was seen. Two code changes came after
the plan and before any result: `devtwins.py --p12` applies the 1.4 intensity after the real area, year and floors
are set (generate_buildings at 1.4 returns a filled spec, so overriding the area afterwards would have left
area-dependent defaults such as DHW volume and occupants at the generated size), and `integrity.py` takes a weights
folder. Log: `training/r10.json` (`training/r10.py`); seen re-score `training/integrity_v11.json`.

**Bank.** p12, 60 training shards (12,000 buildings) + frozen shards 900-904 (1,000), 0 simulation failures; 3
workers (`training/runbank.py`, lock files), 119-131 s per 200-building shard plus 15-24 s of features, 70 min in
all. Campus share as planned (83 of 280 commercial buildings, 30 %, in the first two frozen shards). Physics 1.1's chilled-water
base (42 W/m², 368 kWh/m² a year when the loop runs all year) now dominates campus chilled-water meters. Frozen-set
hashes: `work/frozen_p12_sha256.txt`.

**Re-baseline.** R9 (released weights) on the p12 frozen set: hourly 0.7135, monthly 0.5568, annual 0.4301 (v2 set:
0.8126 / 0.6560 / 0.5412). The biggest drop is base_share (hourly R² 0.22): 1.1's loop base is booked as non-HVAC
energy. R1-R8 cannot be re-scored (no checkpoints). v1.0 dev composite recomputed with today's features: 0.3081
(logged 0.3061; the integrity pass explains the PRISM-tie difference). The chain starts there.

**r10** (MLP 2 x 256, the R9 settings, 60 epochs, seed 0; about 2 min each):

| network | val skill | R9 on p12 | delta | dev composite | floor | decision |
|---|---|---|---|---|---|---|
| hourly | 0.8301 | 0.7135 | +0.117 | 0.2903 | 0.2881 | kept |
| monthly | 0.6736 | 0.5568 | +0.117 | 0.2661 | 0.2703 | rejected (also below v1.0's floor 0.2881) |
| annual | 0.5513 | 0.4301 | +0.121 | 0.2447 | 0.2703 | rejected (also below 0.2881) |

What moved the composite: BDG2 dev floor area 50.8 -> 69.0 % (hourly), 48.5 -> 67.9 % (monthly), 40.2 -> 71.2 %
(annual); CBECS dev area 43.4 -> 52.0 % and RECS 32.9 -> 35.0 % (annual). The hourly network passed by 0.002 because
its area term was already near zero and is clipped there, so a 18-point loss on BDG2 dev area costs the composite
only what the one term held (0.21 / 12). By the rule it is kept and released. I report that the rule let through a
network that is worse on the only real dev quantity it reads (area), and propose in model.yaml `next` that each
composite term be scored against its dev-fitted baseline.

**Dev twin test** (network correction to the dev rule vs what reality needed; v1.0 -> r10): BDG2 area r 0.12 -> 0.03
(hourly), 0.10 -> -0.01 (monthly), 0.22 -> 0.03 (annual); median correction -0.13 -> +0.10, -0.03 -> +0.14, -0.05 ->
+0.09 log10 (needed: 0); error after the network 0.32 -> 0.37, 0.30 -> 0.39, 0.26 -> 0.37 (anchor alone 0.27). RECS
area r 0.60 -> 0.60, correction -0.03 -> +0.04; CBECS area r 0.54 -> 0.48, -0.08 -> +0.06; CBECS hours +36.7 h ->
+36.6 h. **The new physics does not shrink the bias the integrity pass found.** It flips the sign of the area bias:
the networks now read real meters as larger buildings. My reading (not tested): 1.4 lowers generated intensities
(education x0.52, healthcare x0.51, retail x0.46), fitted to survey medians, while BDG2's campus buildings use more
energy per m² than those medians; a network trained on the lower intensities divides real energy by too small an EUI.

**Dev-fitted baselines on dev** (5-fold CV): BDG2 area 34.7 % (log-log line) vs network 50.8 (v1.0) / 69.0 (r10)
hourly and 40.2 / 71.2 annual; RECS area 29.2 vs 32.9 / 35.0; CBECS area 45.2 vs 43.2 / 51.3; RECS heating 91.3 vs
86.9 / 87.9 %; CBECS heating 77.8 vs 70.9 / 70.4 %; RECS heat pump vs resistance AUC 0.66 vs 0.41 / 0.42; CBECS use
25.9 vs 16.5 / 15.1 %; CBECS hours 36.5 vs 44.6 / 44.9 h. The baselines still win on dev, as on the exam.

**Release v1.1** = r10 hourly + v1.0 monthly and annual. BDG2 dev twins rebuilt at p12 (`devtwins.py devtwin_p12 4
--p12`, 208 twins, 0 failures); `calibrate.py` on the hourly network only: floor-area widening x3.43 (v1.0 x3.02),
year x1.00, use temperature 10. Weights CC BY-SA 4.0 (`weights/LICENSE-DATA.md` unchanged; it covers retrained
weights). `buildingdetective.py` version 1.1, inference unchanged. Demo and `tests/golden/buildingdetective.txt`
regenerated (the three BDG2 and the HUE hourly readings changed; e.g. Lamb_education_Bert 1,692 -> 4,390 m², truth
2,403). Instrument re-exported; parity 60 cases, worst relative difference 1.07e-6.

**Seen exam re-scored for v1.1** (labelled seen; nothing chosen from it): 0 wins, 22 losses, 6 ties (v1.0 21 / 7).
BDG2 hourly area 50.0 -> 56.6 % (minus the dev line +13.8 [+10.0, +17.4], loss); electricity-only area 51.0 -> 55.1 %
(+14.1 [+11.0, +17.6], loss); chilled-water AUC 0.751 -> 0.760 (minus the site prior -0.067 [-0.101, -0.034], loss);
steam AUC 0.697 -> 0.740 (-0.087 [-0.122, -0.052], loss); BDG2 use 11.5 -> 12.8 % (loss); year 28.3 -> 27.6 yr
(loss); HUE hourly accuracy 50.0 -> 41.7 % (-33.3 [-58.3, -4.2] vs the majority, now a loss), AUC 0.46 -> 0.55 (tie).
The HEAPO exam was not scored and no HEAPO home was read.

## 2026-11-17 — r10 release choice: hourly kept by the rule, not released; v1.0 stays

The coordinator's decision, following WakeNet's r10 (its NOTEBOOK, 17 Nov 2026): a rung the rule keeps but that is
no better on real dev is not released, and that choice is made on dev. The v1.1 release in the entry above was undone.

- **Reason, from dev.** BDG2 dev hourly floor-area error went from 50.8 % to 69.0 %, worse than the dev median's
  64.3 %. The dev composite fell only 0.018, because area is one of twelve terms and its term was already near zero
  and is clipped there.
- **Not exam-blind.** The seen exam had already been re-scored with r10 hourly in place (`training/integrity_v11.json`,
  entry above) before this choice, and it also got worse: BDG2 hourly area 50.0 to 56.6 % (22 losses, 6 ties against
  v1.0's 21 and 7). So the choice was made knowing an exam result, though the dev evidence alone is enough for it.
- **The gap this exposes.** The composite gate lets a headline output that is 35 % worse (69.0 / 50.8) pass.
- **Next round.** Add a per-output dev gate on the headline real outputs (BDG2, RECS and CBECS floor area; RECS and
  CBECS heating type), each against its dev-fitted baseline, written down before the next rung.
- **What was restored.** `weights/` (all four files byte-identical to the v1.0 backup), `__version__` 1.0, demo and
  `tests/golden/buildingdetective.txt`, the instrument (`site/public/instruments/buildingdetective/`, `ui.ts`),
  model.yaml (integrity-pass rows and returns; r10 added only to `limits` and `next`) and the card's v1.0 numbers.
  Kept as the record: the r10 ladder rungs (hourly kept, not released; monthly and annual rejected),
  `training/r10.py`, `r10.json`, `runbank.py`, `integrity_v11.json`, the p12 code paths in `simone.py`, `simbank.py`
  and `devtwins.py`. The r10 checkpoints and the calibrated r10 hourly network stay in `work/models/` (git-ignored).
  The next rung's dev floor is v1.0's composite, 0.3081.

## 2026-11-17 — Fresh exam scored once: HEAPO (v1.0; r10 hourly for information)

Scored after the round-2 release decision was committed (r10 kept, not released; v1.0 stays). Nothing was changed
after a result was seen; nothing is chosen from it. Script `training/exam2_heapo_score.py`, results
`training/exam2_heapo_results.json`, per-home values in `work/exam2/heapo_scored_*.csv`.

- **Data check.** `exam2_heapo.py build --check` rebuilt all 130 processed homes: the units block matches the frozen
  split (sha256 `fda36a68…`); the scorer re-checks each home's hash before scoring.
- **As the spec says.** v1.0 = released `weights/` through `read_hourly`, `read_monthly_from_hourly`, `read_annual`.
  r10 = the calibrated r10 hourly network (`work/models/r10_rel`), labelled "kept but not released, information only".
  Truth re-read from the protocols. Baselines fitted on the 25 dev homes. Bootstrap over exam homes, paired, 2,000
  resamples, seed 20261117.
- **One baseline could not run as specified.** "Annual elec / dev median kWh per m2 of the TRUE building type": all 25
  dev homes are single-family, so there is no dev value for the 6 exam homes of other types. On the 99 single-family
  homes it equals the all-dev baseline (22.1 % there). It was left out of the best-baseline choice.
- Dev fits: median 56.6 kWh/m2, median area 180 m2, median year 1984, log A = 0.94 + 0.33 log kWh,
  year = 2156 − 41.4 log kWh, majority type single-family.

| Row (n homes) | v1.0 | Best baseline | v1.0 minus best | r10 hourly (info) |
|---|---|---|---|---|
| Floor area MdAPE, hourly (105) | 35.6 % | dev median area 14.6 % | +20.9 [+12.5, +30.0] loss | 39.9 %, +25.2 [+18.2, +30.6] |
| Floor area MdAPE, monthly (105) | 47.7 % | dev median area 14.6 % | +33.1 [+27.1, +36.8] loss | |
| Floor area MdAPE, annual (105) | 19.7 % | dev median area 14.6 % | +5.0 [−0.3, +9.1] tie | |
| Year built MAE, hourly (100) | 20.9 yr | dev median year 13.9 yr | +7.0 [+4.6, +9.7] loss | 21.4 yr, +7.5 [+5.2, +9.9] |
| Year built MAE, monthly (100) | 17.8 yr | dev median year 13.9 yr | +3.9 [+2.4, +5.5] loss | |
| Year built MAE, annual (100) | 14.0 yr | dev median year 13.9 yr | +0.2 [−0.3, +0.6] tie | |
| Single vs multi-family, info (101) | 87 / 96 / 98 % (h / m / a) | majority 98 % | no verdict (2 multi-family homes) | 89 % |
| Heat pump as top heating class, info (105) | 10 / 32 / 56 % (h / m / a) | none | median P(heat pump) 0.01 / 0.19 / 0.33 | 4 %, median P 0.001 |

- **What it says.** On Swiss single-family heat-pump homes the Detective does worse than guessing the dev median area
  and year from hourly or monthly input, and ties it from annual totals. It under-reads area (median predicted / true
  0.79 hourly, 0.53 monthly, 0.81 annual). Its heating-type output misses the one thing every home has: from hourly
  data it puts heat pump on top for 10 % of homes. r10 hourly is worse than v1.0 on area and year and on the heat-pump
  share, which agrees with the dev reason it was not released.

## 27 Nov 2026 — audit: short records fixed; district-meter inputs warned

The two-reviewer audit (`docs/audits/buildingdetective-v1.0.md`) found that a date axis shorter than a year pushed the
two record-span inputs to about 90 sd. HUE_2 at 330 days read 295.6 m² healthcare instead of 99.5 m² residential.
Training only ever saw a full year of weather with masked meters, so `read_hourly` now holds those two inputs at one
year and warns (330 / 180 / 90 days: 95.1 / 76.0 / 88.3 m², all residential). Both exams used full-year axes and are
unchanged. Two district-meter zero-fraction inputs are also far outside training on real BDG2 meters. The reader warns
there but does not clip, since no clip has been shown to be closer to the truth.

## 27 Nov 2026 — rung r11: district meters with real zero runs (plan, written before the run)

Owner-approved follow-up to the audit. Dev and simulation only; no exam unit and no HEAPO home is read.

- **What is out of range.** Input columns 904 and 1197 are the zero fraction of the district-heat and district-cooling
  meters. v1.0 trained on bank f2 (physics v2, always-on base loads), where they are 0.0002 ± 0.002; on the 52 BDG2 dev
  buildings they reach 387 sd and 347 sd (28 of 52 dev buildings have some input beyond 8 sd). r10's bank p12 (district
  plants, physics 1.1) already gives simulated district meters zero fractions up to 0.5 (dev: up to 21.6 sd), but r10
  was not released because its BDG2 dev area error rose from 50.8 % to 69.0 %. Dev district meters: zero fraction above
  0.05 on 8 of 24 heat and 9 of 25 cooling meters, up to 0.88; in f3 training above 0.05 on 13 % and 35 %, up to 0.51.
- **Rung r11 (hourly network only).** r10's recipe (bank p12, MLP 2×256, 60 epochs, seed 0, shards 0-59) with one
  change to the training meter quirks (`features.quirk_hourly(..., Th=, district_zeros=True)`): each district-heat or
  district-cooling meter gets, with probability 0.35, a seasonal plant shutdown (cooling off on days with mean T below
  U(2, 14) °C, heating off on days above U(12, 22) °C), and with probability 0.15 one outage of U(5, 50) % of the
  record. Features f4: training shards re-featurized with the change; the frozen set (shards 900-904) is f3's files,
  copied unchanged. Script `training/r11.py`.
- **Rule (the ladder's).** Keep if frozen-set skill beats the hourly incumbent by more than 0.01 and the dev composite
  (r11 in place of v1.0's hourly network, v1.0 monthly and annual) is no more than 0.02 below v1.0's composite
  (0.3081, floor 0.2881). The hourly ladder's incumbent is r10-hourly (0.8301 on p12, kept but not released); R9 on p12
  is 0.7135. The frozen set holds no long district zero runs, so the rung can only pass the margin by being better in
  simulation generally; expected outcome: within ±0.01 of r10-hourly, rejected. Also logged (information, no
  decision): BDG2 dev area MdAPE and the largest dev |z| on columns 904 and 1197.
- **Bounded inputs (variant, information only).** v1.0 with columns 904 and 1197 clipped at the training mean + 8 sd:
  frozen-set skill and dev composite. An inference change cannot raise frozen skill by 0.01 when the frozen set is
  inside the bound, so it is not a candidate under the rule.
- **Exam:** opened once only if a rung is kept and then released. Otherwise not.

## 27 Nov 2026 — r11 result: rejected; bounded variant logged; no exam

Run as planned (features 7 min on 4 processes, training 60 epochs). Nothing was changed after a result was seen.
`training/r11.json`.

| Hourly network (with v1.0 monthly + annual) | Frozen p12 skill | Dev composite | BDG2 dev area MdAPE | BDG2 dev use F1 | Dev max \|z\|, cols 904 / 1197 | Dev buildings with any \|z\| > 8 |
|---|---|---|---|---|---|---|
| v1.0 (R9) | 0.7135 | 0.3081 | 50.8 % | 0.092 | 387 / 347 | 28 of 52 |
| r10-hourly (kept, not released) | 0.8301 | 0.2903 | 69.0 % | 0.088 | 21.6 / 16.3 | 15 |
| **r11-hourly** | **0.8287** | **0.2998** | **60.7 %** | **0.146** | **6.1 / 6.1** | 12 |
| v1.0, cols 904 / 1197 clipped at mean + 8 sd | 0.7204 | 0.3157 | 43.5 % | 0.070 | (clipped) | 28 before the clip |

- **Decision: r11 rejected.** Frozen skill −0.0014 against r10-hourly (needs > +0.01); the dev gate passes (0.2998,
  floor 0.2881). The repair does what the audit asked of it: the two district inputs are now inside training
  (6 sd), and the use F1 on BDG2 dev rises from 0.09 to 0.15. But the frozen set has no long district zero runs, so
  it cannot reward the change, as the plan expected. And r11 still carries r10's area error (60.7 % vs v1.0's 50.8 %).
- **Bounded variant:** information only, not a candidate. It gains +0.007 on the frozen set (below the margin) and is
  better on dev (composite 0.3157, area 43.5 %). Dev is 52 buildings, and no rule lets dev alone keep a rung, so
  v1.0's reader is unchanged: it warns and does not clip.
- **What would decide it:** a frozen set whose simulated district meters include seasonal shutdowns and outages, built
  once before either candidate is scored again (a frozen-set change for the owner). The exam was not opened.
