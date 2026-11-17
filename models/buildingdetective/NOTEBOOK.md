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
