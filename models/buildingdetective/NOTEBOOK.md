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
