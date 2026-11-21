# The fire platform

The fifth shared platform. **FireSim** is a verified fire-behaviour simulator: Rothermel surface fire on 53 fuel
models, crown fire, spotting, dead and live fuel moisture, an arrival-time spread solver over real terrain,
suppression lines, and an emulator of what a satellite would see. **FireData** is real US wildfire data, split once
and frozen: 12-hourly progression for 7,739 fires (2012-2024), 1.59 million ignition records, live fuel moisture,
LANDFIRE fuels and hourly HRRR weather for 88 fires, and classical baselines already scored on the exam. Four models
sit on it. All four were built on 19 Nov 2026 by parallel agents in a claude.ai sandbox, on top of the platform built
there on 18-19 Nov, then brought into this repository on 21 Nov 2026.

| Model | The question |
|---|---|
| [FireCast](../models/firecast) | Which ground will this fire have burned by tomorrow afternoon, from today's satellite picture and the weather? |
| [FireSize](../models/firesize) | Will this new ignition escape initial attack, and how big will it get, from the first report alone? |
| [FireDetective](../models/firedetective) | What drove this fire: how much crowned, how hard the wind blew, how dry the live fuel was, from its satellite record? |
| [FirePilot](../models/firepilot) | Where should two crews build containment line in the next 72 hours, graded in the simulator on real fires' fuels and weather? |

Code and data: [`shared/fire/`](https://github.com/dogum/ML-Ratchet/tree/main/shared/fire) (FireSim, FireData, the
checks and the data scripts). The models' `training/_paths.py` import the code from there and read data from
`$ML_RATCHET_FIRE` (default the same folder). The platform release (`firesim-firedata-v1.0.zip`, 135 MB) holds the
code, the frozen splits, the fire tables, the exam guard footprints, the baseline summaries, and the LANDFIRE
landscapes and hourly HRRR weather for the 40 dev fires and the 48 replay fires (104 MB of data in 120 files). The
large real tables are not in it: they are rebuilt from the public sources (see [Using it](#using-it)).

**Two caveats come first.**

- **Simulated fires do not stop.** They grow on 91 % of half-days; real fires grow on 50 %. FireSim has no
  burnout, no stalls and too little night slow-down, and the dev-calibrated physics replays still over-predict
  next-day area 4-12×. Three of the four models found this gap on their own twin tests. It caps every simulated
  bank, and FireCast, FireDetective and the physics replays all over-predict growth because of it.
- **The truth is a satellite picture, not the ground.** Next-day truth is the FEDS perimeter built from VIIRS 375 m
  fire pixels at the two daily S-NPP overpasses. On the 4 dev fires that also have GOFER (GOES) perimeters, the two
  satellite products agree at Dice 0.77-0.93 and area ratio 0.80-1.22. Day-to-day new-area scores have a noise floor
  of about 10-20 % of the burned area. FirePilot is graded only in the simulator, so its gains are gains inside
  FireSim.

## FireSim: the simulator

- **Surface fire:** Rothermel 1972 with the Albini 1976 corrections, following Andrews 2018 (RMRS-GTR-371): the Scott
  & Burgan 40 and Anderson 13 fuel models, dynamic herbaceous curing, wind and slope with the BehavePlus vector
  addition, Byram intensity and flame length, and the fire ellipse (L/W = 1 + 0.25 UE, UE in mi/h).
- **Wind:** the 20-ft wind is the 10 m wind / 1.15; a wind adjustment factor gives the midflame wind, unsheltered from
  the fuel-bed depth, sheltered from canopy cover, height and crown ratio.
- **Crown fire:** Van Wagner 1977 initiation, active spread above 3 / CBD, crown spread from Rothermel 1991 (3.34 ×
  fuel model 10) or Cruz et al. 2005, and the crown fraction burned (Scott & Reinhardt 2001).
- **Spotting:** Albini maximum spot distance, Schroeder 1969 ignition probability, and a stochastic ember model from
  crowning cells. It is not calibrated, and the replays run without it.
- **Fuel moisture:** a Nelson-type dead-fuel stick model (1-, 10- and 100-h sticks) from hourly weather; live
  moisture from the growing-season index (Jolly et al. 2005) or held constant.
- **Spread solver:** minimum travel time (Finney 2002) on a 16-cell stencil with two-point updates (default), a
  32-cell `level_set` mode and an 8-cell `ca` mode, integrated through hourly weather. FARSITE-style Huygens wavelets
  are not implemented.
- **Suppression:** control lines with build times, a retardant ROS factor, and Fried & Fried 1996-style initial
  attack (line built against the perimeter of the burned-area ellipse).
- **Satellite emulator:** `observe()` turns a simulated fire into VIIRS, MODIS or GOES fire pixels at real overpass
  times: cell fire radiative power, sensor footprints that grow with scan angle, detection against FRP, cloud
  omission and false alarms.
- **Fire danger:** the Canadian FWI system (FFMC, DMC, DC, ISI, BUI, FWI). NFDRS ERC and BI are not implemented;
  gridMET's are used instead.
- **Training data:** random parametric landscapes and weather, real LANDFIRE windows and real HRRR weather from places
  and dates that pass FireData's exam guard, and `replay.replay_day` to re-run a real fire-day from its observed
  perimeter.
- **Speed:** a 30 × 30 km landscape at 60 m for 24 h of spread in 0.47 s on one core (numba); batch mode 0.09 s per
  ignition at 90 m. Without numba the same kernels run as slow plain Python.

Checks (`shared/fire/checks/*.json` as shipped; the baseline any change must not regress):

| Check | Result |
|---|---|
| Published Rothermel and BehavePlus tables (GTR-371 Tables 8, 12, 13, 17-19, 21-24 and worked examples; Rothermel 1983, GTR-143) | 436 of 436 values within half the last published digit |
| Against pyretechnics (an independent PyPI implementation), 6,720 flat and slope-only cases | ROS median relative error 4.9e-5, maximum 5.2e-5; crown ROS to 1e-7; fire type agrees 100 % |
| ... wind on slope (10,080 cases) | median 1.3 % apart, by design: pyretechnics projects the wind into the slope plane, FireSim adds the vectors horizontally as BehavePlus does |
| Ellipse: uniform fuel, 5 L/W ratios, 30/60/90 m cells, 7 wind directions (55 cases) | default solver area ratio mean 0.981, minimum 0.940, grid-orientation bias ≤ 3.4 %; `level_set` mean 0.992, minimum 0.973 |
| Two-fuel strip against the hand solution; solid fuel break | maximum relative error 2.9e-9; 0 cells burned beyond the break |
| Crown initiation and spread against Van Wagner and Cruz hand calculations | exact |
| Nelson isotherm against the Simard 1968 equilibrium moisture | RMSE 0.41 percentage points |
| Dead moisture on dev fire sites against gridMET NFDRS fm100 | FireSim 2.4 points drier (5.8 % against 8.2 %), r 0.75. No real 10-h stick data were obtainable, so which one is right is unknown |
| FWI: standard test row (Van Wagner & Pickett 1985); against xclim 1.0.0 | all six codes within 0.005; FFMC, DC, ISI identical, FWI within 0.05 |
| Dev replays: 40 dev fires, 470 growth days, F1 on newly burned cells | uncalibrated 0.153; dev-calibrated (a ROS factor per fuel group) 0.194 in sample, 0.184 leave-one-fire-out; area over-predicted 4-12× |
| VIIRS emulator against real dev detections | dev-calibrated count ratio +0.01 log10, Spearman 0.80 |

## FireData: the real data

- **Fire progression:** FEDS-MTBS (Zenodo 20187963): 12-hourly perimeters, active fire lines and VIIRS S-NPP 375 m fire
  pixels for 7,739 MTBS fires, 2012-2024 (171 MB processed). MTBS final perimeters and ignition dates come with it.
- **Hourly perimeters:** GOFER-Combined (Zenodo 14642378), GOES perimeters for 28 California fires, 2019-2021.
- **Benchmark set:** WildfireSpreadTS (Zenodo 8006177), all 607 fires, 2018-2021 (269 MB processed); used for
  context and split-matched, not scored by a model yet.
- **Ignitions:** FPA-FOD 6th edition (Short 2022), 1.59 million CONUS wildfire records 2001-2020; 2012-2020 is used.
- **Live fuel moisture:** Globe-LFMC 2.0 (Yebra et al. 2024), 239,000 CONUS samples.
- **Landscapes:** LANDFIRE fuel model (FBFM40), canopy cover, height, base height and bulk density, elevation, slope and
  aspect, at 90 m on EPSG:5070, for the 40 dev fires and the 48 replay fires. The version is the latest before the fire
  year: LF2016 for 2017-2022 fires, LF2022 for 2023, LF2023 for 2024.
- **Weather:** HRRR analyses (hrrrzarr on AWS) hourly for the same 88 fires; gridMET daily at every fire, every
  FPA-FOD ignition (the discovery day and the day before) and every Globe-LFMC site.
- **Frozen splits** (`data/proc/splits.json`, frozen 18 Nov 2026, seed 20261118). The unit is the fire: every
  half-day of a fire shares its split.

  | Data | Dev | Exam |
  |---|---|---|
  | MTBS fires (FEDS progression) | 40 CONUS wildfires, 2017-2021 | 4,197: early 2012-16 (1,647), main 2017-21 (1,859), **recent 2022-24 (691), held out as whole seasons** |
  | Prescribed fires | 2,859 for calibration only | never scored |
  | GOFER fires | 4 | 24 |
  | WildfireSpreadTS fires | 4 | 568 (35 excluded: their tiles overlap both splits) |
  | FPA-FOD ignitions (FireSize) | 8,913 | 655,869 (all of 2020 is the recent exam) |
  | Globe-LFMC sites | a seeded 10 % of sites (15,361 samples) | the rest (138,113 samples scored, 2012-2022) |
  | Replay subset | | 48 exam fires 2017-2024 (32 main, 16 recent) with landscapes and HRRR, for physics replays and FirePilot |

  Dev fires were picked by region, fuel group, size class and year. They are longer than typical exam fires (median
  10.5 afternoon steps against 3) because they needed at least 2 growth days.
- **The exam guard** (`firedata.guard`): simulated training may use any landscape window that does not touch an exam
  fire's final perimeter plus 2 km (any year), and any weather not within 50 km of an exam fire during its active
  window (ignition − 3 days to its last FEDS step + 3 days). Every loader raises `ExamLeakError` on an exam fire unless
  the call says `for_exam_scoring=True`. Many dev windows touch older exam fires, so banks must run the guard on dev
  windows too.
- **Classical baselines on the exam:**
  - Next-day spread (3,030 fires, 25,105 growth fire-days; AP and F1 on newly burned 90 m cells within 10 km of the
    fire): yesterday's growth measured from the active fire line, AP 0.252, F1 0.184 (the best); the climatological
    median growth 0.204 / 0.004; yesterday's growth from the whole perimeter 0.191 / 0.140; persistence 0.013 / 0, but
    persistence has the best total-area Dice (0.964).
  - Escape from an ignition (≥ 100 acres, 2.7 % of exam ignitions): climatology by region and cause AUC 0.688;
    ERC, BI and wind at discovery 0.634; gradient boosting fitted on the dev ignitions' real outcomes 0.777 (labelled
    "real labels fitted on dev").
  - Live fuel moisture (138,113 exam samples): climatology by functional type and month MAE 27.2 %; gradient boosting
    on gridMET drought features fitted on dev 24.7 %.

Quirks the models had to handle (all in `data/DATA.md`, section 2): each fire's window is cut from its **final**
perimeter, so scoring the whole window would tell a model where the fire stops (score within 10 km of today's burned
area); FEDS `farea` is in m² although its readme does not say; FEDS half-day labels are nominal (`12:00` is the
~13:30 local day overpass); HRRR starts on 25 Aug 2016, so 2012-2016 fires have no hourly weather, and HRRR precipitation
is missing before mid-2018 (14 of 40 dev fires); LANDFIRE 2016 is post-fire for 2012-2016 fires and five years stale
for 2021 ones.

## Licences

FEDS-MTBS (Chen et al. 2022 method), GOFER and WildfireSpreadTS are CC BY 4.0. Globe-LFMC 2.0 is CC0. LANDFIRE, MTBS,
FPA-FOD (USDA Forest Service Research Data Archive), HRRR (NOAA) and gridMET are US public data. VIIRS fire pixels
come inside FEDS-MTBS; the FIRMS archive itself was not used. The Rothermel and BehavePlus reference tables are US
Forest Service General Technical Reports (GTR-143, GTR-153, GTR-371), public domain; FireSim's checks read them, and
the PDFs and their text extracts are not redistributed here. **None of the sources used is share-alike**, so the
example excerpts and the instruments carry attribution only, and every model's weights are MIT like the code. Details
in [DATA_LICENSES.md](../DATA_LICENSES.md).

## What the models found

Every model was trained only on FireSim and examined once on real data. Nothing has been re-trained or re-scored in
this repository. Each model's page has the full table, every row with its verdict.

- **FireCast** (next-day spread, 3,030 exam fires, 25,105 growth fire-days): AP 0.323 [0.318, 0.328] and F1 0.239
  against 0.252 and 0.184 for the best baseline. It wins in every period (recent 2022-24: 0.318 / 0.222), in all 9
  regions and all 5 size bins, and on 71 % of fire-days. On the 48 replay fires it scores AP 0.351 (0.355 with LANDFIRE
  inputs) against 0.165 for the dev-calibrated physics replay, which over-predicts new area 3.5×. Where it loses:
  persistence keeps the best total-area Dice (0.964 against 0.922), and probabilities above about 0.35 are
  over-confident (a 0.9 forecast burned 61 % of the time). LANDFIRE inputs add only 0.002-0.004 AP.
- **FireSize** (escape and final size, 655,869 FPA-FOD exam ignitions, 2012-2020): escape AUC 0.727 against 0.688
  for climatology and 0.777 for gradient boosting fitted on real dev labels: a loss to the GBM. On large fires
  (≥ 1,000 acres) it is the best of all methods, AUC 0.815 against the GBM's 0.803. It beats the GBM in California,
  the Northwest and the Rockies and loses badly in the Southeast (0.566 against 0.704) and the Southern Plains.
  Calibration fails: mean P(escape) 0.055 against a real rate of 0.027. Size runs +0.22 log10 high overall, and the big
  MTBS fires are 3.3 log10 too small.
- **FireDetective** (944 exam fires, 2017-2023): two partial wins and one failure. Burn severity on forest fires
  (n = 357): Spearman 0.58 against 0.54 for FWI; across all fuels the plain Van Wagner crown-fire potential wins (0.55
  against 0.48). Wind on the fire days, read with the wind inputs removed: Spearman 0.30 against 0.13 for a Rothermel
  back-calculation, but the level is worse than the dev median and recent fires drop to 0.14. Live fuel moisture fails:
  0.17 against 0.34 for climatology. The 80 % severity band covers 69 % (too narrow).
- **FirePilot** (48 replay fires, outcomes simulated in FireSim on each fire's real fuels, terrain, HRRR weather and
  first FEDS detection): stops 0.474 of the no-action 72 h growth at 300 m of line per hour and 0.644 at 1,000 m/h,
  against 0.316 / 0.480 for anchor-and-flank, 0.400 / 0.591 for a forecast-world search (MPC) and 0.614 / 0.792 for
  the perfect-foresight bound. It beats anchor-and-flank on 71 % of fires but MPC on only 46 %. Fidelity: the free
  simulator run matches observed FEDS growth in size (×1.1 at 48-72 h) but only partly in shape (Dice 0.40 against
  0.19 for persistence; F1 on new area 0.34).

## What was planned and what was built

| Area | Built | Missing or weak |
|---|---|---|
| Surface fire | Rothermel and Albini on 53 fuel models, wind adjustment, slope; matches the published tables | nothing measured |
| Crown fire | Van Wagner initiation, Rothermel 1991 and Cruz 2005 spread | crown-fire spotting uses the surface plume formula |
| Spread solver | minimum travel time on 16- and 32-cell stencils | no Huygens wavelets; elongated ellipses lose up to 6 % of their area at the default stencil |
| Spotting | Albini distance and stochastic embers | not calibrated; the replays run without it; FireDetective cannot read spotting even in simulation (AUC 0.47) |
| Fuel moisture | Nelson-type dead fuels, GSI live moisture | 2.4 points drier than gridMET NFDRS, and no real dead-fuel stick data to settle it (the National Fuel Moisture Database moved into FEMS, with no public download found); live moisture is constant unless supplied |
| Fire and atmosphere | nothing | no plume-driven winds or terrain wind downscaling: wind-event nights are missed by an order of magnitude (the 2020 Riverside fire burned 496 km² in one night) |
| Suppression | timed lines, ROS rasters, initial attack | no validated suppression data (ICS-209); FireSize's dev-fitted containment race has several parameters at their bounds |
| Fire persistence | nothing | no burnout, stalls or night inactivity: simulated fires grow on 91 % of half-days against 50 % for real ones. This one gap drives the over-prediction in FireCast, FireDetective and the replays |
| Satellite emulator | VIIRS calibrated on dev counts and FRP | MODIS and GOES uncalibrated; no real cloud masks; false alarms cannot be measured |
| Fire danger | FWI, matching xclim | NFDRS ERC and BI not implemented (gridMET's used instead) |
| Real data | strong for CONUS 2012-2024 | hourly weather only from Aug 2016; hourly precipitation missing before mid-2018; LANDFIRE 2016 fuels used up to 2022 (stale, and scarred by the 2012-16 fires); no 2024 disturbance layer |

## Upstream candidates

Each model wrapped FireSim with the physics it found. Those wrappers live in each model's `training/` and are listed
in section 4 of `shared/fire/HANDOFF.md` and on the [backlog](backlog.md) (`firesim-p2`):

1. **U1, a bug: timed suppression lines leak diagonally.** In `firesim/kernel.py` the corner rule blocks a diagonal
   step only when both side cells are non-burnable; it ignores cells blocked by a line (`t_block <= tt`). In FirePilot's
   `training/leak_test.py`, a closed 4-connected ring in uniform grass let 3,551 of 3,972 outside cells burn after 48 h;
   the same ring as non-burnable fuel, or as an 8-connected line, let 0 burn. FirePilot builds 8-connected lines to
   avoid it. The fix: treat a side cell as blocked when `fuel_row < 0 or t_block[c] <= tt`, then rebuild every frozen
   set as physics p2 and re-score.
2. **Fire persistence.** FireDetective added night fine-fuel recovery, anchor-and-flank containment and a
   wind-sensitive WAF; the WAF change raised sim-vs-real per-step growth ρ from 0.36 to 0.44. FireCast found a
   fire-age spread factor (early days too slow, days 5 and later too fast; twin median log10 ratio +0.16). Unify them
   into one documented module.
3. **Agriculture fuel.** LANDFIRE class 93 should burn, for example as GR1 by season, and ignitions on non-burnable
   cells should move to the nearest burnable cell within 180 m. It was FireSize's biggest single gain (dev gate AUC
   0.675 to 0.757).
4. **A sub-cell analytic ellipse** for fires under 0.81 ha (FireSize).
5. **A containment race module:** Fried & Fried style, two attack modes and reporting floors (FireSize's `supp.py`
   and `calib_v1rep2.json`).
6. **A FEDS-like perimeter emulator** from simulated VIIRS pixels, calibrated on dev to a median area ratio of
   +0.012. FireCast and FireDetective each built one; merge them into `firesim.observe`.
7. **Memory-bounded `replay.replay_day` ensembles.** Each member copies the prepared fields, so 12 members on crops
   over 500,000 cells need more than 1.2 GB.
8. **U2, the domain edge.** On 45 of 48 replay fires a 15 km crop reaches the stored landscape edge. Store larger
   windows for replay fires.

## Using it

```bash
pip install -e ".[fire]"                        # add numba for speed (".[fire-data]" for a rebuild)
mlratchet data fire --from-zip firesim-firedata-v1.0.zip   # splits, fire tables, guard, baselines, 88 fires' landscapes and HRRR
mlratchet data fire                             # rebuild from the public sources instead (shared/fire/data/build.py)
pytest -q tests/test_fire.py
```

Without the release, `shared/fire/data/build.py` runs the fetch scripts in `data/scripts/`, each resumable: the
FEDS-MTBS progression (171 MB), WildfireSpreadTS (607 fires, 269 MB processed), FPA-FOD (1.59 million ignitions),
Globe-LFMC, gridMET at fires, ignitions and moisture sites, the LANDFIRE and HRRR packages for the 88 dev and replay
fires, then the splits check and the baselines (the per-fire-day exam rows, `spread_exam.jsonl`, are 89 MB). Not
shipped and not rebuilt by default: the pyretechnics reference environment (`firesim/.refenv`) and every simulated
bank. Hosts that fail are listed at the end; record them, don't route around them.

```python
import sys; sys.path[:0] = ['shared/fire', 'shared/fire/firesim']
import firesim as fs, firedata as fd
b = fs.behave('SH5', m1=4, m10=5, m100=7, mherb=60, mwoody=80, wind_speed=6.0, wind_height='10m', slope=20)
fd.fires('dev')                                    # the 40 dev fires' table, with the release unpacked
fid = 'CA4150012192920210801'                      # a dev fire
pr, W = fd.progression(fid), fd.weather(fid)       # afternoon-to-afternoon burned masks, 90 m; hourly HRRR + daily gridMET
ls = fs.realdata.landscape_from_firedata(fd.landscape(fid))
# fd.guard.check_landscape(bbox) and check_weather(bbox, t0, t1) raise ExamLeakError on an exam footprint
```

`data/proc/splits.json` (seed 20261118, sha256 `d73e2d9f…`) is tracked and never changes; the rest of `data/` is
git-ignored and checked against `data/MANIFEST.json`. Every physics change bumps the FireSim physics version, rebuilds
each fire model's frozen simulated set and re-takes their ratchet decisions in order.

## Rule notes

No exam result was used to choose anything. What the build agents reported, as they reported it:

- **FireDetective built its exam inputs before the ratchet closed.** `exam.py build` (input records and truth values
  for 944 fires) ran during rung R4, before an interruption. On resuming it was checked: no predictions, no scores,
  the records not inspected, and the records do not depend on the rung. R6 was then chosen on simulation and dev only,
  the ratchet closed, and `exam.py score` ran once. Disclosed in its card and `ratchet.json`.
- **FireCast** read one exam fire's landscape array shape outside the exam script, to size a memory guard. Its first
  physics-replay baseline run was killed after 2 fires for memory and re-run from scratch (deterministic); the killed
  attempt is kept. The exam process peaked at about 1.35 GB and was allowed to finish rather than abort a once-only
  run.
- **FireSize** read FPA-FOD records straight from parquet, filtered by its split, because FireData has no FPA-FOD
  accessor. Several dev-calibrated suppression parameters sit at their bounds. After the exam it looks likely (not
  computed) that T = 1 would have given a better exam Brier; the dev gate showed the same direction before the exam,
  and the release was kept as frozen.
- **FirePilot** split its exam into two stages for compute, before any exam output existed. Both stages finished on all
  48 fires, so the split changed nothing.
- **Interruptions:** all four model agents were interrupted once (an interface stall) and resumed from their status
  notes; each checked that no exam had started first. The Phase 1 container restarted once.
- **Dev data in the gates:** dev twins and dev replays fed physics calibration and also the gates in several models, so
  the gates are partly in-sample. The exams were untouched.

## What is still waiting

- **FireSim p2:** fix U1 and add a persistence module, then rebuild every frozen set and re-score every rung. This one
  physics change touches all four models. Then merge the FEDS emulator into `firesim.observe` and add agriculture
  fuel and ignition relocation (backlog `firesim-p2`).
- **FireCast's calibration rule:** a monotone calibration on dev needs a rule change from widen-only, logged as a
  method decision.
- **Data** (backlog `fire-data`): FEMS dead-fuel moisture (no public download found), ICS-209 suppression records, the
  per-fire MTBS severity mosaics (an e-mail queue), and larger replay windows.
- **Not re-run here:** the checks, the dev replays and every model's training (no banks or checkpoints are shipped).
  The exam numbers above are as shipped.
