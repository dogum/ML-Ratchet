> **Lab notebook.** LoadCast was built in a claude.ai cloud session (17 Nov 2026) on the shared BuildingSim +
> BuildingData platform (Phase 2 of the buildings work), which will live in `shared/building/`. The first entry is
> that session's status document as written; its paths (`src/`, `bldg/loadcast/`) refer to the cloud sandbox.
> `MODEL_CARD.md` holds every number.

## 17 Nov 2026 — build notes (loadcast-status.md, as written)

**What it is.** Hourly energy forecasts per building meter (electricity, chilled water, steam, hot water, gas) from weather, calendar and metadata, with 90 % error bars. Two modes: cold (metadata only) and warm (2 weeks, 8 weeks or a year of the meter's own history; the network corrects a TOWT model fitted on that history). Trained only on BuildingSim simulations with real dev noise. Graded on 1,502 BDG2 exam building-meters, forecasting local 2017 from 2016.

- Page: the original page, now `lab/index.html` (lab: pick one of 16 exam meters, shift the weather, change operating hours or close a week, compare against the real meter and the change-point baseline; the network runs in the browser).
- Release: loadcast-v1.0.zip (numpy module, weights, demo on 3 exam meters in ~2 s, model card, training scripts). Work folder: the sandbox workspace bldg/loadcast/.

## Recipe as built
- **Physics:** BuildingSim (Phase 1 checks: conduction 1.8e-9, RC step 0.0035 K, BESTEST 29/34 in range). LoadCast wrappers only (no edits to the shared package): country holidays (US/CA/UK/IE), a BDG2-style spec maker, operating knobs, meter calibration y = a·sim + b·A.
- **Dev calibration (52 dev buildings, 2016):** knob grid (plant for electricity-only buildings, operating hours, outdoor air ×1/×2.5, VAV minimum 0.3/0.8) + per-meter a, b. Sim/real annual ratio generic → best knobs: chilled water 0.29 → 0.47, steam 0.29 → 0.87, hot water 0.18 → 0.57; hourly CV electricity 62 → 24 % in sample. Best knobs: 24-7 in 22 of 52, outdoor air ×2.5 in 37; electricity a = 0.43 with b = 6.3 W/m² (real campus electricity is flatter).
- **Bank:** 1,600 generated buildings × 19 BDG2 sites' real 2016-17 weather, knobs and a, b drawn from dev donors. Frozen set: 300 buildings (650 meters).
- **Real noise:** residuals of 85 dev meters after a TOWT + weather regression, added on the same calendar dates, with real year-to-year level changes and gaps.
- **Network:** two MLPs 3×192, dropout 0.1, Gaussian head, metadata jitter (stops memorising simulated buildings). numpy == torch 1.5e-6; JS == Python 3e-4.
- **Twin test (dev):** cold CV real vs twin: electricity 62.5 vs 51.8 (r 0.86 by building), chilled water 101 vs 99 (r 0.96), steam 104 vs 108 (r 0.95). Warm: similar medians, no per-building correlation (noise borrowed).

## Rungs (frozen objective = mean over meter types of cold skill vs median guess and warm skill vs TOWT; dev score = mean median CV)
| rung | change | frozen | dev | decision |
|---|---|---|---|---|
| R0 | generic physics, 800 buildings | −0.174 | 59.9 | kept |
| R1 | dev-calibrated physics | −0.118 | 54.9 | kept |
| R2 | metadata jitter + dropout + weight decay | −0.018 | 51.9 | kept |
| R3 | 1,600 buildings | +0.014 | 49.2 | kept, released |
| R4 | cold loss aimed at the median magnitude | −0.026 | 52.2 | rejected |
| R5 | noise matched by use type | +0.006 | 51.7 | rejected |

## Exam (median CV(RMSE) %, 2017, n meters)
| meter | cold | warm 2 wk | warm 8 wk | warm 2016 | median guess | TOWT | change point | persistence 24 h (nowcast) | 90 % coverage warm 2016 |
|---|---|---|---|---|---|---|---|---|---|
| electricity (856) | 59.3 | 30.1 | 23.8 | **18.8** | 53.8 | 19.6 | 19.1 | 18.6 | 0.92 |
| chilled water (323) | 104.1 | 87.3 | 62.6 | **42.9** | 101.6 | 45.7 | 42.4 | 40.3 | 0.92 |
| steam (225) | 102.8 | 57.1 | 49.1 | **45.5** | 111.6 | 39.8 | 40.5 | 41.9 | 0.88 |
| hot water (72) | 134.0 | 93.6 | 86.5 | **94.5** | 207.6 | 83.1 | 82.6 | 69.4 | 0.92 |
| gas (26) | 181.5 | 159.9 | 133.1 | **110.5** | 190.4 | 95.7 | 95.8 | 87.7 | 0.73 |

G14 hourly pass rate (warm 2016 / TOWT / CP): electricity 70/70/71 %, chilled water 20/22/24 %, steam 10/25/23 %. Cold NMBE +24 % electricity, +47 % chilled water, +59 % steam (underpredicts).

## Limits
1. Cold mode loses to a median guess on electricity and chilled water and underpredicts thermal meters ~2×.
2. Warm mode ties TOWT/CP on electricity and chilled water, loses on steam, hot water, gas.
3. Two weeks of winter history: chilled water CV 87 %; dev widening hit the cap (30×).
4. Not a nowcaster (yesterday's reading is as good for electricity).
5. Small dev set (5 hot water, 2 gas meters); dev 2017 residuals in the noise bank make dev-gate numbers slightly optimistic.

## Next ideas
- Cold magnitude: a separate log-magnitude head (median-aimed) plus the shape network, or calibrate a/b by use type with more dev-like buildings; R4's fix improved coverage but not CV.
- Physics rung for heating meters: seasonal plant shutdowns, steam distribution losses, multi-building plants.
- Warm mode: train with 4-12 week histories from all seasons (not only year-end) and fall back to cold when history is out of season.

## Sandbox gotchas
- Shared 6.27 GB memory cgroup for all agents: a bank process grew to 1.47 GB (BuildingSim weather caches) and was OOM-killed; banks now restart every 2 parts.
- Training is cheap (~15-30 s/epoch); simulation banks are the bottleneck (~400 s per 100 building-years under load).
- A bash queue that grepped a filtered log for "ACCEPTED" silently picked the wrong inputs; read decisions from ratchet.json instead.

## 17 Nov 2026 — rule notes from the build handoff

The buildings handoff (section 5) records two things about how LoadCast's ladder was run. No exam result was used to
choose anything.

- The frozen-set objective changed twice before R3: from pooled skills floored at 0, to pooled skills without the
  floor, to skills per meter type averaged over meter types (pooled medians mixed meter types and could rank a model
  that is better on every meter as worse). Each time every rung was re-scored and re-decided in order
  (`training/rung.py --redecide`, `--rescore`). `training/ratchet.json` keeps the pooled values as `frozen_history`;
  the floored values were overwritten. `ratchet.yaml` logs both changes as rungs.
- The real-noise bank holds the dev buildings' 2017 residuals, and the dev gate scores the same dev buildings
  forecasting 2017. So the dev-gate numbers are slightly optimistic. The exam does not use dev buildings and is
  unaffected.

## 17 Nov 2026: imported into ML Ratchet

What changed from loadcast-v1.0.zip:

- **Paths.** `training/_paths.py` is new: `WORK` (= `models/loadcast/work/` or `$ML_RATCHET_WORK/loadcast`) replaces
  the sandbox root `bldg/loadcast` (`lc_common.ROOT = WORK`), and `BLDG` (= `shared/building/` or
  `$ML_RATCHET_BUILDING`) replaces `bldg/`, with `BLDG` and `BLDG/buildingsim` on `sys.path`; `webdata.py` finds
  BuildingData's `data/scripts` under `BLDG`. The two shell queues `cd` to their own folder and write logs and banks
  under the same WORK folder. No other line of the training scripts changed. `loadcast.py`'s comment pointing to
  `web/lcnet.js` now points to the lab page's inline script and the site port. The shipped logs (`ratchet.json`,
  `exam_r3.json`, `twin_r3.json`, `widen_r3.json`, `calib_summary.json`) are kept as shipped.
- **Licence folders.** The example files moved from `examples/` to `examples/bdg2/` with a `LICENSE-DATA.md`
  (BDG2, CC BY-SA 4.0); `demo.py` reads them from there (one line). `lab/LICENSE-DATA.md` covers the BDG2 data
  embedded in the page. `MODEL_CARD.md` gained a *Data and licences* section and its path to the platform checks.
- **New:** `model.yaml`, `ratchet.yaml`, this notebook, `README.md` (generated), `web/export_instrument.py` and the
  site instrument (`site/src/instruments/loadcast/`), `tests/golden/loadcast.txt`.

What was re-run:

- `python demo.py` in the repository (numpy 2.5, scipy, pandas): 1.6 s. Every number matches the release's own
  Python predictions stored in the lab page (`ref` per meter and mode, computed by `training/evaluate.py` when the
  page was built): Peacock_lodging_Matthew electricity cold 82.9 / warm 12.7 %, Bull_education_Reina chilled water
  100.5 / 22.2 %, Cockatoo_education_Minh steam 82.0 / 21.4 %. The zip's README printed no demo output to compare.
- `web/export_instrument.py` re-runs the release's feature code and networks on the page's 16 exam meters: weights
  identical to the page's copy; first two weeks within 2.9e-4 of the stored Python predictions (the page stores the
  weather at 0.01 °C); 2017 CV(RMSE) identical to 0.01 points on all 64 meter-modes. The browser port matches numpy
  to 3.9e-6 on 25 reference weeks (float16 weights; float16 vs float32 moves a week's forecast by at most 1.4e-3 of
  its peak). A whole-year forecast takes about 1.2 s in Node, so the instrument runs the shown week (about 25 ms)
  first and the year in slices after it.
- Nothing in `training/` was re-run: it needs BuildingSim and BuildingData (`shared/building/`) and the BDG2 data,
  which arrive with the platform zip (buildingsim-buildingdata-v1.0). The page's sources (`page.html`, `lcnet.js`,
  `page_text.json`, read by `training/build_page.py`) were not in the zip; the built page is `lab/index.html`.

Waiting on the platform zip: re-running calibration, banks, training and the exam; the BuildingSim checks
(`shared/building/checks/`); and the exam's 68 % coverage, which `model.yaml` does not list because the exam JSON
records only the 90 % band's coverage.

## 2026-11-17 — Integrity pass: the plan, written before any re-score runs

The v1.0 exam was taken once in the build session. This pass re-scores it with the released weights; every exam
number below is therefore **seen**. Nothing is chosen from it, and no new rung is tried. `splits.json` is unchanged.
On this machine (Windows, 16 cores shared, at most 3 worker processes) BuildingSim runs a 2-year building in about 1 s,
so simulation is cheap here. Scripts go in `training/`, outputs in `work/` (git-ignored).

1. **Reproduce first.** `training/rescore_seen.py` runs `exam.py`'s pieces with the released `loadcast_weights.npz`
   (numpy forward, `lcnet.np_forward`) and `widen_r3.json` on the unpacked BDG2 data. It also re-scores the 84 dev
   building-meters and re-fits the widening on dev. Report the largest difference from `exam_r3.json`,
   `ratchet.json` (r3 dev) and `widen_r3.json`, row by row. If anything does not reproduce, say so and by how much;
   the re-score then uses the new numbers.
2. **Every row `exam.py` computes goes into `model.yaml` and the card.** Per meter type: cold, warm with 2 weeks,
   8 weeks and a year of history (hourly CV), and daily and monthly totals for cold and a year of history: 8 verdict
   rows × 5 meter types. NMBE, G14 pass rates, 90 % coverage and the nowcasts (persistence 1 h / 24 h / same hour last
   week: they read the real 2017 meter) go in the notes and the card, without verdicts.
3. **Fair baselines, scored on the same building-meters and hours.**
   - Cold: (a) the dumb median guess as in `exam.py`; (b) **the dev-calibrated BuildingSim twin**: each exam
     building's BDG2 spec from metadata only, the knobs most often chosen on the 52 dev buildings (`calib.py`, re-run
     here into `work/data/dev_knobs.json` because the per-building file was not shipped; checked against
     `calib_summary.json`), with HVAC for electricity-only buildings set the same way from the dev electricity-only
     buildings, and per meter type the dev median `a, b`; simulated over 2016-2017 with the real weather, scored on
     local 2017. No exam reading is used. (c) The same twin with `a, b` (and knobs) by use type where dev has ≥ 3
     buildings, else as (b). (d) A **dev TOWT-shape guess**: the dumb guess's dev median intensity per use and meter,
     times the median over dev building-meters of a TOWT fitted on each dev meter's 2017 load / its mean (so it
     answers to the exam building's own weather).
   - Warm, a year: TOWT and the change-point model (BuildingData, fitted on 2016) and LoadCast's own ridge TOWT on
     2016 (`towt_lc`, the network's anchor). There is no dev-tuned variant in `exam.py`.
   - Warm, 8 and 2 weeks: the same ridge TOWT fitted on the same 8 or 2 weeks (the network's anchor), and the cold
     baselines (a user with 2 weeks can still use them).
   - Daily and monthly: the hourly baselines above aggregated the same way (`evaluate.daily_monthly`), plus
     BuildingData's daily and monthly change-point models for the year-of-history rows.
   The best baseline on a row is the one with the lowest median on the exam; picking it on the exam can only favour
   the baselines.
4. **Intervals and verdicts.** 95 % bootstrap interval on median(LoadCast) − median(best baseline) per row: 2,000
   resamples of the exam building-meters (one per building within a meter type), seed 20261117, paired, building-meters
   where either is NaN dropped. Win if the interval is below 0, loss if above, tie if it spans 0. n stated per row.
5. **Dev-gate optimism.** If the smoke test (step 7) says a full rebuild fits in about 2 hours with 3 processes:
   rebuild the noise bank, the v1 bank (1,600 buildings, seed 2) and the frozen set (300, seed 99), then train the R3
   recipe twice with the same seeds: arm A with the noise bank as built, arm B with each dev meter's 2017 residuals
   and gap pattern replaced by its own 2016 ones shifted 364 days (same weekday; the 2017/2016 level ratios are
   kept). Optimism = dev score(B) − dev score(A), set against the same difference on the (seen) exam; the part the
   exam does not share is the dev gate's leak. If the rebuild does not fit, report only the cheaper bound: R3's dev
   numbers against its exam numbers, meter by meter.
6. **Dev twin test of what the network adds (dev only).** Cold: anchor = twin (b) on each dev building (2017);
   correlate the network's forecast minus the anchor with truth minus the anchor, hourly, per meter type, and the
   per-building annual means. Warm (a year): anchor = the ridge TOWT on 2016; the same correlation of the network's
   correction with what reality needed. Bin the needed residuals by hour of week, month, temperature and holiday to
   show where they have structure.
7. **Readiness.** On this machine: the noise bank, a 20-building v1 bank and 1 epoch of cold and warm training into
   `work/`, timed; extrapolate to 1,600 + 300 buildings and 30 epochs; list what blocks a rung here (shell scripts
   call `python3` and `nice`, etc.).
8. Update `model.yaml` (rows, intervals in the notes, verdicts, "seen"), the card, `ratchet.yaml`'s ladder note and a
   results entry here; `readme --write loadcast`; validate; demo test.

## 2026-11-17 — Fresh exam frozen before round-2 rungs: I-BLEND (IIIT Delhi campus)

Written before I-BLEND is downloaded. Nothing is scored in this step: no LoadCast run, no baseline score. Built by
`training/exam2_iblend.py` into `work/exam2/iblend/` (git-ignored); frozen in `training/split_exam2_iblend.json`
(units, hashes, rows, baselines, interval rule). Scored once for v1.0 and once for any round-2 release, after the
round-2 release decision.

**Data.** I-BLEND energy, figshare 10.6084/m9.figshare.6007637.v1 (CC0, `energy_dataset.zip`, 589 MB, md5
754d5c85...): 7 campus buildings (academic, lecture, library, facilities, dining, boys' and girls' hostels), 1-min
electricity, 52 months; semester calendar 6007634 (CC0). Its weather article (6007649) holds four months only, too
short for LoadCast, so weather is built the way BuildingData built BDG2's: NOAA ISD-Lite for Delhi IGI airport
(421810-99999) for T_out, dewpoint and wind; gaps <= 6 h interpolated, longer gaps from NASA POWER hourly at the
campus; GHI from NASA POWER. Unit-clean: no Indian building is in training, dev or exam 1.

**Input contract.** `LoadCast.forecast(weather, tz='Asia/Kolkata', lat, use, area_m2, meter='electricity',
meters_present=['electricity'], history=None or hourly kWh)`. Gaps: (a) the holiday table covers US/CA/UK/IE only and
falls back to US holidays for Asia/Kolkata, so Indian holidays and semester breaks are invisible to the model; (b)
Delhi's climate is outside BDG2's; (c) electricity is the only meter. Use: academic, lecture, library -> education;
dining -> food; hostels -> lodging; facilities -> other. Floor area from the dataset's building details; a building
without an area gets no row. Year built: unknown (None).

**Series and quality control** (BuildingData's BDG2 rules, `build_bdg2.flags_for` and `q_ok`): 1-min power -> UTC
hour-beginning mean kW (= kWh); an hour with < 45 minutes is missing. Electricity flags: missing, zero, zero runs >= 24
h, negative, spikes, flatlines >= 24 h, linear-fill runs >= 12 h, level-shift days. A local calendar year is ok if
>= 85 % of its hours are good, it has < 30 level-shift days and its EUI is 10-2,000 kWh/m2. Per building, history =
local year Y and forecast = Y+1 for the latest consecutive pair where both years are ok and the Y+1 / Y ratio is within
0.4-2.5 (same calendar days). Only good hours are graded. Every dropped building is listed with its reason.

**Split.** All buildings that pass are exam (7 at most: too few to split). Nothing is fitted on them: the dumb guess
and the BuildingSim twin come from BDG2 dev, the history baselines from each building's own history year.

**Rows** (electricity; median over buildings): hourly CV(RMSE) for cold, warm with the history year, warm with its last
8 weeks, warm with its last 2 weeks; daily and monthly totals CV(RMSE) for cold and for warm with the year (8 rows,
ASHRAE G14, LoadCast and dev guesses p = 0, fitted baselines n - p). NMBE and 90 % coverage in notes, no verdict.
Model input: released `loadcast_weights.npz` with its shipped widening.

**Baselines** (as the 17 Nov integrity plan defines them, on the same hours). Cold: the dumb dev median-intensity x
median-shape guess per use (BDG2 dev); the dev-calibrated BuildingSim twin (spec from I-BLEND metadata only, dev knobs
and dev median a, b), if the twin builder accepts a non-BDG2 spec, otherwise the row says it was not run; the dev
TOWT-shape guess. Warm, a year: TOWT and change-point models fitted on the history year (BuildingData's code), and
LoadCast's ridge TOWT (`towt_lc`). Warm, 8 / 2 weeks: the ridge TOWT on the same weeks and the cold baselines. Daily
and monthly: the hourly baselines aggregated, plus daily / monthly change-point models for the year row.

**Intervals.** 95 % bootstrap over buildings, 2,000 resamples, seed 20261117, paired, on median(LoadCast) minus
median(best baseline). Win / loss if the interval is entirely on one side of zero, tie if it spans zero. With <= 7
buildings, most rows will be ties; n is stated.

## 2026-11-17 — Integrity pass: step 5 scaled down (written before the optimism arms run)

The smoke test (step 7) ran: a 20-building v1 bank took 59 s on 1 process (2.9 s per building, generator draws
included), and 1 epoch of 3,552 rows about 2 s per network. Scaled up, the full R3 recipe (1,600 + 300 buildings,
30 epochs, about 336,000 rows per epoch, single-threaded torch as `train.py` sets it) needs about 30 min of bank and
about 95 min per network, so 4 networks on 3 processes is about 3.5-4 hours: more than the 2 hours the plan allowed.
Instead, a half-scale controlled test with everything else as R3 (jitter 1, dropout 0.1, weight decay 1e-3, K 96):
a new v1 bank of 800 buildings (seed 2, `v1_half`), 15 epochs, three runs:
- `oA`: noise bank as built (seed 0);
- `oB`: the 2016-only noise bank (seed 0);
- `oA2`: as `oA` with training seed 1, to show how much two retrains differ by chance.
Each is scored on the 84 dev building-meters (dev score and per meter type) and on the seen exam (cold and a year of
history). The leak is dev(oB) − dev(oA) minus exam(oB) − exam(oA), read against the oA/oA2 spread. Only the
difference between arms is the measurement; none of these networks is a candidate.

## 2026-11-17 — Fresh exam frozen: I-BLEND results (nothing scored)

Built by `training/exam2_iblend.py`; no LoadCast run and no baseline score. `--check` rebuild reproduces the split.

- **Downloaded.** `energy_dataset.zip` (589 MB, md5 matches figshare), the semester calendar and the 4-month weather
  file (not used). NCEI's ISD-Lite path now redirects to an HTML page; the same files come from NOAA's open-data
  bucket (`noaa-isd-pds`), 2013-2018. NASA POWER hourly 2013-2018.
- **Weather.** ISD T_out covers 92.7 % of hours, 99.8 % after filling gaps <= 6 h; the rest is NASA POWER (MAE 2.5
  degC against ISD, bias +0.04 degC). GHI is NASA POWER.
- **Metadata.** Floor areas from the I-BLEND paper's Table 2 (ft2 -> m2): academic 5,696, lecture 1,630, library
  2,453, dining 4,663, facilities 908, boys' hostel 6,758, girls' hostel 3,542 m2. Hostels = mains + UPS meters.
- **Units.** Kept 3, all with history 2016 and forecast 2017: academic, library, facilities. Dropped 4: lecture
  (meter reads zero most hours: 9-23 % good hours in every year), dining (84 % good in 2016, 82 % in 2015), boys'
  hostel (48 / 54 / 79 % in 2015-17), girls' hostel (52 / 45 % in 2015-16). The 85 % rule was kept as planned.
- **What to expect.** Three buildings: a bootstrap over three units cannot separate anything but large gaps, so this
  exam is a stress check of the climate and holiday gaps, not a ranking.
- `split_exam2_iblend.json` sha256 `115d4029abf533c0b82af24be03eeb41b984f71479c02bbd767c2c4127fe135a` (units block
  `0674542c…`; no readings or EUIs stored).

## 2026-11-17 — Integrity pass: step 5 back to full scale (written before the full arms run)

The smoke-test extrapolation above was wrong: with 37 units, one epoch's 2 s was mostly fixed overhead. On the
800-building bank an epoch of 164,160 rows takes 6.5 s (cold) to 7 s (warm), so a network of the R3 recipe
(about 336,000 rows, 30 epochs) takes about 7 minutes, and the 800-building bank took 966 s on 2 processes (2.4 s
per building per process). A full rebuild fits easily, so step 5 runs as first planned, in addition to the half-scale
arms (already trained; they are reported too): bank `v1_train` (1,600 buildings, seed 2) and frozen set `v1_val`
(300, seed 99) rebuilt; `fA` (noise bank as built) and `fB` (2016-only noise bank), R3 recipe (30 epochs, jitter 1,
dropout 0.1, weight decay 1e-3, K 96, seed 0); `fA2` as `fA` with seed 1. `fA` also checks how closely a rebuild
reproduces R3 (frozen objective +0.014, dev score 49.2). Scored as the half-scale arms, plus the frozen-set objective.

## 2026-11-17 — Integrity pass: results

Everything below followed the plan and the two step-5 addenda above; nothing was changed after a result was seen.
Exam numbers are **seen**. Logs: `training/repro_r3.json`, `rescore_seen_r3.json`, `twin_knobs.json`,
`twin_anchor_r3.json`, `optimism_r3.json`. While this ran, another session was editing `shared/building/buildingsim`
(its `__init__` stopped importing mid-way), so every simulation here ran on a copy of the committed v1.0 package
(`$LC_BUILDINGSIM`, new in `_paths.py`); the calibration had already run on the live package and reproduces anyway.

**1. Reproduction.** `calib.py` re-run (52 dev buildings, 3 processes, 103 s): `calib_summary.json` to 1e-9, same
knob counts. Released weights through the training code (numpy forward): all 635 statistics in `exam_r3.json`
within 5.3e-6 (float32 numpy vs torch), the 60 dev statistics in `ratchet.json` within 1.1e-5 (dev score 49.2338 vs
49.2338), widening factors identical.

**2-4. Rows, fair baselines, intervals.** 40 rows (5 meter types × 8). Verdicts by the interval: **6 wins, 16 ties,
18 losses**. Two things change v1.0's reading:
- BuildingData's baselines used n − p; converted to p = 0 they drop a little hourly (TOWT 19.6 → 19.5 % electricity)
  and a lot monthly (change point 11.0 → 9.3 % electricity, 24.2 → 20.0 % steam). I added this conversion after the
  plan was written and before the bootstrap ran; it only helps the baselines.
- The dev-calibrated twin (knobs 24-7, outdoor air ×2.5, VAV minimum 0.3, district HVAC for electricity-only
  buildings; dev median `a, b`) is the strongest cold baseline: electricity 53.5 % (by use 52.4), chilled water 85.8,
  steam 100.2 (by use 99.7), hot water 124.4, gas 179.8. Four Bobcat buildings have no twin (BuildingSim
  `KeyError('kW')`); their twin rows are paired over the rest. The dev TOWT-shape guess beats the median guess on
  chilled water (92.4 vs 101.6) and hot water (192.9 vs 207.6) but is never the best baseline.

| row | v1.0 verdict | new verdict | LoadCast − best [95 % CI] |
|---|---|---|---|
| electricity, a year | win | tie | −0.3 [−0.8, +0.1] vs change point |
| chilled water, a year | loss | tie | +0.5 [−1.5, +2.9] vs change point |
| steam, a year | loss | loss | +6.0 [+2.9, +8.2] vs TOWT |
| hot water, a year | loss | loss | +14.0 [+0.3, +23.9] vs ridge TOWT (its anchor) |
| gas, a year | loss | tie | +15.9 [−4.9, +27.2] vs TOWT (n = 26) |
| electricity, none | loss | loss | +6.8 [+4.7, +9.5] vs twin by use |
| chilled water, none | loss | loss | +18.2 [+9.5, +25.1] vs twin |
| steam, none | win | tie | +3.1 [−1.1, +6.8] vs twin by use |
| hot water, none | win | tie | +9.6 [−10.6, +28.2] vs twin |
| gas, none | win | tie | −10.3 [−65.2, +94.7] vs twin (paired, n = 25) |

New rows: wins for 2 and 8 weeks on electricity (−6.4, −2.4), 8 weeks on chilled water (−5.5) and on hot water (2 weeks
−30.8 against the twin, 8 weeks −20.3), and electricity daily totals with a year (−0.4 [−0.9, −0.1] against its own
TOWT anchor). Every no-history daily and monthly row is a tie or a loss against the twin; monthly with a year loses on
chilled water, steam, hot water and gas.

**5. Dev-gate optimism.** Score = mean of cold and warm (a year) median CV %. Arms on the same rebuilt bank:

| | dev A | dev B (2016-only residuals) | dev A2 (seed 1) | exam A | exam B | exam A2 |
|---|---|---|---|---|---|---|
| half scale (800 bldg, 15 epochs) | 53.5 | 53.7 | 52.0 | 53.5 | 53.9 | 53.7 |
| full scale (R3 recipe) | 53.8 | 53.7 | 52.0 | 53.8 | 52.9 | 51.2 |

Full scale, paired bootstrap over building-meters: dev B − A = −0.1 [−2.7, +2.7], exam B − A = −0.9 [−1.8, −0.1];
dev A2 − A = −1.8 [−5.9, +2.0], exam A2 − A = −2.7 [−3.6, −1.8]. The leak (dev change minus exam change) is +0.8
points at full scale and −0.2 at half scale, both below the seed spread. The released R3 scores 49.2 on dev and 52.8
on the exam; its retrains score 53.8 / 53.8 and 52.0 / 51.2. So R3's dev number is optimistic by about 4 points, and
the cause is selection on a noisy gate, not the 2017 residuals. Frozen objective on the rebuilt `v1_val`: R3 0.004
(0.014 logged; the rebuilt frozen set is not bit-identical), fA 0.002, fA2 −0.018, fB −0.012: the seed spread (0.020)
is twice the ladder margin.

**6. Dev twin test.** Cold, anchor = the calibrated twin on each dev building (2017): median CV twin → network
electricity 53.4 → 62.5 % (47 meters), chilled water 71.1 → 101.0 % (20), heating 96.0 → 103.2 % (17). Correlation of
the network's correction with truth − twin, hourly: 0.73, 0.70, 0.18 (across buildings, of annual means: 0.75, 0.70,
0.16; within buildings, median: 0.19, 0.22, 0.12). It shifts the level the wrong way (needed +13 % / +2 % / +12 % of
the mean; given −11 % / −55 % / −7 %). Structure in truth − twin that the network does not supply: electricity
weekday 8-18 h +11 %, weekends −8 % (given about 0); heating February +56 % (given +7 %), summer −13 to −23 % (given
about −4 %). It does follow the chilled-water season the twin gets wrong (July −68 % needed, −52 % given). Warm, anchor
= its ridge TOWT: 22.3 → 22.3 % electricity (r 0.15), 48.1 → 43.2 % chilled water (r 0.50), 65.9 → 64.5 % heating
(r 0.32); holidays are the one place it adds what electricity needs (−14.6 % needed, −13.7 % given).

**7. Readiness on this machine (Windows, 3 processes).** Dev calibration 103 s (3 processes). Noise banks under a
minute. Twins of 1,001 dev and exam buildings × 2 variants: 1,170 s on 2 processes. Banks: 20 buildings 59 s (1
process, cold start); 800 in 966 s and 1,600 in 1,476 s on 2 processes; frozen 300 in 394 s. Training (single
torch thread, as `train.py` sets it): 15 epochs on 800 buildings about 2.5 min per network; R3 recipe (3,416 units,
327,168 rows per epoch, 30 epochs) 241 s cold, 332 s warm. Scoring dev + exam + frozen: about 25 min to prepare the
units (load from other sessions slowed it), then about 5 min per pair of networks. A full rebuild of R3 (calibration,
noise, 1,900 buildings, two networks, scoring) is about 1.5 hours on 3 processes. Blockers, all worked around and
none in the training code: `train_rung.sh` / `queue_a.sh` call `python3` (missing on Windows) and `nice`, and pipe
through `grep`; `winrun.py` runs the calibration and banks with `multiprocessing` instead. The shared BuildingSim was
being edited by another session (`$LC_BUILDINGSIM` points at a frozen copy). Banks are not bit-reproducible
(generator order depends on the process), so a rebuilt frozen set re-scores R3 at 0.004, not 0.014. Single-seed
decisions are within training noise (step 5).

**8.** `model.yaml` (40 rows with intervals, verdicts, "seen" notes, wins / losses / limits / next), the card's exam,
ratchet and limits sections, and `ratchet.yaml`'s ladder note are updated; the weights, inference code and demo are
unchanged (demo test passes).

## 2026-11-17 — Round 2 rungs r6-r8: the plan (written before any bank is built)

The integrity pass found that two training seeds of the R3 recipe differ by 0.020 in the frozen objective (twice the
0.01 margin) and by 1.8 points in the dev score, and that banks were not reproducible. This round fixes the bank, keeps
the ladder's rule, and judges every rung on a mean over three training seeds. The I-BLEND fresh exam is frozen and is
not scored in this round. The seen BDG2 exam is scored only if something is released.

**Infrastructure (no rung).**
- `bank.py` phys `v1d`: the v1 bank with a deterministic spec per building. Each building draws its generator spec
  with `generate_buildings(1, seed=gs)`, where `gs` is the building's own seeded draw. The spec no longer depends on
  which process asked first. Checked by building the same 20 buildings on 1 and on 2 processes and comparing every
  array. The physics version (`BUILDINGSIM_PHYSICS`) is recorded per building.
- `$LC_KNOBS` names the dev-calibration file that `calib.py` writes and `bank.py` reads (default `dev_knobs.json`).
- Frozen sets, built once and never regenerated for a rung: `v2_val` (300 buildings, seed 99, physics 1.0, real noise
  seed 12345) and its physics 1.1 twin `v2_val_p11` (same seed, so the same buildings and draws).
- Training banks: `v2_train` (1,600 buildings, seed 2, physics 1.0) and `v2_train_p11` (same seed, physics 1.1).
- **Incumbent:** the released R3 weights re-scored on `v2_val`, and on `v2_val_p11` for r6. Its dev score stays 49.2,
  since the dev data do not change.
- **Control (reported, not the rule):** the unchanged R3 recipe retrained on `v2_train` with seeds 0, 1, 2. The
  released R3's dev score is about 4 points better than retrains of its own recipe (integrity pass). So under the
  rule as kept, a rung must beat a selected incumbent. The control shows how much of any gap is the recipe.

**Rule (unchanged ladder rule, 3-seed means).** A rung is kept if both hold:
- the mean over seeds 0, 1, 2 of its frozen objective beats the incumbent's by more than 0.01 on the frozen set of its
  own physics;
- the mean of its three dev scores is no more than 2 % (relative) worse than the incumbent's 49.2.

Each seed and the spread (max − min) is reported. If a rung is released, the shipped weights are seed 0, fixed now (no
pick among seeds). Recipe as R3 (30 epochs, width 192, depth 3, K 96, jitter 1, dropout 0.1, weight decay 1e-3).

**Rungs (each against the incumbent).**
- **r6, physics 1.1.** The dev calibration is re-run at physics 1.1 (`calib.py` with `BUILDINGSIM_PHYSICS=1.1`, dev
  2016 only, into `dev_knobs_p11.json`). The meter fit `a, b` fitted at 1.0 would count the new loop base twice. Then
  `v2_train_p11` and `v2_val_p11` are built and cold and warm are trained (3 seeds each). Judged on `v2_val_p11`; also
  reported on `v2_val`.
- **r7, a median-aimed no-history magnitude head.** The cold forecast becomes M × shape:
  - M = exp(g(metadata)): an MLP on the 38 metadata features, trained on log(target-year mean) of each simulated unit
    (MSE on the log aims at the median; plus 0.1 × Gaussian NLL for its spread; the same metadata jitter).
  - shape: the cold MLP trained on the load divided by the unit's own target-year mean (base 1).
  - σ combines the two: σ = M · sqrt(σ_shape² + (shape · s_logM)²).
  - The warm networks are the control's (seeds 0, 1, 2); only the no-history mode changes.
- **r8, the no-history mode anchored on the dev-calibrated twin.**
  - The cold network reads the cold features plus the twin's hourly reading and the log of its annual mean, and
    predicts a correction to the twin: μ = twin + f(x), with the head initialised to zero.
  - The twin is what `twin_anchor.py` used: each building's metadata-only spec (use, area, year built if shown,
    meters, site), the dev plurality knobs and dev median `a, b` (`twin_knobs.json`), physics 1.0.
  - For simulated buildings it is simulated from the same metadata the network sees, not from the generator spec that
    made the truth. For dev and exam buildings it is the stored `twins.npz` ('med').
  - Warm networks as the control's.
  - **Reference row (not a candidate):** the twin alone as the no-history forecast: its cold skill on `v2_val` and its
    dev no-history CV.
- **Combination.** Only if two or more of r6-r8 are each kept: their combination (3 seeds) is judged by the same rule
  against the best kept single rung. Otherwise nothing is combined.

**If something is released (v1.1).**
- Weights stay CC BY-SA 4.0. The numpy inference must reproduce the trained network.
- r8 would need the twin at inference, which means BuildingSim, not just numpy. I stop and ask before shipping that
  design.
- Then: demo + golden, model.yaml, ladder, card, this notebook. The seen exam is re-scored with `rescore_seen.py`'s
  baselines and intervals (labelled seen), and the instrument is re-exported with parity passing.
- If nothing is kept, v1.0 stays and `ratchet.yaml` logs the rungs.

Limits: at most 3 processes. The live BuildingSim (1.2.0, committed) is used directly; at the default physics it is
v1.0 bit for bit.

## 2026-11-17 — Round 2 rungs r6-r8: results (nothing kept; v1.0 stays)

Everything followed the plan entry above, with one fix. The first r7 magnitude heads trained to NaN: units with no
valid target hours gave log(NaN). They were retrained with `train.py`'s own filter (≥ 200 valid target hours) before
any r7 score existed; the shape networks were fine. Per-seed numbers: `training/r2_ladder.json`.

- **Infrastructure.** `bank.py` phys `v1d` is reproducible: 20 buildings built on 1 and on 2 processes are identical
  array for array. Dev calibration at physics 1.1 (`dev_knobs_p11.json`, 109 s): chilled water a 0.92 → 0.74,
  b 31 → 0 W/m², in-sample CV 52.9 → 50.4 %; hot water 79.0 → 76.6 %; steam and electricity unchanged. Banks:
  `v2_train` 672 s, `v2_val` 115 s, `v2_train_p11` 582 s, `v2_val_p11` 115 s (3 processes). Bank twins (r8) 462 s.
  Training: 21 networks in 27 + 5 min on 3 processes.
- **Incumbent.** Released R3 on `v2_val` −0.0081 (cold skill −0.018, warm +0.002), on `v2_val_p11` −0.0152;
  dev 49.23, gate 50.22.
- **Control (R3 recipe, seeds 0/1/2):** +0.0035 / −0.0151 / −0.0113, mean −0.0076, spread 0.019; dev 53.5 / 53.2 /
  53.3. The seed spread in the dev score is small here (0.3), but the control sits 4.1 points above R3's selected 49.2.

| rung (3-seed mean) | frozen objective (seeds) | vs incumbent | dev score (seeds) | dev gate | decision |
|---|---|---|---|---|---|
| r6 physics 1.1, on v2_val_p11 | −0.0042 (−0.0077, −0.0028, −0.0021) | +0.011 | 52.8 (53.9, 52.1, 52.4) | fail (+7 %) | rejected |
| r7 magnitude head | −0.0276 (−0.0305, −0.0290, −0.0235) | −0.020 | 50.4 (52.2, 50.3, 48.8) | fail (+2.4 %) | rejected |
| r8 twin anchor | −0.0047 (+0.0072, −0.0132, −0.0082) | +0.003 | 53.9 (54.0, 52.9, 54.7) | fail (+9 %) | rejected |

- r6 on the physics 1.0 frozen set: −0.024. Against the control on `v2_val_p11` (−0.0108) it gains 0.007. Dev
  no-history CV 75.9 vs control 77.2; the gain is in warm skill on the 1.1 frozen set (+0.026 vs 0.000).
- r7 moves the no-history level where the plan aimed: dev no-history CV 71.4 vs control 77.2. Dev electricity
  no-history NMBE is −4 / +2 / +16 % for the three seeds, against +38 % (control seed 0). In simulation its cold skill
  falls (−0.065 vs −0.025): the median-aimed level costs CV where magnitudes are known to be skewed.
- r8 starts at the twin and moves away from it. Dev no-history CV is 78.2 against 65.4 for the twin alone; chilled
  water NMBE is +66 % against +2 % for the twin. The cold loss (weights 1/ȳ², which aim unknown magnitudes at about
  half) pulls the level back down, as in v1.0. Reference row: the twin alone has cold skill +0.002 on `v2_val`
  (R3 −0.018) and dev no-history CV 65.4 (R3 70.6).
- **Combination:** none, since no rung was kept (plan). The natural next rung is r8's anchor with r7's median-aimed
  level.
- **The gate:** under the kept rule, a 3-seed mean has to beat R3's single selected seed on dev. The control misses by
  3.1 points (53.3 vs a gate of 50.2), so no rung could pass without a large real gain. Changing that is a rule change
  for the owner to decide, not something to do after seeing these numbers.
- **Release:** none. The inference-design question for r8 does not arise. If a twin-anchored mode is released later,
  the numpy module cannot run BuildingSim (inference is numpy/scipy only). It would take the twin series as an input,
  computed separately by a helper that uses BuildingSim. That design waits for the owner's approval.

## 2026-11-17 — Fresh exam scored once: I-BLEND (v1.0)

Scored after the round-2 decision was committed (no round-2 release; v1.0 stays). Nothing was changed after a
result was seen; nothing is chosen from it. Script `training/exam2_iblend_score.py`, results
`training/exam2_iblend_results.json` (per-building values included).

- **Data check.** `exam2_iblend.py build --check` rebuilt the three buildings from the raw downloads: the units block
  matches the frozen split (sha256 `115d4029…`); the scorer re-checks each building's hash before scoring.
- **As the spec says.** `LoadCast.forecast` with the released weights and shipped widening, tz Asia/Kolkata, lat
  28.5463, electricity only, history 2016 (a year, its last 8 weeks, its last 2 weeks), forecast local 2017. Dev
  guesses from the 47 BDG2 dev electricity meters (education and other keys exist). BuildingData TOWT and
  change-point from `baselines.load_one` (n − p, as its code computes); the ridge TOWT and every daily / monthly
  aggregate of an hourly prediction p = 0, as exam.py. Bootstrap over the 3 buildings, paired, 2,000 resamples, seed
  20261117.
- **Could not run as specified: the dev-calibrated BuildingSim twin.** The integrity-pass builder
  (`rescore_seen._sim_one`) takes a BDG2 building id and a BDG2 site's weather; it has no entry for a non-BDG2 spec.
  The spec says the row then states it was not run. The cold rows have two baselines, not three.

| Row (n = 3), median CV(RMSE) | LoadCast v1.0 | Best baseline | Minus best, 95 % CI | Verdict |
|---|---|---|---|---|
| Cold, hourly | 125.7 % | dev median guess 89.8 % | +35.9 [−33.5, +78.8] | tie |
| Warm, a year, hourly | 41.0 % | change-point (BuildingData) 40.2 % | +0.7 [−2.2, +1.0] | tie |
| Warm, 8 weeks, hourly | 41.1 % | ridge TOWT on the same 8 weeks 46.2 % | −5.1 [−5.1, +2.5] | tie |
| Warm, 2 weeks, hourly | 43.2 % | ridge TOWT on the same 2 weeks 46.7 % | −3.5 [−4.1, −3.5] | win |
| Cold, daily | 120.7 % | dev TOWT-shape guess 82.3 % | +38.4 [−39.4, +78.6] | tie |
| Cold, monthly | 119.6 % | dev TOWT-shape guess 80.7 % | +39.0 [−42.0, +79.5] | tie |
| Warm, a year, daily | 17.4 % | TOWT (BuildingData), daily 17.9 % | −0.5 [−2.4, −0.3] | win |
| Warm, a year, monthly | 12.6 % | ridge TOWT, monthly 13.0 % | −0.5 [−1.9, −0.4] | win |

- **What it says.** Cold, LoadCast over-forecasts Delhi's campus by about 2.2 times (median NMBE −120 %; per building
  CV 56 / 180 / 126 % for academic / library / facilities) and its 90 % band covers every hour because it is that wide.
  The dev guesses are also poor (90 %) but closer. With history it is level with the fitted models (a year) and a few
  points better than its own anchor on 2 and 8 weeks. Three buildings cannot rank anything: the three "wins" are
  differences of 0.5-3.5 points whose intervals come from resampling three values, so they say "not worse", no more.
  Warm 90 % coverage 84 % (a year), 80 % (8 weeks), 84 % (2 weeks). Nowcasts (info, read the real 2017 meter):
  persistence 1 h 17 %, 24 h 35 %, same hour last week 25 %.

## 27 Nov 2026 — audit: buildings without an electricity meter (warning added; re-score queued)

The two-reviewer audit (`docs/audits/loadcast-v1.0.md`) found that a meter list without electricity puts one input at
z = −1000, because every training building had an electricity meter. Two packaged steam and chilled-water forecasts rose
24-50× in every hour. Counted on BDG2 metadata: 58 of the 1,584 exam buildings list no electricity meter and were
scored that way. `forecast()` now warns, and `assume_electricity=True` adds the meter (off by default). A dated
re-score of those buildings waits for the owner's approval.
