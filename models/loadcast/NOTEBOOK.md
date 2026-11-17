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
