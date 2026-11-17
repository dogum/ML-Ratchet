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
