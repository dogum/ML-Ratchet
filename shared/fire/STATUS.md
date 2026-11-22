# Wildfire platform (FireSim + FireData) status, 21 Nov 2026

Greg's fifth big-system project for ML Ratchet, after vehicles, the grid, buildings and watersheds. It was built in a
sandbox in two phases and shipped as zips:
- **Phase 1 (18-19 Nov):** two agents built the shared simulator (FireSim, `firesim/`) and the real-data package
  (FireData, `firedata/` and `data/`) against `FIRE_BRIEF.md`.
- **Phase 2 (19 Nov):** four agents built FireCast, FireSize, FireDetective and FirePilot against `PHASE2_BRIEF.md`.
- **Handoff:** `HANDOFF.md` lists what exists, what is missing and the suggested order of work.

Resume notes from the sandbox: `firesim/STATUS.md` (agent A) and `firedata/STATUS.md` (agent B). Their paths now point
into this folder; the text is otherwise as shipped.

## What was imported (21 Nov 2026, from `firesim-firedata-v1.0.zip`)

Tracked in git (code and small files only):
- `firesim/`: the `firesim` package (19 modules), `demo.py`, `README.md`, `STATUS.md`.
- `firedata/`: the loader, exam guard, grid, HRRR reader and exam metrics, and `STATUS.md`.
- `checks/`: the check scripts, `ref/pyre_run.py` and `ref/xclim_fwi_run.py`, and the 11 shipped result JSONs.
- `data/`: `DATA.md`, the 21 rebuild scripts in `scripts/`, `build.py`, `MANIFEST.json` and the frozen
  `proc/splits.json` (byte-identical to the zip: file sha256 `70f65596...`, its own `sha256` field `d73e2d9f...`).
- `FIRE_BRIEF.md`, `PHASE2_BRIEF.md`, `HANDOFF.md`.

Git-ignored, under the data root (`$ML_RATCHET_FIRE`, default `shared/fire`):
- `data/proc/` except `splits.json`: the fire table, FireSize split, exam guard footprints, LANDFIRE fuel groups,
  landscapes and hourly HRRR for the 40 dev fires, baseline summaries and fire-day tables (104 MB).
- `checks/out/`: pyretechnics values, xclim FWI output, VIIRS and replay caches (17 MB).
- `firesim/refdata/`: the USFS GTR-371, GTR-153 and GTR-143 PDFs and their text (39 MB, public domain). Only
  `checks/check_published.py` reads them; its test skips without them.
- `firesim/demo_out/`: the two README figures (0.8 MB). `demo.py` rewrites `demo.png`.
- `firesim/.refenv/`: the isolated pyretechnics and xclim environment. It is not in the zip. Without it,
  `check_fwi.py` reuses the shipped xclim output when its inputs are identical, and `check_pyretechnics.py` reads
  the shipped `checks/out/pyre_values.json`.
- `data/raw/`, `data/tmp/`, `data/logs/`, `data/work/`: rebuild scratch.

`mlratchet data fire --from-zip firesim-firedata-v1.0.zip` unpacks the 229 data files (161 MB on disk) and checks
all 230 manifest entries, splits included. `mlratchet data fire --list` shows the rebuild stages.

## Changes made on import

- Sandbox paths replaced. Code: the code root is the folder itself; data are read from `$ML_RATCHET_FIRE` (default
  `shared/fire`), as the water platform does. Docs: repo-relative paths. In `HANDOFF.md` the four private page links
  were removed; the zip names stay.
- Dropped: `data/scripts/chain_packages.sh`, `chain_repair.sh` and `chain_wsts.sh`. They waited on sandbox PIDs to
  chain the scripts; `data/build.py` runs the same steps in the same order.
- `checks/_common.py`: `FIRESIM_CHECKS_OUT` sends the result JSONs to another folder (the tests use it), and JSONs are
  written with LF line endings.
- `checks/check_fwi.py`: the xclim comparison runs only when `firesim/.refenv` exists (Windows or POSIX layout);
  otherwise it uses the shipped xclim CSV for identical inputs, or records the comparison as skipped.
- `checks/check_published.py` reads the GTR text as UTF-8 (the Windows default codec failed on it).
- `checks/ref/*.py` take the output folder as an optional argument.

No physics, number, split or calibration was changed.

## Verification on import (21 Nov 2026, Windows, Python 3.14, numpy 2.5)

Re-run against the shipped JSONs:

| Check | Result | Matches HANDOFF section 3 |
|---|---|---|
| Published BehavePlus/Rothermel tables (`check_published.py`) | 436 of 436 within half the last digit; JSON identical | yes |
| Analytic ellipse, strip, fuel break, refraction, time-varying (`check_analytic.py`) | all 220 ellipse runs reproduce to 1e-13; default solver mean area ratio 0.981, minimum 0.940, orientation bias 3.4 %; `level_set` 0.992 / 0.973; strip error 1.5e-7 h; fuel break leaks 0 cells | yes |
| Crown hand calculations (`check_crown.py`) | identical | yes |
| Dead fuel moisture (`check_moisture.py`) | Simard isotherm RMSE 0.41 points; identical to 1e-13 | yes |
| FWI (`check_fwi.py`) | test row within 0.0049 (DMC); vs the shipped xclim season identical | yes |
| pyretechnics cross-check (`check_pyretechnics.py`, from the shipped values) | identical to 2e-15 | yes |
| Speed (`check_speed.py`, numba 0.68.1) | 30 × 30 km at 60 m, 24 h: 0.75 s per call (shipped 0.47 s; kernel 0.21 s vs 0.36 s); worst case 1.73 s; batch 0.06 s per ignition at 90 m; burned areas identical | target met |

numba is optional (`firesim/_nb.py`): the analytic, crown, moisture, FWI and published checks give the same numbers
with and without it (largest difference 1.3e-13). Without numba the full analytic check takes 628 s instead of 15 s,
so `tests/test_fire.py` re-runs a subset (four 90 m ellipse cases, the strip and the time-varying case) and adds the
fuel break and refraction only when numba is installed. `check_speed.py` needs numba. Without scikit-image the
perimeters fall back to matplotlib and the ledger has no `growth_check` (`speed.json` then shows `null` ratios).

Not re-run on import: the dev replays (`replay_dev.py`, hours), the VIIRS emulator check and `check_moisture_real.py`.
They need the FEDS-MTBS progression and gridMET, which are not in the zip.

## Found on import

- **Bug U1 (known, HANDOFF section 4): timed suppression lines leak diagonally.** In `firesim/kernel.py` the corner
  rule blocks a diagonal step only when both side cells are non-burnable. It ignores cells blocked by a line
  (`t_block <= tt`). Reproduced in 2 s on a 31 × 31 grid of GR2 grass with a closed 4-connected ring: 282 of 852
  outside cells burn in 12 h with a timed line at t = 0, and 0 with the same ring as non-burnable fuel.
  `tests/test_fire.py::test_u1_timed_line_holds_like_nonburnable_fuel` is `xfail(strict=True)`: the fix flips it.
  The fix changes physics p1, so it waits for the round that rebuilds every frozen set as p2 and re-scores.
- **`lfmc_us.parquet` has no script** (closed 21 Nov 2026 by a checked reconstruction; see the rebuild entry). `build_lfmc.py` writes `data/proc/lfmc/lfmc_all.parquet`, but
  `build_gridmet.py` and `baselines_moisture.py` read `lfmc_us.parquet` (CONUS samples, a `date` column and Globe-LFMC's
  column names). The step that made it in the sandbox was not shipped. The `lfmc` build stage reports it missing.
- **`moisture_split.json` is not in the zip.** The live-moisture split (10 % of Globe-LFMC sites, seed 20261118 + 3)
  is frozen, but only `baselines_moisture.py` can recreate it, from `lfmc_us.parquet`. It is deterministic given the
  same site list.
- **WildfireSpreadTS order file.** `fetch_wsts.py full` read `proc/wsts/order.txt` (dev first, exam in a seeded order),
  which no script writes. `build.py` writes a dev-first order to `data/work/wsts_order.txt`. Order changes only which
  fires an interrupted run has.
- **The rebuild needs the zip first.** `fires.parquet`, `firesize_split.parquet`, `guard/` and `replay_subset.json`
  come from `make_splits.py` and `make_replay_subset.py`, which are frozen and refuse to rewrite. A rebuild from
  sources therefore starts from `--from-zip`.
- **FireData loader coverage after `--from-zip`.** `fd.fires`, `fd.splits`, `fd.landscape`, `fd.weather` (HRRR) and
  the guard work for the 40 dev fires. `fd.progression`, `fd.perimeters`, `fd.firepix`, `fd.final_perimeter`,
  `fd.gofer`, `fd.wsts` and `fd.weather(...)['daily']` need the `feds`, `gofer`, `wsts` and `gridmet` stages.
  The 48 exam replay fires have no landscapes or HRRR until the `landscapes` stage runs.

## Data hosts (record failures in `help_wanted`; do not route around them)

From `data/DATA.md` section 7: NFMD dead-fuel sticks (wfas.net 404, moved into FEMS, no public download), RAWS via
Synoptic (key needed), FIRMS (MAP_KEY needed; not used). No download stage was run on import. The FPA-FOD GeoPackage
link was renamed by the archive (Format2 to Format3); see the 2026-11-21 rebuild entry below.

## Next

See `HANDOFF.md` section 7. The first item is the U1 fix and a fire persistence module, then physics p2 and a re-score
of every model.

## 2026-11-21 — PR #7 review: a mask ignition's interior was not burned at the start

`parse_ignition` seeds only a mask's edge cells, and `simulate()` reset to 0 only the mask cells the kernel never
reached. The kernel spreads inward from the edge, so the interior got positive arrival times (75 of 121 cells in an
11 × 11 square on a generated grass landscape, 12 h). The interior now counts as burned at t = 0, as replay.py's
docstring says it should. Cells outside the mask are unchanged (bit-identical on that case): by minimum travel time
a later interior arrival cannot make an outside cell earlier. What changes: the arrival raster and anything that
counts burned cells by arrival time inside the initial mask (early area series, early perimeters). Replays and
FirePilot scenarios that score only new burn outside the mask are not affected. The shipped results were not
re-run. Test: tests/test_fire.py::test_mask_ignition_interior_is_burned_at_start.

## 2026-11-21 — PR #7 review: two weather generator fixes

- `hourly_from_daily` rounded UTC times to the hour with pandas' round-half-to-even. At longitudes whose solar
  offset ends in half an hour (lon = -112.5, -97.5) every time fell on :30, alternate hours merged, and the dedup
  dropped half the series (48 of 96 hours). Ties now round up; other longitudes give identical output.
- `realdata.random_weather(start=...)` shifts the real window by whole days to keep its diurnal phase, so the result
  could start after the spin-up start or end short of the horizon (one case: started 19:00 for an 18:00 request).
  It now reads a day either side, shifts, and trims to the requested window; without `start` nothing changes.
Training banks built with the old code are not re-built here. Tests: tests/test_fire.py
(test_hourly_from_daily_keeps_every_hour_at_half_hour_offsets, test_real_weather_covers_the_requested_window).

## 2026-11-21 — PR #7 review: U9, simplex updates use one weather step for the whole trip (not fixed)

The kernel's two-point (simplex) update in `kernel.py` scores its candidate with the spread rates cached at the
departure step for the whole trip, while the single-segment traversal before it integrates through later hourly
steps. When the trip crosses an hour boundary into slower weather, the simplex candidate can beat the correctly
integrated time. Simplex is on by default (`method='mtt'`), so this touches ordinary runs, every simulated bank and
the dev replays. Like U1 it is a physics change: the fix (integrate the simplex path through the hourly forcing, or
reject updates that leave the current step) belongs in physics p2 with U1, with every frozen set rebuilt and every
rung re-scored. Not measured here; the analytic checks (constant weather) cannot see it.

## 2026-11-21 — PR #7 review: scheduled mask ignitions; U10, generated weather flags on UTC dates (not fixed)

- The interior fix above set every mask cell to 0, even for a mask or polygon ignition with a later `time`.
  `parse_ignition` now returns each mask cell's ignition hour, and the interior takes that hour (cells of an
  ignition after the end are left as the kernel set them). Masks at time 0, the replay case, are unchanged.
- U10: `generate_weather` maps its daily event and wet-day flags to hours by UTC date, but `hourly_from_daily` lays
  each local solar day out in UTC, so at lon = -120 the last 8 hours of a local day take the next day's flags (wind
  persistence, direction swing, cloud). Not fixed: it changes the default weather generator that every simulated
  bank was drawn from, so it goes with U1 and U9 into the next physics version.
- `prepare()` took the Rothermel 1991 crown-spread reference bundle (standard fuel model 10) from the default table at
  the row index of the custom table; with `custom_fuels` overriding a fuel numbered 1-10 that row was another fuel
  (FM9 when fuel 1 was overridden). It now always uses FM10 in the default table. Runs without custom fuels are
  unchanged. Test: test_custom_fuels_keep_the_standard_fm10_crown_bundle (fails on the old code).
- `generate_weather` cut the series at `start`, so a start with minutes (18:30) lost the 18:00 row and `simulate()`
  refused the weather for its first partial hour. It now keeps the row at `start.floor('h')`; on-the-hour starts are
  unchanged. Test: test_generated_weather_covers_a_fractional_start.
- Initial attack starting at (or after) the first ignition's hour: the first heap entry saw zero burned cells, a zero
  perimeter, and the solver stopped as "contained" before the fire ignited, even at line rate 0. Containment is now
  only tested once at least one cell has burned. Runs whose initial attack starts after the first cell burns (all
  shipped uses: FireDetective's banks start it at 1-6 h) are unchanged. Test:
  test_initial_attack_at_ignition_time_does_not_contain_an_empty_fire (fails on the old code).
- U11 (not fixed): a timed suppression line is checked against the departure time for every intermediate cell of a
  multi-cell stencil step, and only the destination's build time is checked at arrival, so a line finished while
  the fire is mid-step can be crossed (Codex's example: a (1, 2) step at 10 m/min crosses a line finished at
  0.00001 h). Same family as U1 (lines leaking); both change default physics and belong in physics p2.
- `fwi.daily_from_hourly` converted UTC to local solar time with the longitude sign reversed, so "noon" was local
  night (05:20 UTC at lon = -100; 04:00 UTC at lon = -120). Only the physics checks call it (FireDetective's FWI
  baseline uses `fwi_series` on gridMET daily values and is not affected). After the fix, check_moisture's
  FFMC-implied noon moisture is 3.84 % against the simulator's noon 1-h moisture 3.42 % (r = 0.946); the shipped
  numbers, taken at night, were 12.60 % and r = 0.681. moisture.json is updated for those two values only. The FWI
  check's xclim season comparison now has different inputs from the shipped xclim output, so without
  firesim/.refenv it is skipped rather than compared. Test: test_fwi_daily_inputs_are_taken_at_local_noon.

## 2026-11-21 — Rebuild, first four download stages (gofer, fpafod, lfmc, feds)

Run on Windows from the main checkout with `.venv` (Python 3.14) and the `fire-data` extra installed. About 1.1 GB
downloaded, 7 min in all. Raw files are in `data/raw/` (git-ignored): FEDS-MTBS 619 MB, GOFER 828 MB with its unpacked files,
FPA-FOD 211 MB, Globe-LFMC 70 MB.

- **FEDS-MTBS:** 7,739 fires, 171 MB processed (as the handoff recorded). All 29 columns `fires_raw.parquet` shares
  with the frozen `fires.parquet` match for every fire, except how missing text is stored (NaN vs None). The
  perimeters, fire lines and VIIRS pixels have no shipped hashes to check against.
- **GOFER:** 28 fires, 20,301 progression rows. **FPA-FOD:** 1,593,067 CONUS records 2001-2020 (handoff: 1.59 M).
- **Three fixes to the scripts, none changing outputs:**
  - The FPA-FOD link `..._Data_Format2_GPKG.zip` returns 404. The RDS-2013-0009.6 catalog page lists the same
    edition's GeoPackage as `..._Data_Format3_GPKG.zip` (221 MB, same host). `build.py` now uses it.
  - figshare names the Globe-LFMC file `Globe-LFMC-2.0 final.xlsx` (a space). `build.py` now matches either spelling
    and saves it under the underscore name `build_lfmc.py` reads.
  - `build_feds.py` imported `resource`, which is POSIX only, for the sandbox's 1.1 GB memory guard. On Windows the
    guard is skipped.
- **`lfmc_us.parquet` gap closed (reconstruction):** new `data/scripts/build_lfmc_us.py`, run by the `lfmc` stage. It
  keeps Country == 'USA' without Alaska (238,997 rows, 1,244 sites, 1977-2023) and adds `date`. Checked against
  the shipped `proc/baselines/moisture_exam.json`: with `baselines_moisture.py`'s filters it gives exactly 1,167
  sites and 15,361 dev + 138,113 exam samples, and the seeded site split gives exactly 122 dev sites. So
  `moisture_split.json`, when the `baselines` stage writes it, should match the frozen one. It is still not
  byte-checked, because the shipped file was never in the zip.
- `verify`: 230 of 230 shipped files identical, splits OK, loader smoke test passes (40 dev fires, 4,197 exam fires).
- Not run yet: `wsts` (5 h or more), `gridmet` (tens of GB transferred), `landscapes` for the 48 replay fires,
  `baselines`. The exams are not re-scored.

## 2026-11-22 — Rebuild, long stages, and the four exams re-scored

The rest of `build.py` was run on Windows from the main checkout, and the four fire exams were re-run on the rebuilt
data with the released weights. All outputs went to git-ignored folders (`data/`, `models/*/work/`). Nothing was
chosen from an exam; this was a reproduction check.

- **gridmet:** 33 files, 89 MB in `proc/weather/gridmet`: gridMET daily values sampled at the fires (13 yearly
  tables), the FPA-FOD records (9) and the LFMC sites (11).
- **landscapes:** LANDFIRE landscapes (76 MB) and HRRR weather (68 MB) for 88 fires: the 48 replay fires and the 40
  dev fires.
- **wsts:** all 607 fires, no failures, about 4.5 h by HTTP range reads (the first try failed every fire for
  lack of output folders; fixed in PR #8). 128 MB of daily fire and weather arrays plus 607 header files (1.1 MB).
  The 4 dev fires load through `fd.wsts`; the guard refuses exam fires, as it should. The handoff recorded 269 MB.
  The likely gap is the static layers (elevation, land cover, NDVI) that the sandbox once kept for 299 fires: the
  shipped `fetch_wsts.py meta` writes only headers (the layer code after its `return` never runs), and no code reads
  those layers. Not checked further.
- **baselines:** `spread_exam.jsonl` rebuilt (4,197 fires, 93 MB) and `moisture_split.json` written (122 dev sites
  of 1,167, as the reconstruction predicted). The spread summary matches all 1,844 shipped numbers exactly, the moisture
  metrics to 7e-15 (GBM MAE 24.72 % on 138,113 exam samples), and the FireSize baselines to 6e-7.
- **Side effect, undone:** the `landscapes` stage's precip and repair steps rewrote 21 shipped dev HRRR files. 14
  differed only in container bytes; 7 had one or two wind_speed cells moved by one float16 step (0.00195 m/s).
  Re-running the baselines for comparison also overwrote three shipped baseline files. All were restored with
  `build.py --from-zip <zip> --only verify`; verify then reported 230 of 230 identical. Running `landscapes` on a
  data root that already holds the shipped files should not rewrite them; not fixed yet.
- **numba** was missing from `.venv` and is needed by FireSim's fast path; installed (0.68.1). Fire tests: 31 passed,
  1 xfailed.

Re-scores (details in each model's NOTEBOOK.md, 22 Nov entries):

| Model | What was compared | Result |
|---|---|---|
| FireSize | 560 exam numbers | all match, largest difference 7.8e-16 |
| FireDetective | 391 exam numbers, 944 fires rebuilt from LANDFIRE | largest difference 4.6e-5 |
| FirePilot | 324 planner numbers, stages A and B (48 fires) | within 2e-9; headline 0.474 / 0.644 |
| FireCast | 616 metric numbers, 3,030 fires, 25,105 days; replay 627 days | within 3.7e-5; AP 0.323, replay ensemble 0.165 |

One real change: FirePilot's fidelity check (FireSim vs observed total burned area) now gives Dice 0.49 / 0.44 /
0.40 at 24 / 48 / 72 h, against 0.48 / 0.44 / 0.40 shipped. Running the same 48 fires with FireSim before PR #7's
mask-ignition fix (3383dfa, interior cells of the start mask burn at t = 0) gives the shipped numbers exactly, so
the fix is the cause. Planner outcomes do not change. The model card now says so.

The sandbox exams loaded torch checkpoints that were not shipped. Two small scripts re-run them from the released
numpy weights: `models/firecast/training/src/rescore_np.py` and `models/firedetective/training/rescore_release.py`.
