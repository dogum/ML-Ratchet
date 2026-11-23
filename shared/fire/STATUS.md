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

## 2026-11-22 — Physics p2: U1, U9, U10 and U11 fixed

FireSim has a physics version: `params['physics']` for the solver and `generate_weather(physics=...)` for the
weather generator. The default is now `'p2'`. `'p1'` reproduces the old code exactly: 16 test runs (three solvers,
timed lines, spotting, initial attack, the U1 ring, constant weather) give arrival rasters identical to the code
before this change. Every shipped `checks/*.json` was made with p1, and the tests now re-run them with p1.

- **U1 and U11, timed lines** (`kernel.py`): p2 walks each straight segment cell by cell. A cell blocks the step
  if its line is finished before the fire enters it. A corner crossing is blocked if both side cells are unburnable
  or lined by the time the fire reaches the corner. The two-point update checks the cells and corners of both its
  legs against its arrival time; in p1 it never checked lines on its second leg. Results:
  - The U1 ring (31 × 31 grass, 4-connected line at t = 0) now holds: 0 of 852 outside cells burn, against 282
    in p1.
  - A line of diagonal single cells holds: burned area falls from 4,567 to 2,439 cells and from 3,712 to 1,874.
  - A line finished 0.00001 h after ignition stops (1, 2) steps across it. In p1, 244 cells past it burned.
- **U9, two-point update** (`kernel.py`): the trip is integrated through the hourly weather from its start time.
  Test: fast spread for one hour, then 1-h moisture 14.9 % with no wind (0.009 m/min). After 1 h, p1 reached 62 more
  cells and p2 reaches 2.
- **U10, generated weather** (`weather.py`): event and wet flags follow the local solar day that
  `hourly_from_daily` laid out, using its own hour rounding. At lon = -120, 8 hours a day changed day.
- **Also changed in p2:** when a segment crosses an hour, p1 spread its progress evenly over all its cells; p2
  gives each cell its own part.
  - In strictly constant weather, p2 equals p1 to 2e-7 h. The start step of a two-point trip can use the float32
    look-ahead cache.
  - A first version blocked the two-point update when the straight segment arrived after the end time. That
    dropped cells at the edge of the run and moved the ellipse summary (0.981 to 0.979); fixed before commit.
- **Checks under p2** (`checks/p2/`):
  - Ellipse summaries are identical: default mean 0.981, minimum 0.940, orientation bias 3.4 %; level_set
    0.992 / 0.973.
  - All 2,343 analytic numbers agree to 1.5e-5 relative. The fuel break still leaks 0 cells.
  - Speed: 0.51 s against 0.47 s for 30 × 30 km at 60 m over 24 h (8 % slower).
- **Effect on ordinary runs** (generated landscapes and weather, 36 h, the cases above):
  - Burned area changes by -1.9 % to +0.2 % (straight timed lines included, diagonal ones not); the speed check's 24 h runs change by -0.2 % (mtt) and
    -2.4 % (level_set).
  - Arrival times move by a median 0.02 to 0.19 h; the 90th percentile is 0.13 to 0.82 h.
  - The `replay_dev` calibration, the VIIRS emulator check and the dev replays were not re-run under p2.
- **Not done here:** nothing was re-scored under p2. The models' code still calls FireSim with default params, so
  re-running their training or exam scripts now uses p2. Only the released rung's weights were shipped for each
  model, so "score every rung" needs the other rungs retrained from rebuilt banks. The one released inference path
  that runs FireSim, FirePilot's `pipeline.plan_fire`, now passes `physics='p1'` (its v1.0 physics).

## 2026-11-22 — Released rungs scored on physics p2 frozen sets; U12 fixed, U13 found in p1

For each model a p1 rebuild and a p2 version of the frozen simulated set were built from the same seeds. Each
released rung was scored on both with its shipped weights. The p1 rebuild checks that the environment reproduces
the shipped score; p2 minus the p1 rebuild is the physics change alone. Only released weights were shipped, so
no other rung was scored and no ratchet decision was re-taken. Outputs are in `models/*/work/` (git-ignored); details
are in each model's NOTEBOOK.md (22 Nov entries).

| Model (rung, score) | Shipped (p1) | p1 rebuild | p2 | p2 - p1 rebuild |
|---|---|---|---|---|
| FireCast (r2, AP F) | 0.5866 (149 fires) | 0.5866 (149) | 0.5923 (147) | +0.006 |
| FireDetective (R6, primary) | 0.3759 (~650) | 0.3759 (650) | 0.3677 (641) | -0.008 |
| FirePilot (r1, saved share) | 0.484 (160) | 0.484 (160) | 0.460 (159) | -0.024 |
| FirePilot (r0) | 0.403 | 0.403 | 0.403 | 0.000 |
| FireSize (r7, RPSS) | 0.2799 (1,175) | 0.2841 (1,175) | 0.2846 (1,175) | +0.0005 |

- **Reproduction:**
  - FireCast, FireDetective and FirePilot reproduce exactly (to 2e-8, 2e-5 and to the last printed digit).
  - FireSize's network inputs reproduce (predicted escape rate identical), but its simulated labels do not
    (simulated escape rate 1.563 % against 1.445 %; escape AUC 0.861 against 0.874). Its LANDFIRE windows and
    30-day gridMET series were re-downloaded and the originals were not shipped. Open.
- **Physics was implicit:** the models' training code called FireSim with default params, so since PR #10 it ran p2
  while writing p1-named folders. Each model now names its solver:
  - FirePilot: the `FIREPILOT_PHYSICS` variable.
  - FireDetective: a `_p2_` tag in the bank name.
  - FireCast: a `"physics"` key in `phys/*.json`; `phys/p2.json` is added.
  - FireSize: `v1p1` / `v1p2` in `sim.py`'s table.
  - Absent, they use p1 as before.
- **U12, found here and fixed in p2:** when the straight step to a cell met that cell's (or an intermediate cell's)
  line before arriving, the step counted as blocked and the two-point update was skipped. The two-point update can
  arrive hours earlier, so a line could stop fire that reached its cell before the line existed. Example:
  FirePilot v0117, a cell reached at 0.995 h stayed unburned under a line built at 5 h.
  - p2 now treats such a line as removing only the straight candidate; unburnable fuel still blocks.
  - The merged p2 had the bug in milder form: 1 of 40 cells in the new test, p1 3 of 40.
  - Test: `test_u12_a_line_built_after_a_cell_burns_does_not_change_that_cell`.
  - p1 is unchanged: the 16 comparison runs are identical to the old code. No check or comparison run without
    timed lines moved.
- **U13, p1 only (not a p2 change):** p1 re-checked every cell of a step, the departure cell included, at each hour
  boundary. A line built on a cell the fire had already left therefore cancelled spread still in transit from it.
  p2 checks the departure cell once, at departure.
- **What U12 and U13 did to FirePilot:** plans that close lines near the fire were credited for line built behind
  the front.
  - Same scenario and plan: direct attack saved 0.873 under p1 and 0.007 under p2.
  - Removing the line cells the free fire reaches before they are built brings p1 down to p2 in all five
    scenarios tried.
  - On val_p2, direct attack falls from 0.107 to 0.054 and anchor and flank from 0.195 to 0.114; the indirect plans
    move by at most +0.012.
  - FirePilot v1.0's exam was scored in p1; its model card now says so.
- **Not done:** the FirePilot gate and exam, FireDetective's dev composite and FireCast's dev metrics were not
  re-run under p2. The dev metrics use real data only and are unchanged. No rung was retrained.

## 2026-11-22 — FirePilot gate and exam re-run under physics p2

FirePilot's dev gate was rebuilt as p1 and p2 from the same seeds. The 48-fire exam then ran once under p2 with the
released r1 weights; nothing was retrained or chosen. Details are in `models/firepilot/NOTEBOOK.md`, and the summary
is in `models/firepilot/training/results/exam_summary_p2.json`.

| FirePilot, mean share of no-action growth stopped | p1 | p2 |
|---|---|---|
| Gate (24 cases): r1 / r0 | 0.481 / 0.383 (rebuild identical) | 0.487 / 0.354 |
| Exam 300 m/h: r1 / MPC / anchor and flank | 0.474 / 0.400 / 0.316 | 0.401 / 0.351 / 0.198 |
| Exam 1000 m/h: r1 / MPC / anchor and flank | 0.644 / 0.591 / 0.480 | 0.629 / 0.526 / 0.372 |
| Exam perfect-foresight bound, 300 / 1000 m/h | 0.614 / 0.792 | 0.582 / 0.780 |

- r1 still beats r0 on the p2 gate, so the ratchet decision stands.
- On the exam at 300 m/h, r1's leads over MPC (paired −0.013 to 0.115) and r0 (−0.025 to 0.131) no longer exclude
  zero. At 1000 m/h r1 now beats MPC (0.030 to 0.179).
- Plans that close line near the fire lost most, consistent with U12 and U13 crediting line built behind the front.
- Exam fidelity (free run vs observed perimeters) moves by at most 0.016 in median Dice.
- Still not done: retraining any fire model in p2.

## 2026-11-22 — LawFinder: FireSim's implied growth is a median 36× too large (recorded, not fixed)

LawFinder v1.0 (`models/lawfinder/`) ran symbolic regression on FireSim banks (physics p1 with the dev-calibrated ROS
factors, 1,680 multi-day fires) and graded the laws on FireData's real exam fire-days. Nothing in `shared/fire/` was
changed.

- FireSim's own law, G ≈ (0.32 P)^1.6 km² a day (P = perimeter, km), over-predicts real next-day growth a median 36×
  at the same perimeter, wind and moisture (bias +1.56 decades).
- On all 38,599 exam fire-days it scores 1.80 mean |log₁₀|; no growth at all scores 1.40.
- The released law, log₁₀ G = −0.559 / P, exists only because the dev gate refused every law that kept growth rising
  with perimeter. It beats FireCast on growth days (0.693 against 0.757) and loses to yesterday's growth on all days
  (1.15 against 0.671).
- The same gap shows in FireDetective's dev twins (growth on 91 % of half-days against 50 %) and in FirePilot's
  uncalibrated replays (3.9×). Persistence (stalls, burnout, night recovery) is the known missing piece; p2 did not add
  it, and the 36× was not re-measured under p2.
- Backlog: `firesim-growth` (docs/backlog.yaml).
