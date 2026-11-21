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
- **`lfmc_us.parquet` has no script.** `build_lfmc.py` writes `data/proc/lfmc/lfmc_all.parquet`, but
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
Synoptic (key needed), FIRMS (MAP_KEY needed; not used). No download stage was run on import. The FPA-FOD URL in
`build.py` follows the RDS archive's file naming and was not tested; if it fails, download the GeoPackage zip from the
RDS-2013-0009.6 page by hand.

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
