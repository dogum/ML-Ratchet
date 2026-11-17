# BuildingSim checks: the first re-run here (17 Nov 2026)

The JSONs in this folder are the published v1.0 results, as shipped (Python 3.13 on the 2-core sandbox, 16 Nov 2026).
Re-runs write to `checks/out/` under the data root and never over them (`_common.write`; `BUILDINGSIM_PUBLISH=1`
restores the old behaviour). `python checks/compare.py` compares every numeric leaf (relative tolerance 1e-6; both
values below 1e-9 count as equal; timings reported, not compared). This re-run: Windows 11, Python 3.14.9, numpy 2.5.4,
scipy 1.18.1, pandas 3.1.1, pvlib 0.16.1, single thread, with four other agents' jobs on the machine. Data: first
`buildingsim-minusdata.zip` and five proc chunks, then the full release `buildingsim-buildingdata-v1.0.zip` (sha256
c918802d…), unpacked by `mlratchet data building --from-zip`: 409 of 409 data files identical to MANIFEST.json. The
release's code, docs, check JSONs and splits.json are byte-identical to the parts received earlier.

| Check | Numbers equal | What differs |
|---|---|---|
| `analytic.json` (steady conduction, RC step, degree-day limit) | 53 / 53 | Nothing. Steady state 1.8e-9 from the 5R1C network |
| `multinode.json` (steady state, ISO 13786 wall, BESTEST 600/900) | 82 / 84 | Two round-off residuals themselves: steady-state error 1.67799e-9 vs 1.67800e-9, a ledger closure 1.29815e-9 vs 1.29814e-9 |
| `bestest.json` (ASHRAE 140, 5R1C) | 222 / 222 | Nothing: 29 of 34 values in the reference ranges, as published |
| `faults.json` (26 fault cases, signatures, ledgers) | 544 / 544 | Nothing: 26 of 26 signatures in the expected direction |
| `controller.json` (TOU pre-cool, step API, rollback, PV + battery) | 34 / 34 | Nothing; step API vs callable 0.0 kWh, rollback 0.0 kW. Controller overhead 0.077 s vs 0.066 s (timing) |
| `generator.json` (156 random buildings vs RECS/CBECS prior medians) | 1715 / 1715 | Nothing; 0 errors, max ledger closure 8.2e-9. Per-building wall times 1.33x slower here (median; timing) |
| `speed.json` | timings only | 15-min building-years 0.58-1.33 s cached (v1: 0.38-0.74 s); generated mix 1.33 s (v1: 1.01 s). Shared machine |
| `resstock_twins.json` (`check_resstock.py summary`) | 117 / 117 | Nothing: 126 dwellings, site 1.10x, daily r 0.92. Re-summarised from the shipped per-unit results in `checks/out/resstock/`, not re-simulated |
| `comstock_twins.json` (`check_comstock.py report`) | 204 / 204 | Nothing: held-out 45, site 0.86x (aggregate 0.97), daily r 0.89. Re-summarised from the shipped per-unit results, not re-simulated |
| `bdg2_dev_meters.json` (52 real BDG2 dev buildings, 84 building-meters, 2016) | 1233 / 1236 | Three near-zero values, 1e-6 to 5e-6 relative (a daily r of 0.0187, two temperature slopes of 1e-3 and 6e-5). Annual sim/real median: electricity 0.98 (n 47), chilled water 0.28 (20), steam 0.29 (10), hot water 0.15 (5), gas 0.03 (2), as published. 22 s |
| `buildingsim/demo.py` | runs (1.8 s) | No shipped numbers to compare; every ledger closes (zone 1e-10 to 5e-9) |

Could not run:

- `check_components.py` (psychrometrics vs CoolProp, PMV vs pythermalcomfort): needs the reference venv
  `buildingsim/.refenv` with CoolProp and pythermalcomfort (not shipped). Its path is the Linux `bin/python`.
- `check_resstock.py`, `check_comstock.py` as simulations: need `data/proc/stock/` (never shipped; stage `stock` of
  `data/build.py`) and, for ResStock, `buildingsim/tmp/res_pick.csv`, which no script in the release writes, and
  `buildingsim/refdata/stock_weather/` (`tools/fetch_stock_weather.py`).
- `energyplus_reference.json` is a record that EnergyPlus was blocked, not a check.

## Splits

The file's sha256 is 524aee15… and its own `sha256` field (7ac15e9f…) equals the hash recomputed the way
`make_splits.py` computes it. `mlratchet data building --only splits` re-ran `make_splits.py` on the shipped
`bdg2/buildings.parquet`, `eia/*.parquet` and `faults/runs.parquet` in `data/work/splits_rebuild/` (never over the
frozen file): the `sha256` field reproduces exactly (7ac15e9f…, MATCH), and the file is byte-identical once Windows'
CRLF line endings are turned back into LF (on Windows `json.dump` to a text-mode file writes CRLF: 451,639 bytes vs
424,395). The `faults/runs.parquet` it rewrites equals the shipped one. Same draws: BDG2 52 dev / 1,584 exam, RECS
9,247 / 1,848 / 7,401, CBECS 3,217 / 661 / 2,558, faults 20 dev / 217 exam.

## BuildingData baselines (`data/scripts/baselines.py`)

Re-run on the shipped data in a scratch root (`data/work/baselines_rebuild/`, so the shipped `proc/baselines/` was
not touched): faults 3 s, thermostat 7 s, detective 277 s, load 280 s, summary 1 s.

| Output | Result |
|---|---|
| `summary.json` (every exam baseline number) | 993 / 993 equal (one is NaN on both sides: the AUC of a dataset with one class) |
| `faults_apar.parquet`, `thermostat.parquet` | identical |
| `faults_summary.json`, `thermostat_summary.json`, `detective_summary.json` | equal (94+1 NaN / 95, 367 / 367, 107 / 107) |
| `load_bdg2.parquet` (1,586 building-meters, change-point and TOWT) | every number within 4e-9 relative |
| `prism_bdg2.parquet` (2,091 PRISM fits) | 99.80 % of numbers within 1e-6; the rest within 5e-5 relative (least-squares fits on a 0.5 degC grid); the detective summary built from it is equal |


## BuildingSim 1.0.1: Phase 2 wrappers upstreamed, flags off and on (17 Nov 2026)

BuildingSim 1.0.1 adds the four models' Phase 2 wrappers, all opt-in (`buildingsim/README.md`, "Phase 2 wrappers";
plan and results in `../NOTEBOOK.md`). Same machine and Python as above; three single-thread checks at a time.

**Flags off: identical to v1.0.** Every check that ran before was re-run with no wrapper used, then `compare.py`:

| Check | Numbers equal | Same as the first re-run? |
|---|---|---|
| `analytic.json` | 53 / 53 | yes |
| `multinode.json` | 82 / 84 | yes: the same two round-off residuals (1.67799e-9 vs 1.67800e-9; 1.29815e-9 vs 1.29814e-9) |
| `bestest.json` | 222 / 222 | yes |
| `faults.json` | 544 / 544 | yes |
| `controller.json` | 34 / 34 | yes (2 timings) |
| `generator.json` | 1715 / 1715 | yes (156 timings) |
| `bdg2_dev_meters.json` | 1233 / 1236 | yes: the same three near-zero values (daily r 0.0187, slopes 1e-3 and 6e-5) |
| `speed.json` | timings only | 3 / 3 non-timing values |

`resstock_twins.json` and `comstock_twins.json` are re-summaries of shipped per-unit results (unchanged).
`tests/test_building.py::test_phase2_flags_off_is_v1` checks `options=None`, `{}`, `{'holidays': 'auto'}` at a US
site and `{'holidays': 'US'}` give the v1.0 arrays bit for bit.

**Each port reproduces the model's wrapper** (`check_phase2.py` -> `phase2_equivalence.json`; the models' v1.0
training files, unchanged from dfafdcf, imported read-only):

| Port | Cases | Largest difference from the model |
|---|---|---|
| holiday calendars (`holidays.py`) | 4 countries x 2015-2020 | 0 differing years |
| `options={'holidays': 'auto'}` vs LoadCast's monkeypatch | Robin (UK) office twin over Easter 2016 | 0.0 (2 holidays; electricity 68,501 kWh vs 71,670 with the v1.0 calendar) |
| `phase2.spec_from_meters` | 1,044 (52 dev buildings x the calibration grid + 3) | 0 differing specs |
| `phase2.scale_meters` | random meters and knobs | 0.0 |
| `phase2.apply_intensity` (W1) | 24 generated buildings, 12 uses | 0 differing specs |
| `phase2.make_campus` + `district_eff_chw` (W2) vs `_eff_chw` + meter rescale | 5 campus buildings, 2 weeks | 1.2e-10 kW (relative 2e-16); the port's ledgers close (closure <= 1.3e-10) |
| `phase2.flat_base_loads` (W3) | 20 meter sets | 0.0 |
| `phase2.clean_labels` | 400 label rows | 0.0 |
| `phase2.log_commands` | 6 faults x 2 seeds, 15 min | 3e-8 (float32 day arrays) |
| `phase2.damper_command` / `installed_damper` | 50 + 20 draws | 0.0 |
| `phase2.static_offsets` (v4 large and v3) | 20 draws each | 3.8e-6 K (float32 rounding order) |
| `phase2.OperatorJitter` | 3 seeds, a week at 15 min | 0.0 |
| `stock.recs_to_spec` / `recs_comfort_band` | 24 RECS dev respondents | 0 differing |
| `bd.tariff_arrays`, `bd.bill_arrays` | 19 URDB + 1 synthetic tariff, 2017 | 0.0 |
| `bd.tariff_prices(..., utc=True)` vs the tz frame | 608 frames (19 tariffs x 7 time zones x 2016-2019) | 0 differing |
| `bd.rtp_prices` | ERCOT, 2017 weather -> 2019 prices | 0.0 |

**Flags on: the 52 BDG2 dev twins, 2016** (`check_bdg2.py w2 | w2w3 | lc` -> `bdg2_dev_meters__<v>.json`, published
here as the dev baseline). Annual sim/real median [IQR], and daily r median:

| Meter (n) | v1.0 | `w2` (Detective W1 + W2) | `w2w3` (+ W3 at its means) | `lc` (LoadCast campus knobs + holidays) |
|---|---|---|---|---|
| electricity (47) | 0.98 [0.50-1.42], r 0.34 | 0.68 [0.39-1.49], r 0.34 | 1.02 [0.59-2.24], r 0.34 | 1.23 [0.78-1.85], r 0.17 |
| chilled water (20) | 0.28 [0.20-0.45], r 0.88 | 0.89 [0.52-1.37], r 0.88 | 0.90 [0.52-1.37], r 0.88 | 0.50 [0.33-0.80], r 0.89 |
| steam (10) | 0.29 [0.12-0.56], r 0.70 | 0.88 [0.55-1.52], r 0.72 | 0.87 [0.55-1.61], r 0.72 | 0.88 [0.40-1.66], r 0.85 |
| hot water (5) | 0.15 [0.13-0.22], r 0.76 | 0.89 [0.77-1.19], r 0.79 | 0.87 [0.76-1.15], r 0.79 | 0.53 [0.45-0.64], r 0.75 |
| gas (2) | 0.03, r -0.13 | 0.04, r -0.13 | 0.04, r -0.13 | 0.15, r 0.25 |

Shape statistics (median; real vs sim): summer level = Jun-Aug mean / annual mean; night/day = weekday 22-5 h /
10-16 h.

| Meter | real summer, night/day | `w2` sim | `w2w3` sim | `lc` sim |
|---|---|---|---|---|
| electricity | 0.97, 0.79 | 1.03, 0.40 | 1.02, 0.54 | 1.03, 0.75 |
| chilled water | 1.53, 0.80 | 1.98, 0.10 | 1.49, 0.38 | 2.14, 0.56 |
| steam | 0.66, 0.95 | 0.09, 0.12 | 0.74, 0.70 | 0.18, 1.78 |
| hot water | 0.01, 0.87 | 0.02, 0.12 | 0.70, 0.73 | 0.05, 2.12 |

What changes: W2's factors lift the district meters from about 0.2-0.3x to 0.87-0.90x, but they were fitted on these
same buildings (2017), so this is in-sample. W1 cuts electricity to 0.68x; W3's flat load puts it back to 1.02x.
W3 is affine per meter, so it cannot move any correlation (daily r is unchanged); it fixes the steam summer level
(0.09 -> 0.74 against 0.66 real) and narrows the night/day gaps (chilled water 0.10 -> 0.38, steam 0.12 -> 0.70,
against 0.80 and 0.95 real), but it is wrong for these 5 hot-water meters, which
are off in summer (real 0.01; W3 gives 0.70). LoadCast's campus knobs alone get steam to 0.88x and chilled water to
0.50x without any distribution factor, raise electricity to 1.23x and halve its daily r (0.34 -> 0.17), and make
steam and hot water peak at night (night/day 1.8-2.1 against 0.9 real: 24x7 air handlers with x2.5 outdoor air).
Holidays change nothing here: the only non-US dev buildings are two Robin lodging buildings, and the lodging schedule
ignores the day type (11 UK holidays in 2016, 0.0 kW difference).
