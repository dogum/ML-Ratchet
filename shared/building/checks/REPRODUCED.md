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

