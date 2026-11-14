# GridSim v1 checks, re-run in the repository (14 Nov 2026)

The JSONs in this folder are the published v1 results, as shipped. `run_all.py` re-runs the checks and compares every
number with them (relative tolerance 1e-6; solver timings and values below 1e-9 on both sides are not counted). This is
the record of the first re-run, on Windows 11 (Python 3.14, numpy 2.5, scipy 1.18, pandas 3.1, highspy; reference
libraries pandapower 3.5, PYPOWER, ANDES 2.0 under Python 3.11 / pandas 2.3), with the v1.0 data unpacked by
`mlratchet data grid --from-zip` plus three sources fetched fresh that day: NASA POWER hourly at the 20 ERCOT weather
points, CAMPD Texas hourly 2018-2025 (EPA bulk files), PUDL tables.

| Check | Numbers equal | What differs |
|---|---|---|
| `frequency_analytic.json` | 89 / 90 | One reference value from `scipy.signal.lsim`: hydro nadir time 37.78 s vs 37.69 s (the reference, not GridSim) |
| `frequency_events.json` (GB 9 Aug 2019) | 198 / 198 | Nothing (only the trace's file path). First nadir 49.11 Hz at 24.5 s, demand disconnection at 75.6 s |
| `frequency_dev_events.json` | 167 / 167 | Nothing |
| `frequency_andes.json` | 9 / 9 | Nothing: ANDES Kundur two-area, 15 mHz RMS |
| `powerflow.json` (vs PYPOWER, pandapower) | 150 / 154 | The four DC-OPF reference gaps themselves, e.g. ACTIVSg2000 LMP gap $0.012740 vs $0.012740/MWh (5th digit) |
| `network_dispatch.json` | 93 / 99 | ACTIVSg2000 24 h MILP: prices in hours 14-16 (e.g. $35.87 vs $33.60); every other hour equal. An equally cheap commitment found by a newer HiGHS |
| `generator.json` | 54 / 54 | Nothing |
| `renewables.json` | 69 / 69 | Nothing (NASA POWER fetched fresh returned the same series) |
| `demand.json` | 114 / 114 | Nothing |
| `calibrate_ercot.py` (dev 2019) | — | `gridsim/data/calibration.json` re-fitted identically |
| `ercot_dispatch_2019.json`, calibrated | 688 / 817 | MILP solution noise in the 4th-6th digit: coal +5.63 %, gas −2.17 %, CO2 −1.0 % (all as published); price rank correlation 0.871 |
| `ercot_dispatch_2022.json`, calibrated (held out) | 755 / 817 | Same kind: coal +27.71 %, gas −11.43 % (as published), CO2 +1.844 % vs +1.846 %; median price $46.55 vs real $47.99 (published $46.56), rank correlation 0.881 |
| `ercot_inertia.json` | 23 / 30 | Within 0.1 %, except 2022 hours below 150 GW·s: 943 vs 949 |
| `speed.json` | timings only | This machine, with other jobs running: ACTIVSg2000 AC power flow 71 ms (v1: 36 ms), one frequency event 0.9 s (0.3 s) |

Not re-run: the merit-order and economic-dispatch variants of the ERCOT years (`merit`, `economic`), and the v0
generic-availability runs (kept as history).

Data rebuilt from the raw sources and compared with the shipped tables: the ERCOT event table (329 events) and all 33
network tables are identical; the Henry Hub table byte-identical; the EIA-923 receipts byte-identical; the heat-rate table
identical in content. The models' exams re-scored on this data are recorded in each model's `NOTEBOOK.md`.
