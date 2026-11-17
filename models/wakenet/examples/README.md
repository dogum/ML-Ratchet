# WakeNet real exam examples

Six single-timestamp real SCADA snapshots drawn once by fixed seed (`20261116`) within input-defined wind-direction, yaw, and availability strata (`training/export.py`):

- `01_kelmarsh_sw_wakes.npz` — Kelmarsh (6× MM92), prevailing SW pair wakes (`2020-03-10 20:40 UTC`)
- `02_kelmarsh_nw_3row.npz` — Kelmarsh (6× MM92), aligned NW 3-turbine row `KWF3 -> KWF5 -> KWF6` (`2020-02-23 15:30 UTC`)
- `03_kelmarsh_yaw_offset.npz` — Kelmarsh (6× MM92), upstream yaw misalignment $\ge 10^\circ$ (`2021-12-29 22:50 UTC`)
- `04_kelmarsh_parked_unit.npz` — Kelmarsh (6× MM92), 1 upstream turbine parked/offline (`2021-03-14 03:50 UTC`)
- `05_penmanshiel_sw_14wt.npz` — Penmanshiel (unseen 14× MM82 farm), SW multi-row array (`2020-02-18 02:40 UTC`)
- `06_penmanshiel_nw_yaw.npz` — Penmanshiel (unseen 14× MM82 farm), NW coastal ridge + yaw offset (`2019-01-08 11:10 UTC`)

Data source: Charlie Plumley / Cubico Sustainable Investments, Kelmarsh (`10.5281/zenodo.5841834`) and Penmanshiel (`10.5281/zenodo.5946808`) Wind Farm Open Data, licensed under CC BY 4.0.
