# SnowSense examples

Three SNOTEL exam stations, a fixed random draw (seed 20261118 + 7, `training/make_examples.py`), one from each of the
Pacific, Central Rockies and Northern Rockies groups. Water years 2016-2025 plus the August-September lead-in of WY2016.
SnowSense never trained on these stations or on any snow measurement; `SWE_obs` is used only to score it. The files
are unchanged from the release.

| File | Station |
|---|---|
| `584_OR_SNTL.csv` | Little Meadows, OR, 1,225 m |
| `497_WY_SNTL.csv` | Granite Creek, WY, 2,063 m |
| `349_MT_SNTL.csv` | Black Pine, MT, 2,185 m |

Columns: `date`, `P` (mm, the local-day gauge total), `T_mean`, `T_min`, `T_max` (deg C), `SWE_obs` (mm, start of
day, as SNOTEL reports it). `stations.json` gives name, state, region, latitude and elevation.

## Source and licence

USDA NRCS SNOTEL daily data (National Water and Climate Center report generator, https://wcc.sc.egov.usda.gov/reportGenerator/,
downloaded 18 Nov 2026 by the Watershed WaterData package, units converted to metric). US public domain. Please credit
the USDA Natural Resources Conservation Service SNOTEL network.

The other Watershed sources (Caravan with CAMELS-US and ERA5-Land, USGS NWIS, ResOpsUS with GRanD, CDEC) are not used
by these examples.
