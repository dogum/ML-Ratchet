# FlowCast exam examples

Three CAMELS-US exam basins, one per stratum, drawn by `training/src/webdata.py` with seed 20261118 within the exam
basins (snow: snow share > 0.3; arid: aridity > 1; humid: aridity < 0.6), never hand-picked and never used for
training or for choosing anything:

| Basin | Name | Stratum | Area |
|---|---|---|---|
| camels_01137500 | Ammonoosuc River at Bethlehem Junction, NH | snow | 227 km² |
| camels_07315700 | Mud Creek near Courtney, OK | arid | 1,486 km² |
| camels_12095000 | South Prairie Creek at South Prairie, WA | humid | 206 km² |

- `<id>_forcing.csv`: basin-average daily forcing, 1993-10-01 to 1996-10-01 (WY1994 is the 365-day warm-up, then
  WY1995-96 and one extra day): P, PET (mm/d), T_mean, T_min, T_max (°C), SRAD (W/m², net shortwave), vp (Pa),
  wind (m/s). UTC days. Caravan's ERA5-Land forcing.
- `<id>_obs.csv`: observed daily flow Q_mm (mm/day), 1994-10-01 to 1996-09-30, as distributed in Caravan (USGS
  daily values, local standard-time days). Used only to score the prediction (and, in the gauged nowcast, as its
  gauge input).
- `examples.json`: the basins' attributes in FlowCast units (slope m/km, conductivity cm/h) and the windows.

The site instrument and the offline lab use the same three files (`web/export_instrument.py`).

## Sources and licences
- **Caravan** v1.6 (Kratzert, F. et al. 2023, *Caravan - A global community dataset for large-sample hydrology*,
  Scientific Data 10:61), CC BY 4.0: the forcing and flow excerpts and the attributes.
- **CAMELS-US** (Newman, A. J. et al. 2015, HESS 19:209-223; Addor, N. et al. 2017, HESS 21:5293-5313): the basin set
  and boundaries, through Caravan.
- **ERA5-Land** (Muñoz-Sabater, J. et al. 2021, ESSD 13:4349-4383), Copernicus Climate Change Service (C3S): the
  forcing. Contains modified Copernicus Climate Change Service information; neither the European Commission
  nor ECMWF is responsible for any use of it.
- **HydroATLAS** (Linke, S. et al. 2019, Scientific Data 6:283), CC BY 4.0: the static attributes, through Caravan.
- **USGS NWIS** daily values: public domain (U.S. Government), through Caravan.

None of these is share-alike. The weights and code are MIT like the rest of the repository.
