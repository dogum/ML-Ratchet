# FireCast exam examples

`demo_days.npz` holds three real exam fire-days, never used for training or for choosing anything. They are #0, #4
and #8 of a 12-day draw (`training/src/exam.py examples`, seed 20261119): exam growth fire-days (observed new
burn > 0) in three size strata by burned area at issue (small < 5 km², medium 5-50 km², large > 50 km²), four per
stratum, the first of each stratum kept for the demo. The other nine are in the offline lab (`lab/index.html`).

| Fire | MTBS Event_ID | Issue day | Fire-day | Stratum | Exam period | Burned at issue | Grid |
|---|---|---|---|---|---|---|---|
| Hwy 113 (LA) | LA3098109281120230824 | 2023-08-24 | 0 (first mapped) | small | recent | 0.15 km² | 163 × 145 |
| Coon (CA) | CA4176412395720150801 | 2015-08-10 | 8 | medium | early | 5.48 km² | 134 × 156 |
| Pole Creek (UT) | UT3980611166020180906 | 2018-09-17 | 7 | large | main | 357.6 km² | 559 × 531 |

Per day `d<j>_` (90 m cells, north up, rows north to south; FireData's EPSG:5070 fire grid, the final MTBS
perimeter + 30 %):
- `shape`; `now`, `nxt`, `prev`, `fl`: packed bit grids (`np.unpackbits(...)[:H*W]`): burned at the issue
  afternoon overpass (FEDS), burned by the next afternoon overpass (the truth, used only to score), burned 24 h
  earlier (absent on a first day) and the FEDS active fire line (absent when FEDS has none);
- `pix_pm`, `pix_am`: VIIRS S-NPP detections at the issue overpass and the night overpass 12 h before, rows
  `row`, `col` (fractional grid position of the pixel centre), `ps`, `pt` (pixel size along scan and track, km),
  `frp` (MW);
- `wx`: the 16 gridMET features `firecast.feature_vector(day0, day1, grid_convergence)` of the issue day and the
  next day;
- `meta` (JSON): name, date, fire-day, period, stratum, the first-day and previous-day flags, the burned area and the
  observed new area, and the exam run's own scores of FireCast and of the fire-line baseline on the full grid.

`demo.py` runs the numpy model on them and scores it exactly like the exam (it reproduces the exam run's AP and F1).
The site instrument (`web/export_instrument.py`) uses the same file.

## Sources and licences
- **FEDS-MTBS** fire progression (Zenodo 20187963; FEDS method: Chen, Y. et al. 2022, Scientific Data), CC BY 4.0: the
  burned-area, fire-line and next-day grids.
- **VIIRS S-NPP 375 m active fire** (NASA LANCE / FIRMS), NASA open data, as shipped inside FEDS-MTBS: the
  detections.
- **MTBS** burned-area boundaries (USGS / USDA Forest Service), public domain: the fire set and the grid extent.
- **gridMET** (Abatzoglou, J. T. 2013, Int. J. Climatology 33:121-131), public domain: the weather features.

None of these is share-alike. The weights and code are MIT like the rest of the repository.
