# FireDetective examples

Four exam fires, as shipped with the v1.0 release (`demo_data/` in the zip): a fixed seeded draw (seed 7) within
strata, made after the exam by `training/make_demo.py`, never chosen by score: two forest fires with MTBS severity from
the main period 2017-21, one non-forest fire from 2017-21, one recent fire from 2022-23.

| File | Fire | Year | Region | Stratum | Half-day steps |
|---|---|---|---|---|---|
| `NM3624410503720220417.npz` | Cooks Peak | 2022 | SW | recent | 12 |
| `OK3496309538920170202.npz` | Powder Mill | 2017 | SP | forest, main | 6 |
| `UT4071511223920180915.npz` | Black | 2018 | GB | non-forest, main | 6 |
| `WA4757812313020180804.npz` | Maple | 2018 | NW | forest, main | 12 |

Each file holds one fire's raw observations, cropped to the fire on its 180 m EPSG:5070 grid (row 0 = north):

- `masks`: cumulative burned area at each FEDS half-day step, bit-packed along the last axis (`shape` gives the
  unpacked size; `demo.load_fire` unpacks it);
- `fuel`, `cc`, `ch`, `cbh`, `cbd`, `slope`, `elev`: LANDFIRE layers at 180 m (FBFM40 code, %, m, m, kg/m³, degrees, m);
- `meta` (JSON): name, year, period, region, stratum, step times, day/night flags, the VIIRS detections of each step
  (grid row and column, FRP in MW), the gridMET day for each step, day of year, grid convergence, the truth the exam
  graded (MTBS high-severity share, gridMET wind, Globe-LFMC; none of the four has an LFMC sample) and the classical
  values (Van Wagner crown-fire potential, ERC max, the raw Rothermel back-calculated wind).

The pipeline from these files reproduces the four fires' rows in `training/results/exam.json` to the shipped digits
(`training/import_check.py`).

## Sources and licences

- Burned-area masks and VIIRS pixels: excerpts of **FEDS-MTBS** (Zenodo 20187963; FEDS method of Chen et al. 2022,
  Scientific Data), CC BY 4.0, with VIIRS S-NPP 375 m active fire data (NASA LANCE / FIRMS).
- Fuels, canopy and terrain: **LANDFIRE** (LANDFIRE Product Service), public domain.
- Weather: **gridMET** (Abatzoglou 2013), public domain.
- Severity truth: **MTBS** (USGS/USFS) via LANDFIRE Annual Disturbance, public domain.
- Live fuel moisture truth (none in these four): **Globe-LFMC 2.0** (Yebra et al. 2024), CC0.

None of these is share-alike, so the files stay with the repository's MIT licence, with the credits above.
