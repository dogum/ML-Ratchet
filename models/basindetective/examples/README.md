# BasinDetective examples

Four CAMELS-US exam basins, as shipped with the v1.0 release: a fixed seeded draw (seed 7), one per stratum
(snow/humid, rain/humid, rain/arid by the gauged decade's ERA5-Land snow fraction and aridity, plus one regulated
basin), never chosen by score (`training/release_examples.py`).

- `<id>_cal.csv`: 1999-10-01 .. 2008-09-30, daily `P` (mm/day), `T_mean` (°C), `PET` (mm/day) and `Q_mm` (mm/day over
  the basin): what the network reads.
- `<id>_tgt.csv`: 1989-10-01 .. 1999-09-30, daily `P`, `T_mean`, `PET`: the target decade's forcing (no flow).
- `<id>_truth.json`: the observed change of the six signatures (from the basin's own 1989-99 vs 1999-2008 record),
  the calibrated GR4J's model-implied change, and HydroATLAS storage / annual flow.
- `expected.json`: the released model's outputs on these files (the demo's self-check).

## Sources and licences

- Forcing and streamflow are excerpts of **Caravan v1.6** (Kratzert, F. et al. 2023, *Caravan - A global community
  dataset for large-sample hydrology*, Scientific Data 10, 61; CC BY 4.0), which carries
  - **CAMELS-US** streamflow (Newman, A. J. et al. 2015, HESS 19, 209-223; Addor, N. et al. 2017, HESS 21, 5293-5313),
    originally from USGS NWIS daily values (public domain), and
  - **ERA5-Land** forcing (Muñoz-Sabater, J. et al. 2021, ESSD 13, 4349-4383). Contains modified Copernicus Climate
    Change Service information (2021); neither the European Commission nor ECMWF is responsible for any use of it.
- `reservoir_storage_over_annual_flow` comes from **HydroATLAS / BasinATLAS** (Linke, S. et al. 2019, Scientific Data
  6, 283; CC BY 4.0) with reservoir volumes from **GRanD** (Lehner, B. et al. 2011, Frontiers in Ecology and the
  Environment 9, 494-502; CC BY 4.0), via Caravan's attributes.
- The GR4J changes are WaterData's per-basin calibrated GR4J+snow (ML Ratchet, MIT).

None of these is share-alike, so the weights stay MIT like the rest of ML Ratchet. The exam's NWIS peak-code flag
(public domain) is not in these files. SNOTEL, ResOpsUS and CDEC are not used by BasinDetective.
