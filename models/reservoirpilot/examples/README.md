# Examples: six real exam dams

`exam_dams.json` holds the scored decade of six dams from WaterData's ResOpsUS exam split, as shipped with the release
(`demo_data/exam_dams.json`, unchanged). They are a fixed random draw within main-use strata (two flood-control, two
irrigation, two hydropower dams), not picked by result; none is a dev dam.

Each entry: GRanD id, main use, first scored day, daily inflow (m³/s, 3,653 days, stored to 6 significant digits), the
calibration half's demand calendar D(doy) and inflow climatology (366 values each), Qmax (m³/s), capacity and start
storage (hm³), the observed release and storage (the record), and the exam's J for the pilot, SOP, WaterData's rule
curve, historical operations and perfect-foresight DP on that dam.

## Sources and credit
- **ResOpsUS v2**: J. Steyaert, L. Condon, S. Turner and N. Voisin, "ResOpsUS, a dataset of historical reservoir
  operations in the contiguous United States", Scientific Data 9, 34 (2022), doi:10.5281/zenodo.6612040. CC BY 4.0.
  The inflow, release and storage series and everything derived from them come from here.
- **GRanD** dam attributes (capacity, main use): B. Lehner et al., Frontiers in Ecology and the Environment 9, 494
  (2011), CC BY 4.0, read through ResOpsUS+CARS v1.0 (J. Casado-Rodriguez et al., doi:10.5281/zenodo.15978041).
- Processed by WaterData (Watershed Phase 1, `shared/water/waterdata`): task construction and baselines.

The recent exam's California reservoirs (CDEC, California Department of Water Resources) are not in these excerpts.
The training inflows came from WaterSim driven partly by Caravan / ERA5-Land forcing (Kratzert et al. 2023, CC BY 4.0;
Muñoz-Sabater et al. 2021, Copernicus licence); no forcing is stored here. None of these sources is share-alike.
