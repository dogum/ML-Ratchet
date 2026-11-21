# FireSize exam examples

Eight real FPA-FOD exam ignitions, two per size group (under 1 acre, 1-100, 100-1,000, 1,000 acres and up), drawn
after the single exam run by `training/src/build_release.py` with seed 20261126 (`SEED + 7`) among the exam
ignitions that have a discovery time. They were never used for training or for choosing anything. Each group's two
fires are a seeded draw, not picked by result.

| FOD_ID | Fire | Date | State | Real size | FPA class | Fuel at the point |
|---|---|---|---|---|---|---|
| 400558692 | Peterson Ranch Fire | 2019-04-07 | MT | 0.1 acres | A | agriculture |
| 201639524 | Fy2012-Wilcox-076 | 2012-06-05 | GA | 0.6 acres | B | timber litter |
| 400051766 | 151 | 2016-02-06 | KS | 2.0 acres | B | timber litter |
| 201509321 | Easy | 2012-06-22 | SD | 2.5 acres | B | grass |
| 300020400 | Weaver | 2014-08-17 | NM | 276 acres | D | grass |
| 300207498 | Green | 2015-03-31 | OH | 109 acres | D | timber litter |
| 400025466 | Freedom Hills | 2016-04-03 | OK | 3,900 acres | F | timber litter |
| 400028954 | North Moore | 2016-06-09 | UT | 1,414 acres | F | shrub |

- `demo_examples.csv` (as shipped in the release zip): the record (`FOD_ID`, `fire_name`, `year`, `state`,
  `real_acres`), the exam run's own P(escape) (`exam_p_escape`, which `demo.py` checks to 1e-4), and the 75 network
  inputs exactly as the exam built them (float32 values). Column meanings are in `MODEL_CARD.md`.
- `patches.json` (added on import, 21 Nov 2026): for each ignition the 13 x 13 patch of LANDFIRE 2016 FBFM40 fuel
  codes around it (250 m cells, row 0 to the north, the ignition at row 6, column 6), and the discovery date, NWCG
  general cause, fuel class at the point and FPA-FOD size class. Copied from the data block of the shipped page
  (`lab/index.html`, written by `build_release.py` from the same 250 m grid). The live instrument uses the patches
  to recompute the Rothermel spread-rate inputs when the weather changes. Check on import: recomputing `ros_pt`,
  `ros5`, `ros13` and `ros_pt_m1` from these codes and the CSV weather with `firesize.make_features`' code
  reproduces the CSV columns to 1.1e-7, and the 13 x 13 fuel-class fractions match `f13_*` to 3e-8.

## Sources and licences
- **FPA-FOD**, 6th edition: Short, K. C. (2022), *Spatial wildfire occurrence data for the United States,
  1992-2020*, USDA Forest Service Research Data Archive, https://doi.org/10.2737/RDS-2013-0009.6. U.S. public data;
  cite Short (2022). The records and real sizes.
- **LANDFIRE** 2016 Remap (FBFM40, canopy cover, canopy height, slope), https://landfire.gov/, via the LFPS service.
  U.S. public domain (USGS and USDA Forest Service). The landscape summaries and fuel patches.
- **gridMET**: Abatzoglou, J. T. (2013), *Int. J. Climatol.* 33, 121-131, https://www.climatologylab.org/gridmet.html.
  U.S. public domain. The weather columns of the discovery day and the day before.

None of these is share-alike. The weights and code are MIT like the rest of the repository.
