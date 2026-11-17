# Data licence: CC BY-SA 4.0 (Building Data Genome 2)

The data files in this folder (`examples.json`, `examples.bin`, `reference.json`) are derived from the Building Data
Genome 2 dataset (BDG2): Miller, C., et al. (2020), "The Building Data Genome Project 2, energy meter data from the
ASHRAE Great Energy Predictor III competition", Scientific Data 7, 368, https://doi.org/10.1038/s41597-020-00712-x;
data at https://github.com/buds-lab/building-data-genome-project-2. Licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Changes: the 2017 hourly site weather of five BDG2 sites (Hog,
Bobcat, Bear, Bull, Fox) was cleaned by BuildingData (UTC, gaps interpolated), cut to one winter and one summer week,
run through BuildingSim with twins of three RECS 2020 respondents (public domain) and URDB / ERCOT tariffs, and the
outdoor temperature, solar, the house maps and the simulated runs rounded and stored as float32. You may share and
adapt them under the same licence, with this credit. This licence covers the data files only; the code in this
repository is MIT. The network weights (`weights.*`) are not BDG2 data.
