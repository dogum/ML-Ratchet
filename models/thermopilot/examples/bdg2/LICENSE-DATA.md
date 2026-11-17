# Data licence: CC BY-SA 4.0 (Building Data Genome 2)

The file in this folder, `demo_episodes.npz`, is derived from the Building Data Genome 2 dataset (BDG2): Miller, C.,
et al. (2020), "The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III
competition", Scientific Data 7, 368, https://doi.org/10.1038/s41597-020-00712-x; data at
https://github.com/buds-lab/building-data-genome-project-2. Licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/).

Changes: three exam episodes (one month each, 2017) were simulated in BuildingSim under the hourly 2017 site weather of
three BDG2 sites (Cockatoo, Hog, Gator), cleaned by BuildingData (UTC, gaps interpolated). The file holds the
24 h outdoor-temperature and solar forecasts the controller saw (real weather plus simulated forecast error), the
simulated indoor temperature, HVAC output and electricity, and the price / comfort-band context. The third episode
(`Gator_public_Geraldine`) is a twin of a BDG2 building built from its BDG2 metadata (use, floor area). Stored as
float64. The other two are twins of RECS 2020 respondents (EIA, public domain); tariffs are from the Utility Rate
Database (OpenEI). You may share and adapt the file under the same licence, with this credit.

This licence covers the data file only; the code in this repository is MIT.
