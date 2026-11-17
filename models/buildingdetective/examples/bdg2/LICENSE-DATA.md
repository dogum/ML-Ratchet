# Data licence: CC BY-SA 4.0 (Building Data Genome 2)

The files in this folder (`bdg2_hourly.npz`, `bdg2_hourly.json`) are derived from the Building Data Genome 2 dataset
(BDG2): Miller, C., et al. (2020), "The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy
Predictor III competition", Scientific Data 7, 368, https://doi.org/10.1038/s41597-020-00712-x; data at
https://github.com/buds-lab/building-data-genome-project-2. Licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/).

Changes: three exam buildings (Lamb_education_Bert, Eagle_office_Remedios, Peacock_lodging_Sergio), drawn by a fixed
seed within use types; meters and site weather cleaned by BuildingData (bad hours set to NaN, timestamps fixed to UTC,
Eagle thermal meters divided by 1000); cut to the local calendar year 2017; steam and hot water summed into one
district-heat channel; stored as float32 kW with UTC times. The JSON holds BDG2 metadata (primary use, floor area,
year built, meter list, site latitude) used as truth.

You may share and adapt them under the same licence, with this credit. This licence covers the data files only; the
code in this repository is MIT.
