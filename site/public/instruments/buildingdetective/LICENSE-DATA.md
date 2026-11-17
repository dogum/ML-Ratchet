# Data licence: CC BY-SA 4.0 (Building Data Genome 2) and CC BY 4.0 (HUE)

`examples.json` and `examples.bin` (BuildingDetective's live instrument) hold the hourly 2017 meters, site weather and
metadata of three exam buildings derived from the Building Data Genome 2 dataset (BDG2): Miller, C., et al. (2020),
"The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition",
Scientific Data 7, 368, https://doi.org/10.1038/s41597-020-00712-x; data at
https://github.com/buds-lab/building-data-genome-project-2. Licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Changes: cleaned by BuildingData (bad hours NaN, timestamps fixed
to UTC), local calendar year 2017, steam and hot water summed into one district-heat channel, float32, local time
fields added, a fixed noise series added for the noise control. You may share and adapt them under the same licence,
with this credit.

The same files hold one home from HUE (Makonin, S. (2019), Data in Brief 23, 103744; Harvard Dataverse
doi:10.7910/DVN/N3HGRN; CC BY 4.0) and eight respondents of EIA RECS 2020 and CBECS 2018 (public domain).
This licence covers the data files only; the code in this repository is MIT.
