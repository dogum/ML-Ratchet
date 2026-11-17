# Data licence: CC BY-SA 4.0 (Building Data Genome 2)

The files in this folder (examples.bin, examples.json, reference.json) are derived from the Building Data Genome 2
dataset (BDG2): Miller, C., et al. (2020), "The Building Data Genome Project 2, energy meter data from the ASHRAE Great
Energy Predictor III competition", Scientific Data 7, 368, https://doi.org/10.1038/s41597-020-00712-x; data at
https://github.com/buds-lab/building-data-genome-project-2. Licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Changes: cleaned by BuildingData (timestamps fixed to UTC, quality
flags), subset to 16 exam building-meters at 8 sites; local 2017 meter readings (quantised to int16 steps); site
weather from 2 Dec 2016 to the end of 2017 at 0.01 degC / 0.01 m/s / 1 W/m2 (gaps filled from Iowa Environmental
Mesonet ASOS, irradiance from NASA POWER, by BuildingData); building metadata; 2016 history encodings and
change-point baselines fitted on the 2016 meters; LoadCast outputs on these inputs. You may share and adapt them under
the same licence, with this credit. This licence covers the data files only; the code in this repository is MIT.
The network weights (weights.bin, weights.json) were trained on simulations only.
