# Data licence: CC BY-SA 4.0 (Building Data Genome 2)

`index.html` in this folder embeds data derived from the Building Data Genome 2 dataset (BDG2): Miller, C., et al.
(2020), "The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition",
Scientific Data 7, 368, https://doi.org/10.1038/s41597-020-00712-x; data at
https://github.com/buds-lab/building-data-genome-project-2. Licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Changes: cleaned by BuildingData (timestamps fixed to UTC,
quality flags), subset to 16 exam building-meters at 8 sites (a fixed random draw, seed 20261117); local 2017 meter
readings quantised to int16; site weather from 2 Dec 2016 to the end of 2017 at 0.01 °C / 0.01 m/s / 1 W/m² (gaps
filled from Iowa Environmental Mesonet ASOS, irradiance from NASA POWER, by BuildingData); building metadata; 2016
history encodings (TOWT and change-point coefficients, time-of-week profiles) and change-point baselines fitted on
the 2016 meters; exam and calibration summaries. You may share and adapt them under the same licence, with this
credit. This licence covers the data only; the page's code, like the code in this repository, is MIT.
