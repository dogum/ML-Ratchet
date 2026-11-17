# Data licence: CC BY-SA 4.0 (Building Data Genome 2)

The files in this folder are derived from the Building Data Genome 2 dataset (BDG2): Miller, C., et al. (2020),
"The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition",
Scientific Data 7, 368, https://doi.org/10.1038/s41597-020-00712-x; data at
https://github.com/buds-lab/building-data-genome-project-2. Licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Changes: cleaned by BuildingData (ML Ratchet's building data
loader: timestamps fixed to UTC, quality flags), subset to three exam building-meters and their sites; meter readings
as kWh per hour at 0.001 kWh; site weather (T_out, dewpoint, GHI, wind_speed) at 0.01, with gaps filled from Iowa
Environmental Mesonet ASOS and irradiance from NASA POWER by BuildingData; metadata (use, floor area, meters present,
site latitude and time zone) and BuildingData's 2017 baseline scores in `examples.json`. You may share and adapt them
under the same licence, with this credit. This licence covers the data files only; the code in this repository is MIT.

Files: `Peacock_lodging_Matthew_electricity.csv.gz`, `Bull_education_Reina_chilledwater.csv.gz`,
`Cockatoo_education_Minh_steam.csv.gz` (meters, 2016-2017), `weather_Peacock.csv.gz`, `weather_Bull.csv.gz`,
`weather_Cockatoo.csv.gz` (site weather), `examples.json` (metadata and baseline scores).
