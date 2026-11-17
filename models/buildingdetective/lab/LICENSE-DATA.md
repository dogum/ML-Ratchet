# Data licence: CC BY-SA 4.0 (Building Data Genome 2)

`index.html` (the original BuildingDetective page) embeds data derived from the Building Data Genome 2 dataset (BDG2):
Miller, C., et al. (2020), "The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy
Predictor III competition", Scientific Data 7, 368, https://doi.org/10.1038/s41597-020-00712-x; data at
https://github.com/buds-lab/building-data-genome-project-2. Licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/).

Changes: 29 BDG2 exam buildings, hourly 2017 meter series and site weather as cleaned by BuildingData (bad hours set to
NaN, timestamps fixed to UTC), quantised to 8 bits per hour for the page, with their BDG2 metadata (use, floor area,
year built, meters) as truth, plus summary numbers from the BDG2 exam and dev twin test. You may share and adapt the
embedded BDG2 data under the same licence, with this credit.

The page also embeds five HUE homes (Makonin, S. (2019), "HUE: The hourly usage of energy dataset for buildings in
British Columbia", Data in Brief 23, 103744; Harvard Dataverse doi:10.7910/DVN/N3HGRN; CC BY 4.0) and summary numbers
from EIA RECS 2020 and CBECS 2018 (public domain). This licence covers the data only; the code in this repository is
MIT.
