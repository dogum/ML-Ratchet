# Weights licence: CC BY-SA 4.0 (adapted from Building Data Genome 2)

The files in this folder (`annual.npz`, `monthly.npz`, `hourly.npz`, `calibration.json`) are licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/), not MIT like the code. BuildingDetective learned only from
BuildingSim simulations, but its training banks were calibrated on Building Data Genome 2 (BDG2) dev buildings
(campus district factors, base loads, intensities) and its error-bar factors were fitted partly on the 52 BDG2 dev
buildings. Because BDG2-derived values shaped the training samples, the weights are treated as an adaptation and
shared alike. Decided 17 Nov 2026, on the cautious reading.

Source: Miller, C., et al. (2020), "The Building Data Genome Project 2, energy meter data from the ASHRAE Great
Energy Predictor III competition", Scientific Data 7, 368, https://doi.org/10.1038/s41597-020-00712-x; data at
https://github.com/buds-lab/building-data-genome-project-2 (CC BY-SA 4.0). You may share and adapt them under the
same licence, with this credit. The code (`buildingdetective.py`, `buildingdetective/`, `features.py`, `pipeline.py`,
`demo.py`, `training/`, `web/`) is MIT.
