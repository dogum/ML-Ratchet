# Weights licence: CC BY-SA 4.0 (adapted from Building Data Genome 2)

`loadcast_weights.npz` in this folder is licensed CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/),
not MIT like the code. LoadCast learned only from BuildingSim simulations, but those simulations were shaped by data
from the Building Data Genome 2 dataset (BDG2): real 2017 residual series from 85 BDG2 dev building-meters were added
to the simulated meters as noise, and the dev calibration and widening factors were fitted on BDG2 dev buildings.
Because real BDG2-derived series went into the training samples, the weights are treated as an adaptation and
shared alike. Decided 17 Nov 2026, on the cautious reading.

Source: Miller, C., et al. (2020), "The Building Data Genome Project 2, energy meter data from the ASHRAE Great
Energy Predictor III competition", Scientific Data 7, 368, https://doi.org/10.1038/s41597-020-00712-x; data at
https://github.com/buds-lab/building-data-genome-project-2 (CC BY-SA 4.0). You may share and adapt the weights under
the same licence, with this credit. The code in this folder (`loadcast.py`, `pipeline.py`, `demo.py`, `training/`,
`web/`) is MIT.
