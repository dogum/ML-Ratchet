# Weights licence: CC BY-NC-SA 4.0 (adapted from KW51 railway bridge modal data)

`bridgesense_weights.npz` in this folder (and its fp16 browser export at
`site/public/instruments/bridgesense/weights.bin`) is licensed **CC BY-NC-SA 4.0**
(https://creativecommons.org/licenses/by-nc-sa/4.0/), not MIT like the code.

BridgeSense learned only from 2-D finite-element thermo-mechanical simulations, with zero real
labelled bridge examples in training. However, half of its released training bank (`train3`)
injects real healthy autumn OMA residual jitter and missing-mode dropout windows extracted from
the KW51 railway bridge dataset, and its environmental error-bar widening factor was fitted on
the KW51 dev split. Following the cautious reading established for LoadCast and BuildingDetective
in `DATA_LICENSES.md`, the resulting weights are treated as an adaptation and carry the same
CC BY-NC-SA 4.0 terms.

Source: Maes, K. and Lombaert, G. (2020), "Monitoring railway bridge KW51 before, during, and
after retrofitting", *Journal of Bridge Engineering* 25(6), 04720001; Zenodo record
https://zenodo.org/records/3745914 (CC BY-NC-SA 4.0).

The code in this folder (`bridgesense.py`, `demo.py`, `training/`, `web/`) is MIT. The bundled
real examples (`examples/` and `site/public/instruments/bridgesense/examples.json`) come solely
from CC BY 4.0 sources (Concrete Box-Girder Bridge, Zenodo 20522191; Hell Bridge Test Arena,
Zenodo 14632942) plus NASA POWER open weather data, with zero KW51 rows redistributed.
