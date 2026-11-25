# Climatology (dumb baseline, reference arm)

Trip energy = median simulated energy per km of the vehicle's (architecture, class) on the bake-off's train bank x the
trip's distance. That is EnergyNet's own dumb baseline (the denominator of its skill score, `evaluate.dumb_rates2`).
Band: the 5/25/50/75/95 % quantiles of log(energy / (rate x distance)) for the same group on the train bank.
Inputs used: distance, architecture, class. Both tracks (identical numbers). Fitted by `work/fit.py`; zero real labels.
