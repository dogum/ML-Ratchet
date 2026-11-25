# Climatology (dumb ungauged baseline; harness test arm)

**What it is.** No learner. Every day gets the same flow: the long-term water balance of the forcing,
`mean = 1.198 x Fu(aridity) x p_mean`, where `p_mean` and `aridity` (PET / P) are the climate indices of the
batch's own forcing (static vector entries 11 and 13) and Fu's curve uses w = 2.6. The quantiles are fixed
multiples of `Fu x p_mean` (0.004, 0.13, 0.45, 1.17, 4.58).

**Trained on.** The simulated train bank only: `fit.py` pools daily flow / (Fu x p_mean) over 2,000 random runs.
No real flow is read.

**Why it is here.** It tests the harness end to end with the simplest possible arm, and it is the ungauged "guess
the average" row. A constant series has NSE <= 0 by construction (NSE = -bias^2 / variance), so it shows where
zero skill lies. It differs from WaterData's climatology reference, which is gauged (own day-of-year mean).

**Protocol.** Ungauged. **Inference.** numpy only, microseconds. **Compute.** ~5 s of fitting.
