# FlowCast v1.0 as released (reference arm)

**What it is.** The incumbent: FlowCast v1.0 from the repo (`models/flowcast`), copied byte for byte. One LSTM
layer (64 units, 24,710 weights) over 11 daily forcing features plus 18 static features; a linear head gives the
mean flow and five monotone quantiles in units of the basin's mean precipitation. Trained only on WaterSim
simulations (physics p1, the same recipe as the bake-off bank). Not retrained here.

**Protocols.**
- Ungauged: `flowcast_weights.npz` (rung r1b); bands widened on dev by 1.00 (50 %) and 1.08 (90 %), as released.
- Gauged: `flowcast_nowcast_weights.npz`, the released nowcast (r1b fine-tuned to also read observed flow `lead`
  days earlier). Raw pinball bands (no dev widening was fitted for this mode).

**Inputs used.** Forcing (8 columns), the 18-value static vector of the batch. Nowcast: `qobs` lagged by the lead.

**Adaptation.** `predict.py` calls `FlowCast.predict_arrays` on the batch's forcing rows and static vector; the
harness builds both exactly as `flowcast.predict()` does (checked: identical output on a dev basin, see
harness STATUS.md).

**Files (sha256).** flowcast.py `6d98b1bd…`, flowcast_weights.npz `bbf1830f…`, flowcast_nowcast_weights.npz
`788db343…` (identical to the repo copies).

**Compute.** None in the bake-off. Its own build: about 15 min of training on one shared core plus the bank.

**Known results (FlowCast's own exam, already opened):** ungauged main NSE 0.341 (U-dev 0.372), recent 0.378;
nowcast 0.569 / 0.474 / 0.426 at 1 / 3 / 7 days. The bake-off re-scores it with the harness.
