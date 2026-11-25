# EnergyNet v2.1 as released (reference arm)

The incumbent. `models/energynet` v2.1 (15 Nov 2026), byte copies of its five networks (`weights/r15_s*.npz`), its numpy
forward pass (`model_np.py`) and its error-bar factors (`sigma_factors.json`). Not retrained.

- Inputs: the table track's 151 features (they ARE EnergyNet's inputs: `features.py`, assembled by `training/dataset.py`).
- Network: 5 MLPs (151 -> 256 -> 256 -> 256 -> 128 -> 7, tanh-GELU), mean in log space; predicts log(E / E_phys), log sigma
  and 5 energy components.
- Trained by EnergyNet on 86,400 simulated trips (three banks of the train2 fleet, seeds 101-103, physics v2, version h).
  The bake-off bank's train split is one of those three banks (seed 101), so this reference saw 3x the bake-off's data.
- Bands: log-normal, sigma = sqrt((k x SIM_arch x s)^2 + tau^2) with EnergyNet's factors (simulation factor per
  architecture, then the dev factor per output group, which is read from the inputs: see predict.py).
- Checks (harness): dev terms equal ratchet2.json's r17 record (max |diff| 5.6e-16); val bank MdAPE 0.0746 = val2[h].
