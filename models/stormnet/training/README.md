# Reproduce StormNet

Run from this directory with the ML Ratchet checkout installed in editable mode. NumPy runs the network. SciPy fits the physical opponents and independently checks the simulator. Use one BLAS thread. The reference build used Python 3.13.5, NumPy 2.5.5 and SciPy 1.18.1. All generated training and exam files go in `../work/` or `$ML_RATCHET_WORK/stormnet/`.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python physics_check.py
python check_index.py
python split.py
python baselines.py
python simbank.py
python train.py
python ratchet.py
python twin.py
python stability.py
python exam.py
python onsets.py
```

The split command regenerates the frozen storm and quiet blocks and checks them against the release. The baseline command fits both physical opponents, records source hashes and estimates solar-wind relaxation from training-era wind alone. Training takes the targets in the two simulation banks. Reports go under `work/results/`. The ratchet writes the proposed weights and ladder to `work/release/`. Compare these with `../stormnet_weights.npz` and `../ratchet.yaml`. The seed check is diagnostic. The exam scores the shipped weights and frozen calibration and writes to `work/exam/`. Compare its `exam.json`, `sources_primary.json` and example arrays with the release. The reserve is excluded by the loader. The final onset diagnostic counts first crossings on the original dev blocks.

`sources_dev.json` and `sources_primary.json` pin the exact provider bytes. A changed revision is reported before scoring. `banks.json` pins the array content of each training and validation bank. The raw data can be removed from the work folder and fetched again from the documented NASA endpoint.

From the model directory run `python web/export_instrument.py` to rebuild the site packs and Python references. With the site's npm dependencies installed, `node web/build_lab.mjs` bundles the offline lab. From the site directory run `npm test` for browser parity. The optional dev twin figure uses Matplotlib (`python training/plot_twins.py`).
