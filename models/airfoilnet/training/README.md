# AirfoilNet training scripts (as run, 22 Nov 2026)

The build ran from a flat folder: `src/` (these scripts), `data/` (UIUC downloads), `work/` (banks, runs),
`release/`, `web/`. Order:

1. `uiuc_lsat.py`, `dataset.py` -> data/units.json (82 tested airfoils); `make_splits.py` -> splits.json (frozen).
2. `make_shapes.py` -> work/shapes.npz (CST fits; tested airfoils removed).
3. `verify_fast.py`, `verify_physics.py` -> solver checks.
4. `simbank.py` via `run_bank.sh` / `chain_A.sh`: twin_dev, val, phys_exam, train banks (work/bank/*.jsonl).
5. `twin_dev.py`, `calib_sim.py` (physics p1c), `dev_baselines.py`.
6. `train.py r1 ... r7` (snapshots work/snap_A.json, snap_B.json), `ratchet.py` -> ratchet.json, ratchet_val2.json.
7. `calibrate.py` (widen-only on dev) -> final weights; `exam.py` (once) -> exam.json.
8. `make_examples.py`, `make_release.py`; the lab page: `web_webdata.py` + `web_template.html`.

Solver: `mfoil.py` (K. Fidkowski, MIT; one bug fix at line 2057), `mfoil_fast.py` (numba port + robustness fixes),
`xsolver.py` (sweeps). NeuralFoil baseline: `neuralfoil_np.py` (needs the nn-*.npz files from the neuralfoil 0.3.3
wheel; set NEURALFOIL_WEIGHTS).
