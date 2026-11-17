# Rebuilding WakeNet from scratch

All downloads, processed SCADA tables, simulation banks, and checkpoints are written to `../work/` (or `$ML_RATCHET_WORK/wakenet/`) via `_paths.py`.

```bash
cd models/wakenet/training
python physics_check.py              # independent checks against NREL FLORIS 4.7 -> physics_check.json
python fetch_data.py                 # download Kelmarsh & Penmanshiel Zenodo zips into work/raw/
python realdata.py                   # quality-controlled 10-min farm snapshots -> work/proc/
python split.py                      # verify disjoint split & write split.json
python noise.py                      # extract 2017-18 unwaked front-row backdrop -> work/proc/backdrop_pool.npz
python calib_dev.py                  # measure meandering & hill speedup on 2019 dev -> dev_calibration.json
python simbank.py dev_cache          # cache 2019 dev graphs
python simbank.py val3 4000 303 v3 val
python simbank.py train_v3 24000 3 v3 train
python train.py --name r6 --bank train_v3 --steps 3000 --seed 106
python ratchet.py judge r6 "v3 physics calibrated on dev twins" val3
python twin.py r4 r6                 # dev twin test -> twin_report.json
python export.py                     # write ../wakenet_weights.npz, parity check, draw ../examples/
python exam.py                       # exam 1 (opened once 16 Nov; re-scored 17 Nov) -> exam_results.json
python ../web/export_instrument.py   # export browser packs + 24 parity cases
python build_lab.py                  # build ../lab/index.html
```

Added on 17 Nov 2026 (see NOTEBOOK.md for the order things were decided in):

```bash
python fetch_data.py --exam2         # Kelmarsh 2022-24, Penmanshiel 2021-23
python exam2_split.py                # freeze exam 2 -> split_exam2.json (refuses to overwrite)
WAKENET_ANCHOR=tuned python simbank.py dev_cache
WAKENET_ANCHOR=tuned python simbank.py val3_tuned 4000 303 v3 val
WAKENET_ANCHOR=tuned python simbank.py train_v3_tuned 24000 3 v3 train
python v11_reference.py              # r6 reproduced on rebuilt banks; physics-alone reference rows
python train.py --name r9 --bank train_v3_tuned --steps 3000 --seed 106 --anchor tuned
python ratchet.py judge r9 "..." val3
python twin_residuals.py r9          # dev twin test: what the network adds vs what reality needed
python calib_dev_v4.py               # physics v4 dev grid (ka, ct_scale) -> dev_calibration_v4.json
WAKENET_ANCHOR=tuned_v4 python simbank.py dev_cache
python simbank.py val4 4000 404 v4 val                       # same farms in all three val4 banks
WAKENET_ANCHOR=tuned python simbank.py val4_tuned 4000 404 v4 val
WAKENET_ANCHOR=tuned_v4 python simbank.py val4_tuned_v4 4000 404 v4 val
WAKENET_ANCHOR=tuned_v4 python simbank.py train_v4_tuned_v4 24000 4 v4 train
python v4_reference.py               # r6 and r9 re-scored on val4; physics-alone rows
python train.py --name r10 --bank train_v4_tuned_v4 --steps 3000 --seed 106 --anchor tuned_v4
python calib_sigma.py ../wakenet_weights.npz   # widen-only 68% factors per wind-speed bin, fitted on dev
python exam.py exam2 ../wakenet_weights.npz    # exam 2 -> exam2_results.json
```
