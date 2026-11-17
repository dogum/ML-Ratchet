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
python exam.py                       # run the held-out real exam ONCE -> exam_results.json
python ../web/export_instrument.py   # export browser packs + 24 parity cases
python build_lab.py                  # build ../lab/index.html
```
