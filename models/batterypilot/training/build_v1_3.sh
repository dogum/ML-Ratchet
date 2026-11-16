#!/bin/sh
# v1.3 (GridSim physics 1.2): the banks and frozen set, with the v4 seeds, in 3 parallel parts.
#   sh build_v1_3.sh <k> [v6|v5]      (k = 0, 1, 2; run all three)   logs: WORK/logs/build_v1_3_<k>.log
# v6 (default): GridSim's default 1.2 fit ('ERCO': with the DA-shape term), WORK/data/calibration_v1_2.json.
# v5 (first build, kept as a record): the first 1.2 fit ('ERCO_first_fit'), calibration_v1_2_firstfit.json;
#    to rebuild it, set scar_fit='ERCO_first_fit' in the calibration file's 'calibrated' block and pass --calib.
# The sim-v1 bank is not rebuilt: it has no scarcity of any kind (scar_scale 0), so physics 1.2 leaves v4_train_v1
# bit-identical.
k=$1
S=${2:-v6}
PY=${PY:-../../../.venv/Scripts/python.exe}
export GRIDSIM_PHYSICS=1.2 OMP_NUM_THREADS=1
cd "$(dirname "$0")"
$PY simbank.py ${S}_frozen 24 7001 --part $k/3
$PY simbank.py ${S}_frozen 9 7002 --ba CISO --prefix c --part $k/3
$PY simbank.py ${S}_train_e 120 8101 --part $k/3
$PY simbank.py ${S}_train_c 45 8102 --ba CISO --part $k/3
