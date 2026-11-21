#!/bin/bash
# sequential bank chain (one core): gate -> val -> train   (as run in the sandbox; paths now from the repository
# layout, see _paths.py: logs and banks go under WORK)
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${ML_RATCHET_WORK:+$ML_RATCHET_WORK/firepilot}; WORK=${WORK:-$HERE/../work}
mkdir -p "$WORK/logs"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$WORK/.numba
cd "$HERE"
python3 bank.py gate > "$WORK/logs/bank_gate.log" 2>&1
python3 bank.py val 0 160 > "$WORK/logs/bank_val.log" 2>&1
python3 bank.py train 0 600 > "$WORK/logs/bank_train.log" 2>&1
