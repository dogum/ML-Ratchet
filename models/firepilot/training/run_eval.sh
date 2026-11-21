#!/bin/bash
# usage: run_eval.sh <rung>  -> val then gate   (as run in the sandbox; paths now from the repository layout, see
# _paths.py: logs go under WORK)
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${ML_RATCHET_WORK:+$ML_RATCHET_WORK/firepilot}; WORK=${WORK:-$HERE/../work}
mkdir -p "$WORK/logs"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$WORK/.numba
cd "$HERE"
python3 eval_rung.py $1 val > "$WORK/logs/eval_val_$1.log" 2>&1
python3 eval_rung.py $1 gate > "$WORK/logs/eval_gate_$1.log" 2>&1
