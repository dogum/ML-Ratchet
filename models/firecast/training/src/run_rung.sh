#!/bin/bash
# usage: run_rung.sh <rung>   (train -> evaluate sim+dev); resumable
cd "$(dirname "$0")"                                   # training/src (was the sandbox's fire/firecast/src)
WORK=$(python3 ../_paths.py)                            # models/firecast/work or $ML_RATCHET_WORK/firecast
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$WORK/.numba PYTHONWARNINGS=ignore
nice -n 5 python3 train.py $1 && nice -n 5 python3 evaluate.py $1 both > $WORK/logs/eval_$1.log 2>&1
