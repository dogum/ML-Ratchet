#!/bin/bash
# usage: chain_bank.sh <ver> <n_val> <n_train>
cd "$(dirname "$0")"                                   # training/src (was the sandbox's fire/firecast/src)
WORK=$(python3 ../_paths.py)                            # models/firecast/work or $ML_RATCHET_WORK/firecast
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$WORK/.numba PYTHONWARNINGS=ignore
nice -n 5 python3 simbank.py $1 val $2 && nice -n 5 python3 simbank.py $1 train $3
