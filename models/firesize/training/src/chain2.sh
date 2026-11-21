#!/bin/bash
# train banks: v1 first 4800 ignitions, v0 first 4800, then v1 all
cd "$(dirname "$0")"                       # training/src (was the sandbox's fire/firesize/src)
WORK=${ML_RATCHET_WORK:+$ML_RATCHET_WORK/firesize}; WORK=${WORK:-$(cd ../.. && pwd)/work}   # as training/_paths.py
mkdir -p "$WORK/logs"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR="$WORK/.numba"
while ! grep -q DONE "$WORK"/logs/bank_twin_v0.log 2>/dev/null; do sleep 30; done
nice -n 5 python3 simbank.py train 1 v1 4800 > "$WORK"/logs/bank_train_v1a.log 2>&1
nice -n 5 python3 simbank.py train 1 v0 4800 > "$WORK"/logs/bank_train_v0.log 2>&1
nice -n 5 python3 simbank.py train 1 v1 > "$WORK"/logs/bank_train_v1.log 2>&1
echo CHAIN2 DONE > "$WORK"/logs/chain2.done
