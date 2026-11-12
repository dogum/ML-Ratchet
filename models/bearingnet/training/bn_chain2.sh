#!/bin/sh
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/bearingnet}"; WORK="${WORK:-$SRC/../work}"
# resume after container restart: finish r1, frozen val2 (v2), tr2 bank (v2)
cd $WORK
export OMP_NUM_THREADS=1
nice -n 5 python3 $SRC/bn_train.py --name r1 --banks tr1 --steps 5000 --bs 96 >> logs/train_r1.log 2>&1
nice -n 5 python3 $SRC/bn_simbank.py val2 4000 99 v2 > logs/bank_val2.log 2>&1
nice -n 5 python3 $SRC/bn_simbank.py tr2 40000 2 v2 > logs/bank_tr2.log 2>&1
