#!/bin/sh
# bank_r0: 800 trips, the DP teacher acts (with exploration bursts), untimed DP (v1 recipe) on physics v2
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1
HP_WORKERS=2 $PY $SRC/gen_bank.py r0 800 11 teacher - 0 > logs/bank_r0.log 2>&1
echo DONE >> logs/bank_r0.log
