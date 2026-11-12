#!/bin/sh
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/bearingnet}"; WORK="${WORK:-$SRC/../work}"
cd $WORK
export OMP_NUM_THREADS=1
nice -n 5 python3 $SRC/bn_realbank.py exam > logs/realbank_exam.log 2>&1
