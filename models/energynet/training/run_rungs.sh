#!/bin/sh
# portable paths: code in training/, outputs in models/energynet/work (or $ML_RATCHET_WORK/energynet)
SRC="$(cd "$(dirname "$0")" && pwd)"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/energynet}"; WORK="${WORK:-$SRC/../work}"
VEH="${ML_RATCHET_VEHICLE:-$(cd "$SRC/../../../shared/vehicle" && pwd)}"
cd $WORK
while [ $(ls data/bank_train/chunk_*.npz 2>/dev/null | wc -l) -lt 48 ]; do sleep 60; done
python3 $SRC/train.py r1_abs --version v1 --mode abs > logs/train_r1.log 2>&1
python3 $SRC/train.py r2_corr --version v1 --mode corr > logs/train_r2.log 2>&1
python3 $SRC/train.py r3_maf --version v2 --mode corr > logs/train_r3.log 2>&1
python3 $SRC/train.py r4_rw --version v3 --mode corr > logs/train_r4.log 2>&1
while [ $(ls data/bank_val/chunk_*.npz 2>/dev/null | wc -l) -lt 8 ]; do sleep 60; done
python3 $SRC/ratchet.py > logs/ratchet.log 2>&1
