#!/bin/sh
# portable paths: code in training/, outputs in models/energynet/work (or $ML_RATCHET_WORK/energynet)
SRC="$(cd "$(dirname "$0")" && pwd)"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/energynet}"; WORK="${WORK:-$SRC/../work}"
VEH="${ML_RATCHET_VEHICLE:-$(cd "$SRC/../../../shared/vehicle" && pwd)}"
cd $WORK
while [ $(ls data/bank_train_v4/chunk_*.npz 2>/dev/null | wc -l) -lt 48 ]; do sleep 60; done
python3 $SRC/train.py r5_phev --version v3 --mode corr --bank train_v4 > logs/train_r5.log 2>&1
python3 $SRC/ratchet.py > logs/ratchet2.log 2>&1
for s in 1 2 3 4; do python3 $SRC/train.py r6_s$s --version v3 --mode corr --bank train_v4 --seed $s > logs/train_r6_$s.log 2>&1; done
python3 $SRC/ratchet.py > logs/ratchet3.log 2>&1
