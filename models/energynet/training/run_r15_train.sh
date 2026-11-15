#!/bin/sh
# r15 members (after r14's training ends): three at a time
SRC="$(cd "$(dirname "$0")" && pwd)"; PY="$SRC/../../../.venv/Scripts/python.exe"; cd "$SRC"
A="--version e --mode corr --bank train2di+train2bdi+train2cdi --era"
until grep -q ENS_DONE ../work/logs/ens_r14.log && grep -q R15_BANKS_DONE ../work/logs/bank_r15.log; do sleep 15; done
for s in 0 1 2; do $PY train.py r15_s$s --seed $s $A > ../work/logs/train_r15_s$s.log 2>&1 & done; wait
for s in 3 4; do $PY train.py r15_s$s --seed $s $A > ../work/logs/train_r15_s$s.log 2>&1 & done; wait
echo ENS_DONE r15 > ../work/logs/ens_r15.log
