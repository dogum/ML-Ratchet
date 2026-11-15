#!/bin/sh
# r14 members 2-4 in parallel (the sequential runner was too slow while the machine is shared)
SRC="$(cd "$(dirname "$0")" && pwd)"; PY="$SRC/../../../.venv/Scripts/python.exe"; cd "$SRC"
A="--version e --mode corr --bank train2d+train2bd+train2cd --era"
until [ -f ../work/ckpt/r14_s1.npz ]; do sleep 10; done
$PY train.py r14_s2 --seed 2 $A > ../work/logs/train_r14_s2.log 2>&1 &
until grep -q R15_BANKS_DONE ../work/logs/bank_r15.log; do sleep 10; done
$PY train.py r14_s3 --seed 3 $A > ../work/logs/train_r14_s3.log 2>&1 &
$PY train.py r14_s4 --seed 4 $A > ../work/logs/train_r14_s4.log 2>&1 &
wait
echo ENS_DONE r14 >> ../work/logs/ens_r14.log
