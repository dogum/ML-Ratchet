#!/bin/bash
# usage: train_rung.sh run bank noise "note" [train args: epochs width depth K key=val ...]
cd "$(dirname "$0")"   # was the sandbox workspace src/; outputs go to ../work or $ML_RATCHET_WORK/loadcast
W="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/loadcast}"; W="${W:-../work}"; mkdir -p "$W/tmp"
export OMP_NUM_THREADS=1
R=$1; B=$2; N=$3; NOTE=$4; shift 4
nice -n 5 python3 train.py $R cold $B $N "$@" >> $W/tmp/train_$R.log 2>&1
nice -n 5 python3 train.py $R warm $B $N "$@" >> $W/tmp/train_$R.log 2>&1
nice -n 5 python3 rung.py $R v1_val real "$NOTE" 2>&1 | grep -v Warn | grep -v "^  " >> $W/tmp/rung_$R.log
