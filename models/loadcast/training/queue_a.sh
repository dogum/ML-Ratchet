#!/bin/bash
cd "$(dirname "$0")"   # was the sandbox workspace src/; outputs go to ../work or $ML_RATCHET_WORK/loadcast
W="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/loadcast}"; W="${W:-../work}"; mkdir -p "$W/tmp"
export OMP_NUM_THREADS=1 LC_MAXPARTS=2
run() { while [ $(ls $W/data/bank_$2/part_*.npz 2>/dev/null | wc -l) -lt $(( $3 / 100 )) ]; do nice -n 5 python3 bank.py $1 $2 $3 $4 >> $W/tmp/bank_$2.log 2>&1; sleep 2; done; }
run v1 v1_train 1600 2
