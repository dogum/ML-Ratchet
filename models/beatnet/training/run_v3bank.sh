#!/bin/bash
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/beatnet}"; WORK="${WORK:-$SRC/../work}"
# waits for the v2 bank, then builds the v3 frozen validation set and the v3 training bank (resumable)
cd $WORK; export OMP_NUM_THREADS=1

python3 $SRC/simbank.py --name v3val --seed0 900000 --patients 400 --chunk 200 --phys 3 >> bank_v3.log 2>&1
python3 $SRC/simbank.py --name v3 --seed0 0 --patients 2500 --phys 3 >> bank_v3.log 2>&1
