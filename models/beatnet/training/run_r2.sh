#!/bin/bash
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/beatnet}"; WORK="${WORK:-$SRC/../work}"
# r1 finished + v2 bank finished -> score r1, then train r2 (v2 physics bank, + AF output, warm start from r1)
cd $WORK; export OMP_NUM_THREADS=1
while kill -0 379 2>/dev/null || kill -0 380 2>/dev/null; do sleep 20; done
python3 $SRC/ratchet.py $WORK/ckpt/r1.pt "r1: textbook simulator (physics v1), window only" > ratchet_r1.log 2>&1
python3 $SRC/train.py --name r2 --bank "$WORK/bank/v2_*.npz" --init $WORK/ckpt/r1.pt --nout 7 --steps 3000 --lr 1e-3 >> train_r2.log 2>&1
