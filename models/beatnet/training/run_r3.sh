#!/bin/bash
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/beatnet}"; WORK="${WORK:-$SRC/../work}"
# after r2: physics v1 -> v2, so score r2, then re-score every rung on the v2 frozen set and re-take decisions; then train r3 (+ template channel)
cd $WORK; export OMP_NUM_THREADS=1 BEAT_VAL=v2val
while kill -0 289 2>/dev/null; do sleep 20; done
python3 $SRC/ratchet.py $WORK/ckpt/r2.pt "r2: dev-calibrated simulator (physics v2: APD +47 ms, weaker electrotonic coupling, PVC gain 1.35, less noise) + AF output" > ratchet_r2.log 2>&1
python3 $SRC/ratchet.py --rescore >> ratchet_r2.log 2>&1
INC=$(python3 -c "import json; print([r for r in json.load(open('ratchet.json')) if r['accepted']][-1]['ckpt'])")
python3 $SRC/train.py --name r3 --cin 2 --bank "$WORK/bank/v2_*.npz" --init $WORK/ckpt/$INC --nout 7 --steps 3000 --lr 1e-3 >> train_r3.log 2>&1
