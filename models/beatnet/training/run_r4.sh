#!/bin/bash
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/beatnet}"; WORK="${WORK:-$SRC/../work}"
# after r3 (run_r3.sh, pid 338) and the v3 banks (run_v3bank.sh, pid 288): score r3 on v2val; physics v2 -> v3, so re-score every
# rung on the new frozen set v3val and re-take every decision in order; then train r4 on the v3 bank (warm start from the incumbent) and score it.
cd $WORK; export OMP_NUM_THREADS=1
while kill -0 338 2>/dev/null; do sleep 20; done
BEAT_VAL=v2val python3 $SRC/ratchet.py $WORK/ckpt/r3.pt "r3: + second input channel = the record's median neighbour beat (template), physics v2" > ratchet_r3.log 2>&1
while kill -0 288 2>/dev/null; do sleep 20; done
export BEAT_VAL=v3val
python3 $SRC/ratchet.py --rescore > ratchet_rescore_v3.log 2>&1
INC=$(python3 -c "import json; print([r for r in json.load(open('ratchet.json')) if r['accepted']][-1]['ckpt'])")
CIN=$(python3 -c "import torch; print(torch.load('$WORK/ckpt/$INC', weights_only=False)['cin'])")
nice -n 5 python3 $SRC/train.py --name r4 --cin $CIN --bank "$WORK/bank/v3_*.npz" --init $WORK/ckpt/$INC --nout 7 --steps 3000 --lr 1e-3 >> train_r4.log 2>&1
python3 $SRC/ratchet.py $WORK/ckpt/r4.pt "r4: physics v3 = v2 + dev noise-floor calibration (NSTDB -10 dB + quiet real dev background) + fusion visibility rule" > ratchet_r4.log 2>&1
