#!/bin/bash
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/beatnet}"; WORK="${WORK:-$SRC/../work}"
# Restart-safe driver for the remaining rungs (each step is skipped if already done). Bank v3 runs separately (run_v3bank.sh).
# Usage: run_rest.sh <bank_pid>
cd $WORK; export OMP_NUM_THREADS=1; BANKPID=$1
logged() { python3 -c "import json,sys; sys.exit(0 if any(r['ckpt']=='$1' for r in json.load(open('ratchet.json'))) else 1)"; }
valset() { python3 -c "import json; print(json.load(open('ratchet.json'))[0]['valset'])"; }
inc() { python3 -c "import json; print([r for r in json.load(open('ratchet.json')) if r['accepted']][-1]['ckpt'])"; }
# r2
[ -f ckpt/r2.pt ] || nice -n 5 python3 $SRC/train.py --name r2 --bank "$WORK/bank/v2_*.npz" --init $WORK/ckpt/r1.pt --nout 7 --steps 3000 --lr 1e-3 >> train_r2.log 2>&1
export BEAT_VAL=v2val
if ! logged r2.pt; then
  python3 $SRC/ratchet.py $WORK/ckpt/r2.pt "r2: dev-calibrated simulator (physics v2: APD +47 ms, weaker electrotonic coupling, PVC gain 1.35, less noise) + AF output" > ratchet_r2.log 2>&1
  python3 $SRC/ratchet.py --rescore >> ratchet_r2.log 2>&1
fi
# r3
INC=$(inc)
[ -f ckpt/r3.pt ] || nice -n 5 python3 $SRC/train.py --name r3 --cin 2 --bank "$WORK/bank/v2_*.npz" --init $WORK/ckpt/$INC --nout 7 --steps 3000 --lr 1e-3 >> train_r3.log 2>&1
logged r3.pt || python3 $SRC/ratchet.py $WORK/ckpt/r3.pt "r3: + second input channel = the record's median neighbour beat (template), physics v2" > ratchet_r3.log 2>&1
# physics v3
while kill -0 $BANKPID 2>/dev/null; do sleep 30; done
export BEAT_VAL=v3val
[ "$(valset)" = "v3val_*.npz" ] || python3 $SRC/ratchet.py --rescore > ratchet_rescore_v3.log 2>&1
INC=$(inc); CIN=$(python3 -c "import torch; print(torch.load('$WORK/ckpt/$INC', weights_only=False)['cin'])")
[ -f ckpt/r4.pt ] || nice -n 5 python3 $SRC/train.py --name r4 --cin $CIN --bank "$WORK/bank/v3_*.npz" --init $WORK/ckpt/$INC --nout 7 --steps 3000 --lr 1e-3 >> train_r4.log 2>&1
logged r4.pt || python3 $SRC/ratchet.py $WORK/ckpt/r4.pt "r4: physics v3 = v2 + dev noise-floor calibration (NSTDB -10 dB + quiet real dev background) + fusion visibility rule" > ratchet_r4.log 2>&1
echo ALLDONE >> ratchet_r4.log
