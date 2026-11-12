#!/bin/bash
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/beatnet}"; WORK="${WORK:-$SRC/../work}"
# physics v4 (dev-calibrated ectopic coupling): frozen set v4val + 6-chunk training bank; then (after run_rest.sh finishes)
# rescore every rung on v4val, train r5 from the incumbent on the v4 bank, score it. Restart-safe.
cd $WORK; export OMP_NUM_THREADS=1; WAIT=$1
nice -n 5 python3 $SRC/simbank.py --name v4val --seed0 900000 --patients 400 --chunk 200 --phys 4 >> bank_v4.log 2>&1
nice -n 5 python3 $SRC/simbank.py --name v4 --seed0 0 --patients 1500 --phys 4 >> bank_v4.log 2>&1
python3 $SRC/calib_coupling.py "$WORK/bank/v4val_*.npz" v4 >> bank_v4.log 2>&1
while kill -0 $WAIT 2>/dev/null; do sleep 30; done
export BEAT_VAL=v4val
logged() { python3 -c "import json,sys; sys.exit(0 if any(r['ckpt']=='$1' for r in json.load(open('ratchet.json'))) else 1)"; }
valset() { python3 -c "import json; print(json.load(open('ratchet.json'))[0]['valset'])"; }
[ "$(valset)" = "v4val_*.npz" ] || python3 $SRC/ratchet.py --rescore > ratchet_rescore_v4.log 2>&1
INC=$(python3 -c "import json; print([r for r in json.load(open('ratchet.json')) if r['accepted']][-1]['ckpt'])")
CIN=$(python3 -c "import torch; print(torch.load('$WORK/ckpt/$INC', weights_only=False)['cin'])")
[ -f ckpt/r5.pt ] || nice -n 5 python3 $SRC/train.py --name r5 --cin $CIN --bank "$WORK/bank/v4_*.npz" --init $WORK/ckpt/$INC --nout 7 --steps 3000 --lr 1e-3 >> train_r5.log 2>&1
logged r5.pt || python3 $SRC/ratchet.py $WORK/ckpt/r5.pt "r5: physics v4 = v3 + dev-calibrated ectopic coupling (S beats at 0.55 of local RR on dev, 0.78 in v3 sims)" > ratchet_r5.log 2>&1
echo ALLDONE >> ratchet_r5.log
