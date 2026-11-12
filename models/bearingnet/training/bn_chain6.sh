#!/bin/sh
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/bearingnet}"; WORK="${WORK:-$SRC/../work}"
# restart #3 (~18:00): resume r2 and r3 in parallel, then ratchet them in order (r2 first, then r3)
cd $WORK
export OMP_NUM_THREADS=1
nice -n 5 python3 $SRC/bn_train.py --name r2 --banks tr2 --steps 6000 --bs 96 >> logs/train_r2.log 2>&1 &
P2=$!
nice -n 5 python3 $SRC/bn_train.py --name r3 --arch hyp --banks tr2 --steps 4000 --bs 96 >> logs/train_r3.log 2>&1 &
P3=$!
wait $P2
[ -f ckpt/r2.pt ] && ! grep -q '"r2.pt"' ratchet.json && nice -n 5 python3 $SRC/ratchet.py r2.pt "v2 physics (dev-calibrated: louder faults, longer ringing, loud distractors, shaft-AM background, sensor EQ, OR rotating-load AM), 20k sims, same net" val2 > logs/ratchet_r2.log 2>&1
wait $P3
[ -f ckpt/r3.pt ] && ! grep -q '"r3.pt"' ratchet.json && nice -n 5 python3 $SRC/ratchet.py r3.pt "HypNet: one trunk shared by all hypothesis axes (comb detector) + per-class code; v2 sims (20k)" val2 > logs/ratchet_r3.log 2>&1
