#!/bin/sh
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/bearingnet}"; WORK="${WORK:-$SRC/../work}"
# r4: r2 fine-tuned with band dropout (dev: the same CWRU reading gets different answers on different sensors -> lean on the comb, not on one band)
cd $WORK
export OMP_NUM_THREADS=1
nice -n 5 python3 $SRC/bn_train.py --name r4 --banks tr2 --init ckpt/r2.pt --chdrop 0.3 --steps 2500 --lr 1e-3 --seed 4 > logs/train_r4.log 2>&1
while ! grep -q '"r3.pt"' ratchet.json; do sleep 30; done
[ -f ckpt/r4.pt ] && ! grep -q '"r4.pt"' ratchet.json && nice -n 5 python3 $SRC/ratchet.py r4.pt "r2 fine-tuned 2500 steps with band dropout 0.3 (whole demodulation bands blanked), v2 sims (20k)" val2 > logs/ratchet_r4.log 2>&1
