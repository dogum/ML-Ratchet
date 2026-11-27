#!/bin/bash
# StarQuake rung r6 (27 Nov 2026, audit queue): r5's bank recipe (v1 + v2 physics, 30% non-oscillating in part) rebuilt
# with new seeds, plus one bank whose noise stars are the high-RMS dwarfs screen.py dropped (noise_train_hard: spots,
# eclipses, rotation; 50% of its examples non-oscillating). Trained from scratch (no r1-r5 checkpoint was kept).
# Usage: run_r6.sh banks <0|1>   (two halves of the bank list, run in parallel), then run_r6.sh train
SRC="$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-python}; export OMP_NUM_THREADS=1
B() { [ -f "$SRC/../work/bank/$1" ] || $PY "$SRC/simbank.py" --out "$1" --n 5000 --seed "$2" --calib "$3" --p_neg "$4" --pool "$5"; }
if [ "$1" = banks ] && [ "$2" = 0 ]; then
  B sq6_v1_a.npz 201 '' 0 noise_train; B sq6_v1_b.npz 202 '' 0 noise_train; B sq6_v1_c.npz 203 '' 0 noise_train
  B sq6_v2n_a.npz 221 v2 0.3 noise_train; B sq6_hard_a.npz 231 v2 0.5 noise_train_hard
elif [ "$1" = banks ]; then
  B sq6_v2_a.npz 211 v2 0 noise_train; B sq6_v2_b.npz 212 v2 0 noise_train; B sq6_v2_c.npz 213 v2 0 noise_train
  B sq6_v2n_b.npz 222 v2 0.3 noise_train
elif [ "$1" = train ]; then
  $PY "$SRC/train.py" --name r6 --banks 'sq6_*.npz' --steps 12000 --bs 64 --lr 2e-3 --seed 0 --arch '{"det": true}'
fi
