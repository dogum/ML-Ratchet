#!/bin/sh
# Physics v2: a larger network trained on the GPU (run from WORK after v2_ladder3.sh)
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1
R="$PY $SRC/ratchet.py"
INC() { $PY $SRC/incumbent.py "$@"; }
$PY $SRC/train.py r13_big $(INC banks) --hidden 256 --depth 3 --epochs 40 --lr 1e-3 --bs 2048 --device cuda > logs/train_r13.log 2>&1
$R R13_big_gpu ckpt/r13_big.npz "$(INC params)" "Larger network on the GPU: 3 x 256 tanh (~150k parameters) instead of 2 x 96, 40 epochs on the incumbent's banks; same decision rule" > logs/r13.log 2>&1
echo DONE > logs/ladder4.done
