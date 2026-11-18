#!/bin/bash
# usage: run_bank.sh g1 [g2 ...]   (as run in the sandbox; paths now from the repository layout, see _paths.py)
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${ML_RATCHET_WORK:+$ML_RATCHET_WORK/reservoirpilot}; WORK=${WORK:-$HERE/../work}
mkdir -p "$WORK/logs"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_CACHE_DIR=$WORK/.numba
cd "$WORK"
for g in "$@"; do nice -n 5 python3 "$HERE/bank.py" $g 3000 300 >> logs/bank_$g.log 2>&1; done
