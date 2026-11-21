#!/bin/bash
# usage: run_train.sh NAME [train.py args...]   (as run in the sandbox; paths now from the repository layout, see _paths.py)
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${ML_RATCHET_WORK:+$ML_RATCHET_WORK/reservoirpilot}; WORK=${WORK:-$HERE/../work}
mkdir -p "$WORK/logs"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_CACHE_DIR=$WORK/.numba
cd "$WORK"
name=$1; shift
nice -n 5 python3 "$HERE/train.py" $name "$@" >> logs/$name.log 2>&1
