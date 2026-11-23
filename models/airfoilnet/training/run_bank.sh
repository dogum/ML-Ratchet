#!/bin/bash
# sequential bank jobs for one worker slot; usage: run_bank.sh "<set> <w> <nw> <phys>" ...
# As run, from the flat build folder (src/ = this folder); simbank.py writes under ../work (models/airfoilnet/work).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_CACHE_DIR=$(python3 -c "from _paths import WORK; print(WORK)")/numba_cache
for job in "$@"; do
  nice -n 5 python3 simbank.py $job
done
