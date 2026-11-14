#!/bin/sh
# Physics v2 rebuild, stage 1: baseline caches (frozen, dev) and the first teacher bank. Run from WORK.
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1
HP_WORKERS=3 $PY $SRC/evaluate.py base frozen > logs/base_frozen.log 2>&1
HP_WORKERS=3 $PY $SRC/evaluate.py base dev > logs/base_dev.log 2>&1
echo DONE >> logs/base_dev.log
