#!/bin/sh
# Rung v2-r15: conventional light-duty cars without idle-stop, on top of the r14 banks (2 workers while the GPU trains)
SRC="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$SRC/../../../.venv/Scripts/python.exe}"
export OMP_NUM_THREADS=1
cd "$SRC"
W=${W:-2}
$PY bank_idlestop.py val2d val2 777 10 val2di $W && \
$PY bank_idlestop.py train2d train2 101 12 train2di $W && \
$PY bank_idlestop.py train2bd train2 102 12 train2bdi $W && \
$PY bank_idlestop.py train2cd train2 103 12 train2cdi $W && echo R15_BANKS_DONE
