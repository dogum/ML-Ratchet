#!/bin/bash
cd "$(dirname "$0")"   # training/ (sandbox: bldg/detective/src); banks go to WORK/bank (see _paths.py)
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/buildingdetective}"; WORK="${WORK:-../work}"
export OMP_NUM_THREADS=1
python3 simbank.py v1 900 905
python3 simbank.py v1 0 60
touch "$WORK/bank/v1/DONE"
