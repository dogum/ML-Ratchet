#!/bin/bash
# frozen sim validation set v1 first, then the training bank (resumable)
cd "$(dirname "$0")"   # import: was cd <sandbox>/fire/firedetective/src (logs and models now under WORK, see _paths.py)
WORK=$(python3 -c "import _paths; print(_paths.WORK)"); mkdir -p "$WORK/logs" "$WORK/models"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$WORK/.numba
nice -n 5 python3 simbank.py val_v1 9000000 15 100
nice -n 5 python3 simbank.py train_v1 1000000 120 100
