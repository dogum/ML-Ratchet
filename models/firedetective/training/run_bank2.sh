#!/bin/bash
# v2 physics: frozen sim validation set v2, then training bank v2 (resumable)
cd "$(dirname "$0")"   # import: was cd <sandbox>/fire/firedetective/src (logs and models now under WORK, see _paths.py)
WORK=$(python3 -c "import _paths; print(_paths.WORK)"); mkdir -p "$WORK/logs" "$WORK/models"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$WORK/.numba
nice -n 5 python3 simbank.py val_v2 9000000 15 100
nice -n 5 python3 simbank.py train_v2 2000000 60 100
