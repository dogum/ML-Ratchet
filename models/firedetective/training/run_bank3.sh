#!/bin/bash
# v3 physics (= v2 + wind-sensitivity WAF): frozen sim validation set v3 (same seeds as v1/v2), training bank v3
cd "$(dirname "$0")"   # import: was cd <sandbox>/fire/firedetective/src (logs and models now under WORK, see _paths.py)
WORK=$(python3 -c "import _paths; print(_paths.WORK)"); mkdir -p "$WORK/logs" "$WORK/models"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$WORK/.numba
nice -n 5 python3 simbank.py val_v3 9000000 15 100
nice -n 5 python3 simbank.py train_v3 3000000 60 100
