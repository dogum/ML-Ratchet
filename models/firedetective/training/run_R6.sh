#!/bin/bash
# R6: the R4 recipe (3-seed ensemble of the R3 network) on the full v3 bank (train_v3 60 shards)
cd "$(dirname "$0")"   # import: was cd <sandbox>/fire/firedetective/src (logs and models now under WORK, see _paths.py)
WORK=$(python3 -c "import _paths; print(_paths.WORK)"); mkdir -p "$WORK/logs" "$WORK/models"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$WORK/.numba
for s in 0 1 2; do
  nice -n 5 python3 train.py R6s$s --banks train_v1,train_v2,train_v3 --val val_v3 --seed $s --drop-feats elong > $WORK/logs/train_R6s$s.log 2>&1
done
echo '{"members": ["R6s0", "R6s1", "R6s2"]}' > $WORK/models/R6.ens.json
nice -n 5 python3 ratchet.py R6 "R4 recipe (3-seed ensemble) on the full v3 bank: train_v3 grows from 25 to 60 shards (about 4,100 sims in all)" --frozen val_v3 > $WORK/logs/ratchet_R6.log 2>&1
