#!/bin/bash
# retrain rungs as 3-member ensembles (sequential)
cd "$(dirname "$0")"                       # training/src (was the sandbox's fire/firesize/src)
WORK=${ML_RATCHET_WORK:+$ML_RATCHET_WORK/firesize}; WORK=${WORK:-$(cd ../.. && pwd)/work}   # as training/_paths.py
mkdir -p "$WORK/logs"
export OMP_NUM_THREADS=1
for r in "r0|{\"physics\":\"v0\",\"theta\":\"lit\",\"feats\":\"base\",\"n_ign\":4800}" \
         "r1|{\"physics\":\"v0\",\"theta\":\"v0acc\",\"feats\":\"base\",\"n_ign\":4800}" \
         "r2|{\"physics\":\"v1\",\"theta\":\"v1acc\",\"feats\":\"base\",\"n_ign\":4800}" \
         "r3|{\"physics\":\"v1\",\"theta\":\"v1reg\",\"feats\":\"base\",\"n_ign\":4800}" \
         "r3b|{\"physics\":\"v1\",\"theta\":\"v1rep\",\"feats\":\"base\",\"n_ign\":4800}" \
         "r4|{\"physics\":\"v1\",\"theta\":\"v1acc\",\"feats\":\"phys\",\"n_ign\":4800}"; do
  name=${r%%|*}; cfg=${r#*|}
  nice -n 5 python3 train.py $name "$cfg" 2>&1 | grep -v "Warn\|return np" | tail -1 >> "$WORK"/logs/retrain.log
done
echo DONE >> "$WORK"/logs/retrain.log
