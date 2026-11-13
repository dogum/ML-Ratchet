#!/bin/bash
# portable paths: the shared vehicle platform (shared/vehicle, or $ML_RATCHET_VEHICLE)
VEH="${ML_RATCHET_VEHICLE:-$(cd "$(dirname "$0")/../.." && pwd)}"
# rebuild all VED trips (resumable per bucket via .done2 markers)
cd $VEH/data/scripts
for b in $(seq 0 15); do
  m=$VEH/data/proc/ved/stage2_b$(printf %02d $b).done2
  [ -f $m ] && continue
  FORCE=1 OMP_NUM_THREADS=1 nice -n 5 python3 -I ved_stage2.py $b && touch $m
done
OMP_NUM_THREADS=1 nice -n 5 python3 -I ved_index.py
