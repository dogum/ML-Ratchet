#!/bin/bash
# RoadSim banks for the road law: physics p1 and p2, frozen set first (2,000 trips), then training (6,000), 3 workers
# per version. Vehicle data root: $ML_RATCHET_VEHICLE (RoadData's train_noise routes and the dyno cycles).
cd "$(dirname "$0")"; . ./env_sandbox.sh; export PYTHONDONTWRITEBYTECODE=1
for v in p1 p2; do
  for w in 0 1 2; do
    (python3 -W ignore road_bank.py $v val 2000 1 $w 3 && python3 -W ignore road_bank.py $v train 6000 1 $w 3) &
  done
done
wait
echo DONE
