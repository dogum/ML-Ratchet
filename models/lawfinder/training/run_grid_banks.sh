#!/bin/bash
cd "$(dirname "$0")"; . ./env_sandbox.sh
export PYTHONDONTWRITEBYTECODE=1
nice -n 10 python3 -W ignore grid_bank.py inertia i1 val 2000 1
nice -n 10 python3 -W ignore grid_bank.py inertia i1 train 6000 1
for v in g1 g2; do
  nice -n 10 python3 -W ignore grid_bank.py nadir $v val 3000 1
  nice -n 10 python3 -W ignore grid_bank.py nadir $v train 8000 1
done
echo DONE
