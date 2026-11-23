#!/bin/bash
cd "$(dirname "$0")"; . ./env_sandbox.sh
for v in w1 w2; do
  nice -n 5 python3 -W ignore water_bank.py $v val 3000 1
  nice -n 5 python3 -W ignore water_bank.py $v train 8000 1
done
echo DONE
