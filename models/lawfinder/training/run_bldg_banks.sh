#!/bin/bash
cd "$(dirname "$0")"; . ./env_sandbox.sh
for v in b1 b2; do
  nice -n 10 python3 -W ignore bldg_bank.py $v val 300 1
  nice -n 10 python3 -W ignore bldg_bank.py $v train 1000 1
done
echo DONE
