#!/bin/bash
cd "$(dirname "$0")"; . ./env_sandbox.sh
for v in f1 f2; do
  nice -n 10 python3 -W ignore fire_bank.py $v val 240 1
  nice -n 10 python3 -W ignore fire_bank.py $v train 600 1
done
echo DONE
