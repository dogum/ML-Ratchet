#!/bin/bash
# usage: wait_fit.sh <law> <bank_dir> <n_train_files>
cd "$(dirname "$0")"; . ./env_sandbox.sh; export PYTHONDONTWRITEBYTECODE=1
while [ $(ls $2/train_*.parquet 2>/dev/null | wc -l) -lt $3 ]; do sleep 20; done
nice -n 5 python3 -W ignore fit_law.py $1
