#!/bin/bash
# builds the train bank with 2 workers then finalizes; resumable (re-run after a restart: finished chunks are skipped).
# Runs from any directory; PYTHON overrides the interpreter (default python3). Cloud original: cd /home/claude/bakeoff/vehicle/harness.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
PY="${PYTHON:-python3}"
mkdir -p logs
nice -n 5 "$PY" build_bank.py train 0 2 > logs/bank_train_w0.log 2>&1 &
P0=$!
nice -n 5 "$PY" build_bank.py train 1 2 > logs/bank_train_w1.log 2>&1 &
P1=$!
wait $P0 $P1
nice -n 5 "$PY" build_bank.py finalize train > logs/bank_train_finalize.log 2>&1
