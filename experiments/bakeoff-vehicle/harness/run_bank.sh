#!/bin/bash
# builds the frozen val bank, then the train bank, each with 2 workers, then finalizes each; resumable (re-run after a
# restart: finished chunks are skipped). Runs from any directory; PYTHON overrides the interpreter (default python3).
# Cloud original (train only; val was built the same way before it): cd /home/claude/bakeoff/vehicle/harness.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
PY="${PYTHON:-python3}"
mkdir -p logs
for split in val train; do
  nice -n 5 "$PY" build_bank.py $split 0 2 > logs/bank_${split}_w0.log 2>&1 &
  P0=$!
  nice -n 5 "$PY" build_bank.py $split 1 2 > logs/bank_${split}_w1.log 2>&1 &
  P1=$!
  wait $P0 $P1
  nice -n 5 "$PY" build_bank.py finalize $split > logs/bank_${split}_finalize.log 2>&1
done
