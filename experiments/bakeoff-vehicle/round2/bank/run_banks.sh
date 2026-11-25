#!/bin/bash
# Round-2 bank build: NW single-thread workers over the listed banks, then finalize each. Resumable (finished chunks skip).
# usage: bank/run_banks.sh val train101 train102 train103
cd "$(dirname "$0")/.."
PY=C:/Developer/ML-ratchet/.venv/Scripts/python.exe
NW=${NW:-8}
mkdir -p work/logs
for w in $(seq 0 $((NW-1))); do
  $PY -X utf8 bank/build.py worker $w $NW "$@" > work/logs/bank_w$w.log 2>&1 &
done
wait
for b in "$@"; do $PY -X utf8 bank/build.py finalize $b >> work/logs/bank_finalize.log 2>&1; done
echo BANKS_DONE "$@" >> work/logs/bank_finalize.log
