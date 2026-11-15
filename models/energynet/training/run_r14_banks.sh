#!/bin/sh
# Rung v2-r14: SC03 / 20 F cold-FTP dyno rows under the EPA test conditions (3 worker processes; shared machine)
SRC="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$SRC/../../../.venv/Scripts/python.exe}"
export OMP_NUM_THREADS=1
cd "$SRC"
for b in val2:val2 train2:train2 train2b:train2 train2c:train2; do
  $PY bank_dyno.py ${b%%:*} ${b##*:} 3 || exit 1
done
echo R14_BANKS_DONE
