#!/bin/sh
# Phase D: fleets and banks on RoadSim physics v2 (4 worker processes + 1 for real records; the machine is shared with two other agents)
SRC="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$SRC/../../../.venv/Scripts/python.exe}"
export OMP_NUM_THREADS=1
cd "$SRC"
$PY fleet.py val2 400 202 4 && $PY fleet.py train2 2400 201 4 && \
$PY bank.py val2 10 777 4 && $PY bank.py train2 12 101 4
echo BANKS_DONE
