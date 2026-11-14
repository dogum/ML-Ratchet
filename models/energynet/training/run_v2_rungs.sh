#!/bin/sh
# Phase D ladder on RoadSim physics v2: train rungs on the GPU (train2 bank), then score and decide (ratchet2.py).
SRC="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$SRC/../../../.venv/Scripts/python.exe}"
L="$SRC/../work/logs"
cd "$SRC"
while [ $(ls ../work/data/bank_train2/chunk_*.npz 2>/dev/null | wc -l) -lt 48 ]; do sleep 30; done
$PY train.py r1_abs --version a --mode abs --bank train2 > $L/train_r1.log 2>&1
$PY train.py r2_corr --version a --mode corr --bank train2 > $L/train_r2.log 2>&1
$PY train.py r3_lab --version b --mode corr --bank train2 > $L/train_r3.log 2>&1
$PY train.py r4_rw --version c --mode corr --bank train2 > $L/train_r4.log 2>&1
$PY ratchet2.py > $L/ratchet2_a.log 2>&1
echo RUNGS_DONE
