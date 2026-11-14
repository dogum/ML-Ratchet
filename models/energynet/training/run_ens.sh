#!/bin/sh
# Train ensemble members on the GPU: run_ens.sh <name prefix> <seeds, e.g. "0 1 2 3 4"> <train.py args...>
SRC="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$SRC/../../../.venv/Scripts/python.exe}"
cd "$SRC"
P=$1; S=$2; shift 2
for s in $S; do
  [ -f ../work/ckpt/${P}_s$s.npz ] || $PY train.py ${P}_s$s --seed $s "$@" > ../work/logs/train_${P}_s$s.log 2>&1
done
echo ENS_DONE $P
