#!/bin/sh
# r18 repair rung (audit 25 Nov 2026, run 27 Nov 2026): v2.1's recipe (r15/r17 networks: label version e on the h banks, corr, era, same banks)
# retrained with percentile-clipped input statistics (--robust-norm) and the component loss masked like the NLL
# (--mask-comp). Four members at a time, one thread each (shared machine).
SRC="$(cd "$(dirname "$0")" && pwd)"; PY="${PY:-$SRC/../../../.venv/Scripts/python.exe}"; cd "$SRC"
A="--version e --mode corr --bank train2di+train2bdi+train2cdi --era --robust-norm --mask-comp"
for s in 0 1 2 3; do $PY train.py r18_s$s --seed $s $A > ../work/logs/train_r18_s$s.log 2>&1 & done; wait
$PY train.py r18_s4 --seed 4 $A > ../work/logs/train_r18_s4.log 2>&1
echo ENS_DONE r18 > ../work/logs/ens_r18.log
