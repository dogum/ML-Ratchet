#!/bin/bash
# usage: run_rung.sh <config name>   (train -> export candidate -> harness evaluate on the full frozen sim val + dev gate)
set -e
source /home/claude/bakeoff/vehicle/harness/env.sh
export NUMBA_CACHE_DIR=/home/claude/bakeoff/vehicle/arms_trace/.numba
TRC=/home/claude/bakeoff/vehicle/arms_trace
NAME=$1
ARM=$(python3 -c "import json;print(json.load(open('$TRC/train/configs/$NAME.json'))['arm'])")
RUNG=$(python3 -c "import json;print(json.load(open('$TRC/train/configs/$NAME.json'))['rung'])")
cd $TRC/train
python3 train.py $NAME
python3 /home/claude/bakeoff/vehicle/harness/evaluate.py $TRC/$ARM/work/cand_$RUNG --track trace --tag $RUNG --out $TRC/$ARM/work/eval
echo "RUNG_DONE $NAME"
