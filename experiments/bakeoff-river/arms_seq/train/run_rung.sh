#!/bin/bash
# usage: run_rung.sh <config name>   (train -> export candidate -> harness evaluate on all 800 sim val runs + dev gate)
set -e
source /home/claude/bakeoff/river/harness/env.sh
export NUMBA_CACHE_DIR=/home/claude/bakeoff/river/arms_seq/.numba
SEQ=/home/claude/bakeoff/river/arms_seq
NAME=$1
ARM=$(python3 -c "import json;print(json.load(open('$SEQ/train/configs/$NAME.json'))['arm'])")
RUNG=$(python3 -c "import json;print(json.load(open('$SEQ/train/configs/$NAME.json'))['rung'])")
cd $SEQ/train
python3 train.py $NAME
python3 /home/claude/bakeoff/river/harness/evaluate.py $SEQ/$ARM/work/cand_$RUNG --tag $RUNG --out $SEQ/$ARM/work/eval ${EVAL_ARGS}
echo "RUNG_DONE $NAME"
