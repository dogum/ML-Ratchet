#!/bin/bash
# usage: eval_only.sh <arm> <rung>   (harness evaluate of an already-exported candidate; used after a restart killed the eval step)
source /home/claude/bakeoff/vehicle/harness/env.sh
export NUMBA_CACHE_DIR=/home/claude/bakeoff/vehicle/arms_trace/.numba
TRC=/home/claude/bakeoff/vehicle/arms_trace
echo $$ > $TRC/logs/eval_$1_$2.pid
echo "=== eval $1 $2 $(date -u +%FT%T)"
python3 /home/claude/bakeoff/vehicle/harness/evaluate.py $TRC/$1/work/cand_$2 --track trace --tag $2 --out $TRC/$1/work/eval && echo "RUNG_DONE $1_$2"
echo "=== end eval $1 $2 $(date -u +%FT%T)"
