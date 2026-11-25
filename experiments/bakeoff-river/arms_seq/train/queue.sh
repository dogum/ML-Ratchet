#!/bin/bash
# usage: queue.sh <qname> <config> [<config> ...]   runs rungs one after another; skips rungs already evaluated.
SEQ=/home/claude/bakeoff/river/arms_seq
Q=$1; shift
for NAME in "$@"; do
  ARM=$(python3 -c "import json;print(json.load(open('$SEQ/train/configs/$NAME.json'))['arm'])")
  RUNG=$(python3 -c "import json;print(json.load(open('$SEQ/train/configs/$NAME.json'))['rung'])")
  if [ -f $SEQ/$ARM/work/eval/${RUNG}_summary.json ]; then echo "skip $NAME"; continue; fi
  echo "=== $NAME $(date -u +%FT%T)"
  $SEQ/train/run_rung.sh $NAME > $SEQ/logs/$NAME.log 2>&1 || echo "FAILED $NAME"
  echo "=== end $NAME $(date -u +%FT%T)"
done
