#!/bin/bash
# usage: queue.sh <qname> <config> [<config> ...]   runs rungs one after another; skips rungs already evaluated.
TRC=/home/claude/bakeoff/vehicle/arms_trace
Q=$1; shift
echo $$ > $TRC/logs/$Q.pid
for NAME in "$@"; do
  ARM=$(python3 -c "import json;print(json.load(open('$TRC/train/configs/$NAME.json'))['arm'])")
  RUNG=$(python3 -c "import json;print(json.load(open('$TRC/train/configs/$NAME.json'))['rung'])")
  if [ -f $TRC/$ARM/work/eval/${RUNG}_trace_summary.json ]; then echo "skip $NAME"; continue; fi
  echo "=== $NAME $(date -u +%FT%T)"
  $TRC/train/run_rung.sh $NAME >> $TRC/logs/$NAME.log 2>&1 || echo "FAILED $NAME"
  echo "=== end $NAME $(date -u +%FT%T)"
done
