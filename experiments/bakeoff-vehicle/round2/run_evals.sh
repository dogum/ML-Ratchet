#!/bin/bash
# sequential evaluations: each argument is "<arm dir>|<extra eval.py args>"; log per job in work/logs/eval_<n>.log
cd "$(dirname "$0")"
PY=C:/Developer/ML-ratchet/.venv/Scripts/python.exe
for job in "$@"; do
  arm="${job%%|*}"; args="${job#*|}"; [ "$args" = "$job" ] && args=""
  tag=$(basename "$arm")$(echo "$args" | tr ' -' '__')
  $PY -X utf8 eval.py $arm --no-compare $args > work/logs/eval_$tag.log 2>&1
  echo "$tag: $(grep -h 'simval:\|dev: D16' work/logs/eval_$tag.log | cut -c1-60 | tr '\n' ' ')"
done
