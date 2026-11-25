#!/bin/bash
# prints one line per newly finished training member / MLP set, and per training log with a Traceback
cd "$(dirname "$0")"
declare -A seen
for f in work/trace/*/m*/train.json; do seen[$f]=1; done
for f in work/logs/*.log; do grep -q Traceback "$f" 2>/dev/null && seen[$f]=1; done
while true; do
  for f in work/trace/*/m*/train.json; do
    [ -e "$f" ] || continue
    if [ -z "${seen[$f]}" ]; then echo "DONE $f $(date +%H:%M)"; seen[$f]=1; fi
  done
  n=$(grep -c exported work/logs/p3_mlp.log 2>/dev/null)
  if [ "${n:-0}" -ge 5 ] && [ -z "${seen[p3]}" ]; then echo "P3 MLP done $(date +%H:%M)"; seen[p3]=1; fi
  for f in work/logs/*.log; do
    if grep -q Traceback "$f" 2>/dev/null && [ -z "${seen[$f]}" ]; then echo "ERROR in $f"; seen[$f]=1; fi
  done
  [ -e work/logs/p_done.flag ] && { echo P_DONE; exit 0; }
  sleep 30
done
