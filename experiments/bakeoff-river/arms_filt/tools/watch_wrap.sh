#!/bin/bash
# agent C's watcher (only while agent C is active): wraps newly registered arms, one at a time. Stop: touch tools/STOP
source /home/claude/bakeoff/river/harness/env.sh
cd /home/claude/bakeoff/river/arms_filt
while [ ! -f tools/STOP ]; do
  if [ -f ../harness/results/EXAM_RUNNING.json ] || [ -f ../harness/results/EXAM_DONE.json ]; then echo "exam started; stop"; break; fi
  n=$(python3 conformal/wrap_pending.py 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then echo "$(date -u) wrapping $n"; nice -n 5 python3 conformal/wrap_pending.py --run; fi
  sleep 120
done
