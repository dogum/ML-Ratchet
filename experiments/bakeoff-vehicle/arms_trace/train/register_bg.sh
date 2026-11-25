#!/bin/bash
# usage: register_bg.sh <arm> [<arm> ...]   harness register.py, one arm after another (one thread, nice via caller)
source /home/claude/bakeoff/vehicle/harness/env.sh
export NUMBA_CACHE_DIR=/home/claude/bakeoff/vehicle/arms_trace/.numba
cd /home/claude/bakeoff/vehicle
echo $$ > arms_trace/logs/register.pid
for A in "$@"; do
  echo "=== register $A $(date -u +%FT%T)"
  python3 harness/register.py arms_trace/$A && echo "REGISTER_DONE $A" || echo "REGISTER_FAILED $A"
done
