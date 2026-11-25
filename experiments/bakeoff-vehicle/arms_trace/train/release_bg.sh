#!/bin/bash
source /home/claude/bakeoff/vehicle/harness/env.sh
export NUMBA_CACHE_DIR=/home/claude/bakeoff/vehicle/arms_trace/.numba
cd /home/claude/bakeoff/vehicle/arms_trace/train
echo $$ > ../logs/release_$1.pid
python3 release.py $1 && echo "RELEASE_DONE $1"
