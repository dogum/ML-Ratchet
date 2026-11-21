#!/bin/bash
cd "$(dirname "$0")"                       # training/src (was the sandbox's fire/firesize/src)
WORK=${ML_RATCHET_WORK:+$ML_RATCHET_WORK/firesize}; WORK=${WORK:-$(cd ../.. && pwd)/work}   # as training/_paths.py
mkdir -p "$WORK/logs"
export OMP_NUM_THREADS=1
nice -n 5 python3 calib.py v1 > "$WORK"/logs/calib_v1.log 2>&1
nice -n 5 python3 calib.py v0 > "$WORK"/logs/calib_v0.log 2>&1
