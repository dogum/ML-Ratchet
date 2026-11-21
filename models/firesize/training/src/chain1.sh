#!/bin/bash
# waits for gridMET points, then features of sim ignitions, twin sims (v1 x4, v0 x2), calibrations
cd "$(dirname "$0")"                       # training/src (was the sandbox's fire/firesize/src)
WORK=${ML_RATCHET_WORK:+$ML_RATCHET_WORK/firesize}; WORK=${WORK:-$(cd ../.. && pwd)/work}   # as training/_paths.py
mkdir -p "$WORK/logs"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR="$WORK/.numba"
while [ ! -f "$WORK"/data/wx_points.npz ]; do sleep 30; done
while pgrep -f "python3 fetch_windows.py" > /dev/null; do sleep 30; done
nice -n 5 python3 trainfeat.py > "$WORK"/logs/trainfeat.log 2>&1
nice -n 5 python3 simbank.py twin 4 v1 > "$WORK"/logs/bank_twin_v1.log 2>&1
nice -n 5 python3 calib.py v1 > "$WORK"/logs/calib_v1.log 2>&1 &
nice -n 5 python3 simbank.py twin 2 v0 > "$WORK"/logs/bank_twin_v0.log 2>&1
wait
nice -n 5 python3 calib.py v0 > "$WORK"/logs/calib_v0.log 2>&1
echo CHAIN1 DONE > "$WORK"/logs/chain1.done
