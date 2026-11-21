#!/bin/bash
cd "$(dirname "$0")"                       # training/src (was the sandbox's fire/firesize/src)
WORK=${ML_RATCHET_WORK:+$ML_RATCHET_WORK/firesize}; WORK=${WORK:-$(cd ../.. && pwd)/work}   # as training/_paths.py
mkdir -p "$WORK/logs"
export OMP_NUM_THREADS=1
nice -n 5 python3 train.py r6 '{"physics":"v1","theta":"v1rep2","feats":"base"}' 2>&1 | grep -v "Warn\|return np" | tail -1 >> "$WORK"/logs/final_rungs.log
nice -n 5 python3 train.py r7 '{"physics":"v1","theta":"v1rep2","feats":"phys"}' 2>&1 | grep -v "Warn\|return np" | tail -1 >> "$WORK"/logs/final_rungs.log
nice -n 5 python3 ratchet.py 'v1|v1rep2' 2>&1 | grep -v "Warn\|return np\|warn" >> "$WORK"/logs/final_rungs.log
echo DONE >> "$WORK"/logs/final_rungs.log
