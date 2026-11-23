#!/bin/bash
# rungs on physics v1 (tr1 bank, residual spread 1.5 s): r1 no pick times; r2 + pick times; r3 + v1.2 summaries; r4 + P cross-correlation lags
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 train2.py --name r1_notime --no_time
python3 train2.py --name r2_time
python3 train2.py --name r3_emb --emb
python3 train2.py --name r4_cc --emb --cc
