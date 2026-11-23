#!/bin/bash
# frozen simulated set first (val_v1, seeds 5000-5003, 2,000 events), then training chunks (seeds 100-119, 10,000 events)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 sim2.py val1 5000 4 500
python3 sim2.py tr1 100 20 500
