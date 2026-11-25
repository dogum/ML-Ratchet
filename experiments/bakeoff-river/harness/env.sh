# source this before running any harness script (one thread, local numba cache, data root)
export ML_RATCHET_WATER=/home/claude/water
export WATERSHED_ROOT=/home/claude/water
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/home/claude/bakeoff/river/harness/.numba
export PYTHONDONTWRITEBYTECODE=1
