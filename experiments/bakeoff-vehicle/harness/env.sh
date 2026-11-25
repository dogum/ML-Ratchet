# source this before running any harness script (one thread, vendored platform, local data, numba cache)
export ML_RATCHET_VEHICLE=/home/claude/bakeoff/vehicle/harness/vendor/vehicle
export ML_RATCHET_WORK=/home/claude/bakeoff/vehicle/harness/work
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/home/claude/bakeoff/vehicle/harness/.numba
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/claude/bakeoff/vehicle/harness
