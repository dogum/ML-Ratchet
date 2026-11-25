# source this before running any harness script: one thread, numba cache. Paths resolve from this file, and a value the
# caller already exported wins. harness/config.py sets the water data root ($ML_RATCHET_WATER, default shared/water).
# As run in the cloud (22-23 Nov 2026): ML_RATCHET_WATER=WATERSHED_ROOT=/home/claude/water,
# NUMBA_CACHE_DIR=/home/claude/bakeoff/river/harness/.numba.
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$H/.numba}"
export PYTHONDONTWRITEBYTECODE=1
