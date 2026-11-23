# Environment for the shell wrappers (sourced from training/). One thread each; numba cache and temp files in WORK.
# The platform code and data roots come from training/_paths.py ($ML_RATCHET_WATER, _FIRE, _GRID, _BUILDING, _VEHICLE).
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORK=$(python3 _paths.py)
export NUMBA_CACHE_DIR=$WORK/numba TMPDIR=$WORK/tmp
