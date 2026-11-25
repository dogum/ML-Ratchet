# source this before running any harness script: one thread, numba cache, work folder. Paths resolve from this file,
# and a value the caller already exported wins. harness/config.py finds the repository's shared/vehicle (data under
# $ML_RATCHET_VEHICLE/data, default shared/vehicle/data) and models/energynet by itself.
# As run in the cloud (23-24 Nov 2026): ML_RATCHET_VEHICLE=/home/claude/bakeoff/vehicle/harness/vendor/vehicle,
# ML_RATCHET_WORK=/home/claude/bakeoff/vehicle/harness/work, PYTHONPATH=/home/claude/bakeoff/vehicle/harness.
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ML_RATCHET_WORK="${ML_RATCHET_WORK:-$H/work}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$H/.numba}"
export PYTHONDONTWRITEBYTECODE=1
