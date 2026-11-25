# source this for every arms_table job (one thread, harness on the path, own numba cache)
source /home/claude/bakeoff/vehicle/harness/env.sh
export NUMBA_CACHE_DIR=/home/claude/bakeoff/vehicle/arms_table/.numba
export PYTHONPATH=/home/claude/bakeoff/vehicle/harness:/home/claude/bakeoff/vehicle/arms_table/src
export TABPFN_DISABLE_TELEMETRY=1 TABPFN_NO_BROWSER=1 HF_HUB_OFFLINE=1
