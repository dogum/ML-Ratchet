#!/bin/bash
# Re-run every GridSim check in order (single thread) and compare with the published v1 numbers.
# Long ones: ERCOT years (~10-25 min each, MILP). Reference checks need an interpreter with pandapower/PYPOWER/ANDES:
#   REF=/path/to/refenv/python ./run_all.sh
set -e
cd "$(dirname "$0")"
exec python3 run_all.py ${REF:+--ref "$REF"} "$@"
