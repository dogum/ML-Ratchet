#!/bin/bash
# builds val then train then the manifest; resumable (re-run after a restart). Runs from any directory; PYTHON overrides
# the interpreter (default python3). Cloud original: cd /home/claude/bakeoff/river/harness.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
PY="${PYTHON:-python3}"
nice -n 5 "$PY" build_bank.py val && nice -n 5 "$PY" build_bank.py train && "$PY" build_bank.py finalize
