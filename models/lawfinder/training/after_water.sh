#!/bin/bash
# As run in the sandbox: wait for the water banks' process (PID given as $1; it was 1066 there), then the grid banks.
while kill -0 "$1" 2>/dev/null; do sleep 30; done
bash "$(dirname "$0")"/run_grid_banks.sh
