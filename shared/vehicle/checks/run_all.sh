#!/bin/bash
# portable paths: the shared vehicle platform (shared/vehicle, or $ML_RATCHET_VEHICLE)
VEH="${ML_RATCHET_VEHICLE:-$(cd "$(dirname "$0")/.." && pwd)}"
# Re-run the full verification chain (physics version change => recalibrate on dev, then every check).
cd $VEH/checks
export OMP_NUM_THREADS=1
L=$VEH/roadsim/logs
set -e
nice -n 5 python3 epa_vehicles.py > $L/run_epa_vehicles.log 2>&1
nice -n 5 python3 calibrate_dev.py > $L/calib.log 2>&1
nice -n 5 python3 check_cycles.py > $L/check_cycles.log 2>&1
nice -n 5 python3 check_coastdown.py > $L/coastdown.log 2>&1
nice -n 5 python3 check_fastsim.py > $L/fastsim.log 2>&1
nice -n 5 python3 check_hd.py > $L/hd.log 2>&1
nice -n 5 python3 check_strategies.py > $L/strat.log 2>&1
nice -n 5 python3 check_ledger.py > $L/ledger.log 2>&1
nice -n 5 python3 check_generator.py > $L/gen.log 2>&1
python3 check_speed.py > $L/speed.log 2>&1
echo ALL_DONE > $L/run_all.done
