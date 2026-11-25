#!/bin/bash
# usage: wrap_bg.sh [--only ids]   arm C's conformal wrap_pending.py --run (dev residuals only), one thread
source /home/claude/bakeoff/vehicle/arms_phys/env.sh
echo $$ > /home/claude/bakeoff/vehicle/arms_trace/logs/wrap.pid
cd /
python3 /home/claude/bakeoff/vehicle/arms_phys/conformal/wrap_pending.py --run "$@"
echo "WRAP_EXIT $?"
