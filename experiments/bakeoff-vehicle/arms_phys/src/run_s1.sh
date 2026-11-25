#!/bin/bash
# search s1: one law per powertrain group, all 7 variables, 20 min wall each (resumable: finished groups are skipped)
source /home/claude/bakeoff/vehicle/arms_phys/env.sh
cd /home/claude/bakeoff/vehicle/arms_phys
for g in ice hev bev; do
  python3 src/search_law.py s1 $g 1200 --seed 1
done
echo ALLDONE
