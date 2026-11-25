#!/bin/bash
# search s2 (rung r2): + the dynamometer-test flag 'dyn' as a law variable; waits for s1 to finish (one search at a time)
source /home/claude/bakeoff/vehicle/arms_phys/env.sh
cd /home/claude/bakeoff/vehicle/arms_phys
while [ ! -f work/search/s1_bev.json ]; do sleep 30; done
for g in ice hev bev; do
  python3 src/search_law.py s2 $g 1200 --seed 2 --vars p,n,t,m,T,c,d,dyn
done
echo ALLDONE
