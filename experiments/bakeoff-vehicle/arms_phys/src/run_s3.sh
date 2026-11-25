#!/bin/bash
# search s3 (rung r3): a dedicated plug-in law (vars + f, the charge-depleting share); waits for s2 to finish
source /home/claude/bakeoff/vehicle/arms_phys/env.sh
cd /home/claude/bakeoff/vehicle/arms_phys
while [ ! -f work/search/s2_bev.json ]; do sleep 30; done
python3 src/search_law.py s3 phev 1200 --seed 3 --vars p,n,t,m,T,c,d,dyn,f
echo ALLDONE
