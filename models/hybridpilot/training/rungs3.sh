#!/bin/sh
# portable paths: code in training/, outputs in models/hybridpilot/work (or $ML_RATCHET_WORK/hybridpilot)
SRC="$(cd "$(dirname "$0")" && pwd)"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/hybridpilot}"; WORK="${WORK:-$SRC/../work}"
VEH="${ML_RATCHET_VEHICLE:-$(cd "$SRC/../../../shared/vehicle" && pwd)}"
cd $WORK
export OMP_NUM_THREADS=1
nice -n 5 python3 $SRC/train.py r3_dagger2 bank_r0 bank_r1 bank_r2 --epochs 25 > logs/train_r3.log 2>&1
nice -n 5 python3 $SRC/ratchet.py R6_dagger2 ckpt/r3_dagger2.npz '{"Kp":6,"onoff":"ecms","preview_s":0}' "DAgger round 2: +600 trips driven by the R3 policy (cost-based on/off, Kp 8, beta 0.2), retrained on all 2,000 trips" > logs/r6.log 2>&1
nice -n 5 python3 $SRC/ratchet.py R7_dagger2_preview30 ckpt/r3_dagger2.npz '{"Kp":6,"onoff":"ecms","preview_s":30}' "R6 + 30 s speed/grade preview" > logs/r7.log 2>&1
echo DONE > logs/rungs3.done
