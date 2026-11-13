#!/bin/sh
# portable paths: code in training/, outputs in models/hybridpilot/work (or $ML_RATCHET_WORK/hybridpilot)
SRC="$(cd "$(dirname "$0")" && pwd)"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/hybridpilot}"; WORK="${WORK:-$SRC/../work}"
VEH="${ML_RATCHET_VEHICLE:-$(cd "$SRC/../../../shared/vehicle" && pwd)}"
cd $WORK
R="nice -n 5 python3 $SRC/ratchet.py"
export OMP_NUM_THREADS=1
$R R0_aecms - '{"baseline":"aecms"}' "Incumbent: RoadSim's default A-ECMS (SOC-feedback equivalence factor, Onori et al. 2010)" > logs/r0.log 2>&1
$R R1_bc ckpt/r1_bc.npz '{"Kp":0,"preview_s":0}' "Behaviour cloning of DP (800 teacher trips with exploration bursts); s and on/off both from the network; no preview" > logs/r1.log 2>&1
$R R2_dagger ckpt/r2_dagger.npz '{"Kp":6,"preview_s":0}' "DAgger round 1 (+600 trips driven by R1, beta 0.3, DP labels at visited states) + A-ECMS-style SOC feedback Kp 6 on s" > logs/r2.log 2>&1
$R R3_costonoff ckpt/r2_dagger.npz '{"Kp":6,"onoff":"ecms","preview_s":0}' "Same network; engine on/off decided by the cost P_fuel + s P_batt + start fuel with the learned s (on/off head unused)" > logs/r3.log 2>&1
$R R4_preview30 ckpt/r2_dagger.npz '{"Kp":6,"onoff":"ecms","preview_s":30}' "R3 + 30 s speed/grade preview (as from navigation)" > logs/r4.log 2>&1
echo DONE > logs/rungs1.done
