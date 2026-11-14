#!/bin/sh
# Physics v2: re-take the v1 ladder R0-R4 (run from WORK after v2_stage1.sh and v2_bank0.sh)
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1
R="$PY $SRC/ratchet.py"
$PY $SRC/train.py r1_bc bank_r0 --epochs 30 --device cuda > logs/train_r1.log 2>&1
until grep -q DONE logs/base_dev.log; do sleep 20; done
$R R0_aecms - '{"baseline":"aecms"}' "Incumbent: RoadSim's default A-ECMS (SOC-feedback equivalence factor, Onori et al. 2010)" > logs/r0.log 2>&1
$R R1_bc ckpt/r1_bc.npz '{"Kp":0}' "Behaviour cloning of DP (800 teacher trips with exploration bursts); s and on/off both from the network; no preview" > logs/r1.log 2>&1
HP_WORKERS=5 $PY $SRC/gen_bank.py r1 600 12 mix ckpt/r1_bc.npz 0.3 '{"Kp":0}' > logs/bank_r1.log 2>&1
$PY $SRC/train.py r2_dagger bank_r0 bank_r1 --epochs 25 --device cuda > logs/train_r2.log 2>&1
$R R2_dagger ckpt/r2_dagger.npz '{"Kp":6}' "DAgger round 1 (+600 trips driven by R1, beta 0.3, DP labels at visited states) + A-ECMS-style SOC feedback Kp 6 on s" > logs/r2.log 2>&1
$R R3_costonoff ckpt/r2_dagger.npz '{"Kp":6,"onoff":"ecms"}' "Same network; engine on/off decided by the cost P_fuel + s P_batt + start fuel with the learned s (on/off head unused)" > logs/r3.log 2>&1
$R R4_preview30 ckpt/r2_dagger.npz '{"Kp":6,"onoff":"ecms","preview_s":30}' "R3 + 30 s speed/grade preview (as from navigation)" > logs/r4.log 2>&1
echo DONE > logs/ladder1.done
