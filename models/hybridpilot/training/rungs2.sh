#!/bin/sh
# portable paths: code in training/, outputs in models/hybridpilot/work (or $ML_RATCHET_WORK/hybridpilot)
SRC="$(cd "$(dirname "$0")" && pwd)"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/hybridpilot}"; WORK="${WORK:-$SRC/../work}"
VEH="${ML_RATCHET_VEHICLE:-$(cd "$SRC/../../../shared/vehicle" && pwd)}"
cd $WORK
export OMP_NUM_THREADS=1
nice -n 5 python3 $SRC/ratchet.py R5_tuned ckpt/r2_dagger.npz '{"onoff":"ecms","preview_s":0,"s_scale":0.995,"s_bias":0.017,"Kp":8.82,"guard_k":0.112,"min_on_s":4.87,"min_off_s":1.40,"pon_w":0.276}' "Policy-search fine-tune (black-box (1+3)-ES, 10 iterations, on 60 TRAINING-distribution trips) of the head knobs: s scale/bias, SOC-feedback Kp, SOC guard, min on/off times, on/off-head prior weight; objective mean %DP - 0.002 x chatter/km - CS penalty" > logs/r5.log 2>&1
nice -n 8 python3 $SRC/evaluate.py base exam > logs/base_exam.log 2>&1
