#!/bin/sh
# Physics v2: re-take R5-R7 (run from WORK after v2_ladder1.sh)
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1
R="$PY $SRC/ratchet.py"
$PY $SRC/evaluate.py base tune > logs/base_tune.log 2>&1
$PY $SRC/tune.py ckpt/r2_dagger.npz data/tune_r5.json 0 10 > logs/tune_r5.log 2>&1
P5=$($PY -c "import json; b=json.load(open('data/tune_r5.json'))['best']; b['onoff']='ecms'; print(json.dumps(b))")
$R R5_tuned ckpt/r2_dagger.npz "$P5" "Policy-search fine-tune (black-box (1+3)-ES, 10 iterations, on 60 TRAINING-distribution trips) of the head knobs: s scale/bias, SOC-feedback Kp, SOC guard, min on/off times, on/off-head prior weight; objective mean %DP - 0.002 x chatter/km - CS penalty" > logs/r5.log 2>&1
$PY $SRC/gen_bank.py r2 600 13 mix ckpt/r2_dagger.npz 0.2 '{"Kp":8,"onoff":"ecms"}' > logs/bank_r2.log 2>&1
$PY $SRC/train.py r3_dagger2 bank_r0 bank_r1 bank_r2 --epochs 25 --device cuda > logs/train_r3.log 2>&1
$R R6_dagger2 ckpt/r3_dagger2.npz '{"Kp":6,"onoff":"ecms"}' "DAgger round 2: +600 trips driven by the R3 policy (cost-based on/off, Kp 8, beta 0.2), retrained on all 2,000 trips" > logs/r6.log 2>&1
$R R7_dagger2_preview30 ckpt/r3_dagger2.npz '{"Kp":6,"onoff":"ecms","preview_s":30}' "R6 + 30 s speed/grade preview" > logs/r7.log 2>&1
echo DONE > logs/ladder2.done
