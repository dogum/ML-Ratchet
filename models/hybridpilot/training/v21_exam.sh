#!/bin/sh
# v2.1 pass: the final exam on physics v3.1, once, after the release choice was committed (d9fd0fd). Run from WORK with
# HP_PHYS=roadsim-physics-v3.1. Every baseline, the v2.1 release, v2.0 (its network and settings) and the v1 release are
# scored on the same exam and physics.
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1 HP_WORKERS=${HP_WORKERS:-3}
INC() { $PY $SRC/incumbent.py "$@"; }
$PY $SRC/sets.py exam > logs/set_exam_p3.1.log 2>&1
$PY $SRC/evaluate.py base exam > logs/base_exam_p3.1.log 2>&1
$PY $SRC/run_pol.py exam HPv21 "$(INC ckpt)" "$(INC params)" > logs/pol_exam_v21_p3.1.log 2>&1
$PY $SRC/run_pol.py exam HPv20 ckpt/r2_dagger.npz '{"Kp":12,"onoff":"ecms","Ki":0.02}' > logs/pol_exam_v20_p3.1.log 2>&1
$PY $SRC/run_pol.py exam HPv1 ../weights/hybridpilot_v1.npz '{"Kp":6,"onoff":"ecms"}' > logs/pol_exam_v1_p3.1.log 2>&1
$PY $SRC/exam_report.py HPv21 HPv20 HPv1 > logs/exam_report_p3.1.log 2>&1
$PY $SRC/simcheck.py exam "$(INC ckpt)" "$(INC params)" 3 _HPv21 > logs/simcheck_exam_v21_p3.1.log 2>&1
$PY $SRC/simcheck.py dev "$(INC ckpt)" "$(INC params)" 12 _HPv21 > logs/simcheck_dev_v21_p3.1.log 2>&1
echo DONE > logs/exam_p3.1.done
