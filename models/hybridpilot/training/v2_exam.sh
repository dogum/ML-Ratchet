#!/bin/sh
# Physics v2: the final exam, once, after the release choice is committed (run from WORK).
# HPv2 = the release (incumbent.py); HPv1 = the v1 release (its weights and decision settings) re-scored on the same exam.
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1
INC() { $PY $SRC/incumbent.py "$@"; }
$PY $SRC/evaluate.py base exam > logs/base_exam.log 2>&1
$PY $SRC/run_pol.py exam HPv2 "$(INC ckpt)" "$(INC params)" > logs/pol_exam_v2.log 2>&1
$PY $SRC/run_pol.py exam HPv1 ../weights/hybridpilot_v1.npz '{"Kp":6,"onoff":"ecms"}' > logs/pol_exam_v1.log 2>&1
$PY $SRC/exam_report.py HPv2 HPv1 > logs/exam_report.log 2>&1
$PY $SRC/simcheck.py exam "$(INC ckpt)" "$(INC params)" 3 _HPv2 > logs/simcheck_exam.log 2>&1
echo DONE > logs/exam.done
