#!/bin/sh
# Round 2: the final exam on physics v3.1, once, after the v2.2 release choice was committed (48e9985). Run from WORK with
# HP_PHYS=roadsim-physics-v3.1. Same exam set and baseline cache as v2.1; v2.1 is re-scored alongside (its params are
# R17's), and v2.0 / v1 re-scores from the v2.1 exam are reused.
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1 HP_WORKERS=${HP_WORKERS:-3}
INC() { $PY $SRC/incumbent.py "$@"; }
V21='{"Kp":12,"onoff":"ecms","Ki":0.02,"fb_dead":0.03,"s_prior":{"bus_hev_series":[1,2.8],"bus_hev_parallel_p2":[1,2.6],"mild":[1,2.8]}}'
$PY $SRC/run_pol.py exam HPv22 "$(INC ckpt)" "$(INC params)" > logs/pol_exam_v22_p3.1.log 2>&1
$PY $SRC/run_pol.py exam HPv21r2 ckpt/r2_dagger.npz "$V21" > logs/pol_exam_v21r2_p3.1.log 2>&1
$PY $SRC/exam_report.py HPv22 HPv21r2 HPv20 HPv1 > logs/exam_report_v22_p3.1.log 2>&1
$PY $SRC/simcheck.py exam "$(INC ckpt)" "$(INC params)" 3 _HPv22 > logs/simcheck_exam_v22_p3.1.log 2>&1
echo DONE > logs/exam_v22_p3.1.done
