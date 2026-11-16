#!/bin/sh
# 15 Nov 2026 scoring correction (no new version, chooses nothing): build_exam draws routes with replacement, so the
# exam set holds 432 cases but 430 case_ids; results were keyed by case_id and the last finisher won. Cases are now keyed
# uniquely (sets.assign_keys). The two ambiguous baseline entries were removed from each base cache; this script scores
# the four duplicate cases' baselines and re-scores every exam policy on all 432 trips, physics v3.1 and v2.
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1 HP_WORKERS=${HP_WORKERS:-3}
V21='{"Kp":12,"onoff":"ecms","Ki":0.02,"fb_dead":0.03,"s_prior":{"bus_hev_series":[1,2.8],"bus_hev_parallel_p2":[1,2.6],"mild":[1,2.8]}}'
V22='{"Kp":12,"onoff":"ecms","Ki":0.02,"fb_dead":0.03,"s_prior":{"bus_hev_series":[1,2.8],"bus_hev_parallel_p2":[1,2.6],"mild":[1,2.8]},"cls_fb":{"phev":{"fb_dead":0.0,"Kp":24},"bus_hev_series":{"Kp":3,"Ki":0.0},"bus_hev_parallel_p2":{"Kp":3,"Ki":0.0}},"phev_cs_s":{"phev_powersplit":2.42,"phev_series":2.7}}'
export HP_PHYS=roadsim-physics-v3.1
$PY $SRC/evaluate.py base exam > logs/dupfix_base_exam_p3.1.log 2>&1
$PY $SRC/run_pol.py exam HPv22 ckpt/r2_dagger.npz "$V22" > logs/dupfix_pol_v22_p3.1.log 2>&1
$PY $SRC/run_pol.py exam HPv21r2 ckpt/r2_dagger.npz "$V21" > logs/dupfix_pol_v21_p3.1.log 2>&1
$PY $SRC/run_pol.py exam HPv20 ckpt/r2_dagger.npz '{"Kp":12,"onoff":"ecms","Ki":0.02}' > logs/dupfix_pol_v20_p3.1.log 2>&1
$PY $SRC/run_pol.py exam HPv1 ../weights/hybridpilot_v1.npz '{"Kp":6,"onoff":"ecms"}' > logs/dupfix_pol_v1_p3.1.log 2>&1
$PY $SRC/exam_report.py HPv22 HPv21r2 HPv20 HPv1 > logs/dupfix_report_p3.1.log 2>&1
export HP_PHYS=roadsim-physics-v2
$PY $SRC/evaluate.py base exam > logs/dupfix_base_exam_p2.log 2>&1
$PY $SRC/run_pol.py exam HPv2 ckpt/r2_dagger.npz '{"Kp":12,"onoff":"ecms","Ki":0.02}' > logs/dupfix_pol_v2_p2.log 2>&1
$PY $SRC/run_pol.py exam HPv1 ../weights/hybridpilot_v1.npz '{"Kp":6,"onoff":"ecms"}' > logs/dupfix_pol_v1_p2.log 2>&1
$PY $SRC/exam_report.py HPv2 HPv1 > logs/dupfix_report_p2.log 2>&1
echo DONE > logs/dupfix.done
