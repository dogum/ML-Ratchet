#!/bin/sh
# Physics v2: new rungs for the V2_PLAN targets (run from WORK after v2_ladder2.sh). Each builds on incumbent.py.
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1
R="$PY $SRC/ratchet.py"
INC() { $PY $SRC/incumbent.py "$@"; }
# R8: charge holding as RoadSim's A-ECMS holds it (its SOC feedback gains)
$R R8_aecms_gains "$(INC ckpt)" "$(INC params '{"Kp":12,"Ki":0.02}')" "SOC feedback with RoadSim A-ECMS's own gains (Kp 12, Ki 0.02 1/s, integral clipped at 30) instead of Kp 6: v2 dev charge holding fell to 87-91 %" > logs/r8.log 2>&1
# R9: re-teach on the calibrated twins' hardware classes: mild P1 (CR-Z-type twins) added to the training fleet
$PY $SRC/gen_bank.py tw 600 31 mix "$(INC ckpt)" 0.3 "$(INC params '{"fleet":"gen+mild"}')" > logs/bank_tw.log 2>&1
$PY $SRC/train.py r9_twin $(INC banks) bank_tw --epochs 25 --device cuda > logs/train_r9.log 2>&1
$R R9_twin_classes ckpt/r9_twin.npz "$(INC params)" "Re-teach with the calibrated twins' hardware: 40 mild P1 hybrids built with RoadSim's CR-Z twin recipe join the 400 generated training vehicles; +600 DAgger trips driven by the incumbent (beta 0.3), retrained on all banks" > logs/r9.log 2>&1
# R10: causal proxy of DP with the physics' minimum on/off times: the teacher is the timed DP (8 s / 3 s)
$PY $SRC/gen_bank.py t0 800 21 teacher - 0 '{"timed":true,"fleet":"gen+mild"}' > logs/bank_t0.log 2>&1
$PY $SRC/train.py t0_bc bank_t0 --epochs 30 --device cuda > logs/train_t0.log 2>&1
$PY $SRC/gen_bank.py t1 600 22 mix ckpt/t0_bc.npz 0.3 "$(INC params '{"timed":true,"fleet":"gen+mild"}')" > logs/bank_t1.log 2>&1
$PY $SRC/train.py r10_timed bank_t0 bank_t1 --epochs 25 --device cuda > logs/train_r10.log 2>&1
$R R10_timed_teacher ckpt/r10_timed.npz "$(INC params)" "Teacher = DP with the physics' minimum on/off times (8 s / 3 s, roadsim dp_min_on_off), the platform every causal strategy runs on: 800 teacher trips + 600 DAgger trips (beta 0.3), fleet with the mild class; same decision rule as the incumbent" > logs/r10.log 2>&1
# R11: drivability term: a virtual start cost in the decision (20 kJ, RoadSim v2's start_penalty_J, as its ECMS uses)
$R R11_start_pen "$(INC ckpt)" "$(INC params '{"start_pen":20000}')" "Drivability term: virtual start cost 20 kJ per engine start in the decision (RoadSim v2's start_penalty_J), on top of the 8 s / 3 s minimum on/off times" > logs/r11.log 2>&1
# R12: distance-to-go plug-in plan (nav known)
$R R12_dtg "$(INC ckpt)" "$(INC params '{"dtg":true,"Kp_dtg":6}')" "Distance-to-go plug-in plan: with a known trip length (nav), the SOC reference falls linearly with distance from the start SOC to the CS level at the trip end, as the DP teacher depletes over the known length; SOC feedback Kp 6 both ways" > logs/r12.log 2>&1
echo DONE > logs/ladder3.done
