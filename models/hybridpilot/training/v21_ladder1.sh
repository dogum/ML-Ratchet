#!/bin/sh
# v2.1 pass (15 Nov 2026), new rungs on physics v3.1 after the re-take (v21_ladder0.sh). Each builds on incumbent.py.
# Decision-rule settings were chosen on the training-distribution probe set (sets.build_probe; v21_explore.py), never
# on dev or exam. Run from WORK with HP_PHYS=roadsim-physics-v3.1.
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1 HP_WORKERS=${HP_WORKERS:-3}
R="$PY $SRC/ratchet.py"
INC() { $PY $SRC/incumbent.py "$@"; }
$R R16_deadband "$(INC ckpt)" "$(INC params '{"fb_dead":0.03}')" "Charge-sustaining term with a dead band: the proportional SOC feedback (Kp 12) acts only on the part of the SOC error beyond +-0.03 (the integral term is unchanged), so short swings inside the window no longer move s. Dev showed the strong feedback costing fuel on buses (series bus 95.3 % vs constant-s ECMS 98.3 %) and mild hybrids" > logs/r16_p3.1.log 2>&1
$R R17_class_priors "$(INC ckpt)" "$(INC params '{"fb_dead":0.03,"s_prior":{"bus_hev_series":[1,2.8],"bus_hev_parallel_p2":[1,2.6],"mild":[1,2.8]}}')" "Per-class prior of the equivalence factor where the network is weakest: buses and mild P1 hybrids use a constant s0 (series bus 2.8, P2 bus 2.6, mild 2.8, fitted on the 30 probe trips of those classes, training distribution) in place of the network's s, with the same SOC feedback; every other class keeps the network" > logs/r17_p3.1.log 2>&1
$R R18_reteach_v31 ckpt/p31_dagger.npz "$(INC params)" "Re-teach on physics v3.1: the v2.0 recipe (800 DP-teacher trips with exploration, behaviour cloning, +600 DAgger trips driven by that net with beta 0.3, retrain on both, 25 epochs on the GPU) with every bank simulated on v3.1; the incumbent's decision rule" > logs/r18_p3.1.log 2>&1
echo DONE > logs/ladder1_p3.1.done
