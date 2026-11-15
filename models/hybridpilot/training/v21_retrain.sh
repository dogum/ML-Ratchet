#!/bin/sh
# v2.1 pass (15 Nov 2026): re-teach on physics v3.1. The v2.0 recipe (800 DP-teacher trips with exploration bursts,
# behaviour cloning, +600 DAgger trips driven by that net, retrain on both) with banks simulated on v3.1 (generator
# hardware v3, run-on, converter rule). Run from WORK with HP_PHYS=roadsim-physics-v3.1 after v21_ladder0.sh.
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1 HP_WORKERS=${HP_WORKERS:-3}
$PY $SRC/gen_bank.py p31r0 800 11 teacher - 0 > logs/bank_p31r0.log 2>&1
$PY $SRC/train.py p31_bc bank_p31r0 --epochs 30 --device cuda > logs/train_p31_bc.log 2>&1
$PY $SRC/gen_bank.py p31r1 600 12 mix ckpt/p31_bc.npz 0.3 '{"Kp":0}' > logs/bank_p31r1.log 2>&1
$PY $SRC/train.py p31_dagger bank_p31r0 bank_p31r1 --epochs 25 --device cuda > logs/train_p31_dagger.log 2>&1
echo DONE > logs/retrain_p3.1.done
