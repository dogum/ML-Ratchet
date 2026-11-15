#!/bin/sh
# v2.1 pass (15 Nov 2026), stage 1 on RoadSim physics v3.1: case sets (frozen, dev, tune, probe) and their baseline
# caches. Run from WORK with HP_PHYS=roadsim-physics-v3.1 (sets, twins and caches carry the physics in their names).
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1 HP_WORKERS=${HP_WORKERS:-3}
$PY $SRC/sets.py frozen dev tune probe > logs/sets_p3.1.log 2>&1
for S in dev frozen tune probe; do
  $PY $SRC/evaluate.py base $S > logs/base_${S}_p3.1.log 2>&1
done
echo DONE > logs/stage1_p3.1.done
