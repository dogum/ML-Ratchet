#!/bin/bash
# Re-run every BuildingSim check in order (single thread, niced). Long ones: resstock (~10 min), comstock (~15 min).
# Logs and JSONs go to $ML_RATCHET_BUILDING/checks/out (default shared/building/checks/out, git-ignored); the shipped
# JSONs next to this file are not touched (BUILDINGSIM_PUBLISH=1 writes over them). PYTHON overrides the interpreter.
set -e
cd "$(dirname "$0")"
OUT="${ML_RATCHET_BUILDING:-$(cd .. && pwd)}/checks/out"
PY="${PYTHON:-python3}"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1
for s in check_components.py check_analytic.py check_bestest.py check_faults.py check_controller.py check_generator.py \
         check_speed.py check_bdg2.py check_multinode.py; do
  [ -f "$s" ] && echo "== $s" && nice -n 5 "$PY" "$s" > "$OUT/${s%.py}.log" 2>&1 && tail -1 "$OUT/${s%.py}.log"
done
# Phase 2 wrappers (opt-in, BuildingSim 1.0.1): ports vs the models' own wrappers, and the dev twins with flags on
echo "== check_phase2.py" && nice -n 5 "$PY" check_phase2.py > "$OUT/check_phase2.log" 2>&1 && tail -1 "$OUT/check_phase2.log"
for v in w2 w2w3 lc; do
  echo "== check_bdg2.py $v" && nice -n 5 "$PY" check_bdg2.py $v > "$OUT/check_bdg2__$v.log" 2>&1 && tail -1 "$OUT/check_bdg2__$v.log"
done
rm -f "$OUT"/resstock/*.json && nice -n 5 "$PY" check_resstock.py > "$OUT/resstock.log" 2>&1 && tail -1 "$OUT/resstock.log"
rm -rf "$OUT/comstock" "$OUT/comstock_heldout12" && nice -n 5 "$PY" check_comstock.py 11 > "$OUT/comstock_cal.log" 2>&1 && \
  nice -n 5 "$PY" check_comstock.py 12 > "$OUT/comstock_held.log" 2>&1 && tail -1 "$OUT/comstock_held.log"
