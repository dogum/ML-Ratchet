#!/bin/bash
# Round 2: P rungs on the phevcold banks (PREREG A3): P2 transformer members 0-2 in parallel; P3 MLP (5 nets) then
# P1 GRU members 0-4 in one queue. Waits for the last _pc bank. Resumable (finished members are skipped).
cd "$(dirname "$0")"
PY=C:/Developer/ML-ratchet/.venv/Scripts/python.exe
until [ -f work/banks/train103_pc/MANIFEST.json ]; do sleep 10; done
for m in 0 1 2; do
  $PY -X utf8 trace/train_ens.py p2_transformer $m > work/logs/p2_tf_m$m.log 2>&1 &
done
(
  for s in 0 1 2 3 4; do $PY -X utf8 table/train_mlp3.py p3_mlp $s train101_pc train102_pc train103_pc; done > work/logs/p3_mlp.log 2>&1
  for m in 0 1 2 3 4; do $PY -X utf8 trace/train_ens.py p1_gru $m > work/logs/p1_gru_m$m.log 2>&1; done
) &
wait
echo P_DONE > work/logs/p_done.flag
