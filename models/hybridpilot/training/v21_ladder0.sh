#!/bin/sh
# v2.1 pass (15 Nov 2026): physics changed (v2 -> v3.1), so every v2 decision is re-taken in order on the v3.1 frozen
# set and dev gate. Each rung keeps its v2 network and settings (trained on physics-v2 banks): a re-score, no retraining.
# Run from WORK with HP_PHYS=roadsim-physics-v3.1 after v21_stage1.sh. Log: WORK/ratchet_v2_p3.1.json
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1 HP_WORKERS=${HP_WORKERS:-3}
$PY - "$SRC" <<'PYEOF' > logs/ladder0_p3.1.cmds
import json, sys, os
R = json.load(open('ratchet_v2.json'))
for r in R:
    note = 'Re-scored on physics v3.1 (v2 network and settings, no retraining). v2 rung: ' + r['note']
    print(json.dumps([r['rung'], r['ckpt'] or '-', json.dumps(r['params']), note]))
PYEOF
while IFS= read -r line; do
  N=$(echo "$line" | $PY -c "import json,sys; print(json.loads(sys.stdin.read())[0])")
  echo "$line" | $PY -c "
import json,sys,subprocess
a=json.loads(sys.stdin.read())
import os; sys.exit(subprocess.call([sys.executable, os.path.join(os.path.dirname(os.getcwd()), 'training', 'ratchet.py')]+a))" > logs/r_${N}_p3.1.log 2>&1
done < logs/ladder0_p3.1.cmds
echo DONE > logs/ladder0_p3.1.done
