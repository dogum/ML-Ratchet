#!/bin/sh
# Physics v2: preview and a head re-tune on the new incumbent (run from WORK after v2_ladder4.sh)
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1
R="$PY $SRC/ratchet.py"
INC() { $PY $SRC/incumbent.py "$@"; }
$R R14_preview30 "$(INC ckpt)" "$(INC params '{"preview_s":30}')" "Incumbent + 30 s speed/grade preview (as from navigation); v1's R4 idea re-tried on the new incumbent" > logs/r14.log 2>&1
PV=$($PY -c "import json,sys; print(json.loads(sys.argv[1]).get('preview_s', 0))" "$(INC params)")
$PY $SRC/tune.py "$(INC ckpt)" data/tune_r15.json $PV 10 "$(INC params)" > logs/tune_r15.log 2>&1
P15=$($PY -c "import json,sys; b=json.load(open('data/tune_r15.json'))['best']; p=json.loads(sys.argv[1]); p.update(b); print(json.dumps(p))" "$(INC params)")
$R R15_retuned "$(INC ckpt)" "$P15" "Policy-search re-tune of the head knobs (as R5, plus the integral gain Ki) starting from the incumbent, on the 60 TRAINING-distribution tune trips; objective mean %DP - 0.002 x chatter/km - CS penalty" > logs/r15.log 2>&1
echo DONE > logs/ladder5.done
