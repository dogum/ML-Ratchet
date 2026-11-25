#!/bin/bash
# usage: evalq.sh <arm>:<rung> ...   harness evaluate of existing candidates (all 800 sim val runs + dev gate)
source /home/claude/bakeoff/river/harness/env.sh
export NUMBA_CACHE_DIR=/home/claude/bakeoff/river/arms_seq/.numba
SEQ=/home/claude/bakeoff/river/arms_seq
for x in "$@"; do a=${x%%:*}; r=${x##*:}
  python3 /home/claude/bakeoff/river/harness/evaluate.py $SEQ/$a/work/cand_$r --tag $r --out $SEQ/$a/work/eval > $SEQ/logs/${a}_${r}_eval.log 2>&1
done
