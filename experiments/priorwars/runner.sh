#!/bin/bash
# Single sequential runner (Amendment A1): ONE TabPFN process at a time, 2 GiB address-space cap, nice 10, one thread.
# Queue: $W/runq.txt lines '<model> <stage>' (stage: tune_dev | tune_sim | prereg | exam_predict | score).
# Done/failed lines -> $W/runq_done.txt. Waits while the shared memory group is above 4.2 GB before each launch.
# Repository paths (the sandbox ran from priorwars/scripts with ../work, ../logs, ../venv and ../PREREG.md):
#   W = pw_paths.WORK; logs in $W/logs; a rerun's PREREG.md is $W/PREREG.md; PY = the TabPFN venv's python ($PW_PYTHON).
# The memory check read the sandbox's cgroup-v1 file; elsewhere set PW_MEMFILE to a usage file, or the check is skipped.
cd "$(dirname "$0")"
PY=${PW_PYTHON:-python3}
W=$($PY -c 'import pw_paths; print(pw_paths.WORK)')
mkdir -p $W/logs
Q=$W/runq.txt; DONE=$W/runq_done.txt
CG=${PW_MEMFILE:-$(ls /sys/fs/cgroup/memory/process_api/*/claude-code-bash/memory.usage_in_bytes 2>/dev/null | head -1)}
[ -n "$CG" ] || { CG=$W/.mem0; echo 0 > $CG; }
echo $$ > $W/logs/runner.pid
idle=0
while [ $idle -lt 60 ]; do
  job=$(grep -m1 -v '^\s*$' $Q 2>/dev/null)
  if [ -z "$job" ]; then sleep 30; idle=$((idle+1)); continue; fi
  idle=0
  source ./jobenv.sh
  while [ $(cat $CG) -gt $MEMWAIT ]; do echo "$(date +%T) waiting: shared memory $(($(cat $CG)/1000000)) MB" >> $W/logs/runner.log; sleep 60; done
  m=${job% *}; s=${job#* }
  t0=$(date +%s)
  echo "$(date +%T) start $job (shared memory $(($(cat $CG)/1000000)) MB)" >> $W/logs/runner.log
  if [ "$s" = prereg ]; then
    n=$(grep -cE '^## [0-9]+\.' $W/PREREG.md)
    { nice -n 10 prlimit --as=$CAP $PY prereg_section.py $m $n > $W/$m/prereg_section.md \
      && printf '\n' >> $W/PREREG.md && cat $W/$m/prereg_section.md >> $W/PREREG.md \
      && $PY freeze.py $m; } >> $W/logs/${m}_prereg.log 2>&1
    rc=$?
  else
    nice -n 10 prlimit --as=$CAP $PY harness.py $m $s >> $W/logs/${m}_${s}.log 2>&1
    rc=$?
  fi
  # drop the job from the head of the queue (the file may have been appended to meanwhile)
  # on failure, the model's later stages are moved to $W/runq_held.txt (re-queued by hand after a fix)
  python3 - "$Q" "$job" "$rc" <<'PY'
import sys
q, job, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
L = [l.rstrip('\n') for l in open(q) if l.strip()]
if L and L[0].strip() == job.strip(): L = L[1:]
if rc != 0:
    m = job.split()[0]
    held = [l for l in L if l.split()[0] == m]
    L = [l for l in L if l.split()[0] != m]
    with open(q.replace('runq.txt', 'runq_held.txt'), 'a') as f:
        f.write(''.join(l + '\n' for l in [job] + held))
open(q, 'w').write(''.join(l + '\n' for l in L))
PY
  echo "$(date +%T) $job rc=$rc $(( $(date +%s) - t0 ))s" >> $DONE
  echo "$(date +%T) end $job rc=$rc" >> $W/logs/runner.log
done
