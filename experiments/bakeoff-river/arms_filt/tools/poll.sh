#!/bin/bash
# wait up to $1 seconds for a change in harness/registry or arms_tab/STATUS.md "READY for C"; print what changed
R=/home/claude/bakeoff/river/harness/registry
a=$(ls $R | md5sum); b=$(grep -c "READY for C:" /home/claude/bakeoff/river/arms_tab/STATUS.md)
for i in $(seq 1 $(( $1 / 20 ))); do
  sleep 20
  if [ "$(ls $R | md5sum)" != "$a" ] || [ "$(grep -c 'READY for C:' /home/claude/bakeoff/river/arms_tab/STATUS.md)" != "$b" ]; then break; fi
done
date -u; ls $R; grep -i "ready for c" /home/claude/bakeoff/river/arms_tab/STATUS.md | head -3; ls /home/claude/bakeoff/river/harness/results | grep -i exam
