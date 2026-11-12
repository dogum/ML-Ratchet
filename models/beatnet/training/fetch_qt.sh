#!/bin/bash
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/beatnet}"; WORK="${WORK:-$SRC/../work}"
D=$WORK/data; mkdir -p $D/qtdb
get() { db=$1; ver=$2; f=$3; mkdir -p $D/$db; [ -s $D/$db/$f ] || curl -sSLf -o $D/$db/$f https://physionet.org/files/$db/$ver/$f; }
for r in $(curl -sSLf https://physionet.org/files/qtdb/1.0.0/RECORDS); do
  [ -s $D/qtdb/$r.q1c ] || curl -sSLf -o $D/qtdb/$r.q1c https://physionet.org/files/qtdb/1.0.0/$r.q1c || { rm -f $D/qtdb/$r.q1c; continue; }
  for e in hea dat; do get qtdb 1.0.0 $r.$e; done
done
echo qtdb done
