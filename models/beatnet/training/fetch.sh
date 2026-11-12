#!/bin/bash
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/beatnet}"; WORK="${WORK:-$SRC/../work}"
# Download MIT-BIH Arrhythmia (mitdb), Noise Stress Test (nstdb) and Supraventricular Arrhythmia (svdb) from PhysioNet.
set -e
D=$WORK/data
get() { db=$1; ver=$2; f=$3; mkdir -p $D/$db; [ -s $D/$db/$f ] || curl -sSLf -o $D/$db/$f https://physionet.org/files/$db/$ver/$f; }
for r in $(curl -sSLf https://physionet.org/files/mitdb/1.0.0/RECORDS); do for e in hea dat atr; do get mitdb 1.0.0 $r.$e; done; done
for r in bw em ma; do for e in hea dat; do get nstdb 1.0.0 $r.$e; done; done
for r in $(curl -sSLf https://physionet.org/files/svdb/1.0.0/RECORDS); do for e in hea dat atr; do get svdb 1.0.0 $r.$e; done; done
echo done
# QT database (manual waveform boundaries, q1c) for the QRS-width check
for r in $(curl -sSLf https://physionet.org/files/qtdb/1.0.0/RECORDS); do
  [ -s $D/qtdb/$r.q1c ] || curl -sSLf -o $D/qtdb/$r.q1c https://physionet.org/files/qtdb/1.0.0/$r.q1c || { rm -f $D/qtdb/$r.q1c; continue; }
  for e in hea dat; do get qtdb 1.0.0 $r.$e; done
done
echo qtdb done
