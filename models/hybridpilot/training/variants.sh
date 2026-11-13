#!/bin/sh
# usage: variants.sh SET CKPT LOG  then lines "tag|prev|json"
SET=$1; CK=$2; LOG=$3; shift 3
for v in "$@"; do
  tag=$(echo "$v" | cut -d'|' -f1); prev=$(echo "$v" | cut -d'|' -f2); js=$(echo "$v" | cut -d'|' -f3)
  echo "== $tag prev=$prev $js" >> $LOG
  OMP_NUM_THREADS=1 nice -n 5 python3 quick.py $SET $CK $prev "$js" _$tag 2>&1 | grep -v Warn >> $LOG
done
echo DONE >> $LOG
