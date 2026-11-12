#!/bin/sh
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/sundoctor}"; WORK="${WORK:-$SRC/../work}"
# every real-data exam, the export and the page for the released checkpoint ($1, e.g. ckpt/s7_v2_polish.pt)
set -e
cd $WORK
C=$1
python3 $SRC/episodes2.py $C > logs_fin_ep.txt 2>&1
python3 $SRC/episodes2.py $C --home > logs_fin_eph.txt 2>&1
python3 $SRC/inject2.py $C > logs_fin_inj.txt 2>&1
python3 $SRC/inject2.py $C --home > logs_fin_injh.txt 2>&1
python3 $SRC/rdtools_cmp.py $C > logs_fin_rd.txt 2>&1
python3 $SRC/export2.py $C release/sundoctor_weights.npz > logs_fin_export.txt 2>&1
python3 $SRC/export_js.py release/sundoctor_weights.npz web/weights.json >> logs_fin_export.txt 2>&1
python3 $SRC/build_web_data2.py $C > logs_fin_web.txt 2>&1
python3 $SRC/build_page.py >> logs_fin_web.txt 2>&1
python3 $SRC/js_parity.py > logs_fin_parity.txt 2>&1
python3 $SRC/model_card2.py > logs_fin_card.txt 2>&1
echo FINISHED > logs_fin_done.txt
