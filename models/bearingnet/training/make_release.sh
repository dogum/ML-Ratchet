#!/bin/sh
# portable paths: code in training/, outputs in models/<name>/work (or $ML_RATCHET_WORK/<name>)
SRC="$(cd "$(dirname "$0")" && pwd)"
LAB="$SRC/../lab/src"
WORK="${ML_RATCHET_WORK:+$ML_RATCHET_WORK/bearingnet}"; WORK="${WORK:-$SRC/../work}"
# assemble release/ and zip it; then test the demo from an unzipped copy
set -e
SP=$WORK/scratch
cd $WORK
cp $SRC/features.py release/pipeline.py
mkdir -p release/training
cp $SRC/kin.py $SRC/bearsim.py $SRC/features.py $SRC/realdata.py $SRC/fetch_cwru.py $SRC/fetch_pb.py $SRC/bn_simbank.py $SRC/bn_realbank.py $SRC/bn_train.py $SRC/model.py $SRC/bnet.py \
   $SRC/ratchet.py $SRC/classical.py $SRC/exam.py $SRC/export.py $SRC/physics_check.py $SRC/calib_dev.py $SRC/fig_twin.py $SRC/diag_dev.py $SRC/webdata.py $SRC/calib_lines.py $SRC/copy_text.py $SRC/build_page.py $SRC/py_ref.py $SRC/model.py $SRC/bn_chain*.sh $SRC/make_release.sh release/training/
mkdir -p release/training/web && cp web/page.html web/bnet.js web/lab.js web/test_js.js release/training/web/
cp ratchet.json release/training/ratchet.json
cp checks/*.json release/training/
rm -rf release/__pycache__ release/training/__pycache__
mkdir -p $WORK/dist
rm -f $WORK/dist/bearingnet-v1.0.zip
cd $WORK && rm -rf $SP/bnrel && mkdir -p $SP/bnrel && cp -r release $SP/bnrel/bearingnet-v1.0
cd $SP/bnrel && zip -qr $WORK/dist/bearingnet-v1.0.zip bearingnet-v1.0
rm -rf $SP/bnrel_test && mkdir -p $SP/bnrel_test && cd $SP/bnrel_test && unzip -q $WORK/dist/bearingnet-v1.0.zip && cd bearingnet-v1.0 && OMP_NUM_THREADS=1 python3 -I demo.py
ls -la $WORK/dist/bearingnet-v1.0.zip
