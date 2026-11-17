#!/bin/sh
# build_release.sh WEIGHTS.npz : assemble WORK/release/ and zip it to WORK/thermopilot-v1.0.zip
# (WORK = $ML_RATCHET_WORK/thermopilot or models/thermopilot/work; in the sandbox this wrote to a user outputs folder).
# Repo layout: tp/, scripts/, configs/, tp.js, page_src.html, build_page.py are all in training/.
set -e
cd "$(dirname "$0")/.."
W=$1
WORK=${ML_RATCHET_WORK:+$ML_RATCHET_WORK/thermopilot}; WORK=${WORK:-../work}
mkdir -p $WORK/release
cp ../thermopilot_core.py ../demo.py ../pipeline.py ../MODEL_CARD.md $WORK/release/
cp $W $WORK/release/thermopilot_v1.npz
python3 scripts/export_demo.py $W $WORK/release/demo_episodes.npz
rm -rf $WORK/release/training; mkdir -p $WORK/release/training
cp -r tp scripts configs $WORK/release/training/
cp tp.js page_src.html build_page.py _paths.py $WORK/release/training/
cp $WORK/ratchet.json $WORK/release/training/ 2>/dev/null || cp ratchet.json $WORK/release/training/
mkdir -p $WORK/release/training/results
cp $WORK/results/exam_*.json $WORK/release/training/results/ 2>/dev/null || true
cp -r $WORK/results/wpuq $WORK/release/training/results/ 2>/dev/null || true
find $WORK/release/training -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
rm -f $WORK/thermopilot-v1.0.zip
cd $WORK/release && zip -qr ../thermopilot-v1.0.zip . -x '*.pyc' && cd -
ls -la $WORK/thermopilot-v1.0.zip
