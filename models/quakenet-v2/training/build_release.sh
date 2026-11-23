#!/bin/bash
# Assemble the release zip: release files at the top, training/ (scripts as run), and for_repo/models/quakenet-v2/ in the repo's layout.
set -e
H=$(cd "$(dirname "$0")" && python3 -c "from _paths import H; print(H.rstrip('/').rstrip(chr(92)))"); R=$H/dist/quakenet-v2-v1.0; rm -rf $H/dist; mkdir -p $R/training/v12 $R/results
cp $H/release/{quakenet2.py,quakenet.py,pipeline.py,quakenet_weights.npz,quakenet2_weights.npz,tt_ak135.npz,demo.py} $R/
cp -r $H/release/examples $R/
cp $H/docs/{MODEL_CARD.md,README.md,model.yaml,ratchet.yaml,NOTEBOOK.md,BESIDE_V1.2.md,DATA_LICENSES.md} $R/
cp $H/src/*.py $H/src/*.sh $R/training/; cp $H/src/v12/*.py $R/training/v12/
cp $H/splits.json $H/ratchet.json $H/exam_results.json $H/physics_check.json $R/results/
cp $H/data/{sigma_factors.json,twin_compare.json,calib_picks.json,js_parity.json,export_check.json,station_order.json} $R/results/
# repo layout
F=$R/for_repo/models/quakenet-v2; mkdir -p $F/training/v12 $F/examples $F/lab $F/web
cp $R/{quakenet2.py,quakenet.py,pipeline.py,quakenet_weights.npz,quakenet2_weights.npz,tt_ak135.npz,demo.py,MODEL_CARD.md,model.yaml,ratchet.yaml,NOTEBOOK.md,BESIDE_V1.2.md,DATA_LICENSES.md} $F/
cp -r $R/examples/* $F/examples/; cp $R/training/*.py $R/training/*.sh $F/training/; cp $R/training/v12/*.py $F/training/v12/
cp $H/splits.json $F/splits.json; cp $H/ratchet.json $F/training/ratchet.json
cp $H/docs/_paths.py $F/training/_paths.py; mkdir -p $R/for_repo/tests/golden; cp $H/docs/golden_quakenet-v2.txt $R/for_repo/tests/golden/quakenet-v2.txt
cp $H/web/quakenet2.html $F/lab/index.html; cp $H/web/qnet2.js $H/web/page.html $H/src/export_web.py $F/web/
# portable paths: the sandbox ran sed here to swap its absolute paths for _paths imports; the scripts in this folder are already the swapped copies.
cd $H/dist && mkdir -p $H/out && rm -f $H/out/quakenet-v2-v1.0.zip && zip -qr $H/out/quakenet-v2-v1.0.zip quakenet-v2-v1.0 && ls -la $H/out/quakenet-v2-v1.0.zip
