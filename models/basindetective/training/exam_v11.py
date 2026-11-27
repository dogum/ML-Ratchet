"""The one exam of BasinDetective v1.1 (R6s, promoted by the ratchet on 27 Nov 2026), on the 619 CAMELS-US exam basins.

Runs the shipped exam path unchanged (realfeats.build('exam') and exam.main) with the stand-ins rescore.py documents
(WaterData's daily GR4J simulations re-made from the shipped parameters; the NWIS peaks table from the data root).
Run once: it refuses if WORK/exam_v1.1/exam.json exists, and every start is appended to WORK/exam_v1.1/exam.started
with the model's sha256. Outputs: WORK/exam_v1.1/{exam.json, exam_pred.parquet}; nothing is chosen from them.

python exam_v11.py <v1.1 model npz>
"""
import common  # noqa: F401
import datetime
import hashlib
import json
import os
import shutil
import sys

from common import D
from _paths import WATER_PROC
import waterdata as wd

OUT = f'{D}/exam_v1.1'


def main(model):
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f'{OUT}/exam.json'):
        raise SystemExit('exam already run for v1.1: exam_v1.1/exam.json exists (the exam is run once)')
    sha = hashlib.sha256(open(model, 'rb').read()).hexdigest()
    with open(f'{OUT}/exam.started', 'a') as fh:
        fh.write(f'{datetime.datetime.utcnow().isoformat()}Z model={model} sha256={sha}\n')
    import rescore
    import realfeats
    import exam
    rescore._sklearn_shim()
    realfeats.B = rescore.stage_baselines()
    realfeats.WATER_PROC, peaks_src = rescore.stage_peaks()
    ids = wd.basins('camels', 'exam')
    split = json.load(open(f'{WATER_PROC}/splits.json'))
    assert sorted(ids) == sorted(split['camels_us']['exam']) and len(ids) == 619
    mark = f'{D}/feats/real_exam.peaks'
    cached = open(mark).read().strip() if os.path.exists(mark) else 'empty stand-in'
    if not os.path.exists(f'{D}/feats/real_exam.npz') or cached != peaks_src:
        realfeats.build('exam')
        open(mark, 'w').write(peaks_src)
    exam.main(model)                                  # -> WORK/exam.json, WORK/feats/exam_pred.parquet
    shutil.move(f'{D}/exam.json', f'{OUT}/exam.json')
    shutil.move(f'{D}/feats/exam_pred.parquet', f'{OUT}/exam_pred.parquet')
    print('written', OUT, 'sha256', sha, 'peaks', peaks_src)


if __name__ == '__main__':
    main(sys.argv[1])
