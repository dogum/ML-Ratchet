"""Page data: weights (float32 base64), exam runs (raw day arrays rounded, labels, APAR verdict, Python prediction for
the JS check), lab days, exam summary and ratchet. Usage: python3 webdata.py <model> -> web/data.json, web/weights.json"""
import os
import sys
import json
import base64
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # models/faultfinder (faultfinder.py)
from common import ROOT, CLASSES, RAW, load_real_runs, EXAM2CLASS  # noqa: E402
import faultfinder as ffr  # noqa: E402

DEC = {'T_oa': 2, 'T_ra': 2, 'T_ma': 2, 'T_sa': 2, 'T_sa_sp': 2, 'T_zone': 2, 'damper_oa': 3, 'valve_cc': 3,
       'valve_hc': 3, 'fan_speed': 3, 'fan_power': 3, 'fan_on': 3, 'occ': 3}


def rnd(x):
    """[13,96] -> rounded float array (NaN kept) and JSON lists (None for NaN)."""
    x = np.array(x, np.float64)
    for i, c in enumerate(RAW):
        x[i] = np.round(x[i], DEC[c])
    js = [[None if not np.isfinite(v) else float(v) for v in row] for row in x]
    return x, js


def weights_pack(path):
    W = dict(np.load(path))
    man, arrs, off = [], [], 0
    for k in ['c1_W', 'c1_b', 'c2_W', 'c2_b', 'c3_W', 'c3_b', 'c4_W', 'c4_b', 'fc_W', 'fc_b', 'cls_W', 'cls_b',
              'par_W', 'par_b', 'waste_W', 'waste_b']:
        a = W[k].astype('<f4').ravel()
        man.append(dict(k=k, shape=list(W[k].shape), off=off))
        arrs.append(a)
        off += a.size
    b64 = base64.b64encode(np.concatenate(arrs).tobytes()).decode()
    return dict(b64=b64, man=man, meta=dict(temperature=float(W['temperature']), classes=CLASSES,
                                            labels=ffr.LABELS))


def main(model):
    ff = ffr.FaultFinder(f'{ROOT}/release/faultfinder_weights.npz')
    wp = weights_pack(f'{ROOT}/release/faultfinder_weights.npz')
    json.dump(wp, open(f'{ROOT}/web/weights.json', 'w'))
    import buildingdata as bd
    summ = json.load(open(os.path.join(__import__('common').BLDG, 'data', 'proc', 'baselines', 'faults_summary.json')))
    ap = bd.baselines('faults_apar').set_index('run_id')
    ex = pd.read_parquet(f'{ROOT}/exam/exam_runs.parquet').set_index('run_id')
    runs = load_real_runs('exam')
    out = []
    for r, x in runs:
        xr, js = rnd(x)
        pred = ff.predict_raw(xr[None])[0]
        rtu = r.unit == 'RTU'
        thr = summ['rtu' if rtu else 'ahu_all']['threshold_dev']
        sc = float(ap.score.get(r.run_id, np.nan))
        rules = []
        if r.run_id in ap.index:
            for c in ap.columns:
                if c.startswith('rule_R') and np.nan_to_num(ap.loc[r.run_id, c]) > 0:
                    rules.append(c.replace('rule_', ''))
        e = ex.loc[r.run_id]
        out.append(dict(id=r.run_id, src=e.source, ds=r.dataset, unit=r.unit, date=str(r.date)[:10],
                        fault=r.fault_type, sev=None if str(r.severity) in ('nan', 'None') else str(r.severity),
                        diag=e.diag_class, apar=dict(score=None if not np.isfinite(sc) else round(sc, 4),
                                                     thr=round(thr, 4), flag=bool(np.isfinite(sc) and sc > thr),
                                                     rules=rules, rule='fan on while unoccupied' if rtu else 'APAR (12 rules)'),
                        py=dict(p=round(pred['p_fault'], 5), top=[d[0] for d in pred['diagnosis'][:3]],
                                topp=[round(d[2], 5) for d in pred['diagnosis'][:3]]),
                        x=js))
    # lab
    lab = dict(np.load(f'{ROOT}/web/lab_raw.npz'))
    lm = json.load(open(f'{ROOT}/web/lab_meta.json'))
    labj = {k: rnd(v)[1] for k, v in lab.items()}
    data = dict(model=model, channels=RAW, runs=out, lab=labj, lab_meta=lm,
                exam=json.load(open(f'{ROOT}/exam/exam.json')),
                ratchet=json.load(open(f'{ROOT}/ratchet.json')),
                other_why=__import__('common').OTHER_WHY)
    tx = json.load(open(f'{ROOT}/web/text.json')) if os.path.exists(f'{ROOT}/web/text.json') else {}
    data['text'] = tx.get('text', {})
    data['start_run'] = tx.get('start_run')
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return clean(o.item())
        return o
    json.dump(clean(data), open(f'{ROOT}/web/data.json', 'w'), separators=(',', ':'), default=float, allow_nan=False)
    print('runs', len(out), 'data.json MB', os.path.getsize(f'{ROOT}/web/data.json') / 1e6,
          'weights MB', os.path.getsize(f'{ROOT}/web/weights.json') / 1e6)


if __name__ == '__main__':
    main(sys.argv[1])
