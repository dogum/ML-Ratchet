"""Ladder of every evaluated rung of the four trace arms (harness evaluate.py outputs) + paired bootstrap of a rung
against its incumbent on the full frozen sim val set (clusters = vehicles, 10 trips each).
usage: python3 ladder.py                     (table of all rungs)
       python3 ladder.py <arm> <rung> <incumbent>   (paired sim-val skill bootstrap)"""
import json
import os
import sys

import numpy as np
import pandas as pd

TRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ['cnn', 'gru', 'transformer', 's4d']
SETS = ['VED_ICE', 'VED_HEV', 'VED_EV', 'VED_PHEV', 'EPA_conv_FTP', 'EPA_conv_HWY', 'Mersin_diesel', 'Mersin_cng', 'ICCT']
DUMB_MED = None


def rows(arm):
    d = os.path.join(TRC, arm, 'work', 'eval')
    out = []
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not f.endswith('_trace_summary.json'):
            continue
        s = json.load(open(os.path.join(d, f)))
        tag = s['tag']
        tj = os.path.join(TRC, arm, 'work', f'train_{tag}.json')
        cpu = json.load(open(tj))['train_cpu_h'] if os.path.exists(tj) else np.nan
        r = {'arm': arm, 'rung': tag, 'cpu_h': cpu}
        if 'simval' in s:
            v = s['simval']
            r.update(skill=v['skill'], MdAPE=v['MdAPE'], bias=v['bias'], cov50=v['cov50'], cov90=v['cov90'],
                     crps=v['crps_log'])
        if 'dev' in s:
            r['D16'] = s['dev']['D16']
            for k in SETS:
                r[k] = s['dev']['sets'].get(k, {}).get('MdAPE', np.nan)
        out.append(r)
    return out


def boot(arm, rung, inc, nb=2000):
    d = os.path.join(TRC, arm, 'work', 'eval')
    a = pd.read_parquet(os.path.join(d, f'{rung}_trace_simval_preds.parquet'))
    b = pd.read_parquet(os.path.join(d, f'{inc}_trace_simval_preds.parquet'))
    ea = np.abs(np.log(a.pred / a.y)).values; eb = np.abs(np.log(b.pred / b.y)).values
    s = json.load(open(os.path.join(d, f'{rung}_trace_summary.json')))['simval']
    den = s['med_log_err'] / (1 - s['skill'])            # the dumb baseline's median |log error|
    veh = (a.row.values // 10)
    u, inv = np.unique(veh, return_inverse=True)
    rng = np.random.default_rng(20261123)
    groups = [np.where(inv == k)[0] for k in range(len(u))]
    diffs = []
    for _ in range(nb):
        pick = np.concatenate([groups[k] for k in rng.integers(0, len(u), len(u))])
        diffs.append((np.median(eb[pick]) - np.median(ea[pick])) / den)
    d0 = (np.median(eb) - np.median(ea)) / den
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'d_skill': round(float(d0), 4), 'ci95': [round(float(lo), 4), round(float(hi), 4)], 'clusters': len(u)}


if __name__ == '__main__':
    if len(sys.argv) == 4:
        print(json.dumps(boot(*sys.argv[1:])))
    else:
        df = pd.DataFrame([r for a in ARMS for r in rows(a)])
        pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
        print(df.round(4).to_string(index=False))
