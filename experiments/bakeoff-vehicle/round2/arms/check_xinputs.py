"""PREREG section 3 check: the table members' inputs rebuilt from the trace batch (xlib.table_from_trace) vs the table
batch, judged on the table members' predictions. Sim val (first 400 scored trips) + the first 200 records of each dev
set. Pass: median |log(pred_rebuilt / pred_table)| < 0.002 for every member. -> round2/arms/check_xinputs.json"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths as P  # noqa: E402
P.setup()
import numpy as np  # noqa: E402
import bake  # noqa: E402
sys.path.insert(0, os.path.join(HERE, 'common'))
import xlib  # noqa: E402

MEMBERS = {k: os.path.normpath(os.path.join(HERE, p)) for k, p in json.loads(sys.argv[1]).items()} if len(sys.argv) > 1 else {
    'tab_mlp': os.path.join(P.BAKEOFF, 'arms_table', 'tab_mlp', 'predict.py'),
    'tab_gp': os.path.join(P.BAKEOFF, 'arms_table', 'tab_gp', 'predict.py')}


def main():
    mods = {k: xlib._load_module(p, 'chk_' + k) for k, p in MEMBERS.items()}
    pairs = []
    B = bake.load_bank('val'); T = bake.val_tiers(); idx = bake.simval_rows()[:400]
    pairs.append(('simval', bake.sim_batch(B, T, idx, 'trace')[0], bake.sim_batch(B, T, idx, 'table')[0]))
    for name in bake.SETS:
        _, _, bt = next(bake._real_batches('dev', 'trace', sets=(name,), chunk=200))
        _, _, bb = next(bake._real_batches('dev', 'table', sets=(name,), chunk=200))
        pairs.append((name, bt, bb))
    out = {}
    for name, bt, bb in pairs:
        X, E = xlib.table_from_trace(bt)
        tb = dict(bb); tb['X'] = X; tb['E_phys'] = E
        r = {'ephys_med_abs_log': float(np.median(np.abs(np.log(E / bb['E_phys']))))}
        for k, m in mods.items():
            a = m.predict(bb)['pred']; b = m.predict(tb)['pred']
            d = np.abs(np.log(b / a))
            r[k] = {'median': float(np.median(d)), 'p95': float(np.percentile(d, 95)), 'max': float(d.max())}
        out[name] = r
        print(name, json.dumps(r), flush=True)
    ok = all(v[k]['median'] < 0.002 for v in out.values() for k in MEMBERS)
    out['PASS'] = ok
    json.dump(out, open(os.path.join(HERE, 'check_xinputs.json'), 'w'), indent=1)
    print('PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
