"""JS == Python check: run web/engine.js (node) and the Python policy / B's SOP / B's DP on a few DEV dams.
usage: python3 check_js.py weights.npz  -> web/check_js.json"""
import sys, json, pickle, subprocess, os
import numpy as np
import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from policy import load_policy, rollout_np
from task import simulate, dp, cost
ROOT = _paths.WORK   # the old workspace root (data/, runs/, exam/, web/, tmp/, ratchet.json)


def net_json(path):
    z = np.load(path); nl = sum(1 for k in z.files if k.startswith('W'))
    return {'fset': str(z['fset']), 'mu': z['mu'].tolist(), 'sd': z['sd'].tolist(),
            'W': [z[f'W{i}'].tolist() for i in range(nl)], 'b': [z[f'b{i}'].tolist() for i in range(nl)]}


def task_json(tk, n=None):
    n = n or len(tk['I'])
    return {'I': tk['I'][:n].tolist(), 'doy': tk['dates'][:n].dayofyear.tolist(), 'Dcurve': list(map(float, tk['Dcurve'])),
            'Icurve': list(map(float, tk['Icurve'])), 'Qmax': tk['Qmax'], 'cap': tk['cap'], 'S0': tk['S0']}


if __name__ == '__main__':
    w = sys.argv[1]; dev = pickle.load(open(f'{ROOT}/data/real_dev.pkl', 'rb'))[:5]
    for t in dev:                                   # 3 years each keeps the DP check quick
        for k in ('I', 'D', 'Ss'):
            t[k] = t[k][:1100]
        t['dates'] = t['dates'][:1100]
    payload = {'net': net_json(w), 'tasks': [task_json(t) for t in dev]}
    json.dump(payload, open(f'{ROOT}/tmp/js_in.json', 'w'))
    js = """const RP=require('%s/engine.js');const P=require('%s/tmp/js_in.json');const out=[];
for(const t of P.tasks){const a=RP.run(t,'pilot',P.net),b=RP.run(t,'sop'),c=RP.dp(t);
out.push({pilot:Array.from(a.R),sop:Array.from(b.R),dp:Array.from(c.R)});}
require('fs').writeFileSync('%s/tmp/js_out.json',JSON.stringify(out));""" % (_paths.SRC, ROOT, ROOT)
    open(f'{ROOT}/tmp/js_run.js', 'w').write(js)
    subprocess.run(['node', f'{ROOT}/tmp/js_run.js'], check=True)
    out = json.load(open(f'{ROOT}/tmp/js_out.json'))
    Rp, _ = rollout_np(load_policy(w), dev)
    res = []
    for t, o, rp in zip(dev, out, Rp):
        rs, _ = simulate('sop', t['I'], t['D'], t['Qmax'], t['cap'], t['S0']); rd = dp(t['I'], t['D'], t['Qmax'], t['cap'], t['S0'])
        Jp, Js, Jd = [np.mean(cost(np.array(x), t['D'], t['Qmax'])) for x in (o['pilot'], o['sop'], o['dp'])]
        res.append({'res_id': t['res_id'], 'max_rel_release_diff_pilot': float(np.max(np.abs(np.array(o['pilot']) - rp)) / t['Qmax']),
                    'max_rel_release_diff_sop': float(np.max(np.abs(np.array(o['sop']) - rs)) / t['Qmax']),
                    'max_rel_release_diff_dp': float(np.max(np.abs(np.array(o['dp']) - rd)) / t['Qmax']),
                    'dJ_pilot': float(abs(Jp - np.mean(cost(rp, t['D'], t['Qmax'])))),
                    'dJ_dp': float(abs(Jd - np.mean(cost(rd, t['D'], t['Qmax']))))})
        print(res[-1])
    json.dump(res, open(f'{ROOT}/web/check_js.json', 'w'), indent=1)
