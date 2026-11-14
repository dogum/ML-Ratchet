"""JS (web/dcnet.js) vs Python on 9 page days: physics and network (float16 web weights) -> checks/js_vs_python.json"""
import json, numpy as np, torch
from common import *
import real, evaluate as E

J = json.load(open(f'{ROOT}/web/verify_js.json')); D = json.load(open(f'{ROOT}/web/data.json'))
nets = E.load('r8_wide')
for k, n in nets.items():
    with torch.no_grad():
        for p in n.state_dict().values(): p.copy_(p.half().float())
res = dict(phys_rel=0, mu_rel=0, w_abs=0, s_abs=0, phys_hot_rel=0)
for j in J:
    d = D['days'][j['i']]; real._CACHE.clear(); g = real.real_region(d['ba'])
    i0 = day_start_index(d['ba'], d['day']); s = g.sample(i0)
    Bk = {k: s[k][None] for k in ('xd', 'xw', 'pw', 'xs', 'ps', 'night') if k in s}
    if 'xw' not in Bk: Bk['xw'] = np.zeros((1, 332)); Bk['pw'] = np.zeros((1, 24))
    if 'xs' not in Bk: Bk['xs'] = np.zeros((1, 324)); Bk['ps'] = np.zeros((1, 24)); Bk['night'] = np.ones((1, 24))
    p = E.predict(nets, Bk)
    res['phys_rel'] = max(res['phys_rel'], float(np.max(np.abs(np.array(j['phys']) / s['pd_total'] - 1))))
    res['mu_rel'] = max(res['mu_rel'], float(np.max(np.abs(np.array(j['mu']) / (s['pd_total'] * np.exp(p['d'][0][0])) - 1))))
    if j['w']: res['w_abs'] = max(res['w_abs'], float(np.max(np.abs(np.array(j['w']) / s['capw'] - p['w'][0][0]))))
    if j['s']: res['s_abs'] = max(res['s_abs'], float(np.max(np.abs(np.array(j['s']) / s['caps'] - p['s'][0][0]))))
    fc = dict(T=np.full(38, 3.0), rh=np.full(38, 0.1), ghi=np.full(38, 0.8), v=np.ones((g.W['pws50'].shape[0], 38)))
    sf = g.sample(i0, fc)
    res['phys_hot_rel'] = max(res['phys_hot_rel'], float(np.max(np.abs(np.array(j['phys_hot']) / sf['pd_total'] - 1))))
res['note'] = ('phys/mu/w/s: neutral sliders, JS vs Python with the same float16 weights; phys_hot: +3 C, humidity x1.1, '
               'irradiance x0.8 (Python re-warms its filters over 720 h, JS continues full-series states)')
jdump(res, f'{ROOT}/checks/js_vs_python.json'); print(res)
