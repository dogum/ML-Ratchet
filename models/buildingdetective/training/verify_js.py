"""Check the JS port (web/detective.js) against Python: features (hourly / monthly / annual) and, when weights exist,
the network outputs. Writes web/verify_js.json. usage: python3 verify_js.py [weights_dir]"""
import json
import os
import subprocess
import sys
import warnings

import numpy as np

import common as C
import features as F
import realdata as R

warnings.filterwarnings('ignore')
WEB = os.path.join(C.ROOT, 'web')
TMP = os.path.join(C.ROOT, 'tmp')


def time_arrays(loc):
    day = loc.tz_localize(None).values.astype('datetime64[D]').astype(np.int64)
    return dict(hr=np.asarray(loc.hour).tolist(), dow=np.asarray(loc.dayofweek).tolist(), day=day.tolist(),
                month=np.asarray(loc.month).tolist())


def net_json(fn):
    z = np.load(fn)
    n = int(z['n_layers'])
    return dict(W=[z[f'W{i}'].ravel().tolist() for i in range(n)], b=[z[f'b{i}'].tolist() for i in range(n)],
                Wr=z['Wr'].ravel().tolist(), br=z['br'].tolist(), Wc=z['Wc'].ravel().tolist(), bc=z['bc'].tolist(),
                xm=z['xm'].tolist(), xs=z['xs'].tolist(), ym=z['ym'].tolist(), ys=z['ys'].tolist(),
                widen=(z['widen'] if 'widen' in z else np.ones(len(z['ym']))).tolist(),
                temp=(z['temp'] if 'temp' in z else np.ones(3)).tolist())


def main(wdir=None):
    meta = R.bdg2_buildings('dev')
    cases = []
    for bid in list(meta.id[[0, 7, 20, 33]]):
        c = R.bdg2_case(bid, 2017, meta.set_index('id').loc[bid])
        if c is not None:
            cases.append(c)
    hue = R.hue_cases()[:1]
    cases += hue
    js_in, py = [], []
    nets = {}
    if wdir:
        for m in ('hourly', 'monthly', 'annual'):
            fn = os.path.join(wdir, f'{m}.npz')
            if os.path.exists(fn):
                nets[m] = net_json(fn)
    from detective import Net
    for c in cases:
        m = {k: [None if not np.isfinite(v) else float(v) for v in x] for k, x in c['meters'].items()}
        T = [None if not np.isfinite(v) else float(v) for v in c['T']]
        js_in.append(dict(id=c['id'], meters=m, T=T, GHI=[float(v) for v in c['GHI']], tm=time_arrays(c['loc']), lat=c['lat']))
        h = F.feat_hourly(c['meters'], c['T'], c['GHI'], c['loc'], c['lat'])
        mm, Tm, Hm, Cm, nd = F.monthly_from_hourly(c['meters'], c['T'], c['loc'])
        mo = F.feat_monthly(mm, Tm, Hm, Cm, nd, c['lat'])
        d = dict(id=c['id'], hourly=h.tolist(), monthly=mo.tolist())
        for k in nets:
            X = h if k == 'hourly' else mo if k == 'monthly' else None
            if X is None:
                continue
            mu, sg, lg = Net(os.path.join(wdir, f'{k}.npz')).raw(X)
            d[k + '_mu'] = mu[0].tolist()
            d[k + '_sg'] = sg[0].tolist()
        py.append(d)
    os.makedirs(TMP, exist_ok=True)
    json.dump(dict(cases=js_in, nets=nets), open(os.path.join(TMP, 'js_in.json'), 'w'))
    js = r"""
const BD = require('%s/detective.js'); const fs = require('fs');
const D = JSON.parse(fs.readFileSync('%s/js_in.json'));
const fix = a => a.map(v => v === null ? NaN : v);
const out = D.cases.map(c => { const m = {}; for (const k in c.meters) m[k] = fix(c.meters[k]);
  const T = fix(c.T); const h = BD.featHourly(m, T, c.GHI, c.tm, c.lat);
  const mo = BD.featMonthly(BD.monthlyFromHourly(m, T, c.tm)); const r = {id: c.id, hourly: Array.from(h), monthly: Array.from(mo)};
  for (const k of ['hourly', 'monthly']) if (D.nets[k]) { const o = BD.runNet(D.nets[k], k === 'hourly' ? h : mo); r[k + '_mu'] = o.mu; r[k + '_sg'] = o.sg; }
  return r; });
fs.writeFileSync('%s/js_out.json', JSON.stringify(out));
""" % (WEB, TMP, TMP)
    open(os.path.join(TMP, 'verify.js'), 'w').write(js)
    subprocess.run(['node', os.path.join(TMP, 'verify.js')], check=True)
    jo = json.load(open(os.path.join(TMP, 'js_out.json')))
    rep = []
    for a, b in zip(py, jo):
        r = dict(id=a['id'])
        for k in ('hourly', 'monthly', 'hourly_mu', 'monthly_mu', 'hourly_sg', 'monthly_sg'):
            if k in a and k in b:
                x, y = np.array(a[k], float), np.array([np.nan if v is None else v for v in b[k]], float)
                dd = np.abs(x - y)
                r[k] = dict(max_abs=float(np.nanmax(dd)), n_bad=int((dd > 1e-4).sum()),
                            worst=[int(i) for i in np.argsort(-np.nan_to_num(dd, nan=9))[:5]])
        rep.append(r)
        print(json.dumps(r))
    json.dump(rep, open(os.path.join(WEB, 'verify_js.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
