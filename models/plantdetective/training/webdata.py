"""Data for the web page (web/data.js): model weights, explorer units (fixed stratified random draw of exam units),
lab price background (ERCOT 2022, a train-input year), exam tables, ratchet ladder, twin test summary.

usage: python webdata.py <model> <exam grade pkl> <dev grade pkl> <factors json>
"""
from common import *
import json, glob, base64, zlib
import numpy as np, pandas as pd
from pdcore import truth, TECHS, REG, model_input, flex_class
from evaluate import real_unit
from report import table, coverage, natural
import realdata
sys.path.insert(0, MODEL)   # the release: plantdetective.py
from plantdetective import Detective, window_vector


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def b64(a, dt):
    return base64.b64encode(np.asarray(a, dt).tobytes()).decode()


def weights_js(model):
    fs = sorted(glob.glob(f'{DATA}/models/{model}_s[0-9].npz')) or [f'{DATA}/models/{model}.npz']
    out = []
    for f in fs:
        W = dict(np.load(f))
        layers = []
        i = 0
        while f'l{i}w' in W:
            w = W[f'l{i}w']
            layers.append(dict(w=b64(w.ravel(), '<f4'), b=b64(W[f'l{i}b'], '<f4'), nin=int(w.shape[1]), nout=int(w.shape[0])))
            i += 1
        out.append(dict(mu=b64(W['mu'], '<f4'), sd=b64(W['sd'], '<f4'), layers=layers))
    return out


def explorer_units(n_per=4, seed=20261113):
    """Fixed random draw within technology strata among exam units with a heat-rate truth (not cherry-picked)."""
    U = realdata.units(); U = U[U.split == 'exam'].copy()
    U['tech'] = [realdata.truth_tech(r) for _, r in U.iterrows()]
    rng = np.random.default_rng(seed)
    pick = []
    for tech in TECHS:
        cand = sorted(U[U.tech == tech].unit_id)
        rng.shuffle(cand)
        k = 0
        for uid in cand:
            rec, t = real_unit('exam', uid)
            tr = truth(rec['op'], rec['P'], rec['HI'], rec['meas'])
            yrs = pd.to_datetime(t, unit='s').year
            ok = np.isfinite(tr.get('hr_full', np.nan)) and ((rec['op'] > 0) & (yrs == 2024)).sum() > 200
            if ok:
                pick.append(uid); k += 1
            if k == n_per:
                break
    return pick


def unit_payload(uid, det, factors):
    rec, t = real_unit('exam', uid)
    tr = truth(rec['op'], rec['P'], rec['HI'], rec['meas'])
    U = realdata.units().set_index('unit_id').loc[uid]
    tech = realdata.truth_tech(U)
    ctx = realdata.context(uid)
    tt = pd.to_datetime(t, unit='s')
    x = model_input(rec['op'], rec['P'])
    m24 = tt.year == 2024
    pr = realdata.prices()
    price = pr.price.reindex(pd.to_datetime(t, unit='s', utc=True)).values
    gas = pr.gas.reindex(pd.to_datetime(t, unit='s', utc=True)).values
    # Python reference predictions: 2024 window only (what the page recomputes live) and all years pooled
    r24 = det.infer(x[m24], t[m24], price[m24], gas[m24], ctx)
    rall = det.infer(x, t, price, gas, ctx)
    rout = det.infer(x, t)
    # heat-rate scatter: steady measured hours, binned by load fraction (all years)
    full = (rec['op'] >= 0.999) & (rec['P'] > 0)
    steady = full & np.r_[False, full[:-1]] & np.r_[full[1:], False]
    ms = steady & rec['meas'] & np.isfinite(rec['HI']) & (rec['HI'] > 0)
    f = rec['P'][ms] / tr['pmax']; hr = rec['HI'][ms] / rec['P'][ms]
    bins = np.arange(0.0, 1.125, 0.025)
    pts = []
    for a, b in zip(bins[:-1], bins[1:]):
        k = (f >= a) & (f < b)
        if k.sum() >= 5:
            pts.append([round(float(np.median(f[k])), 4), round(float(np.median(hr[k])), 3),
                        round(float(np.percentile(hr[k], 25)), 3), round(float(np.percentile(hr[k], 75)), 3), int(k.sum())])
    c = tr.get('hr_coef')
    curve_true = []
    if c is not None:
        for fr in np.linspace(max(tr['pmin_frac'], 0.3), 1.0, 15):
            P = fr * tr['pmax']; curve_true.append([round(float(fr), 3), round(float((c[0] + c[1] * P + c[2] * P * P) / P), 3)])
    ctx_js = {k: v for k, v in ctx.items() if k != 'hr923'}
    ctx_js['hr923'] = {str(k): v for k, v in ctx['hr923'].items()}
    return dict(id=uid, name=str(U.facility_name), unit=str(U.unit_id_epa), tech=tech, unit_type=str(U.unit_type),
                t0=int(t[m24][0]), x=b64(np.nan_to_num(x[m24]), '<i2'), ctx=ctx_js,
                truth={k: (None if not np.isfinite(tr.get(k, np.nan)) else round(float(tr[k]), 4))
                       for k in REG + ['pmax', 'f_part']},
                flex_true=flex_class(tr['starts_yr'], tr['cf'], tr['med_up']),
                hr_pts=pts, hr_curve_true=curve_true,
                py24=dict(tech=r24['tech'], est={k: v['value'] for k, v in r24['estimates'].items()}),
                pyall=dict(tech=rall['tech'], est=rall['estimates'], curve=rall['heat_rate_curve'], flex=rall['flex_class']),
                pyout=dict(tech=rout['tech'], est={k: v['value'] for k, v in rout['estimates'].items()}),
                years=sorted(set(tt.year.tolist())))


def main(model, exam_pkl, dev_pkl, fac_json):
    det = Detective(os.path.join(MODEL, 'plantdetective_weights.npz'))
    factors = json.load(open(fac_json))
    D = dict(factors=factors, weights=weights_js(model))
    pr = realdata.prices()
    p24 = pr[pr.index.year == 2024]; p22 = pr[pr.index.year == 2022]
    D['price2024'] = dict(t0=int(epoch_s(p24.index)[0]), price=b64(p24.price.ffill().bfill().values, '<f4'),
                          gas=b64(p24.gas.values, '<f4'))
    D['price2022'] = dict(t0=int(epoch_s(p22.index)[0]), price=b64(p22.price.ffill().bfill().values, '<f4'),
                          gas=b64(p22.gas.values, '<f4'))
    D['units'] = [unit_payload(u, det, factors) for u in explorer_units()]
    ex = pd.read_pickle(exam_pkl)
    tabs = {}
    for mode in ('ctx', 'out'):
        tabs[mode] = table(ex, mode).to_dict('records')
        for tech in TECHS:
            tabs[f'{mode}_{tech}'] = table(ex, mode, kinds=('year', 'pooled'), techs=[tech]).to_dict('records')
    D['exam'] = tabs
    D['coverage'] = {k: coverage(ex, 'ctx', k, factors) for k in ('month', 'year', 'pooled')}
    D['coverage_raw'] = coverage(ex, 'ctx', 'year', None)
    hi = pd.read_csv(exam_pkl.replace('grade_', 'heatinput_').replace('.pkl', '.csv'))
    D['heatinput'] = {f'{mode}': {t: dict(n=int(len(g)), model=float(g.err.abs().median()),
                                           e923=float(g.err923.abs().median()) if g.err923.notna().any() else None,
                                           n923=int(g.err923.notna().sum()))
                                   for t, g in hi[hi['mode'] == mode].groupby('tech')} for mode in ('ctx', 'out')}
    for mode in ('ctx', 'out'):
        g = hi[hi['mode'] == mode]; both = g[g.err923.notna()]
        D['heatinput'][mode]['all'] = dict(n=int(len(g)), model=float(g.err.abs().median()),
                                          model_on923=float(both.err.abs().median()), e923=float(both.err923.abs().median()),
                                          n923=int(len(both)))
    D['ratchet'] = json.load(open(f'{ROOT}/ratchet.json'))['rungs']
    for k in ('v1', 'v2'):
        f = f'{ROOT}/out/twin_{k}.json'
        if os.path.exists(f):
            D[f'twin_{k}'] = json.load(open(f))
    tw = pd.read_csv(f'{ROOT}/out/twin_v2.csv') if os.path.exists(f'{ROOT}/out/twin_v2.csv') else None
    if tw is not None:
        D['twin_rows'] = tw.round(3).to_dict('records')
    os.makedirs(f'{ROOT}/web', exist_ok=True)
    s = json.dumps(clean(D), separators=(',', ':'), allow_nan=False)
    open(f'{ROOT}/web/data.js', 'w').write('window.PDDATA=' + s + ';')
    print('web/data.js', len(s) / 1e6, 'MB', [u['id'] for u in D['units']])


if __name__ == '__main__':
    main(*sys.argv[1:5])
