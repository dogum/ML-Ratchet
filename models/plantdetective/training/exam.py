"""Grade the Detective and the baselines on real CAMPD units.

split 'dev'  : 40 dev units, dev years 2019 + 2021 (calibration of the coverage factors, gates)
split 'exam' : 407 exam units, all years 2018-2025 (final report only; never used for choosing anything)

For every unit: truth = pdcore.truth over the whole period (CAMPD op time, gross load, heat input); inputs = windows
of hourly output (month, quarter, year) and the pooled prediction over all years; two model modes:
  'ctx' : output + ERCOT price + Henry Hub gas + EIA-860/923 context      'out' : output only
Baselines: simple output statistics, technology defaults (oracle technology), EIA-923 annual heat rate, dev medians.
Also the per-unit-year annual heat input error (the GridData baseline metric).

usage: python exam.py <model> <split> [out_tag]
"""
from common import *
import json
import numpy as np, pandas as pd
from pdcore import truth, TECHS, REG, model_input
from trainset import tgt_vector, window_features
from evaluate import real_unit, windows_for, pool, DEV_YEARS
from model import load_any as load
import baselines as B
import realdata
sys.path.insert(0, MODEL)   # the release: plantdetective.py
from plantdetective import tech_rule

LOGIT = {'pmin_frac', 'hot', 'warm', 'cold', 'cf'}


def to_tgt(d):
    """Natural-unit dict -> transformed target vector (NaN-safe), same transform as trainset.tgt_vector."""
    return tgt_vector({k: (None if (d.get(k) is None or not np.isfinite(d.get(k))) else d.get(k)) for k in REG})


def decode(m):
    """Transformed means -> natural units dict."""
    sg = lambda a: 1 / (1 + np.exp(-a))  # noqa
    o = dict(pmin_frac=sg(m[0]), ramp_up=np.exp(m[1]), ramp_dn=np.exp(m[2]), starts_yr=max(np.expm1(m[3]), 0),
             min_up=np.exp(m[4]), med_up=np.exp(m[5]), min_dn=np.exp(m[6]), hot=sg(m[7]), warm=sg(m[8]), cold=sg(m[9]),
             hr_full=np.exp(m[10]), hr_min=np.exp(m[10] + m[11]), cf=sg(m[12]))
    s = o['hot'] + o['warm'] + o['cold']
    for k in ('hot', 'warm', 'cold'):
        o[k] /= s
    return o


def heat_input_err(rec, t, hr_full, hr_min, pmin_frac, years):
    """Annual heat input error per year on steady measured full hours (GridData baseline construction)."""
    op, P, HI, meas = rec['op'], rec['P'], rec['HI'], rec['meas']
    full = (op >= 0.999) & (P > 0)
    steady = full & np.r_[False, full[:-1]] & np.r_[full[1:], False]
    m0 = steady & meas & np.isfinite(HI) & (HI > 0)
    tt = pd.to_datetime(t, unit='s')
    out = {}
    pc = np.percentile(P[full], 99.5) if full.sum() > 24 else np.nan
    fp = max(pmin_frac, 0.4)
    beta = (hr_min / hr_full - 1) / max(1 / fp - 1, 1e-3) if fp < 0.98 else 0.0
    for y in years:
        m = m0 & (tt.year == y)
        if m.sum() < 50:
            continue
        f = np.clip(P[m] / pc, 0.05, 1.2)
        pred = (P[m] * hr_full * (1 + beta * (1 / f - 1))).sum()
        out[int(y)] = dict(err=float(pred / HI[m].sum() - 1), hours=int(m.sum()), hr_true=float(HI[m].sum() / P[m].sum()))
    return out


def run(model, split, tag=''):
    W = load(model)
    U = realdata.units(); U = U[U.split == split]
    years = DEV_YEARS if split == 'dev' else tuple(range(2018, 2026))
    dumb = pd.read_csv(f'{DATA}/dev_medians.csv', index_col=0) if os.path.exists(f'{DATA}/dev_medians.csv') else None
    rows, hi_rows = [], []
    for n_u, uid in enumerate(U.unit_id):
        try:
            rec, t = real_unit(split, uid)
        except FileNotFoundError:
            continue
        tt = pd.to_datetime(t, unit='s')
        m = np.isin(tt.year, years)
        rec = {k: v[m] for k, v in rec.items()}; t = t[m]
        if len(t) < 24 * 30:
            continue
        tr = truth(rec['op'], rec['P'], rec['HI'], rec['meas'])
        if not np.isfinite(tr['pmax']):
            continue
        tech = realdata.truth_tech(realdata.units().set_index('unit_id').loc[uid])
        if tech not in TECHS:
            continue
        y = tgt_vector(tr)
        ctx = realdata.context(uid)
        x_all = model_input(rec['op'], rec['P'])
        for kind in ('month', 'quarter', 'year', 'pooled'):
            wins = windows_for(t, 'year' if kind == 'pooled' else kind)
            if not wins:
                continue
            preds = {}
            for mode, force in (('ctx', dict(price=True, ctx=True, net=False)), ('out', dict(price=False, ctx=False, net=False))):
                X = []
                pr = realdata.prices()
                price = pr.price.reindex(pd.to_datetime(t, unit='s', utc=True)).values
                gas = pr.gas.reindex(pd.to_datetime(t, unit='s', utc=True)).values
                for s, L in wins:
                    v, _ = window_features(rec, t, price, gas, -6, ctx, np.random.default_rng(0), L // 24 + 1, start=s,
                                           gap=False, force=force)
                    X.append(v)
                p, mm, ss = W(np.array(X))
                if kind == 'pooled':
                    p, mm, ss = [a[None] for a in pool(p, mm, ss)]
                preds[mode] = (p, mm, ss)
            nwin = 1 if kind == 'pooled' else len(wins)
            for i in range(nwin):
                if kind == 'pooled':
                    xw = x_all; yrs = sorted(set(tt.year[m])) if False else sorted(set(pd.to_datetime(t, unit='s').year))
                else:
                    s, L = wins[i]; xw = x_all[s:s + L]; yrs = sorted(set(pd.to_datetime(t[s:s + L], unit='s').year))
                hr = [ctx['hr923'].get(yy) for yy in yrs]; hr = [h for h in hr if h]
                row = dict(unit=uid, plant=int(realdata.units().set_index('unit_id').loc[uid].plant_id_eia), tech=tech,
                           kind=kind, win=i, C=TECHS.index(tech), y=y.tolist(),
                           chp=ctx.get('chp'), pmax=tr['pmax'],
                           rule=None if tech_rule(ctx) is None else tech_rule(ctx).tolist())
                for mode in preds:
                    p, mm, ss = preds[mode]
                    row[f'p_{mode}'] = p[i].tolist(); row[f'm_{mode}'] = mm[i].tolist(); row[f's_{mode}'] = ss[i].tolist()
                row['b_stats'] = to_tgt(B.stats(xw)).tolist()
                row['b_default'] = to_tgt(B.tech_default(tech)).tolist()
                row['b_923'] = to_tgt(B.eia923(float(np.mean(hr)) if hr else None)).tolist()
                if dumb is not None and tech in dumb.index:
                    row['b_dumb'] = to_tgt(dumb.loc[tech].to_dict()).tolist()
                rows.append(row)
        # annual heat input per unit-year: model from that year's window ('ctx'), EIA-923 from the same year
        ywins = windows_for(t, 'year')
        if ywins:
            pr = realdata.prices()
            price = pr.price.reindex(pd.to_datetime(t, unit='s', utc=True)).values
            gas = pr.gas.reindex(pd.to_datetime(t, unit='s', utc=True)).values
            for mode, force in (('ctx', dict(price=True, ctx=True, net=False)), ('out', dict(price=False, ctx=False, net=False))):
                X = [window_features(rec, t, price, gas, -6, ctx, np.random.default_rng(0), L // 24 + 1, start=s, gap=False,
                                     force=force)[0] for s, L in ywins]
                p, mm, ss = W(np.array(X))
                for (s, L), mi in zip(ywins, mm):
                    d = decode(mi)
                    yy = int(pd.to_datetime(t[s + L // 2], unit='s').year)
                    e = heat_input_err(rec, t, d['hr_full'], d['hr_min'], d['pmin_frac'], [yy])
                    if yy in e:
                        r = dict(unit=uid, tech=tech, year=yy, mode=mode, err=e[yy]['err'], hours=e[yy]['hours'],
                                 hr_true=e[yy]['hr_true'], hr_full=d['hr_full'], hr_min=d['hr_min'])
                        h923 = ctx['hr923'].get(yy)
                        if h923 and mode == 'ctx':
                            r['err923'] = h923 / e[yy]['hr_true'] - 1
                        hi_rows.append(r)
        if n_u % 20 == 0:
            print(n_u, uid, len(rows), flush=True)
    os.makedirs(f'{ROOT}/out', exist_ok=True)
    pd.DataFrame(rows).to_pickle(f'{ROOT}/out/grade_{split}_{model}{tag}.pkl')
    pd.DataFrame(hi_rows).to_csv(f'{ROOT}/out/heatinput_{split}_{model}{tag}.csv', index=False)
    print('done', len(rows), len(hi_rows))


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else '')
