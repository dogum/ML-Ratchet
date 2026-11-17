"""Real-data metrics for a set of Detective weights (dev gate or final exam), with the B baselines beside them.

Metrics
  BDG2 (hourly / monthly / annual from the same 2017 meters):
    floor area MdAPE and share within x2; primary use accuracy and macro-F1 (residential_* predictions count as
    lodging, BDG2's 'Lodging/residential'); year built MAE; electricity-only copy: AUC of P(district cooling)
    against 'has a chilled-water meter' and of P(district heat) against 'has a steam/hot-water meter'.
  RECS (annual): floor area MdAPE; vintage MAE (to the survey's year-range midpoint) ; heating class accuracy
    (fuel / electric) and heat pump vs resistance balanced accuracy; UA/m2 vs insulation / draftiness / vintage
    (Spearman, a consistency check, not a truth).
  CBECS (annual): floor area MdAPE; use accuracy among commercial classes; heating fuel class accuracy; chiller
    presence AUC.
  HUE (hourly, exam only): AUC of P(electric heat) for the 27 homes; heat pump vs resistance among electric homes.
"""
import json
import os

import numpy as np

import common as C
from detective import Net, decode

OD = os.path.join(C.ROOT, 'feats', 'real')
COM_USES = [u for u in C.USES if not u.startswith('residential')]


def load(split, name):
    z = np.load(os.path.join(OD, f'{split}_{name}.npz'))
    return {k: z[k] for k in z.files}, json.load(open(os.path.join(OD, f'{split}_{name}.json')))


def auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    m = np.isfinite(score)
    score, y = score[m], y[m]
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(score)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def macro_f1(t, p):
    f = []
    for c in set(t):
        tp = sum((a == c) and (b == c) for a, b in zip(t, p))
        fp = sum((a != c) and (b == c) for a, b in zip(t, p))
        fn = sum((a == c) and (b != c) for a, b in zip(t, p))
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def bacc(t, p):
    t, p = np.asarray(t), np.asarray(p)
    return float(np.mean([np.mean(p[t == c] == c) for c in np.unique(t)]))


def spearman(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).correlation) if m.sum() > 10 else np.nan


def run_net(net, X):
    ok = np.isfinite(X).all(1)
    mu = np.full((len(X), len(C.REG)), np.nan)
    sg = np.full((len(X), len(C.REG)), np.nan)
    lg = np.full((len(X), len(C.USES) + len(C.HEAT) + len(C.COOL)), np.nan)
    if ok.any():
        mu[ok], sg[ok], lg[ok] = net.raw(X[ok])
    return mu, sg, lg


def _probs(lg, a, b, temp=1.0):
    z = lg[:, a:b] / temp
    z = z - np.nanmax(z, 1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


NU, NH = len(C.USES), len(C.HEAT)
J = {n: j for j, n in enumerate(C.REG)}


def area_metrics(mu, sg, A):
    A = np.asarray(A, float)
    m = np.isfinite(A) & (A > 0) & np.isfinite(mu[:, J['logA']])
    err = mu[m, J['logA']] - np.log10(A[m])
    ape = np.abs(10 ** mu[m, J['logA']] - A[m]) / A[m]
    return dict(n=int(m.sum()), MdAPE=float(np.median(ape) * 100), within_x2=float(np.mean(np.abs(err) < np.log10(2))),
                bias_log10=float(np.median(err)), cov1=float(np.mean(np.abs(err) <= sg[m, J['logA']])),
                z=(err / sg[m, J['logA']]).tolist())


def eval_bdg2(nets, split, modes=('hourly', 'monthly', 'annual')):
    mats, labs = load(split, 'bdg2')
    out = {}
    for mode in modes:
        if mode not in nets:
            continue
        mu, sg, lg = run_net(nets[mode], mats[mode])
        r = dict(area=area_metrics(mu, sg, [l['A'] for l in labs]))
        pu = _probs(lg, 0, NU)
        pred = [C.USES[i] for i in pu.argmax(1)]
        pred = ['lodging' if p.startswith('residential') else p for p in pred]
        t = [l['use'] for l in labs]
        r['use'] = dict(n=len(t), accuracy=float(np.mean(np.array(pred) == np.array(t))), macroF1=macro_f1(t, pred))
        y = np.array([l['year'] for l in labs], float)
        m = np.isfinite(y)
        e = mu[m, J['year']] - y[m]
        r['year'] = dict(n=int(m.sum()), MAE=float(np.mean(np.abs(e))), cov1=float(np.mean(np.abs(e) <= sg[m, J['year']])),
                         z=(e / sg[m, J['year']]).tolist())
        out[mode] = r
    if 'hourly' in nets:
        X = mats['hourly_elec']
        mu, sg, lg = run_net(nets['hourly'], X)
        ph = _probs(lg, NU, NU + NH)
        pc = _probs(lg, NU + NH, NU + NH + len(C.COOL))
        ok = np.isfinite(X).all(1)
        out['elec_only'] = dict(
            n=int(ok.sum()),
            auc_district_cooling=auc(pc[ok, C.COOL.index('district')], [l['has_chw'] for l, o in zip(labs, ok) if o]),
            auc_district_heat=auc(ph[ok, C.HEAT.index('district')], [l['has_dheat'] for l, o in zip(labs, ok) if o]),
            n_chw=int(sum(l['has_chw'] for l, o in zip(labs, ok) if o)),
            n_dheat=int(sum(l['has_dheat'] for l, o in zip(labs, ok) if o)))
        A = np.array([l['A'] for l in labs], float)
        out['elec_only']['area'] = area_metrics(mu[ok], sg[ok], A[ok])
    return out


def eval_recs(net, split):
    mats, labs = load(split, 'recs')
    mu, sg, lg = run_net(net, mats['annual'])
    r = dict(area=area_metrics(mu, sg, [l['A'] for l in labs]))
    ylo = np.array([l['year_lo'] for l in labs], float)
    yhi = np.array([l['year_hi'] for l in labs], float)
    ymid = np.where(np.isfinite(ylo) & np.isfinite(yhi), (ylo + yhi) / 2, np.where(np.isfinite(yhi), yhi - 10, ylo + 5))
    m = np.isfinite(ymid)
    e = mu[m, J['year']] - ymid[m]
    r['year'] = dict(n=int(m.sum()), MAE=float(np.mean(np.abs(e))), cov1=float(np.mean(np.abs(e) <= sg[m, J['year']] + 5)),
                     z=(e / np.sqrt(sg[m, J['year']] ** 2 + 5 ** 2)).tolist())
    ph = _probs(lg, NU, NU + NH)
    t = np.array([l['heat'] or 'nan' for l in labs])
    # fuel vs electric (incl. heat pump) vs none
    grp = {'heat_pump': 'electric', 'resistance': 'electric', 'fuel': 'fuel', 'none': 'none', 'district': 'fuel'}
    pg = np.array([grp[C.HEAT[i]] for i in ph.argmax(1)])
    tg = np.array([grp.get(x, 'nan') for x in t])
    k = tg != 'nan'
    r['heat_fuel_vs_electric'] = dict(n=int(k.sum()), accuracy=float(np.mean(pg[k] == tg[k])))
    e_ = np.isin(t, ['heat_pump', 'resistance'])
    s = ph[e_, C.HEAT.index('heat_pump')] / (ph[e_, C.HEAT.index('heat_pump')] + ph[e_, C.HEAT.index('resistance')])
    th = 0.5
    r['hp_vs_resistance'] = dict(n=int(e_.sum()), auc=auc(s, t[e_] == 'heat_pump'),
                                 balanced_accuracy=bacc(t[e_] == 'heat_pump', s > th))
    ua = mu[:, J['logUAm2']]
    ins = np.array([{'well': 0, 'adequate': 1, 'poor': 2, 'none': 3}.get(l['insulation'], np.nan) for l in labs], float)
    dr = np.array([{'never': 0, 'sometimes': 1, 'mostly': 2, 'always': 3}.get(l['drafty'], np.nan) for l in labs], float)
    r['UAm2_consistency'] = dict(spearman_vs_insulation_poorness=spearman(ua, ins), spearman_vs_draftiness=spearman(ua, dr),
                                 spearman_vs_vintage=spearman(ua, ymid))
    return r


def eval_cbecs(net, split):
    mats, labs = load(split, 'cbecs')
    mu, sg, lg = run_net(net, mats['annual'])
    r = dict(area=area_metrics(mu, sg, [l['A'] for l in labs]))
    pu = _probs(lg, 0, NU)
    ci = [C.USES.index(u) for u in COM_USES]
    pred = [COM_USES[i] for i in pu[:, ci].argmax(1)]
    t = [l['use'] for l in labs]
    r['use'] = dict(n=len(t), accuracy=float(np.mean(np.array(pred) == np.array(t))), macroF1=macro_f1(t, pred))
    ylo = np.array([l['year_lo'] for l in labs], float)
    yhi = np.array([l['year_hi'] for l in labs], float)
    ymid = np.where(np.isfinite(ylo) & np.isfinite(yhi), (ylo + yhi) / 2, np.where(np.isfinite(yhi), yhi - 10, ylo + 5))
    m = np.isfinite(ymid)
    e = mu[m, J['year']] - ymid[m]
    r['year'] = dict(n=int(m.sum()), MAE=float(np.mean(np.abs(e))), z=(e / np.sqrt(sg[m, J['year']] ** 2 + 5 ** 2)).tolist())
    ph = _probs(lg, NU, NU + NH)
    grp = {'heat_pump': 'electric', 'resistance': 'electric', 'fuel': 'fuel', 'none': 'none', 'district': 'district'}
    t = np.array([grp.get(l['heat'], 'nan') if l['heat'] else 'nan' for l in labs])
    pg = np.array([grp[C.HEAT[i]] for i in ph.argmax(1)])
    k = t != 'nan'
    r['heat_class'] = dict(n=int(k.sum()), accuracy=float(np.mean(pg[k] == t[k])))
    pc = _probs(lg, NU + NH, NU + NH + len(C.COOL))
    ch = np.array([l['chiller'] for l in labs], float)
    m = np.isfinite(ch)
    r['chiller'] = dict(n=int(m.sum()), auc=auc(pc[m, C.COOL.index('chiller')], ch[m] == 1))
    hp = np.array([l['hp'] for l in labs], float)
    m = np.isfinite(hp)
    r['heat_pump_present'] = dict(n=int(m.sum()), auc=auc(ph[m, C.HEAT.index('heat_pump')], hp[m] == 1))
    return r


def eval_hue(nets):
    mats, labs = load('exam', 'hue')
    out = {}
    for mode in ('hourly', 'monthly'):
        mu, sg, lg = run_net(nets[mode], mats[mode])
        ph = _probs(lg, NU, NU + NH)
        pe = ph[:, C.HEAT.index('heat_pump')] + ph[:, C.HEAT.index('resistance')]
        y = np.array([l['electric_heat'] for l in labs])
        t = np.array([l['heat'] for l in labs])
        pred = np.array([C.HEAT[i] for i in ph.argmax(1)])
        pg = np.where(np.isin(pred, ['heat_pump', 'resistance']), 'electric', 'other')
        out[mode] = dict(n=len(labs), n_electric=int(y.sum()), auc_electric_heat=auc(pe, y),
                         accuracy_electric_vs_not=float(np.mean((pg == 'electric') == y)),
                         per_home=[dict(id=l['id'], codes=l['codes'], truth=l['heat'], p_electric=float(p), pred=str(q))
                                   for l, p, q in zip(labs, pe, pred)])
    return out


def dev_gate(nets):
    """Composite real-dev score in [0,1] (higher better), used only as a non-inferiority gate."""
    b = eval_bdg2(nets, 'dev')
    s = []
    for mode in ('hourly', 'monthly', 'annual'):
        if mode in b:
            s.append(max(0.0, 1 - b[mode]['area']['MdAPE'] / 64.3))        # vs dumb dev-median area (B: 64 %)
            s.append(b[mode]['use']['macroF1'])
    rc = eval_recs(nets['annual'], 'dev') if 'annual' in nets else None
    cb = eval_cbecs(nets['annual'], 'dev') if 'annual' in nets else None
    if rc:
        s.append(max(0.0, 1 - rc['area']['MdAPE'] / 50.0))
        s.append(max(0.0, 2 * rc['hp_vs_resistance']['balanced_accuracy'] - 1))
        s.append(rc['heat_fuel_vs_electric']['accuracy'])
    if cb:
        s.append(max(0.0, 1 - cb['area']['MdAPE'] / 76.0))
        s.append(cb['use']['macroF1'])
        s.append(cb['heat_class']['accuracy'])
    return float(np.mean(s)), dict(bdg2=b, recs=rc, cbecs=cb)


def strip(d):
    """Drop the long per-unit lists for logging."""
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if k not in ('z', 'per_home')}
    return d


def load_nets(wdir):
    return {m: Net(os.path.join(wdir, f'{m}.npz')) for m in ('hourly', 'monthly', 'annual')
            if os.path.exists(os.path.join(wdir, f'{m}.npz'))}
