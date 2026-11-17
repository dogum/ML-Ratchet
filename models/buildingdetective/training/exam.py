"""FINAL EXAM (run once, after every choice is frozen). Real labels only where they exist; UA / tau / balance points /
sensitivities / schedule are reported on simulated buildings only (frozen validation set + dev twins).

Baselines on exactly the same units:
  BDG2: floor area = annual electricity / dev median electricity EUI of the TRUE use type (B's baseline; it is given
        the use, the Detective is not); dumb = dev median area; use = majority class (dev) and B's weekly-shape
        centroid number (quoted); year = dev median.
  RECS / CBECS: floor area = site energy / dev median EUI by type x climate (B's, quoted + recomputed); heating fuel
        rule (largest non-electric fuel, else electricity); HP vs resistance rule (kWh/m2/HDD, uses true area).
  Simulated: PRISM 5P change-point on daily site energy of the same quirked meters (balance points, heating and
        cooling slopes) vs the Detective; floor area from energy with the true use; median by use type.
usage: python3 exam.py [weights_dir=models/final]
"""
import json
import os
import sys
import warnings

import numpy as np

import common as C
import evaluate as E
import features as F
from detective import Net

warnings.filterwarnings('ignore')
J = {n: j for j, n in enumerate(C.REG)}
NU, NH = len(C.USES), len(C.HEAT)


def wmed(x):
    x = np.asarray(x, float)
    return float(np.nanmedian(x))


def bdg2_baselines(labs_dev, labs_ex):
    def eui(l):
        k = l['kwh'].get('elec', 0)
        return k / l['A'] if (k and l['A'] and l['A'] > 0) else np.nan
    dev_eui = {}
    for u in set(l['use'] for l in labs_dev):
        dev_eui[u] = wmed([eui(l) for l in labs_dev if l['use'] == u])
    all_eui = wmed([eui(l) for l in labs_dev])
    ape, ok = [], []
    for l in labs_ex:
        k = l['kwh'].get('elec', 0)
        A = l['A']
        if not k or not A or not np.isfinite(A):
            ape.append(np.nan)
            continue
        e = dev_eui.get(l['use'], all_eui)
        e = all_eui if not np.isfinite(e) else e
        ape.append(abs(k / e - A) / A)
    ape = np.array(ape)
    dmed = wmed([l['A'] for l in labs_dev])
    dumb = np.array([abs(dmed - l['A']) / l['A'] if l['A'] else np.nan for l in labs_ex])
    from collections import Counter
    maj = Counter(l['use'] for l in labs_dev).most_common(1)[0][0]
    t = [l['use'] for l in labs_ex]
    ymed = wmed([l['year'] for l in labs_dev])
    ye = np.array([abs(ymed - l['year']) for l in labs_ex], float)
    return dict(area_eui_by_true_use=dict(MdAPE=float(np.nanmedian(ape) * 100), n=int(np.isfinite(ape).sum())),
                area_dumb=dict(MdAPE=float(np.nanmedian(dumb) * 100)),
                use_majority=dict(cls=maj, accuracy=float(np.mean(np.array(t) == maj)), macroF1=E.macro_f1(t, [maj] * len(t))),
                year_dev_median=dict(MAE=float(np.nanmean(ye)), n=int(np.isfinite(ye).sum())))


def sim_only(net, X, Y, K, mode):
    """Regression metrics on simulated buildings (MAE, R2, 1-sigma coverage) and class balanced accuracy."""
    mu, sg, lg = net.raw(X)
    out = {}
    for j, n in enumerate(C.REG):
        m = np.isfinite(Y[:, j])
        e = mu[m, j] - Y[m, j]
        base = Y[m, j] - np.median(Y[m, j])
        out[n] = dict(n=int(m.sum()), MAE=float(np.mean(np.abs(e))), MAE_median_predictor=float(np.mean(np.abs(base))),
                      R2=float(1 - np.mean(e ** 2) / np.var(Y[m, j])), cov1=float(np.mean(np.abs(e) <= sg[m, j])),
                      cov2=float(np.mean(np.abs(e) <= 2 * sg[m, j])))
    for (a, b, name, k) in ((0, NU, 'use', 0), (NU, NU + NH, 'heat', 1), (NU + NH, lg.shape[1], 'cool', 2)):
        p = lg[:, a:b].argmax(1)
        out[name] = dict(balanced_accuracy=E.bacc(K[:, k], p), accuracy=float(np.mean(K[:, k] == p)))
    # heat pump vs resistance among electrically heated sims
    ph = lg[:, NU:NU + NH]
    e_ = np.isin(K[:, 1], [0, 1])
    s = ph[e_, 0] - ph[e_, 1]
    out['hp_vs_resistance'] = dict(n=int(e_.sum()), auc=E.auc(s, K[e_, 1] == 0),
                                   balanced_accuracy=E.bacc(K[e_, 1] == 0, s > 0))
    return out


# ----------------------------------------------------------------------------- classical baselines on simulated units
PH = 3 + 2 * F.NB + 168 + 48            # PRISM block offset inside an hourly channel block
PM = 3 + 12                             # ... inside a monthly channel block
ETA = dict(fuel=0.85, elec=1.0, dheat=1.0, dcool=1.0)


def prism_from_feats(X, mode):
    """PRISM 5P baseline read from the PRISM block of the same (quirked) meters the Detective sees.
    Heating meter: fuel if present, else district heat, else electricity; cooling meter: district chw, else electricity.
    Returns dict of arrays (NaN where the fit found no heating / cooling slope)."""
    nw, nc, pp = (F.N_W_H, F.N_C_H, PH) if mode == 'hourly' else (F.N_W_M, F.N_C_M, PM)
    o = {ch: nw + i * nc for i, ch in enumerate(C.CHANNELS)}
    n = len(X)
    out = {k: np.full(n, np.nan) for k in ('Tb_heat', 'Tb_cool', 'logS_heat', 'logS_cool', 'logUA')}
    for i in range(n):
        hc = next((c for c in ('fuel', 'dheat', 'elec') if X[i, o[c]] > 0), None)
        cc = next((c for c in ('dcool', 'elec') if X[i, o[c]] > 0), None)
        if hc:
            b = o[hc] + pp
            mu = 10 ** X[i, o[hc] + 2]
            h = X[i, b + 1] / 10 * mu * 1000          # W/K
            if h > 0:
                out['Tb_heat'][i] = X[i, b + 3] * 10
                out['logS_heat'][i] = np.log10(h)
                out['logUA'][i] = np.log10(h * ETA[hc])
        if cc:
            b = o[cc] + pp
            mu = 10 ** X[i, o[cc] + 2]
            c = X[i, b + 2] / 10 * mu * 1000
            if c > 0:
                out['Tb_cool'][i] = X[i, b + 4] * 10
                out['logS_cool'][i] = np.log10(c)
    return out


def sim_baselines(mode, X, Y, K, Ytr, Ktr, Xa=None, Xa_tr=None, Ytr_a=None, Ktr_a=None):
    """Baselines on simulated units: PRISM (hourly/monthly), median by TRUE use type, floor area from energy with the
    TRUE use (annual features), heating-fuel rule, majority class."""
    R = {}
    if mode in ('hourly', 'monthly'):
        P = prism_from_feats(X, mode)
        for k, v in P.items():
            j = J[k]
            m = np.isfinite(v) & np.isfinite(Y[:, j])
            e = v[m] - Y[m, j]
            R['prism_' + k] = dict(n=int(m.sum()), share_fitted=float(m.sum() / max(np.isfinite(Y[:, j]).sum(), 1)),
                                   MAE=float(np.mean(np.abs(e))), MdAE=float(np.median(np.abs(e))),
                                   bias=float(np.median(e)))
    # median by true use type (all regression targets)
    for j, k in enumerate(C.REG):
        med = {u: np.nanmedian(Ytr[Ktr[:, 0] == u, j]) for u in range(len(C.USES)) if np.isfinite(Ytr[Ktr[:, 0] == u, j]).any()}
        pr = np.array([med.get(u, np.nan) for u in K[:, 0]])
        m = np.isfinite(pr) & np.isfinite(Y[:, j])
        R['median_by_use_' + k] = dict(n=int(m.sum()), MAE=float(np.mean(np.abs(pr[m] - Y[m, j]))),
                                       MdAE=float(np.median(np.abs(pr[m] - Y[m, j]))))
    # heating-fuel rule from which meters exist (hourly/monthly: channel present flags; annual: same)
    nw, nc = {'hourly': (F.N_W_H, F.N_C_H), 'monthly': (F.N_W_M, F.N_C_M), 'annual': (4, 2)}[mode]
    pres = {ch: X[:, nw + i * nc] > 0 for i, ch in enumerate(C.CHANNELS)}
    elec_maj = int(np.bincount(Ktr[np.isin(Ktr[:, 1], [0, 1]), 1]).argmax())
    rule = np.where(pres['fuel'], C.HEAT.index('fuel'), np.where(pres['dheat'], C.HEAT.index('district'), elec_maj))
    R['heat_fuel_rule'] = dict(balanced_accuracy=E.bacc(K[:, 1], rule), accuracy=float(np.mean(rule == K[:, 1])))
    for t, name in ((0, 'use'), (1, 'heat'), (2, 'cool')):
        maj = int(np.bincount(Ktr[:, t]).argmax())
        R['majority_' + name] = dict(cls=[C.USES, C.HEAT, C.COOL][t][maj], accuracy=float(np.mean(K[:, t] == maj)),
                                     balanced_accuracy=E.bacc(K[:, t], np.full(len(K), maj)))
    if Xa is not None:
        e_tr = 10 ** (Xa_tr[:, 5] + 5) / 10 ** Ytr_a[:, J['logA']]
        med = {u: np.nanmedian(e_tr[(Ktr_a[:, 0] == u) & (Xa_tr[:, 4] > 0)]) for u in range(len(C.USES))}
        ok = Xa[:, 4] > 0
        A_hat = 10 ** (Xa[ok, 5] + 5) / np.array([med[u] for u in K[ok, 0]])
        A = 10 ** Y[ok, J['logA']]
        R['area_from_energy_true_use'] = dict(n=int(ok.sum()), MdAPE=float(np.median(np.abs(A_hat - A) / A) * 100))
    return R


def survey_rules(split, name):
    """Heating-fuel rule ('a non-electric fuel is used -> fuel, else electric') and majority on our grouping, same units."""
    mats, labs = E.load(split, name)
    if name == 'recs':
        grp = {'heat_pump': 'electric', 'resistance': 'electric', 'fuel': 'fuel', 'none': 'none', 'district': 'fuel'}
    else:
        grp = {'heat_pump': 'electric', 'resistance': 'electric', 'fuel': 'fuel', 'none': 'none', 'district': 'district'}
    t = np.array([grp.get(l['heat'], 'nan') if l['heat'] else 'nan' for l in labs])
    rule = []
    for l in labs:
        k = l['kwh']
        if (k.get('fuel') or 0) > 0:
            rule.append('fuel')
        elif (k.get('dheat') or 0) > 0:
            rule.append('district')
        else:
            rule.append('electric')
    rule = np.array(rule)
    m = t != 'nan'
    from collections import Counter
    _, ld = E.load('dev', name)
    maj = Counter(grp.get(l['heat'], 'nan') for l in ld if l['heat']).most_common(1)[0][0]
    def mid(L):
        lo = np.array([l['year_lo'] for l in L], float); hi = np.array([l['year_hi'] for l in L], float)
        return np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2, np.where(np.isfinite(hi), hi - 10, lo + 5))
    ym, yd = mid(labs), mid(ld)
    k = np.isfinite(ym)
    yb = float(np.nanmedian(yd))
    A = np.array([l['A'] for l in labs], float); Ad = np.array([l['A'] for l in ld], float)
    use = [l['use'] for l in labs]; used = [l['use'] for l in ld]
    medA = {u: np.nanmedian([a for a, v in zip(Ad, used) if v == u]) for u in set(used)}
    pa = np.array([medA.get(u, np.nanmedian(Ad)) for u in use])
    ka = np.isfinite(A) & (A > 0)
    return dict(n=int(m.sum()), fuel_rule_accuracy=float(np.mean(rule[m] == t[m])), majority=maj,
                majority_accuracy=float(np.mean(t[m] == maj)),
                year_dev_median=yb, year_dev_median_MAE=float(np.mean(np.abs(ym[k] - yb))),
                area_dev_median_by_true_use_MdAPE=float(np.median(np.abs(pa[ka] - A[ka]) / A[ka]) * 100),
                area_dev_median_MdAPE=float(np.median(np.abs(np.nanmedian(Ad) - A[ka]) / A[ka]) * 100))


def cbecs_hours(net, split):
    """Weekly operating hours: Detective (annual input) vs CBECS answer; baselines dev median overall and by TRUE use."""
    mats, labs = E.load(split, 'cbecs')
    mu, sg, lg = E.run_net(net, mats['annual'])
    wk = np.clip(mu[:, J['op_hours']], 0, 24) * (5 + 2 * np.clip(mu[:, J['weekend']], 0, 1.5))
    h = np.array([l['hours_per_week'] if l['hours_per_week'] is not None else np.nan for l in labs], float)
    _, ld = E.load('dev', 'cbecs')
    hd = np.array([l['hours_per_week'] if l['hours_per_week'] is not None else np.nan for l in ld], float)
    med = float(np.nanmedian(hd))
    byuse = {u: float(np.nanmedian([x for x, l in zip(hd, ld) if l['use'] == u])) for u in set(l['use'] for l in ld)}
    bu = np.array([byuse.get(l['use'], med) for l in labs])
    m = np.isfinite(h) & np.isfinite(wk)
    from scipy.stats import spearmanr
    return dict(n=int(m.sum()), detective_MAE_h=float(np.mean(np.abs(wk[m] - h[m]))),
                spearman=float(spearmanr(wk[m], h[m]).correlation),
                dev_median_MAE_h=float(np.mean(np.abs(med - h[m]))),
                median_by_true_use_MAE_h=float(np.nanmean(np.abs(bu[m] - h[m]))))


def stock_eval(nets):
    """Simulator vs simulator: Detective on ResStock / ComStock EnergyPlus meters; UA / tau / balance-point truth from
    a BuildingSim twin of the same characteristics; heat / cool class, area, vintage from the stock characteristics."""
    out = {}
    for kind in ('resstock', 'comstock'):
        fn = os.path.join(C.ROOT, 'feats', 'stock', f'{kind}.npz')
        if not os.path.exists(fn):
            continue
        z = np.load(fn)
        labs = json.load(open(fn.replace('.npz', '.json')))
        Y = C.clean_labels(z['Y'])
        Y[:, J['logA']] = np.log10([l['A'] for l in labs])
        Y[:, J['year']] = [l['year'] for l in labs]
        r = {}
        for mode in ('hourly', 'monthly', 'annual'):
            if mode not in nets:
                continue
            X = z[mode]
            mu, sg, lg = nets[mode].raw(X)
            rr = {}
            for k in ('logA', 'year', 'logUA', 'logUAm2', 'logtau', 'Tb_heat', 'Tb_cool'):
                j = J[k]
                m = np.isfinite(Y[:, j])
                e = mu[m, j] - Y[m, j]
                rr[k] = dict(n=int(m.sum()), MAE=float(np.mean(np.abs(e))), bias=float(np.median(e)),
                             MAE_median_predictor=float(np.mean(np.abs(Y[m, j] - np.median(Y[m, j])))),
                             cov1=float(np.mean(np.abs(e) <= sg[m, j])))
            if mode != 'annual':
                P = prism_from_feats(X, mode)
                for k in ('logUA', 'Tb_heat', 'Tb_cool'):
                    v = P[k]
                    m = np.isfinite(v) & np.isfinite(Y[:, J[k]])
                    rr['prism_' + k] = dict(n=int(m.sum()), MAE=float(np.mean(np.abs(v[m] - Y[m, J[k]]))))
                    m2 = m & np.isfinite(mu[:, J[k]])
                    rr['detective_on_prism_units_' + k] = dict(n=int(m2.sum()), MAE=float(np.mean(np.abs(mu[m2, J[k]] - Y[m2, J[k]]))))
            ph = lg[:, NU:NU + NH]
            t = np.array([l['heat'] or 'nan' for l in labs])
            pred = np.array([C.HEAT[i] for i in ph.argmax(1)])
            k = t != 'nan'
            rr['heat'] = dict(n=int(k.sum()), accuracy=float(np.mean(pred[k] == t[k])), balanced_accuracy=E.bacc(t[k], pred[k]))
            e_ = np.isin(t, ['heat_pump', 'resistance'])
            if e_.sum() > 3:
                sc = ph[e_, 0] - ph[e_, 1]
                rr['hp_vs_resistance'] = dict(n=int(e_.sum()), auc=E.auc(sc, t[e_] == 'heat_pump'))
            if kind == 'comstock':
                tc = np.array([l['cool'] for l in labs])
                pc = np.array([C.COOL[i] for i in lg[:, NU + NH:].argmax(1)])
                rr['cool'] = dict(accuracy=float(np.mean(pc == tc)), balanced_accuracy=E.bacc(tc, pc))
                ci = [C.USES.index(u) for u in COM_USES]
                pu = np.array([COM_USES[i] for i in lg[:, ci].argmax(1)])
                tu = np.array([l['use'] for l in labs])
                rr['use'] = dict(accuracy=float(np.mean(pu == tu)), macroF1=E.macro_f1(list(tu), list(pu)))
            r[mode] = rr
        out[kind] = r
    return out


COM_USES = [u for u in C.USES if not u.startswith('residential')]


def main(wdir, feat_tag='f2', twin_tag='devtwin_v2'):
    import train as T
    nets = E.load_nets(wdir)
    R = dict(weights=wdir, feat_tag=feat_tag, twin_tag=twin_tag, note='exam run once after all choices were frozen')
    # ---------------- real: BDG2
    R['bdg2'] = E.eval_bdg2(nets, 'exam')
    _, ld = E.load('dev', 'bdg2')
    _, lx = E.load('exam', 'bdg2')
    R['bdg2_baselines'] = bdg2_baselines(ld, lx)
    R['bdg2_baselines_B'] = bd_baselines()
    # ---------------- real: RECS / CBECS / HUE
    R['recs'] = E.eval_recs(nets['annual'], 'exam')
    R['recs_rules'] = survey_rules('exam', 'recs')
    R['cbecs'] = E.eval_cbecs(nets['annual'], 'exam')
    R['cbecs_rules'] = survey_rules('exam', 'cbecs')
    R['cbecs_hours'] = cbecs_hours(nets['annual'], 'exam')
    from collections import Counter
    _, cd = E.load('dev', 'cbecs')
    _, cx = E.load('exam', 'cbecs')
    maj = Counter(l['use'] for l in cd).most_common(1)[0][0]
    t = [l['use'] for l in cx]
    R['cbecs_use_majority'] = dict(cls=maj, accuracy=float(np.mean(np.array(t) == maj)), macroF1=E.macro_f1(t, [maj] * len(t)))
    if os.path.exists(os.path.join(E.OD, 'exam_hue.npz')):
        R['hue'] = E.eval_hue(nets)
    # ---------------- simulator vs simulator
    R['stock'] = stock_eval(nets)
    # ---------------- simulated only: frozen validation set and dev twins, with classical baselines
    R['sim_val'], R['dev_twins'], R['sim_baselines'] = {}, {}, {}
    Xa, Ya, Ka, _ = T.load_feats(feat_tag, 'annual', True)
    Xat, Yat, Kat, _ = T.load_feats(feat_tag, 'annual', False)
    for mode, net in nets.items():
        Xv, Yv, Kv, Iv = T.load_feats(feat_tag, mode, True)
        _, Ytr, Ktr, _ = T.load_feats(feat_tag, mode, False)
        R['sim_val'][mode] = sim_only(net, Xv, Yv, Kv, mode)
        R['sim_baselines'][mode] = sim_baselines(mode, Xv, Yv, Kv, Ytr, Ktr,
                                                 *((Xa, Xat, Yat, Kat) if mode == 'annual' else (None,) * 4))
        tw = np.load(os.path.join(C.ROOT, 'feats', 'real', f'{twin_tag}_{mode}.npz'))
        R['dev_twins'][mode] = sim_only(net, tw['X'], C.clean_labels(tw['Y']), tw['K'], mode)
    json.dump(R, open(os.path.join(C.ROOT, 'exam.json'), 'w'), indent=1, default=float)
    print(json.dumps(E.strip(R), indent=1, default=float)[:3000])


def bd_baselines():
    import buildingdata as bd
    return bd.baselines()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.ROOT, 'models', 'final'), *sys.argv[2:4])
