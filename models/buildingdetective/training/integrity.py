"""Integrity pass (17 Nov 2026; the exam is SEEN): re-score the real exam rows with the released weights, add fair
baselines fitted on dev only, and put a 95 % bootstrap interval on Detective minus the best baseline for every row.
Also the dev twin test of what the network adds. Plan: NOTEBOOK.md, entry of 17 Nov 2026 (written before this ran).

Reads WORK/feats/real/<split>_<set>.{npz,json} (integrity_feats.py) and weights/. Writes WORK/integrity.json and
prints a summary. Nothing here is used to choose anything about the model.
usage: python integrity.py
"""
import json
import os
import sys
import warnings
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr, pearsonr

warnings.filterwarnings('ignore')
import common as C            # noqa: E402
import evaluate as E          # noqa: E402
import exam as X              # noqa: E402
import features as F          # noqa: E402
from _paths import MODEL, SRC, WORK  # noqa: E402

SEED, NB = 20261117, 2000
J = {n: j for j, n in enumerate(C.REG)}
NU, NH = len(C.USES), len(C.HEAT)
LAM = 1.0                        # L2 penalty of every dev-fitted linear / logistic baseline (standardised inputs)


# ----------------------------------------------------------------------------- statistics
def s_mdape(p, y):
    return np.median(np.abs(p - y) / y, axis=-1) * 100


def s_mae(p, y):
    return np.mean(np.abs(p - y), axis=-1)


def s_acc(p, y):
    return np.mean(p == y, axis=-1) * 100


def s_bacc(p, y):
    """p, y: (..., n) arrays of labels; balanced accuracy over the classes present in y (per row)."""
    p, y = np.atleast_2d(p), np.atleast_2d(y)
    out = []
    for pi, yi in zip(p, y):
        out.append(np.mean([np.mean(pi[yi == c] == c) for c in np.unique(yi)]))
    return np.array(out) if len(out) > 1 else out[0]


def s_auc(s, y):
    s, y = np.atleast_2d(s), np.atleast_2d(y).astype(bool)
    out = []
    for si, yi in zip(s, y):
        n1, n0 = yi.sum(), (~yi).sum()
        if n1 == 0 or n0 == 0:
            out.append(np.nan)
            continue
        r = rankdata(si)
        out.append((r[yi].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return np.array(out) if len(out) > 1 else out[0]


STAT = dict(mdape=(s_mdape, 'lower'), mae=(s_mae, 'lower'), acc=(s_acc, 'higher'), bacc=(s_bacc, 'higher'),
            auc=(s_auc, 'higher'))


def boot(stat, pm, pb, y, chunk=250):
    """95 % interval of stat(model) - stat(baseline), resampling units (same resample for both)."""
    f, _ = STAT[stat]
    n = len(y)
    rng = np.random.default_rng(SEED)
    d = []
    for k in range(0, NB, chunk):
        idx = rng.integers(0, n, (min(chunk, NB - k), n))
        if stat in ('bacc', 'auc'):
            d.extend(np.asarray(f(pm[idx], y[idx])) - np.asarray(f(pb[idx], y[idx])))
        else:
            d.extend(f(pm[idx], y[idx]) - f(pb[idx], y[idx]))
    d = np.array(d, float)
    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)), int(np.isnan(d).sum())


def verdict(lo, hi, better):
    if better == 'lower':
        return 'win' if hi < 0 else ('loss' if lo > 0 else 'tie')
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')


def row(name, stat, y, pm, bases, notes=''):
    """bases: list of (label, pred, needs_truth). Each baseline is scored on the units where model, baseline and
    truth are all defined. Interval against the best baseline (and against the best one that needs no truth)."""
    f, better = STAT[stat]
    y = np.asarray(y)
    pm = np.asarray(pm)

    def ok(a):
        a = np.asarray(a)
        return np.array([v is not None and v == v for v in a]) if a.dtype == object else np.isfinite(a.astype(float))
    out = dict(row=name, stat=stat, better=better, notes=notes, baselines=[])
    mm = ok(y) & ok(pm)
    out['n'] = int(mm.sum())
    out['model'] = float(f(pm[mm], y[mm]))
    for lab, pb, truth in bases:
        pb = np.asarray(pb)
        m = mm & ok(pb)
        out['baselines'].append(dict(name=lab, needs_truth=truth, n=int(m.sum()), value=float(f(pb[m], y[m])),
                                     model_same_rows=float(f(pm[m], y[m])), _m=m, _p=pb))
    sgn = 1 if better == 'lower' else -1

    def best(sel):
        c = [b for b in out['baselines'] if sel(b) and np.isfinite(b['value'])]
        return min(c, key=lambda b: sgn * b['value']) if c else None
    for key, sel in (('vs_best', lambda b: True), ('vs_best_no_truth', lambda b: not b['needs_truth'])):
        b = best(sel)
        if b is None:
            continue
        m = b['_m']
        lo, hi, nn = boot(stat, pm[m], b['_p'][m], y[m])
        out[key] = dict(baseline=b['name'], n=int(m.sum()), diff=b['model_same_rows'] - b['value'], lo=lo, hi=hi,
                        nan_resamples=nn, verdict=verdict(lo, hi, better))
    for b in out['baselines']:                      # interval against every baseline too (same resamples)
        m = b.pop('_m')
        p = b.pop('_p')
        if np.isfinite(b['value']) and m.sum() > 1:
            lo, hi, _ = boot(stat, pm[m], p[m], y[m])
            b.update(diff=b['model_same_rows'] - b['value'], lo=lo, hi=hi, verdict=verdict(lo, hi, better))
    return out


# ----------------------------------------------------------------------------- dev-fitted classical models
def _std(Xd):
    mu, sd = np.nanmean(Xd, 0), np.nanstd(Xd, 0)
    sd[sd == 0] = 1
    return mu, sd


def ridge(Xd, yd):
    mu, sd = _std(Xd)
    Z = np.c_[np.ones(len(Xd)), (Xd - mu) / sd]
    P = LAM * np.eye(Z.shape[1])
    P[0, 0] = 0
    w = np.linalg.solve(Z.T @ Z + P, Z.T @ yd)
    return lambda Xe: np.c_[np.ones(len(Xe)), (Xe - mu) / sd] @ w


def logistic(Xd, yd, classes=None):
    """Multinomial logistic regression, L2 = LAM. Returns predict_proba(Xe) -> (n, K) and the class list."""
    classes = list(classes or sorted(set(yd)))
    mu, sd = _std(Xd)
    Z = np.c_[np.ones(len(Xd)), (Xd - mu) / sd]
    Y = np.array([[c == k for k in classes] for c in yd], float)
    K, D = len(classes), Z.shape[1]

    def fg(w):
        W = w.reshape(K, D)
        s = Z @ W.T
        s -= s.max(1, keepdims=True)
        P = np.exp(s)
        P /= P.sum(1, keepdims=True)
        reg = W.copy()
        reg[:, 0] = 0
        loss = -np.sum(Y * np.log(P + 1e-12)) + 0.5 * LAM * np.sum(reg ** 2)
        g = (P - Y).T @ Z + LAM * reg
        return loss, g.ravel()
    w = minimize(fg, np.zeros(K * D), jac=True, method='L-BFGS-B').x.reshape(K, D)

    def proba(Xe):
        s = np.c_[np.ones(len(Xe)), (Xe - mu) / sd] @ w.T
        s -= s.max(1, keepdims=True)
        P = np.exp(s)
        return P / P.sum(1, keepdims=True)
    return proba, classes


def ymid(labs):
    lo = np.array([l['year_lo'] for l in labs], float)
    hi = np.array([l['year_hi'] for l in labs], float)
    return np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2, np.where(np.isfinite(hi), hi - 10, lo + 5))


def wmedian(x, w):
    i = np.argsort(x)
    cw = np.cumsum(w[i])
    return float(x[i][np.searchsorted(cw, cw[-1] / 2)])


def dev_threshold(x, t):
    """BuildingData's procedure: threshold on x (low -> positive) maximising dev balanced accuracy over 91 quantiles."""
    ths = np.quantile(x, np.linspace(0.05, 0.95, 91))
    ba = lambda h: 0.5 * ((x[t] < h).mean() + (x[~t] >= h).mean())
    return max(ths, key=ba)


# ----------------------------------------------------------------------------- reproduction against exam.json
def flat(d, p=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f'{p}.{k}' if p else k))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[p] = float(d)
    elif isinstance(d, list) and d and all(isinstance(v, (int, float)) for v in d):
        out[p] = np.array(d, float)
    return out


def reproduce(nets):
    R = {}
    R['bdg2'] = E.eval_bdg2(nets, 'exam')
    _, ld = E.load('dev', 'bdg2')
    _, lx = E.load('exam', 'bdg2')
    R['bdg2_baselines'] = X.bdg2_baselines(ld, lx)
    R['recs'] = E.eval_recs(nets['annual'], 'exam')
    R['recs_rules'] = X.survey_rules('exam', 'recs')
    R['cbecs'] = E.eval_cbecs(nets['annual'], 'exam')
    R['cbecs_rules'] = X.survey_rules('exam', 'cbecs')
    R['cbecs_hours'] = X.cbecs_hours(nets['annual'], 'exam')
    _, cd = E.load('dev', 'cbecs')
    _, cx = E.load('exam', 'cbecs')
    maj = Counter(l['use'] for l in cd).most_common(1)[0][0]
    t = [l['use'] for l in cx]
    R['cbecs_use_majority'] = dict(cls=maj, accuracy=float(np.mean(np.array(t) == maj)), macroF1=E.macro_f1(t, [maj] * len(t)))
    R['hue'] = E.eval_hue(nets)
    old = json.load(open(os.path.join(SRC, 'exam.json')))
    a, b = flat({k: old[k] for k in R}), flat(R)
    rep = dict(n_compared=0, max_abs_diff=0.0, worst=None, missing=[], lists=0, list_max_abs_diff=0.0)
    for k, v in b.items():
        if k not in a:
            rep['missing'].append(k)
            continue
        w = a[k]
        if isinstance(v, np.ndarray):
            if len(v) != len(w):
                rep['missing'].append(k + ' (length %d vs %d)' % (len(v), len(w)))
                continue
            rep['lists'] += 1
            dd = float(np.nanmax(np.abs(v - w))) if len(v) else 0.0
            rep['list_max_abs_diff'] = max(rep['list_max_abs_diff'], dd)
            continue
        rep['n_compared'] += 1
        dd = abs(v - w)
        if dd > rep['max_abs_diff']:
            rep['max_abs_diff'], rep['worst'] = dd, (k, v, w)
    rep['old_keys_not_recomputed'] = [k for k in a if k not in b][:20]
    return R, rep


# ----------------------------------------------------------------------------- the rows
def bdg2_rows(nets, rows, twin):
    md, ld = E.load('dev', 'bdg2')
    mx, lx = E.load('exam', 'bdg2')
    A = np.array([l['A'] for l in lx], float)
    Ad = np.array([l['A'] for l in ld], float)
    el = np.array([l['kwh'].get('elec', 0) or np.nan for l in lx], float)
    eld = np.array([l['kwh'].get('elec', 0) or np.nan for l in ld], float)
    el[el <= 0], eld[eld <= 0] = np.nan, np.nan
    eui_d = eld / Ad
    use_d = np.array([l['use'] for l in ld])
    use_x = np.array([l['use'] for l in lx])
    eui_all = np.nanmedian(eui_d)
    eui_use = {u: np.nanmedian(eui_d[use_d == u]) for u in set(use_d)}
    eui_use = {u: (v if np.isfinite(v) else eui_all) for u, v in eui_use.items()}
    b_true = el / np.array([eui_use.get(u, eui_all) for u in use_x])
    b_all = el / eui_all
    k = np.isfinite(eld) & np.isfinite(Ad) & (Ad > 0)
    co = np.polyfit(np.log10(eld[k]), np.log10(Ad[k]), 1)
    b_ll = 10 ** np.polyval(co, np.log10(el))
    b_dumb = np.full(len(lx), np.nanmedian(Ad))
    out = {}
    for mode, key in (('hourly', 'hourly'), ('monthly', 'monthly'), ('annual', 'annual'), ('electricity only', 'hourly_elec')):
        net = nets['hourly' if key == 'hourly_elec' else key]
        mu, sg, lg = E.run_net(net, mx[key])
        out[key] = (mu, sg, lg)
        rows.append(row(f'BDG2 floor area, {mode} input', 'mdape', A, 10 ** mu[:, J['logA']], [
            ('Electricity ÷ dev median EUI of the TRUE use', b_true, True),
            ('Electricity ÷ dev median EUI of all uses', b_all, False),
            ('Dev log-log line: log A = %.2f + %.2f log kWh' % (co[1], co[0]), b_ll, False),
            ('Dev median floor area', b_dumb, False)]))
    # primary use: majority, and nearest dev centroid of the 168-h electricity week profile (same rows)
    o = F.N_W_H + 3 + 2 * F.NB
    has_e = lambda M: np.abs(M[:, F.N_W_H]) > 0
    Pd, Px = md['hourly'][:, o:o + 168], mx['hourly'][:, o:o + 168]
    cents = {u: Pd[(use_d == u) & has_e(md['hourly'])].mean(0) for u in set(use_d)
             if ((use_d == u) & has_e(md['hourly'])).any()}
    cl = list(cents)
    dist = np.stack([((Px - cents[u]) ** 2).sum(1) for u in cl], 1)
    cen = np.array([cl[i] for i in dist.argmin(1)], object)
    cen[~has_e(mx['hourly'])] = None
    maj = Counter(use_d).most_common(1)[0][0]
    for key in ('hourly', 'monthly', 'annual'):
        lg = out[key][2]
        pu = E._probs(lg, 0, NU)
        pred = np.array(['lodging' if C.USES[i].startswith('residential') else C.USES[i] for i in pu.argmax(1)], object)
        pred[~np.isfinite(lg).all(1)] = None
        rows.append(row(f'BDG2 primary use, {key} input', 'acc', use_x.astype(object), pred, [
            (f'Majority class ({maj})', np.full(len(lx), maj, object), False),
            ('Nearest dev centroid of the 168-h electricity week profile', cen, False)],
            notes='macro-F1: Detective %.3f, majority %.3f, centroid %.3f' % (
                E.macro_f1(list(use_x), list(pred)), E.macro_f1(list(use_x), [maj] * len(lx)),
                E.macro_f1([u for u, c in zip(use_x, cen) if c], [c for c in cen if c]))))
    # year built: dev median; dev median at the same site
    yx = np.array([l['year'] for l in lx], float)
    yd = np.array([l['year'] for l in ld], float)
    sd = np.array([l['site'] for l in ld])
    ymed = np.nanmedian(yd)
    ysite = {s: np.nanmedian(yd[sd == s]) for s in set(sd) if np.isfinite(yd[sd == s]).any()}
    b_ys = np.array([ysite.get(l['site'], ymed) for l in lx])
    for key in ('hourly', 'monthly', 'annual'):
        rows.append(row(f'BDG2 year built, {key} input', 'mae', yx, out[key][0][:, J['year']], [
            ('Dev median (%d)' % ymed, np.full(len(lx), ymed), False),
            ('Dev median of the same site', b_ys, False)]))
    # which meters exist, from electricity alone
    pe = out['hourly_elec']
    oe = F.N_W_H
    pp = oe + X.PH
    st = pp + 6 + 12 + 10

    def efeat(M):
        return np.c_[M[:, pp + 1], M[:, pp + 2], M[:, pp + 3], M[:, pp + 4], M[:, pp + 5], M[:, st + 0], M[:, st + 3],
                     M[:, st + 4], M[:, oe + 2]]
    okx = np.isfinite(mx['hourly_elec']).all(1)
    okd = np.isfinite(md['hourly_elec']).all(1)
    sx = np.array([l['site'] for l in lx])
    for lab, key, cls in (('chilled-water', 'has_chw', 'district'), ('steam or hot-water', 'has_dheat', 'district')):
        y = np.array([l[key] for l in lx], float)
        yd_ = np.array([l[key] for l in ld], bool)
        if key == 'has_chw':
            score = E._probs(pe[2], NU + NH, NU + NH + len(C.COOL))[:, C.COOL.index('district')]
        else:
            score = E._probs(pe[2], NU, NU + NH)[:, C.HEAT.index('district')]
        score[~okx] = np.nan
        p_all = yd_.mean()
        prior = {s: (yd_[sd == s].sum() + p_all) / ((sd == s).sum() + 1) for s in set(sd)}
        b_site = np.array([prior.get(s, p_all) for s in sx])
        b_site[~okx] = np.nan
        proba, cls_ = logistic(efeat(md['hourly_elec'][okd]), list(yd_[okd]), [False, True])
        b_log = np.full(len(lx), np.nan)
        b_log[okx] = proba(efeat(mx['hourly_elec'][okx]))[:, 1]
        y[~okx] = np.nan
        rows.append(row(f'BDG2 has a {lab} meter, from electricity alone (ROC AUC)', 'auc', y, score, [
            ('Dev site prior (share of dev buildings at the same site with one)', b_site, False),
            ('Dev-fitted logistic model on the electricity PRISM and load-shape statistics', b_log, False),
            ('Chance', np.zeros(len(lx)), False)]))
    # dev twin test, BDG2 dev: area correction to the no-truth rule
    tw = {}
    for key in ('hourly', 'monthly', 'annual'):
        mu, _, _ = E.run_net(nets[key], md[key])
        anc = np.log10(eld / eui_all)
        c = mu[:, J['logA']] - anc
        r = np.log10(Ad) - anc
        tw[key] = corr(c, r)
        tw[key]['use_breakdown'] = breakdown(c, r, use_d)
    twin['bdg2_area_vs_elec_over_dev_eui'] = tw


def corr(c, r):
    m = np.isfinite(c) & np.isfinite(r)
    if m.sum() < 5:
        return dict(n=int(m.sum()))
    return dict(n=int(m.sum()), pearson=float(pearsonr(c[m], r[m])[0]), spearman=float(spearmanr(c[m], r[m]).correlation),
                median_needed=float(np.median(r[m])), median_correction=float(np.median(c[m])),
                mae_anchor=float(np.mean(np.abs(r[m]))), mae_network=float(np.mean(np.abs(r[m] - c[m]))))


def breakdown(c, r, g, min_n=5):
    out = {}
    for u in sorted(set(g), key=str):
        m = (g == u) & np.isfinite(c) & np.isfinite(r)
        if m.sum() >= min_n:
            out[str(u)] = dict(n=int(m.sum()), median_needed=float(np.median(r[m])), median_correction=float(np.median(c[m])))
    return out


def survey_frame(name):
    import buildingdata as bd
    d = (bd.recs if name == 'recs' else bd.cbecs)()
    return d.set_index('id')


def survey_rows(nets, rows, twin):
    net = nets['annual']
    for name in ('recs', 'cbecs'):
        md, ld = E.load('dev', name)
        mx, lx = E.load('exam', name)
        Xd, Xx = md['annual'], mx['annual']
        mu, sg, lg = E.run_net(net, Xx)
        mud, _, lgd = E.run_net(net, Xd)
        full = survey_frame(name)
        tcol, czc = ('housing_type', 'climate_zone') if name == 'recs' else ('pba', 'climate_class')
        site = lambda L: np.array([sum(v for v in l['kwh'].values() if v is not None and np.isfinite(v)) for l in L], float)
        Sx, Sd = site(lx), site(ld)
        A, Ad = np.array([l['A'] for l in lx], float), np.array([l['A'] for l in ld], float)
        tx = np.array([full.loc[l['id'], tcol] for l in lx], object)
        td = np.array([full.loc[l['id'], tcol] for l in ld], object)
        cx = np.array([full.loc[l['id'], czc] for l in lx], object)
        cd = np.array([full.loc[l['id'], czc] for l in ld], object)
        okd = (Sd > 0) & (Ad > 0)
        eui = Sd / Ad
        e_tc = {}
        for k in set(zip(td[okd], cd[okd])):
            e_tc[k] = np.median(eui[okd & (td == k[0]) & (cd == k[1])])
        e_t = {t: np.median(eui[okd & (td == t)]) for t in set(td[okd])}
        b_tc = np.array([s / e_tc.get((t, c), e_t.get(t, np.nan)) if s > 0 else np.nan for s, t, c in zip(Sx, tx, cx)])
        a_t = {t: np.median(Ad[td == t]) for t in set(td)}
        b_dt = np.array([a_t.get(t, np.nan) for t in tx])
        # W1 targets: weighted median site EUI of the prior + dev respondents by primary use (as calib_intensity.py)
        fr = full[full.split != 'exam']
        cols = (['elec_kWh', 'gas_kWh', 'fueloil_kWh', 'propane_kWh', 'wood_kWh'] if name == 'recs'
                else ['elec_kWh', 'gas_kWh', 'fueloil_kWh', 'district_heat_kWh'])
        s_ = fr[cols].fillna(0).sum(axis=1)
        okf = (s_ > 0) & (fr.floor_area_m2 > 0)
        W1 = {u: wmedian((s_[g.index] / g.floor_area_m2).to_numpy(), g.weight.to_numpy()) for u, g in fr[okf].groupby('primary_use')}
        ux = np.array([l['use'] for l in lx])
        b_w1 = np.array([s / W1[u] if s > 0 and u in W1 else np.nan for s, u in zip(Sx, ux)])
        okr = okd & np.isfinite(Xd).all(1)
        reg = ridge(Xd[okr], np.log10(Ad[okr]))
        b_reg = 10 ** reg(Xx)
        rows.append(row(f'{name.upper()} floor area, annual energy', 'mdape', A, 10 ** mu[:, J['logA']], [
            ('Energy ÷ dev median EUI by TRUE type × climate (BuildingData rule, recomputed)', b_tc, True),
            ('Energy ÷ W1 target EUI of the TRUE use (prior + dev weighted median)', b_w1, True),
            ('Dev median by TRUE type', b_dt, True),
            ('Dev-fitted linear model of log area on the 12 annual features', b_reg, False),
            ('Energy ÷ dev median EUI of all respondents', np.where(Sx > 0, Sx / np.median(eui[okd]), np.nan), False)]))
        # twin test: network's area correction to the no-truth and the TRUE-type rule, on dev
        b_tc_d = np.array([s / e_tc.get((t, c), e_t.get(t, np.nan)) if s > 0 else np.nan for s, t, c in zip(Sd, td, cd)])
        anc0 = np.log10(np.where(Sd > 0, Sd / np.median(eui[okd]), np.nan))
        for lab, anc in (('area_vs_energy_over_dev_eui', anc0), ('area_vs_true_type_climate_rule', np.log10(b_tc_d))):
            c, r = mud[:, J['logA']] - anc, np.log10(Ad) - anc
            t_ = corr(c, r)
            t_['type_breakdown'] = breakdown(c, r, td)
            t_['climate_breakdown'] = breakdown(c, r, cd)
            twin[f'{name}_{lab}'] = t_
        # vintage
        ym, ymd = ymid(lx), ymid(ld)
        okv = np.isfinite(ymd) & np.isfinite(Xd).all(1)
        regv = ridge(Xd[okv], ymd[okv])
        ymed = np.nanmedian(ymd)
        rows.append(row(f'{name.upper()} vintage, annual energy', 'mae', ym, mu[:, J['year']], [
            ('Dev median (%d)' % ymed, np.full(len(lx), ymed), False),
            ('Dev-fitted linear model on the 12 annual features', regv(Xx), False)]))
        c, r = mud[:, J['year']] - ymed, ymd - ymed
        twin[f'{name}_vintage_vs_dev_median'] = corr(c, r)
        # heating class
        grp = ({'heat_pump': 'electric', 'resistance': 'electric', 'fuel': 'fuel', 'none': 'none', 'district': 'fuel'}
               if name == 'recs' else
               {'heat_pump': 'electric', 'resistance': 'electric', 'fuel': 'fuel', 'none': 'none', 'district': 'district'})
        tg = np.array([grp.get(l['heat'], None) if l['heat'] else None for l in lx], object)
        tgd = np.array([grp.get(l['heat'], None) if l['heat'] else None for l in ld], object)
        ph = E._probs(lg, NU, NU + NH)
        pg = np.array([grp[C.HEAT[i]] for i in ph.argmax(1)], object)

        def rule(L):
            out = []
            for l in L:
                k = l['kwh']
                out.append('fuel' if (k.get('fuel') or 0) > 0 else ('district' if (k.get('dheat') or 0) > 0 else 'electric'))
            return np.array(out, object)
        rx, rd = rule(lx), rule(ld)
        m = np.array([t is not None for t in tgd])
        proba, cls = logistic(Xd[m], list(tgd[m]))
        b_lr = np.array([cls[i] for i in proba(Xx).argmax(1)], object)
        maj = Counter(tgd[m]).most_common(1)[0][0]
        lab = 'fuel / electric / none' if name == 'recs' else 'fuel / electric / district / none'
        rows.append(row(f'{name.upper()} main heating, {lab}', 'acc', tg, pg, [
            ('Rule: fuel used → fuel, district used → district, else electric', rx, False),
            ('Dev-fitted multinomial logistic model on the 12 annual features', b_lr, False),
            (f'Majority ({maj})', np.full(len(lx), maj, object), False)]))
        phd = E._probs(lgd, NU, NU + NH)
        pfuel_d = np.array([sum(phd[i, j] for j, h in enumerate(C.HEAT) if grp[h] == 'fuel') for i in range(len(ld))])
        yf = np.array([float(t == 'fuel') if t is not None else np.nan for t in tgd])
        rf = (rd == 'fuel').astype(float)
        twin[f'{name}_heat_fuel_vs_meter_rule'] = corr(pfuel_d - rf, yf - rf)
        if name == 'recs':
            t = np.array([l['heat'] or 'nan' for l in lx])
            td_ = np.array([l['heat'] or 'nan' for l in ld])
            e_, ed = np.isin(t, ['heat_pump', 'resistance']), np.isin(td_, ['heat_pump', 'resistance'])
            s = ph[:, 0] / (ph[:, 0] + ph[:, 1])
            yhp = (t == 'heat_pump').astype(float)
            yhp[~e_] = np.nan
            elx = np.array([l['kwh'].get('elec') or np.nan for l in lx], float)
            hddx = np.array([l['hdd'] for l in lx], float)
            # BuildingData's rule: dev = electricity-fuelled heat_pump_air / baseboard / electric furnace homes
            fd = full.loc[[l['id'] for l in ld]]
            bdv = (fd.heating_fuel == 'electricity') & fd.heating_equipment.isin(['heat_pump_air', 'baseboard_resistance', 'electric_furnace'])
            xa = (fd.elec_kWh / fd.floor_area_m2 / np.maximum(fd.hdd18C_year, 100))[bdv].to_numpy()
            ta = (fd.heating_equipment == 'heat_pump_air')[bdv].to_numpy()
            th_a = dev_threshold(xa, ta)
            xb = (fd.elec_kWh / np.maximum(fd.hdd18C_year, 100))[bdv].to_numpy()
            th_b = dev_threshold(xb, ta)
            sa = elx / A / np.maximum(hddx, 100)
            sb = elx / np.maximum(hddx, 100)
            proba, cls = logistic(Xd[ed], list(td_[ed] == 'heat_pump'), [False, True])
            plr = proba(Xx)[:, 1]
            hp = lambda v: np.where(np.isfinite(v), v, np.nan)
            rows.append(row('RECS heat pump vs electric resistance, balanced accuracy', 'bacc', yhp,
                            hp((s > 0.5).astype(float)), [
                                ('kWh/m²/HDD < dev threshold, TRUE floor area (BuildingData rule, recomputed)', (sa < th_a).astype(float), True),
                                ('kWh/HDD < dev threshold (no floor area)', (sb < th_b).astype(float), False),
                                ('Dev-fitted logistic model on the 12 annual features', (plr > 0.5).astype(float), False),
                                ('Majority', np.zeros(len(lx)), False)]))
            rows.append(row('RECS heat pump vs electric resistance, ROC AUC', 'auc', yhp, s, [
                ('kWh/m²/HDD, TRUE floor area (low → heat pump)', -sa, True),
                ('kWh/HDD (low → heat pump)', -sb, False),
                ('Dev-fitted logistic model on the 12 annual features', plr, False),
                ('Chance', np.zeros(len(lx)), False)]))
        else:
            # primary use (commercial classes), chiller, heat-pump presence, weekly hours
            ci = [C.USES.index(u) for u in E.COM_USES]
            pu = E._probs(lg, 0, NU)
            pred = np.array([E.COM_USES[i] for i in pu[:, ci].argmax(1)], object)
            ud = np.array([l['use'] for l in ld], object)
            proba, cls = logistic(Xd, list(ud))
            b_lr = np.array([cls[i] for i in proba(Xx).argmax(1)], object)
            maj = Counter(ud).most_common(1)[0][0]
            ux_ = ux.astype(object)
            rows.append(row('CBECS primary use (10 commercial classes), accuracy', 'acc', ux_, pred, [
                (f'Majority class ({maj})', np.full(len(lx), maj, object), False),
                ('Dev-fitted multinomial logistic model on the 12 annual features', b_lr, False)],
                notes='macro-F1: Detective %.3f, majority %.3f, dev logistic %.3f' % (
                    E.macro_f1(list(ux_), list(pred)), E.macro_f1(list(ux_), [maj] * len(lx)), E.macro_f1(list(ux_), list(b_lr)))))
            pc = E._probs(lg, NU + NH, NU + NH + len(C.COOL))
            for lab, key, score in (('has a chiller', 'chiller', pc[:, C.COOL.index('chiller')]),
                                    ('has a heat pump', 'hp', ph[:, C.HEAT.index('heat_pump')])):
                y = np.array([l[key] for l in lx], float)
                y = np.where(np.isfinite(y), (y == 1).astype(float), np.nan)
                ydv = np.array([l[key] for l in ld], float)
                m = np.isfinite(ydv)
                proba, _ = logistic(Xd[m], list(ydv[m] == 1), [False, True])
                rows.append(row(f'CBECS {lab}, ROC AUC', 'auc', y, score, [
                    ('Dev-fitted logistic model on the 12 annual features', proba(Xx)[:, 1], False),
                    ('Chance', np.zeros(len(lx)), False)]))
            wk = np.clip(mu[:, J['op_hours']], 0, 24) * (5 + 2 * np.clip(mu[:, J['weekend']], 0, 1.5))
            h = np.array([l['hours_per_week'] if l['hours_per_week'] is not None else np.nan for l in lx], float)
            hd = np.array([l['hours_per_week'] if l['hours_per_week'] is not None else np.nan for l in ld], float)
            med = np.nanmedian(hd)
            byuse = {u: np.nanmedian(hd[ud == u]) for u in set(ud)}
            m = np.isfinite(hd)
            regh = ridge(Xd[m], hd[m])
            rows.append(row('CBECS weekly operating hours, mean abs. error', 'mae', h, wk, [
                ('Dev median', np.full(len(lx), med), False),
                ('Dev median of the TRUE use', np.array([byuse.get(u, med) for u in ux]), True),
                ('Dev-fitted linear model on the 12 annual features', regh(Xx), False)]))
            wkd = np.clip(mud[:, J['op_hours']], 0, 24) * (5 + 2 * np.clip(mud[:, J['weekend']], 0, 1.5))
            anc = np.array([byuse.get(u, med) for u in ud])
            twin['cbecs_hours_vs_true_use_median'] = corr(wkd - anc, hd - anc)
            twin['cbecs_hours_vs_true_use_median']['use_breakdown'] = breakdown(wkd - anc, hd - anc, ud)


def hue_rows(nets, rows):
    import realdata as R
    mx, lx = E.load('exam', 'hue')
    cs = {c['id']: c for c in R.hue_cases()}
    share = []
    for l in lx:
        c = cs[l['id']]
        x, T, loc = c['meters']['elec'], c['T'], c['loc']
        d = np.asarray(loc.normalize().asi8)
        ud, inv = np.unique(d, return_inverse=True)
        v = np.isfinite(x)
        nv = np.bincount(inv, weights=v.astype(float))
        Ed = np.bincount(inv, weights=np.where(v, x, 0)) / np.maximum(nv, 1)
        Ed[nv < 12] = np.nan
        Td = np.bincount(inv, weights=T) / np.bincount(inv)
        b, hs, cs_, tbh, tbc, r2 = F.prism(Td, Ed)
        share.append(hs * np.mean(np.maximum(tbh - Td, 0)) / np.nanmean(Ed))
    share = np.array(share)
    y = np.array([float(l['electric_heat']) for l in lx])
    for mode in ('hourly', 'monthly'):
        mu, sg, lg = E.run_net(nets[mode], mx[mode])
        ph = E._probs(lg, NU, NU + NH)
        pe = ph[:, C.HEAT.index('heat_pump')] + ph[:, C.HEAT.index('resistance')]
        pred = np.isin(np.array([C.HEAT[i] for i in ph.argmax(1)]), ['heat_pump', 'resistance']).astype(float)
        rows.append(row(f'HUE electric main heat, {mode} electricity, accuracy', 'acc', y, pred, [
            ('Majority (not electric)', np.zeros(len(lx)), False),
            ('PRISM heating share of electricity ≥ 0.20 (threshold fixed before HUE was read)', (share >= 0.2).astype(float), False)]))
        rows.append(row(f'HUE electric main heat, {mode} electricity, ROC AUC', 'auc', y, pe, [
            ('PRISM heating share of electricity (change-point fit of daily kWh)', share, False),
            ('Chance', np.zeros(len(lx)), False)]))
    return dict(prism_heating_share=dict(zip([l['id'] for l in lx], share.round(4).tolist())))


def main():
    nets = E.load_nets(os.path.join(MODEL, 'weights'))
    R, rep = reproduce(nets)
    print('reproduction:', json.dumps(rep, default=str)[:2000], flush=True)
    rows, twin = [], {}
    bdg2_rows(nets, rows, twin)
    survey_rows(nets, rows, twin)
    hue = hue_rows(nets, rows)
    out = dict(note='integrity pass 17 Nov 2026; exam SEEN; nothing chosen from it', seed=SEED, n_boot=NB, lam=LAM,
               reproduction=rep, rows=rows, twin=twin, hue=hue, recomputed=E.strip(R))
    json.dump(out, open(os.path.join(WORK, 'integrity.json'), 'w'), indent=1, default=float)
    for r in rows:
        vb = r.get('vs_best', {})
        vn = r.get('vs_best_no_truth', {})
        print('%-70s n=%5d model %.3f | best %s %.3f [%.3f, %.3f] %s | no-truth %.3f [%.3f, %.3f] %s' % (
            r['row'][:70], r['n'], r['model'], vb.get('baseline', '')[:30], vb.get('diff', np.nan), vb.get('lo', np.nan),
            vb.get('hi', np.nan), vb.get('verdict'), vn.get('diff', np.nan), vn.get('lo', np.nan), vn.get('hi', np.nan),
            vn.get('verdict')))
    print(json.dumps(twin, indent=1, default=float)[:6000])


if __name__ == '__main__':
    main()
