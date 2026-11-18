"""Rung r10 (physics 1.1 + 1.2 + 1.4, bank p12, features f3): plan in NOTEBOOK.md, entry of 17 Nov 2026 (written
before the bank ran). Dev and simulation only: no exam unit and no HEAPO home is read here.

    python r10.py rebaseline          # R9 (weights/) on the f3 frozen set -> WORK/r10.json['rebaseline']
    python r10.py decide              # rule per network, chained hourly -> monthly -> annual -> WORK/r10.json['rungs']
    python r10.py devreport           # dev twin test + 5-fold-CV dev baselines, v1.0 vs r10 -> WORK/r10.json['dev']

Candidates are WORK/models/r10_<mode>/final.pt (train.py r10_<mode> <mode> f3), exported to WORK/models/r10/<mode>.npz.
The rule is the ladder's: keep if frozen-val skill > incumbent + 0.01 AND the dev composite is no more than 0.02
below the composite of the last kept set (starting from v1.0's, recomputed with today's features).
"""
import json
import os
import shutil
import sys
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings('ignore')
import common as C            # noqa: E402
import evaluate as E          # noqa: E402
import integrity as I         # noqa: E402
from detective import Net     # noqa: E402
from _paths import MODEL, WORK  # noqa: E402

FEAT = 'f3'
MODES = ('hourly', 'monthly', 'annual')
MARGIN, GATE_TOL = 0.01, 0.02
OUT = os.path.join(WORK, 'r10.json')
J = I.J
NU, NH = I.NU, I.NH
SEED = 20261117


def load_out():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(R):
    json.dump(R, open(OUT, 'w'), indent=1, default=float)


def val_feats(mode):
    import glob
    fs = sorted(f for f in glob.glob(os.path.join(C.ROOT, 'feats', FEAT, f'{mode}_09*.npz')) if not f.endswith('.tmp.npz'))
    z = [np.load(f) for f in fs]
    return (np.concatenate([a['X'] for a in z]), C.clean_labels(np.concatenate([a['Y'] for a in z])),
            np.concatenate([a['K'] for a in z]), [os.path.basename(f) for f in fs])


def skills(mu, sg, lg, Y, K):
    """train.skills without torch (same formula)."""
    S = {}
    for j, n in enumerate(C.REG):
        m = np.isfinite(Y[:, j])
        if m.sum() < 20:
            continue
        S[n] = float(max(1 - np.mean((mu[m, j] - Y[m, j]) ** 2) / np.var(Y[m, j]), 0))
    NC = len(C.COOL)
    for (a, b, name) in ((0, NU, 'use'), (NU, NU + NH, 'heat'), (NU + NH, NU + NH + NC, 'cool')):
        p = lg[:, a:b].argmax(1)
        t = K[:, ['use', 'heat', 'cool'].index(name)]
        rec = [np.mean(p[t == c] == c) for c in range(b - a) if (t == c).sum() >= 10]
        S[name] = float(max((np.mean(rec) - 1 / len(rec)) / (1 - 1 / len(rec)), 0))
    S['_overall'] = float(np.mean([v for k, v in S.items() if not k.startswith('_')]))
    return S


def rebaseline():
    R = load_out()
    rb = {}
    for mode in MODES:
        X, Y, K, fs = val_feats(mode)
        mu, sg, lg = Net(os.path.join(MODEL, 'weights', f'{mode}.npz')).raw(X)
        S = skills(mu, sg, lg, Y, K)
        rb[mode] = dict(val_skill=S['_overall'], per_task=S, n_val=int(len(X)), files=fs)
        print(mode, 'R9 on the f3 frozen set: %.4f (n %d)' % (S['_overall'], len(X)))
    g, _ = E.dev_gate(E.load_nets(os.path.join(MODEL, 'weights')))
    R['rebaseline'] = rb
    R['v10_dev_composite'] = g
    R['v10_dev_composite_logged'] = 0.30605407964980746
    print('v1.0 dev composite now %.4f (logged 0.3061)' % g)
    save(R)


def export():
    from detective import export_torch
    od = os.path.join(WORK, 'models', 'r10')
    os.makedirs(od, exist_ok=True)
    for mode in MODES:
        export_torch(os.path.join(WORK, 'models', f'r10_{mode}', 'final.pt'), os.path.join(od, f'{mode}.npz'))


def decide():
    R = load_out()
    if 'rebaseline' not in R:
        raise SystemExit('run rebaseline first')
    export()
    kept = os.path.join(WORK, 'models', 'r10_kept')
    if os.path.exists(kept):
        shutil.rmtree(kept)
    os.makedirs(kept)
    for mode in MODES:
        shutil.copy(os.path.join(MODEL, 'weights', f'{mode}.npz'), os.path.join(kept, f'{mode}.npz'))
    gate_prev = R['v10_dev_composite']
    rungs = []
    for mode in MODES:
        res = json.load(open(os.path.join(WORK, 'models', f'r10_{mode}', 'result.json')))
        sk = res['val_skill']['_overall']
        inc = R['rebaseline'][mode]['val_skill']
        cand = os.path.join(WORK, 'models', 'r10_cand')
        if os.path.exists(cand):
            shutil.rmtree(cand)
        shutil.copytree(kept, cand)
        shutil.copy(os.path.join(WORK, 'models', 'r10', f'{mode}.npz'), os.path.join(cand, f'{mode}.npz'))
        g, det = E.dev_gate(E.load_nets(cand))
        dv = sk - inc
        okg = g >= gate_prev - GATE_TOL
        keep = dv > MARGIN and okg
        why = (f'val skill {sk:.4f} vs R9 on the p12 frozen set {inc:.4f} (delta {dv:+.4f}, need > {MARGIN}); '
               f'dev composite {g:.4f} vs {gate_prev:.4f} (floor {gate_prev - GATE_TOL:.4f}, {"ok" if okg else "FAIL"})')
        rungs.append(dict(name=f'r10-{mode}', mode=mode, val_skill=sk, val_per_task=res['val_skill'], incumbent=inc,
                          dev_gate=g, gate_floor=gate_prev - GATE_TOL, kept=bool(keep), reason=why, cfg=res['cfg'],
                          dev=E.strip(det)))
        print(mode, 'KEPT' if keep else 'rejected', '|', why)
        if keep:
            shutil.copy(os.path.join(cand, f'{mode}.npz'), os.path.join(kept, f'{mode}.npz'))
            gate_prev = g
    R['rungs'] = rungs
    R['kept_set_dev_composite'] = gate_prev
    save(R)


# ----------------------------------------------------------------------------- dev twin test + CV baselines
def folds(n, k=5):
    rng = np.random.default_rng(SEED)
    f = np.empty(n, int)
    f[rng.permutation(n)] = np.arange(n) % k
    return f


def cv(fit, Xd, yd, k=5, ok=None):
    """Out-of-fold predictions of fit(X_train, y_train) -> predict(X); rows with ok False are not fitted on."""
    n = len(Xd)
    ok = np.ones(n, bool) if ok is None else ok
    f = folds(n, k)
    out = None
    for i in range(k):
        tr, te = (f != i) & ok, f == i
        p = fit(Xd[tr], yd[tr])(Xd[te])
        if out is None:
            out = np.full((n,) + np.shape(p)[1:], np.nan) if np.asarray(p).dtype != object else np.empty(n, object)
        out[te] = p
    return out


def logit_fit(classes=None):
    def f(X, y):
        proba, cls = I.logistic(X, list(y), classes)
        return lambda Z: proba(Z)
    return f


def twin_and_baselines(nets):
    """Dev-only: the integrity pass's twin test for one set of nets, and the networks' dev numbers on the rows the
    5-fold-CV baselines score."""
    tw, bl = {}, {}
    # BDG2 area: correction to the no-truth rule; CV log-log line
    md, ld = E.load('dev', 'bdg2')
    Ad = np.array([l['A'] for l in ld], float)
    eld = np.array([l['kwh'].get('elec', 0) or np.nan for l in ld], float)
    eld[eld <= 0] = np.nan
    use_d = np.array([l['use'] for l in ld])
    eui_all = np.nanmedian(eld / Ad)
    k = np.isfinite(eld) & np.isfinite(Ad) & (Ad > 0)
    line = lambda X, y: (lambda co: (lambda Z: 10 ** np.polyval(co, np.log10(Z))))(np.polyfit(np.log10(X), np.log10(y), 1))
    b_ll = cv(line, eld, Ad, ok=k)
    for mode in MODES:
        mu, _, lg = E.run_net(nets[mode], md[mode])
        anc = np.log10(eld / eui_all)
        c, r = mu[:, J['logA']] - anc, np.log10(Ad) - anc
        tw[f'bdg2_area_{mode}'] = dict(I.corr(c, r), use_breakdown=I.breakdown(c, r, use_d))
        m = k & np.isfinite(mu[:, J['logA']])
        bl[f'BDG2 area MdAPE, {mode}'] = dict(n=int(m.sum()), network=float(I.s_mdape(10 ** mu[m, J['logA']], Ad[m])),
                                             baseline=float(I.s_mdape(b_ll[m], Ad[m])), baseline_name='CV log-log line on electricity')
        pu = E._probs(lg, 0, NU)
        pred = np.array(['lodging' if C.USES[i].startswith('residential') else C.USES[i] for i in pu.argmax(1)])
        maj = Counter(use_d).most_common(1)[0][0]
        bl[f'BDG2 use accuracy, {mode}'] = dict(n=len(ld), network=float(np.mean(pred == use_d) * 100),
                                               baseline=float(np.mean(use_d == maj) * 100), baseline_name=f'majority ({maj}, in sample)')
    # surveys
    for name in ('recs', 'cbecs'):
        md, ld = E.load('dev', name)
        Xd = md['annual']
        mud, _, lgd = E.run_net(nets['annual'], Xd)
        full = I.survey_frame(name)
        tcol, czc = ('housing_type', 'climate_zone') if name == 'recs' else ('pba', 'climate_class')
        Sd = np.array([sum(v for v in l['kwh'].values() if v is not None and np.isfinite(v)) for l in ld], float)
        Ad = np.array([l['A'] for l in ld], float)
        td = np.array([full.loc[l['id'], tcol] for l in ld], object)
        cd = np.array([full.loc[l['id'], czc] for l in ld], object)
        okd = (Sd > 0) & (Ad > 0)
        eui = Sd / Ad
        e_tc = {kk: np.median(eui[okd & (td == kk[0]) & (cd == kk[1])]) for kk in set(zip(td[okd], cd[okd]))}
        e_t = {t: np.median(eui[okd & (td == t)]) for t in set(td[okd])}
        b_tc_d = np.array([s / e_tc.get((t, c_), e_t.get(t, np.nan)) if s > 0 else np.nan for s, t, c_ in zip(Sd, td, cd)])
        anc0 = np.log10(np.where(Sd > 0, Sd / np.median(eui[okd]), np.nan))
        for lab, anc in (('area_vs_energy_over_dev_eui', anc0), ('area_vs_true_type_climate_rule', np.log10(b_tc_d))):
            c, r = mud[:, J['logA']] - anc, np.log10(Ad) - anc
            tw[f'{name}_{lab}'] = dict(I.corr(c, r), type_breakdown=I.breakdown(c, r, td), climate_breakdown=I.breakdown(c, r, cd))
        okr = okd & np.isfinite(Xd).all(1)
        b_reg = 10 ** cv(I.ridge, Xd, np.log10(np.where(Ad > 0, Ad, np.nan)), ok=okr)
        m = okr & np.isfinite(b_reg)
        bl[f'{name.upper()} area MdAPE'] = dict(n=int(m.sum()), network=float(I.s_mdape(10 ** mud[m, J['logA']], Ad[m])),
                                               baseline=float(I.s_mdape(b_reg[m], Ad[m])), baseline_name='CV ridge on the 12 annual features')
        ymd = I.ymid(ld)
        ymed = np.nanmedian(ymd)
        tw[f'{name}_vintage_vs_dev_median'] = I.corr(mud[:, J['year']] - ymed, ymd - ymed)
        okv = np.isfinite(ymd) & np.isfinite(Xd).all(1)
        b_v = cv(I.ridge, Xd, ymd, ok=okv)
        m = okv & np.isfinite(b_v)
        bl[f'{name.upper()} vintage MAE (yr)'] = dict(n=int(m.sum()), network=float(I.s_mae(mud[m, J['year']], ymd[m])),
                                                     baseline=float(I.s_mae(b_v[m], ymd[m])), baseline_name='CV ridge on the 12 annual features')
        grp = ({'heat_pump': 'electric', 'resistance': 'electric', 'fuel': 'fuel', 'none': 'none', 'district': 'fuel'}
               if name == 'recs' else
               {'heat_pump': 'electric', 'resistance': 'electric', 'fuel': 'fuel', 'none': 'none', 'district': 'district'})
        tgd = np.array([grp.get(l['heat'], None) if l['heat'] else None for l in ld], object)
        phd = E._probs(lgd, NU, NU + NH)
        pg = np.array([grp[C.HEAT[i]] for i in phd.argmax(1)], object)
        rd = np.array(['fuel' if (l['kwh'].get('fuel') or 0) > 0 else ('district' if (l['kwh'].get('dheat') or 0) > 0 else 'electric')
                       for l in ld], object)
        pfuel = np.array([sum(phd[i, j] for j, h in enumerate(C.HEAT) if grp[h] == 'fuel') for i in range(len(ld))])
        yf = np.array([float(t == 'fuel') if t is not None else np.nan for t in tgd])
        rf = (rd == 'fuel').astype(float)
        tw[f'{name}_heat_fuel_vs_meter_rule'] = I.corr(pfuel - rf, yf - rf)
        mh = np.array([t is not None for t in tgd])
        cls = sorted(set(tgd[mh]))
        P = cv(logit_fit(cls), Xd, tgd, ok=mh)
        b_h = np.array([cls[i] for i in np.nan_to_num(P, nan=-1).argmax(1)], object)
        bl[f'{name.upper()} heating class accuracy'] = dict(n=int(mh.sum()), network=float(np.mean(pg[mh] == tgd[mh]) * 100),
                                                           baseline=float(np.mean(b_h[mh] == tgd[mh]) * 100),
                                                           baseline_name='CV multinomial logistic on the 12 annual features',
                                                           meter_rule=float(np.mean(rd[mh] == tgd[mh]) * 100))
        if name == 'recs':
            t_ = np.array([l['heat'] or 'nan' for l in ld])
            ed = np.isin(t_, ['heat_pump', 'resistance'])
            s = phd[:, 0] / (phd[:, 0] + phd[:, 1])
            yhp = t_ == 'heat_pump'
            P = cv(logit_fit([False, True]), Xd, yhp, ok=ed)[:, 1]
            bl['RECS heat pump vs resistance AUC'] = dict(n=int(ed.sum()), network=float(I.s_auc(s[ed], yhp[ed])),
                                                         baseline=float(I.s_auc(P[ed], yhp[ed])), baseline_name='CV logistic on the 12 annual features')
            bl['RECS heat pump vs resistance bal. acc.'] = dict(n=int(ed.sum()), network=float(I.s_bacc((s[ed] > 0.5), yhp[ed])),
                                                               baseline=float(I.s_bacc((P[ed] > 0.5), yhp[ed])), baseline_name='CV logistic on the 12 annual features')
        else:
            ud = np.array([l['use'] for l in ld], object)
            ci = [C.USES.index(u) for u in E.COM_USES]
            pred = np.array([E.COM_USES[i] for i in E._probs(lgd, 0, NU)[:, ci].argmax(1)], object)
            cls = sorted(set(ud))
            P = cv(logit_fit(cls), Xd, ud)
            b_u = np.array([cls[i] for i in P.argmax(1)], object)
            bl['CBECS use accuracy'] = dict(n=len(ld), network=float(np.mean(pred == ud) * 100), baseline=float(np.mean(b_u == ud) * 100),
                                            baseline_name='CV multinomial logistic on the 12 annual features',
                                            network_macroF1=E.macro_f1(list(ud), list(pred)), baseline_macroF1=E.macro_f1(list(ud), list(b_u)))
            hd = np.array([l['hours_per_week'] if l['hours_per_week'] is not None else np.nan for l in ld], float)
            wkd = np.clip(mud[:, J['op_hours']], 0, 24) * (5 + 2 * np.clip(mud[:, J['weekend']], 0, 1.5))
            med = np.nanmedian(hd)
            byuse = {u: np.nanmedian(hd[ud == u]) for u in set(ud)}
            anc = np.array([byuse.get(u, med) for u in ud])
            tw['cbecs_hours_vs_true_use_median'] = dict(I.corr(wkd - anc, hd - anc), use_breakdown=I.breakdown(wkd - anc, hd - anc, ud))
            mh = np.isfinite(hd)
            b_hr = cv(I.ridge, Xd, hd, ok=mh)
            bl['CBECS weekly hours MAE (h)'] = dict(n=int(mh.sum()), network=float(I.s_mae(wkd[mh], hd[mh])),
                                                    baseline=float(I.s_mae(b_hr[mh], hd[mh])), baseline_name='CV ridge on the 12 annual features')
    return tw, bl


def devreport():
    R = load_out()
    sets = {'v1.0': os.path.join(MODEL, 'weights'), 'r10 (all three)': os.path.join(WORK, 'models', 'r10')}
    if os.path.exists(os.path.join(WORK, 'models', 'r10_kept')):
        sets['kept set'] = os.path.join(WORK, 'models', 'r10_kept')
    R['dev'] = {}
    for k, d in sets.items():
        tw, bl = twin_and_baselines(E.load_nets(d))
        R['dev'][k] = dict(twin=tw, baselines=bl)
    save(R)
    keys = list(R['dev'])
    print('TWIN (pearson | median corr vs needed | MAE anchor -> network)')
    for t in R['dev']['v1.0']['twin']:
        line = '%-44s' % t
        for k in keys:
            v = R['dev'][k]['twin'][t]
            line += ' | %s: r %.2f, %+.3f vs %+.3f, %.3f->%.3f' % (k[:4], v.get('pearson', np.nan), v.get('median_correction', np.nan),
                                                                   v.get('median_needed', np.nan), v.get('mae_anchor', np.nan), v.get('mae_network', np.nan))
        print(line)
    print('DEV BASELINES (5-fold CV) vs networks')
    for b in R['dev']['v1.0']['baselines']:
        v0 = R['dev']['v1.0']['baselines'][b]
        print('%-42s n %5d baseline %.3f | ' % (b, v0['n'], v0['baseline']) + ' | '.join(
            '%s %.3f' % (k[:4], R['dev'][k]['baselines'][b]['network']) for k in keys))


if __name__ == '__main__':
    {'rebaseline': rebaseline, 'decide': decide, 'devreport': devreport}[sys.argv[1]]()
