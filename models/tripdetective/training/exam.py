"""Final exam on real VED exam cars (never used for choosing anything). Run once, at the end, on the final rung.
Also: sigma widening factors fitted on dev only (widen-only), baselines on the same trips, pooling curve.
usage: python3 exam.py <ckpt>"""
from common import *
import pandas as pd
import evaluate as E, realbank, baselines as BL
from model import from_reg
import realdata as rd

LBF = 4.4482216; MPH = 0.44704
CV_MED = 0.004       # Crr speed coefficient used to turn (mass, Crr, CdA) into road-load force (bank median order)


def roadload_F(m, crr, cda, mph):
    v = mph * MPH
    return m * G * crr * (1 + CV_MED * v) + 0.5 * 1.2 * cda * v * v


def sig_factor(z):
    """widen-only factor so that 68.3 % of |z| fall inside 1 sigma"""
    z = np.abs(np.asarray(z)); z = z[np.isfinite(z)]
    return float(max(1.0, np.percentile(z, 68.3))) if len(z) else 1.0


def car_inputs(rb, c):
    o = rb['offs']; js = np.where(rb['vid'] == c)[0]
    return [rb['raw'][o[j]:o[j + 1]].T.astype(np.float32) for j in js], [rb['flags'][j] for j in js], js


def run(ckpt):
    net = E.load_net(f'{ROOT}/ckpt/{ckpt}')
    vt = rd.ved_vehicles()
    dev = realbank.load('dev'); ex = realbank.load('exam')
    B = BL.Baselines(); bfit = B.fit_dev(dev, vt)
    # ---- sigma factors on dev (per car pooled) ----
    _, Rd = E.real_scores(net, 'dev', rb=dev)
    cb = np.array([vt.loc[c, 'gen_weight_lb'] * LB for c in Rd['cars']]); ok = np.isfinite(cb)
    zm = (Rd['Pc']['mu'][ok, 0] - np.log((cb[ok] + CURB_OFFSET) / 1800)) / np.sqrt(Rd['Pc']['sig'][ok, 0] ** 2 + (113 / 1700) ** 2 / 3)
    dd = np.array([vt.loc[c, 'displacement_L'] for c in Rd['cars']]); okd = np.isfinite(dd)
    zd = (Rd['Pc']['mu'][okd, 1] - np.log(dd[okd] / 2.5)) / Rd['Pc']['sig'][okd, 1]
    # per-trip too
    cbt = np.array([vt.loc[c, 'gen_weight_lb'] * LB for c in Rd['trip_vid']]); okt = np.isfinite(cbt)
    zmt = (Rd['Pt']['mu'][okt, 0] - np.log((cbt[okt] + CURB_OFFSET) / 1800)) / np.sqrt(Rd['Pt']['sig'][okt, 0] ** 2 + (113 / 1700) ** 2 / 3)
    ddt = np.array([vt.loc[c, 'displacement_L'] for c in Rd['trip_vid']]); okdt = np.isfinite(ddt)
    zdt = (Rd['Pt']['mu'][okdt, 1] - np.log(ddt[okdt] / 2.5)) / Rd['Pt']['sig'][okdt, 1]
    sigk = dict(mass_car=sig_factor(zm), disp_car=sig_factor(zd), mass_trip=sig_factor(zmt), disp_trip=sig_factor(zdt))
    # dumb priors fitted on dev
    curb_med = bfit['curb_med']; disp_med = bfit['disp_med']
    # ---- exam predictions ----
    _, R = E.real_scores(net, 'exam', rb=ex)
    rows = []
    for i, c in enumerate(R['cars']):
        raws, fls, js = car_inputs(ex, c)
        b = B.predict(raws, fls)
        r = vt.loc[c]
        mu, sg = R['Pc']['mu'][i], R['Pc']['sig'][i]
        rows.append(dict(vid=c, ved_type=r.ved_type, arch_rd=r.architecture, arch_conf=r.arch_confidence, cls=r.cls,
                         curb_true=r.gen_weight_lb * LB, disp_true=r.displacement_L, n_trips=len(set(ex['trip'][js])),
                         arch_pred=ARCHS[int(R['Pc']['arch'][i].argmax())], arch_p=float(R['Pc']['arch'][i].max()),
                         mass_pred=float(from_reg(0, mu[0]) - CURB_OFFSET), mass_sig_log=float(sg[0] * sigk['mass_car']),
                         disp_pred=float(from_reg(1, mu[1])), disp_sig_log=float(sg[1] * sigk['disp_car']),
                         CdA_pred=float(from_reg(2, mu[2])), Crr_pred=float(from_reg(3, mu[3])), mass_tot_pred=float(from_reg(0, mu[0])),
                         strat_pred=STRATS[int(R['Pc']['strat'][i].argmax())], ev_frac=float(mu[4]), P_on=float(from_reg(5, mu[5])),
                         v_ev=float(from_reg(6, mu[6])),
                         b_arch=b['arch'], b_mass=b['mass'], b_disp=b['disp'], b_CdA=b['CdA'], b_Crr=b['Crr']))
    d = pd.DataFrame(rows); d.to_csv(f'{ROOT}/data/exam_cars.csv', index=False)
    # per-trip table
    trows = []
    for i, t in enumerate(R['trips']):
        c = R['trip_vid'][i]; r = vt.loc[c]; mu, sg = R['Pt']['mu'][i], R['Pt']['sig'][i]
        trows.append(dict(trip=t, vid=c, ved_type=r.ved_type, curb_true=r.gen_weight_lb * LB, disp_true=r.displacement_L,
                          arch_pred=ARCHS[int(R['Pt']['arch'][i].argmax())], mass_pred=float(from_reg(0, mu[0]) - CURB_OFFSET),
                          mass_sig_log=float(sg[0] * sigk['mass_trip']), disp_pred=float(from_reg(1, mu[1])), disp_sig_log=float(sg[1] * sigk['disp_trip'])))
    dt = pd.DataFrame(trows)
    # trip-level baselines (rule arch, physics mass, idle-fuel displacement) on the same trips
    o = ex['offs']; bt = []
    for t in R['trips']:
        js = np.where(ex['trip'] == t)[0]
        raws = [ex['raw'][o[j]:o[j + 1]].T.astype(np.float32) for j in js]; fls = [ex['flags'][j] for j in js]
        b = B.predict(raws, fls); bt.append(b)
    dt['b_arch'] = [b['arch'] for b in bt]; dt['b_mass'] = [b['mass'] for b in bt]; dt['b_disp'] = [b['disp'] for b in bt]
    dt.to_csv(f'{ROOT}/data/exam_trips.csv', index=False)
    rep = dict(ckpt=ckpt, sigk=sigk, baseline_fit=bfit, n_cars=len(d), n_trips=len(dt))
    rep['arch'] = arch_report(d, dt)
    rep['mass'] = mass_report(d, dt, curb_med)
    rep['disp'] = disp_report(d, dt, disp_med)
    rep['roadload'] = roadload_report(d)
    rep['pooling'] = pooling_curve(net, ex, vt, sigk)
    jsave(rep, f'{ROOT}/exam.json')
    print(json.dumps({k: rep[k] for k in ['sigk', 'arch', 'mass', 'disp', 'roadload']}, indent=1, default=str)[:6000])
    return rep


def arch_report(d, dt):
    out = {}
    for name, df in [('car', d), ('trip', dt)]:
        t = df.ved_type.values
        p = np.array([VEDTYPE[a] for a in df.arch_pred]); pb = np.array([VEDTYPE[a] for a in df.b_arch])
        cm = pd.crosstab(pd.Categorical(t, VEDTYPES), pd.Categorical(p, VEDTYPES), dropna=False)
        cmb = pd.crosstab(pd.Categorical(t, VEDTYPES), pd.Categorical(pb, VEDTYPES), dropna=False)
        bal = lambda tt, pp: float(np.mean([np.mean(pp[tt == k] == k) for k in VEDTYPES if (tt == k).any()]))
        out[name] = dict(acc=float((t == p).mean()), bal_acc=bal(t, p), rule_acc=float((t == pb).mean()), rule_bal_acc=bal(t, pb),
                         majority_acc=float((t == 'ICE').mean()), majority_bal_acc=0.25,
                         confusion=cm.values.tolist(), confusion_rule=cmb.values.tolist(), labels=VEDTYPES)
    # finer architecture vs RoadData inference (not hard truth), by confidence
    fa = {}
    for conf in ['high', 'medium', 'low']:
        x = d[d.arch_conf == conf]
        if len(x): fa[conf] = dict(n=len(x), agree=float((x.arch_pred == x.arch_rd).mean()))
    out['fine_vs_roaddata'] = fa
    out['fine_confusion'] = pd.crosstab(d.arch_rd, d.arch_pred).to_dict()
    return out


def _err(pred, true):
    e = pred - true; ok = np.isfinite(e)
    return dict(n=int(ok.sum()), mae=float(np.mean(np.abs(e[ok]))), med_ae=float(np.median(np.abs(e[ok]))), bias=float(np.median(e[ok])),
                mdape=float(np.median(np.abs(e[ok]) / true[ok])))


def mass_report(d, dt, curb_med):
    out = {}
    for name, df in [('car', d), ('trip', dt)]:
        ok = np.isfinite(df.curb_true)
        x = df[ok]
        r = dict(detective=_err(x.mass_pred.values, x.curb_true.values), physics=_err(x.b_mass.values, x.curb_true.values),
                 dumb=_err(np.full(len(x), curb_med), x.curb_true.values))
        r['detective']['within_bin'] = float(np.mean(np.abs(x.mass_pred - x.curb_true) <= 113.4))
        r['physics']['within_bin'] = float(np.nanmean(np.abs(x.b_mass - x.curb_true) <= 113.4))
        r['dumb']['within_bin'] = float(np.mean(np.abs(curb_med - x.curb_true) <= 113.4))
        z = np.log((x.mass_pred + CURB_OFFSET) / (x.curb_true + CURB_OFFSET)) / np.sqrt(x.mass_sig_log ** 2 + (113 / 1700) ** 2 / 3)
        r['coverage_1s'] = float(np.mean(np.abs(z) < 1)); r['coverage_2s'] = float(np.mean(np.abs(z) < 2))
        r['by_type'] = {k: _err(x[x.ved_type == k].mass_pred.values, x[x.ved_type == k].curb_true.values) for k in VEDTYPES if (x.ved_type == k).sum() >= 2}
        r['by_type_physics'] = {k: _err(x[x.ved_type == k].b_mass.values, x[x.ved_type == k].curb_true.values) for k in VEDTYPES if (x.ved_type == k).sum() >= 2}
        r['rounding_floor_mae'] = 56.7    # MAE of a perfect estimate vs a value rounded to 500 lb (uniform +-113 kg)
        out[name] = r
    return out


def disp_report(d, dt, disp_med):
    out = {}
    for name, df in [('car', d), ('trip', dt)]:
        x = df[np.isfinite(df.disp_true)]
        r = dict(detective=_err(x.disp_pred.values, x.disp_true.values), idle_fuel=_err(x.b_disp.values, x.disp_true.values),
                 dumb=_err(np.full(len(x), disp_med), x.disp_true.values))
        z = np.log(x.disp_pred / x.disp_true) / x.disp_sig_log
        r['coverage_1s'] = float(np.mean(np.abs(z) < 1)); r['coverage_2s'] = float(np.mean(np.abs(z) < 2))
        r['by_type'] = {k: _err(x[x.ved_type == k].disp_pred.values, x[x.ved_type == k].disp_true.values) for k in VEDTYPES if (x.ved_type == k).sum() >= 2}
        out[name] = r
    return out


def roadload_report(d):
    t = pd.read_csv(f'{ROOT}/data/roadload_truth.csv'); t = t[t.split == 'exam']
    x = d.merge(t.drop(columns=['ved_type', 'split']), on='vid')
    out = dict(n=len(x), note='EPA coast-down target road-load force (median over candidate test records) for VED exam cars '
                               'with a high-confidence fueleconomy.gov match that exist in the EPA test-car list')
    for s in [20, 40, 60]:
        Fp = roadload_F(x.mass_tot_pred, x.Crr_pred, x.CdA_pred, s)
        Fb = roadload_F(x.b_mass + CURB_OFFSET, x.b_Crr, x.b_CdA, s)
        Ft = x[f'F{s}_N']
        out[f'{s}mph'] = dict(detective_mdape=float(np.median(np.abs(Fp / Ft - 1))), detective_bias=float(np.median(Fp / Ft - 1)),
                              physics_mdape=float(np.nanmedian(np.abs(Fb / Ft - 1))),
                              truth_spread=float(np.median((x[f'F{s}_hi'] - x[f'F{s}_lo']) / Ft)))
    out['etw'] = _err(x.mass_tot_pred.values, x.etw_kg.values)
    out['cars'] = x[['vid', 'ved_type', 'models', 'F60_N', 'etw_kg', 'mass_tot_pred', 'CdA_pred', 'Crr_pred']].to_dict('records')
    return out


def pooling_curve(net, ex, vt, sigk, ks=(1, 2, 4, 8, 16, 32), seed=0):
    """mass/displacement error vs number of trips pooled, fixed random subsets per car"""
    import features as F
    rng = np.random.default_rng(seed); o = ex['offs']
    out = {}
    cars = [c for c in dict.fromkeys(ex['vid'].tolist()) if np.isfinite(vt.loc[c, 'gen_weight_lb'])]
    for k in ks:
        inp, grp, truth_m, truth_d = [], [], [], []; g = 0
        for c in cars:
            trips = list(dict.fromkeys(ex['trip'][ex['vid'] == c].tolist()))
            if len(trips) < k: continue
            for rep in range(2 if k < 8 else 1):
                sel = set(rng.choice(trips, k, replace=False))
                for j in np.where((ex['vid'] == c) & np.isin(ex['trip'], list(sel)))[0]:
                    inp.append(F.net_input(ex['raw'][o[j]:o[j + 1]].T.astype(np.float32), ex['flags'][j])); grp.append(g)
                truth_m.append(vt.loc[c, 'gen_weight_lb'] * LB); truth_d.append(vt.loc[c, 'displacement_L']); g += 1
        if g < 10: continue
        P = E.predict(net, inp, np.array(grp))
        mp = from_reg(0, P['mu'][:, 0]) - CURB_OFFSET; dp = from_reg(1, P['mu'][:, 1])
        tm = np.array(truth_m); td = np.array(truth_d)
        okd = np.isfinite(td)
        out[k] = dict(n=g, mass_mae=float(np.mean(np.abs(mp - tm))), mass_mdape=float(np.median(np.abs(mp / tm - 1))),
                      mass_sig_kg=float(np.median(P['sig'][:, 0] * (mp + CURB_OFFSET))),
                      disp_mae=float(np.mean(np.abs(dp[okd] - td[okd]))), disp_mdape=float(np.median(np.abs(dp[okd] / td[okd] - 1))))
        print('pool', k, out[k], flush=True)
    return out


if __name__ == '__main__':
    run(sys.argv[1])
