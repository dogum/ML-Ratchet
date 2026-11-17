"""Integrity pass (17 Nov 2026; plan in NOTEBOOK.md, written before this ran). Usage, after
`python exam.py ../faultfinder_weights.npz` (which writes WORK/exam/exam_runs.parquet):

    python integrity.py            # writes WORK/integrity/{integrity.json, rows.json, baseline_scores.parquet}

1. Fair baselines tuned on the 20 dev days only: the APAR rules with a dev-tuned grid of tolerances (a parameterised
   copy of shared/building/data/scripts/baselines.py::apar; the textbook setting is checked against
   faults_apar.parquet first) and a mixing-box residual detector with dev-fitted damper characteristic and threshold.
2. Every exam row: FaultFinder, every baseline, model minus the best classical baseline with 95 % bootstrap intervals
   (2,000 resamples, seed 20261117) over unit-days and over facility-days.
3. Dev twin test: the network's correction to the dev-tuned rule vs what the truth needed (dev only).
4. Unit habit on dev: same-unit vs other-unit similarity of healthy days, between-unit share of the network's
   log-odds, and a leave-one-out unit offset (diagnostic, not a rung). Exam dates per unit (no labels) for the plan.
The exam is already seen (v1.0 took it once): nothing here is chosen from it.
"""
import os
import sys
import json
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, RI, BLDG, load_real_runs  # noqa: E402
from features import featurize  # noqa: E402
import evaluate as ev  # noqa: E402
import buildingdata as bd  # noqa: E402

OUT = f'{ROOT}/integrity'
SEED, NB = 20261117, 2000
WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'faultfinder_weights.npz')


# ------------------------------------------------------------------ APAR, parameterised (copy of baselines.py::apar)
def apar(d, ET=1.5, DTSF=1.1, EV=0.02):
    g = lambda c: d[c].to_numpy(float) if c in d and d[c].notna().any() else np.full(len(d), np.nan)
    Toa, Tra, Tma, Tsa = g('T_oa'), g('T_ra'), g('T_ma'), g('T_sa')
    sp = g('T_sa_sp'); sph, spc = g('T_sa_sp_heat'), g('T_sa_sp_cool')
    sph = np.where(np.isfinite(sph), sph, sp); spc = np.where(np.isfinite(spc), spc, sp)
    hc, cc, oa = np.nan_to_num(g('valve_hc')), np.nan_to_num(g('valve_cc')), g('damper_oa')
    fan = g('fan_sa_status'); spd = g('fan_sa_speed')
    on = np.where(np.isfinite(fan), fan > 0.5, np.where(np.isfinite(spd), spd > 0.05, True))
    occ = g('occ'); on &= np.where(np.isfinite(occ), occ > 0.5, True)
    oamin = np.nanpercentile(oa[on], 10) if np.isfinite(oa[on]).any() else 0.0
    M1 = on & (hc > EV) & (cc <= EV)
    M2 = on & (hc <= EV) & (cc <= EV) & (oa > oamin + 0.05)
    M3 = on & (cc > EV) & (oa >= 0.95)
    M4 = on & (cc > EV) & (oa < 0.95) & (hc <= EV)
    R = {}
    with np.errstate(invalid='ignore'):
        R['R1'] = M1 & (Tsa < Tma + DTSF - ET)
        R['R3'] = M1 & (hc >= 0.98) & (Tsa < sph - ET)
        R['R7'] = M2 & (Toa > spc - DTSF + ET)
        R['R8'] = M2 & (Tsa > Tra + ET)
        R['R9'] = M2 & (np.abs(Tsa - DTSF - Tma) > ET)
        R['R10'] = M2 & (oa >= 0.95) & (np.abs(Toa - Tma) > ET)
        R['R11'] = M3 & (Toa < spc - DTSF - ET)
        R['R12'] = (M3 | M4) & (Tsa > Tma + DTSF + ET)
        R['R13'] = (M3 | M4) & (cc >= 0.98) & (Tsa > spc + ET)
        R['R14'] = M3 & (np.abs(Toa - Tma) > ET)
        R['R23'] = on & (hc > 0.1) & (cc > 0.1)
        R['R24'] = on & ((Tma < np.fmin(Tra, Toa) - ET) | (Tma > np.fmax(Tra, Toa) + ET))
    return {k: np.nan_to_num(v).astype(bool) for k, v in R.items()}, on


def persist(b, n):
    if n <= 1:
        return b
    s = pd.Series(b.astype(int)).rolling(n, min_periods=n).sum().to_numpy()
    return np.nan_to_num(s) >= n


def apar_score(d, dt, ET=1.5, DTSF=1.1, pers_s=900):
    R, on = apar(d, ET, DTSF)
    n = max(1, int(round(pers_s / dt)))
    anyr = np.zeros(len(d), bool)
    for c in R:
        anyr |= persist(R[c], n)
    return float(anyr[on].mean()) if on.any() else 0.0


def fan_rule(d):
    occ = d.occ.to_numpy(float)
    fan = d.fan_sa_status.to_numpy(float)
    un = occ < 0.5
    return float((fan[un] > 0.5).mean()) if un.any() else 0.0


def tune_threshold(score, t):
    """Best dev balanced accuracy among 0 and the midpoints of the dev scores; ties -> the lowest (as baselines.py)."""
    u = np.unique(score)
    cand = np.unique(np.r_[0.0, (u[1:] + u[:-1]) / 2])
    best = max(cand, key=lambda h: (bacc(t, score > h), -h))
    return float(best)


def bacc(t, p):
    t, p = np.asarray(t, bool), np.asarray(p, bool)
    tpr = (p[t]).mean() if t.any() else np.nan
    tnr = (~p[~t]).mean() if (~t).any() else np.nan
    return float(0.5 * (tpr + tnr))


# ------------------------------------------------------------------ mixing-box residual detector (15-min bins)
def mb_components(x, char=None, delta=None):
    """x: raw day [13, 96]. Returns dict of raw residual components (NaN if fewer than 4 qualifying bins) and the
    qualifying bins for the damper fit. char = (a, b, g); delta = healthy-dev median of SA - MA with coils off."""
    g = lambda c: x[RI[c]].astype(float)
    fan = g('fan_on')
    on = np.where(np.isfinite(fan), fan > 0.5, np.isfinite(g('T_sa')))
    Toa, Tra, Tma, Tsa, sp = g('T_oa'), g('T_ra'), g('T_ma'), g('T_sa'), g('T_sa_sp')
    cmd = g('damper_oa')
    out = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        den = Toa - Tra
        q = on & np.isfinite(den) & (np.abs(den) > 3) & np.isfinite(Tma) & np.isfinite(cmd)
        fobs = np.clip((Tma - Tra) / np.where(q, den, 1), -0.5, 1.5)
        out['fit_bins'] = (np.clip(cmd[q], 0, 1), fobs[q])
        if char is not None and q.sum() >= 4:
            a, b, gg = char
            out['C1'] = float(np.median(np.abs(fobs[q] - (a + (b - a) * np.clip(cmd[q], 0, 1) ** gg))))
        else:
            out['C1'] = np.nan
        vc, vh = g('valve_cc'), g('valve_hc')
        logged = [v for v in (vc, vh) if np.isfinite(v).any()]
        if logged:
            off = on.copy()
            for v in logged:
                off &= np.isfinite(v) & (v <= 0.02)
            off &= np.isfinite(Tsa) & np.isfinite(Tma)
            r = Tsa[off] - Tma[off]
            out['raw_C2'] = r
            out['C2'] = float(np.median(np.abs(r - delta))) if (delta is not None and off.sum() >= 4) else np.nan
        else:
            out['raw_C2'] = np.array([])
            out['C2'] = np.nan
        k = on & np.isfinite(Tsa) & np.isfinite(sp)
        out['C3'] = float(np.median(np.abs(Tsa[k] - sp[k]))) if k.sum() >= 4 else np.nan
    return out


def fit_char(bins):
    c = np.concatenate([b[0] for b in bins])
    f = np.concatenate([b[1] for b in bins])
    best = (np.inf, None)
    for a in np.linspace(0, 0.3, 16):
        for b in np.linspace(0.5, 1.0, 26):
            for gg in np.arange(0.5, 2.01, 0.1):
                e = np.sum((f - (a + (b - a) * c ** gg)) ** 2)
                if e < best[0]:
                    best = (e, (float(a), float(b), float(gg)))
    return best[1], int(len(c))


class MB:
    FLOOR = {'C1': 0.02, 'C2': 0.2, 'C3': 0.2}

    def fit(self, dev):
        """dev: [(row, x)]. Characteristic, delta and component scales from the healthy dev days."""
        healthy = [(r, x) for r, x in dev if r.fault_type == 'none']
        self.char, self.n_fit_bins = fit_char([mb_components(x)['fit_bins'] for _, x in healthy])
        r2 = np.concatenate([mb_components(x)['raw_C2'] for _, x in healthy])
        self.delta = float(np.median(r2)) if len(r2) else 0.0
        comps = pd.DataFrame([self.components(x) for _, x in healthy])
        self.med = {k: float(np.nanmedian(comps[k])) for k in comps}
        self.iqr = {k: max(float(np.nanpercentile(comps[k], 75) - np.nanpercentile(comps[k], 25)), self.FLOOR[k]) for k in comps}
        self.n_healthy = {k: int(comps[k].notna().sum()) for k in comps}
        return self

    def components(self, x):
        o = mb_components(x, self.char, self.delta)
        return {k: o[k] for k in ('C1', 'C2', 'C3')}

    def score(self, x):
        c = self.components(x)
        z = [(c[k] - self.med[k]) / self.iqr[k] for k in c if np.isfinite(c[k])]
        return float(max(z)) if z else 0.0


# ------------------------------------------------------------------ metrics and bootstrap
def auc(t, s):
    return ev.auc(t, s)


def recall_fa(t, s, fa):
    return ev.recall_at_fpr(np.asarray(t, bool), np.asarray(s, float), fa)


METRICS = {   # name: (fn(t, score, flag) -> value, better)
    'auc': (lambda t, s, f: auc(t, s), 'higher'),
    'bacc': (lambda t, s, f: bacc(t, f), 'higher'),
    'far': (lambda t, s, f: float(f[~t].mean()) if (~t).any() else np.nan, 'lower'),
    'tpr': (lambda t, s, f: float(f[t].mean()) if t.any() else np.nan, 'higher'),
    'r2': (lambda t, s, f: recall_fa(t, s, 0.02), 'higher'),
    'r5': (lambda t, s, f: recall_fa(t, s, 0.05), 'higher'),
    'r10': (lambda t, s, f: recall_fa(t, s, 0.10), 'higher'),
}


def boot(fn, n, groups=None, seed=SEED):
    """Percentile 95 % interval of fn(idx) over NB resamples of rows (or of groups)."""
    rng = np.random.default_rng(seed)
    if groups is None:
        draws = (rng.integers(0, n, n) for _ in range(NB))
    else:
        ug = np.unique(groups)
        gi = [np.where(groups == g)[0] for g in ug]
        draws = (np.concatenate([gi[k] for k in rng.integers(0, len(ug), len(ug))]) for _ in range(NB))
    v = np.array([fn(i) for i in draws], float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], int(len(v))


def verdict(lo, hi, better):
    if better == 'higher':
        return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')
    return 'win' if hi < 0 else ('loss' if lo > 0 else 'tie')


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    R = {}
    runs = bd.faults(gradable=True)
    runs = runs[runs.split.isin(['dev', 'exam'])].reset_index(drop=True)
    # ---- 1a. APAR textbook check + grid scores on the contract logs (full resolution, as baselines.py)
    ET_G, DT_G, PS_G = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0], [0.5, 1.1, 1.7], [900, 1800, 3600]
    grid = list(itertools.product(ET_G, DT_G, PS_G))
    rows = []
    for ds, g in runs.groupby('dataset'):
        data = bd.fault_data(ds)
        for _, r in g.iterrows():
            d = data[data.run_id == r.run_id]
            o = dict(run_id=r.run_id, split=r.split, rtu=r.unit == 'RTU', is_fault=r.fault_type != 'none')
            if r.unit == 'RTU':
                o['fan_rule'] = fan_rule(d)
            else:
                for (et, dt_, ps) in grid:
                    o[f'apar_{et}_{dt_}_{ps}'] = apar_score(d, r.dt_s, et, dt_, ps)
            rows.append(o)
    S = pd.DataFrame(rows)
    ref = bd.baselines('faults_apar')[['run_id', 'score']]
    chk = S.merge(ref, on='run_id')
    textbook = 'apar_1.5_1.1_900'
    a_ = chk[~chk.rtu]
    R['apar_textbook_check'] = dict(n_ahu=int(len(a_)), max_abs_diff=float(np.abs(a_[textbook] - a_.score).max()),
                                    rtu_max_abs_diff=float(np.abs(chk[chk.rtu].fan_rule - chk[chk.rtu].score).max()))
    print('APAR textbook vs faults_apar.parquet', R['apar_textbook_check'], flush=True)
    # ---- 1b. grid on the 18 AHU dev days
    dA = S[(S.split == 'dev') & ~S.rtu]
    tA = dA.is_fault.to_numpy()
    gr = []
    for (et, dt_, ps) in grid:
        c = f'apar_{et}_{dt_}_{ps}'
        s = dA[c].to_numpy()
        th = tune_threshold(s, tA)
        gr.append(dict(ET=et, DTSF=dt_, pers_min=ps // 60, dev_auc=auc(tA, s), dev_bacc=bacc(tA, s > th), thr=th,
                       dist=abs(et - 1.5) + abs(dt_ - 1.1) + abs(ps - 900) / 900, col=c))
    gr = pd.DataFrame(gr)
    best = gr.sort_values(['dev_auc', 'dev_bacc', 'dist'], ascending=[False, False, True]).iloc[0]
    R['apar_grid'] = dict(n_dev_ahu=int(len(dA)), n_dev_ahu_faulty=int(tA.sum()), chosen=best.drop('col').to_dict(),
                          textbook=gr[gr.col == textbook].iloc[0].drop('col').to_dict(),
                          n_settings=len(gr), n_tied_best_auc=int((gr.dev_auc == best.dev_auc).sum()),
                          table=gr.drop(columns='col').round(4).to_dict('records'))
    print('APAR grid chosen', R['apar_grid']['chosen'], 'textbook', R['apar_grid']['textbook'], flush=True)
    tuned = best.col
    dR = S[(S.split == 'dev') & S.rtu]
    thr_fan = tune_threshold(dR.fan_rule.to_numpy(), dR.is_fault.to_numpy())
    fs = json.load(open(os.path.join(BLDG, 'data', 'proc', 'baselines', 'faults_summary.json')))
    thr_tb, thr_fan_rel = fs['ahu_all']['threshold_dev'], fs['rtu']['threshold_dev']
    thr_tb_own = tune_threshold(dA[textbook].to_numpy(), tA)
    R['thresholds'] = dict(apar_textbook_released=thr_tb, apar_textbook_dev_midpoint_rule=thr_tb_own,
                           apar_tuned=float(best.thr), fan_rule_released=thr_fan_rel, fan_rule_midpoint_rule=thr_fan,
                           fan_rule_dev_scores=dict(zip(dR.run_id, dR.fan_rule.round(4))))
    # ---- 1c. mixing-box residual detector (15-min raw days)
    dev = load_real_runs('dev')
    exam = load_real_runs('exam')
    mb = MB().fit(dev)
    xs = {r.run_id: x for r, x in dev + exam}
    S['mb'] = [mb.score(xs[r]) for r in S.run_id]
    comp = pd.DataFrame([dict(run_id=r, **mb.components(xs[r])) for r in S.run_id])
    dv = S[S.split == 'dev']
    thr_mb = tune_threshold(dv.mb.to_numpy(), dv.is_fault.to_numpy())
    R['mb'] = dict(char=mb.char, n_fit_bins=mb.n_fit_bins, delta=mb.delta, med=mb.med, iqr=mb.iqr, n_healthy=mb.n_healthy,
                   thr=thr_mb, dev_auc=auc(dv.is_fault, dv.mb), dev_bacc=bacc(dv.is_fault, dv.mb > thr_mb),
                   availability_exam={k: int(comp.merge(S[['run_id', 'split']])[lambda z: z.split == 'exam'][k].notna().sum()) for k in ('C1', 'C2', 'C3')})
    print('MB', {k: v for k, v in R['mb'].items()}, flush=True)
    S.to_parquet(f'{OUT}/baseline_scores.parquet', index=False)

    # ---- 2. exam rows
    E = pd.read_parquet(f'{ROOT}/exam/exam_runs.parquet')
    E = E.merge(S, on=['run_id'], how='left', suffixes=('', '_b'))
    t = E.is_fault.to_numpy()
    rtu = E.rtu.to_numpy()
    B = {   # name: (score, flag at its dev threshold). RTU: the fan rule in both rule sets (released / midpoint threshold)
        'rules_textbook': (np.where(rtu, E.fan_rule, E[textbook]), np.where(rtu, E.fan_rule > thr_fan_rel, E[textbook] > thr_tb)),
        'rules_tuned': (np.where(rtu, E.fan_rule, E[tuned]), np.where(rtu, E.fan_rule > thr_fan, E[tuned] > best.thr)),
        'mixing_box': (E.mb.to_numpy(), E.mb.to_numpy() > thr_mb),
    }
    B = {k: (np.nan_to_num(np.asarray(s, float)), np.asarray(f, bool)) for k, (s, f) in B.items()}
    NA = {'RTU (ORNL)': {'mixing_box'}}     # the ORNL RTU logs no OA/MA temperature or setpoint: no residual exists
    model = (E.d.to_numpy(), E.p.to_numpy() > 0.5)
    subsets = {'All units': np.ones(len(E), bool), 'AHU': ~rtu, 'RTU (ORNL)': rtu,
               'Iowa ERS': (E.source == 'Iowa ERS (lab)').to_numpy(), 'LBNL FLEXLAB': (E.source == 'LBNL FLEXLAB (lab)').to_numpy(),
               'Nesbitt Hall': (E.dataset == 'nesbitt').to_numpy()}
    groups = E.group.to_numpy()
    out = []
    for sname, m in subsets.items():
        idx0 = np.where(m)[0]
        tt = t[idx0]
        for mname, (fn, better) in METRICS.items():
            val = lambda sc, fl, ii: fn(t[ii], sc[ii], fl[ii])
            mv = val(*model, idx0)
            if not np.isfinite(mv):
                continue
            bv = {k: val(s, f, idx0) for k, (s, f) in B.items() if k not in NA.get(sname, ())}
            sign = 1 if better == 'higher' else -1
            bestk = max(bv, key=lambda k: sign * bv[k] if np.isfinite(bv[k]) else -np.inf)
            dfn = lambda ii: val(*model, ii) - val(*B[bestk], ii)
            ci_u, n_u = boot(lambda ii: dfn(idx0[ii]), len(idx0))
            gsub = groups[idx0]
            ci_g, n_g = boot(lambda ii: dfn(idx0[ii]), len(idx0), gsub)
            maj = None
            if mname in ('bacc',):
                maj = 0.5
            elif mname == 'far':
                maj = 1.0 if (tt.mean() >= 0.5 and sname == 'RTU (ORNL)') else 0.0
            v = verdict(*ci_u, better)
            if maj is not None and ((better == 'higher' and maj > mv) or (better == 'lower' and maj < mv)):
                v = 'loss'
            out.append(dict(subset=sname, metric=mname, better=better, n=int(len(idx0)), n_fault=int(tt.sum()),
                            model=mv, baselines=bv, best=bestk, diff=mv - bv[bestk], ci_unitday=ci_u, ci_facilityday=ci_g,
                            n_groups=int(len(np.unique(gsub))), boot_valid=[n_u, n_g], majority=maj,
                            verdict=v, verdict_facilityday=verdict(*ci_g, better),
                            report_only=mname in ('r2', 'r5', 'r10')))
    # diagnosis rows (66 faulty days with a simulable class)
    dg = E[E.is_fault & ~E.diag_class.isin(['other'])].reset_index(drop=True)
    c1 = dg.pred1.to_numpy() == dg.diag_class.to_numpy()
    c3 = np.array([c in p for c, p in zip(dg.diag_class, dg.pred3)])
    vc = dg.diag_class.value_counts()
    top1c, top3c = vc.index[0], list(vc.index[:3])
    b1 = (dg.diag_class == top1c).to_numpy()
    b3 = dg.diag_class.isin(top3c).to_numpy()
    for name, mm, bb, lab in (('diag_top1', c1, b1, top1c), ('diag_top3', c3, b3, ', '.join(top3c))):
        ci, _ = boot(lambda ii: mm[ii].mean() - bb[ii].mean(), len(dg))
        ci_g, _ = boot(lambda ii: mm[ii].mean() - bb[ii].mean(), len(dg), dg.group.to_numpy())
        out.append(dict(subset='Diagnosis', metric=name, better='higher', n=int(len(dg)), model=float(mm.mean()),
                        baselines={f'always {lab}': float(bb.mean())}, best=f'always {lab}', diff=float(mm.mean() - bb.mean()),
                        ci_unitday=ci, ci_facilityday=ci_g, verdict=verdict(*ci, 'higher'), verdict_facilityday=verdict(*ci_g, 'higher'),
                        report_only=False, tie_counts=vc.head(4).to_dict()))
    # info rows: calibration on the value itself
    p = E.p.to_numpy()
    ece = lambda ii: _ece(p[ii], t[ii].astype(float))
    for name, fn in (('ece_all', ece), ('brier_all', lambda ii: float(np.mean((p[ii] - t[ii]) ** 2))),
                     ('brier_uncal_minus_cal', lambda ii: float(np.mean((1 / (1 + np.exp(-E.d.to_numpy()[ii])) - t[ii]) ** 2) - np.mean((p[ii] - t[ii]) ** 2)))):
        ci, _ = boot(fn, len(E))
        out.append(dict(subset='Calibration', metric=name, better='lower', n=len(E), model=fn(np.arange(len(E))), ci_value=ci, verdict='info'))
    json.dump(out, open(f'{OUT}/rows.json', 'w'), indent=1, default=float)
    for o in out:
        if 'ci_unitday' in o:
            print(f"{o['subset']:13s} {o['metric']:10s} n={o['n']:3d} model={o['model']:.3f} best={o['best']}"
                  f" {o['baselines'][o['best']]:.3f} diff={o['diff']:+.3f} CI_u=[{o['ci_unitday'][0]:+.3f},{o['ci_unitday'][1]:+.3f}]"
                  f" CI_g=[{o['ci_facilityday'][0]:+.3f},{o['ci_facilityday'][1]:+.3f}] {o['verdict']}/{o['verdict_facilityday']}"
                  f" | {', '.join(f'{k} {v:.3f}' for k, v in o['baselines'].items())}", flush=True)
        else:
            print(o, flush=True)

    # ---- 3. dev twin test
    R['twin'] = twin_test(dev, S, tuned, best.thr, thr_fan)
    # ---- 4. unit habit (dev only) + exam dates per unit (metadata, no labels)
    R['habit'] = unit_habit(dev)
    ex = bd.faults(split='exam')
    R['exam_prior_days'] = prior_days(ex)
    json.dump(R, open(f'{OUT}/integrity.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in R.items() if k not in ('apar_grid',)}, indent=1, default=float)[:12000])


def _ece(p, t, nb=5):
    bins = np.linspace(0, 1, nb + 1)
    e = 0.0
    for a, b in zip(bins[:-1], bins[1:]):
        s = (p >= a) & (p < b if b < 1 else p <= b)
        if s.any():
            e += s.mean() * abs(p[s].mean() - t[s].mean())
    return float(e)


def logistic_fit(x, y, l2=1.0):
    from scipy.optimize import minimize
    mu, sd = x.mean(), x.std() + 1e-9
    z = (x - mu) / sd

    def f(w):
        q = w[0] + w[1] * z
        return np.sum(np.logaddexp(0, q) - y * q) + 0.5 * l2 * (w[0] ** 2 + w[1] ** 2)
    w = minimize(f, np.zeros(2)).x
    return lambda xx: 1 / (1 + np.exp(-(w[0] + w[1] * (xx - mu) / sd)))


def net_dev(dev):
    m = ev.NpzModel(WEIGHTS)
    lo, _, _ = ev.run_predict(m, dev)
    d = ev.det_logit(lo)
    t = np.array([r.fault_type != 'none' for r, _ in dev], float)
    import exam
    T = exam.fit_temperature(d, t)
    return d, 1 / (1 + np.exp(-d / T)), T, lo


def twin_test(dev, S, tuned, thr_tuned, thr_fan):
    d, p, T, lo = net_dev(dev)
    ids = [r.run_id for r, _ in dev]
    s = S.set_index('run_id').loc[ids]
    rtu = s.rtu.to_numpy()
    score = np.where(rtu, s.fan_rule, s[tuned]).astype(float)
    flag = np.where(rtu, s.fan_rule > thr_fan, s[tuned] > thr_tuned)
    y = s.is_fault.to_numpy().astype(float)
    # anchor: one logistic map of each rule's score (fitted on dev); AHU and RTU scores share it (2 RTU days)
    z = np.where(rtu, score, score)
    pr = logistic_fit(z, y)(z)
    corr_ = float(np.corrcoef(p - pr, y - pr)[0, 1])
    days = []
    for i, (r, _) in enumerate(dev):
        days.append(dict(run_id=r.run_id, unit=f'{r.dataset}:{r.unit}', fault=r.fault_type, truth=int(y[i]), p_net=round(float(p[i]), 3),
                         net_flag=bool(p[i] > 0.5), rule_score=round(float(score[i]), 3), rule_flag=bool(flag[i]), p_rule=round(float(pr[i]), 3),
                         net_right=bool((p[i] > 0.5) == y[i]), rule_right=bool(flag[i] == y[i])))
    D = pd.DataFrame(days)
    return dict(T=T, corr_correction_vs_needed=corr_, n=len(D), net_auc=ev.auc(y.astype(bool), d), rule_auc=ev.auc(y.astype(bool), score),
                net_right=int(D.net_right.sum()), rule_right=int(D.rule_right.sum()),
                both_right=int((D.net_right & D.rule_right).sum()), net_only=int((D.net_right & ~D.rule_right).sum()),
                rule_only=int((~D.net_right & D.rule_right).sum()), both_wrong=int((~D.net_right & ~D.rule_right).sum()),
                mean_abs_needed=float(np.mean(np.abs(y - pr))), mean_abs_after_net=float(np.mean(np.abs(y - p))),
                days=days)


def unit_habit(dev):
    d, p, T, lo = net_dev(dev)
    unit = np.array([f'{r.dataset}:{r.unit}' for r, _ in dev])
    y = np.array([r.fault_type != 'none' for r, _ in dev])
    V = []
    for _, x in dev:
        F = featurize(x)
        on = np.nan_to_num(x[RI['fan_on']]) > 0.5
        if on.sum() < 8:
            on = np.ones(F.shape[1], bool)
        V.append(np.r_[F[:, on].mean(1), F[:, on].std(1)])
    V = np.array(V)
    sd = V.std(0)
    keep = sd > 1e-6
    Z = (V[:, keep] - V[:, keep].mean(0)) / sd[keep]
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).mean(-1))
    near = []
    for i in np.where(~y)[0]:
        same = [j for j in range(len(dev)) if j != i and unit[j] == unit[i]]
        other = [j for j in range(len(dev)) if unit[j] != unit[i]]
        if not same:
            continue
        same_h = [j for j in same if not y[j]]
        near.append(dict(run_id=dev[i][0].run_id, unit=unit[i], d_same=float(D[i, same].min()),
                         d_same_healthy=float(D[i, same_h].min()) if same_h else None, d_other=float(D[i, other].min()),
                         nearest_is_same_unit=bool(D[i, same].min() < D[i, other].min()),
                         log_odds=float(d[i]), p=float(p[i])))
    # network log-odds of healthy days: between-unit vs within-unit variance (units with >= 2 healthy days)
    h = pd.DataFrame(dict(unit=unit[~y], d=d[~y]))
    hh = h[h.groupby('unit').unit.transform('size') >= 2]
    within = float(hh.groupby('unit').d.var(ddof=1).mean())
    between = float(hh.groupby('unit').d.mean().var(ddof=1))
    # leave-one-out unit offset: subtract the mean log-odds of the unit's OTHER healthy dev days
    T_ = T
    loo = []
    for i in range(len(dev)):
        o = [j for j in range(len(dev)) if j != i and unit[j] == unit[i] and not y[j]]
        if o:
            loo.append(dict(run_id=dev[i][0].run_id, unit=unit[i], truth=bool(y[i]), d=float(d[i]), ref=float(np.mean(d[o])),
                            d_adj=float(d[i] - np.mean(d[o])), p=float(p[i]), p_adj=float(1 / (1 + np.exp(-(d[i] - np.mean(d[o])) / T_)))))
    L = pd.DataFrame(loo)
    return dict(n_healthy=int((~y).sum()), units=dict(pd.Series(unit).value_counts()),
                healthy_by_unit=dict(pd.Series(unit[~y]).value_counts()),
                nearest=near, share_nearest_same_unit=float(np.mean([n['nearest_is_same_unit'] for n in near])) if near else None,
                median_d_same=float(np.median([n['d_same'] for n in near])), median_d_other=float(np.median([n['d_other'] for n in near])),
                logodds_healthy=dict(units=list(hh.unit.unique()), n=int(len(hh)), within_var=within, between_var=between,
                                     between_share=between / (between + within),
                                     by_unit={k: [round(v, 2) for v in g.d] for k, g in hh.groupby('unit')}),
                fa_all_healthy_dev=dict(n=int((~y).sum()), false_alarms=int((p[~y] > 0.5).sum())),
                loo=dict(n=len(L), n_healthy=int((~L.truth).sum()), n_faulty=int(L.truth.sum()),
                         fa_before=int(((L.p > 0.5) & ~L.truth).sum()), fa_after=int(((L.p_adj > 0.5) & ~L.truth).sum()),
                         det_before=int(((L.p > 0.5) & L.truth).sum()), det_after=int(((L.p_adj > 0.5) & L.truth).sum()),
                         auc_before=ev.auc(L.truth, L.d) if L.truth.any() and (~L.truth).any() else None,
                         auc_after=ev.auc(L.truth, L.d_adj) if L.truth.any() and (~L.truth).any() else None,
                         rows=L.round(3).to_dict('records')))


def prior_days(ex):
    """Exam metadata only (dates per unit, no labels): how many earlier same-unit exam days lie within 7/14/30 days."""
    ex = ex.assign(key=ex.dataset + ':' + ex.unit, date=pd.to_datetime(ex.date))
    out = {}
    for w in (3, 7, 14, 30):
        n = []
        for _, g in ex.groupby('key'):
            ds = np.sort(g.date.to_numpy())
            for dd in ds:
                n.append(int(((ds < dd) & (ds >= dd - np.timedelta64(w, 'D'))).sum()))
        n = np.array(n)
        out[f'{w}d'] = dict(share_with_any=float((n > 0).mean()), share_with_3plus=float((n >= 3).mean()), median=float(np.median(n)))
    out['by_dataset_7d_any'] = {}
    for ds_, g in ex.groupby('dataset'):
        c = 0
        for _, gg in g.groupby('unit'):
            dd = np.sort(pd.to_datetime(gg.date).to_numpy())
            c += sum(((dd < x) & (dd >= x - np.timedelta64(7, 'D'))).any() for x in dd)
        out['by_dataset_7d_any'][ds_] = f'{c}/{len(g)}'
    return out


if __name__ == '__main__':
    main()
