"""r7_context helpers (NOTEBOOK 2026-11-17 r7 plan): bank v5 index and references, noise v5, frozen v5, the dev days
with their causal references, scoring of 35-row (r6) and 82-row (r7) networks, and r7b (Platt on dev).

    python r7.py frozen          build WORK/data/frozen/val_v5.npz (seed 505; never rebuilt if it exists)
    python r7.py check           bank v5 vs the shipped v3 code, noise v5 vs v4 (writes WORK/r7/check_v4.json)
    python r7.py score <names>   score rungs (r6 = released numpy weights; others WORK/models/<name>.pt) on frozen v5
                                 and dev, plus r7b; writes WORK/r7/scores.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, RI, load_real_runs  # noqa: E402
from features import featurize  # noqa: E402
from context import context, context_batch, draw_refs, NF1, MAX_REF  # noqa: E402
import evaluate as ev  # noqa: E402

BANKS = f'{ROOT}/data/banks'
OUT = f'{ROOT}/r7'
FROZEN5 = f'{ROOT}/data/frozen/val_v5.npz'
# r6 = the v1.0 release weights. Since v1.1 (r8) the model folder holds 82-row weights, so r6 is read from a copy:
#   git show 7e3f51d:models/faultfinder/faultfinder_weights.npz > WORK/models/r6_v1_0.npz
WEIGHTS = f'{ROOT}/models/r6_v1_0.npz'
EFF_MIN = 1.0
P_NONE = 0.3
FROZEN_SEED = 505


def noise5():
    from noise import Noise
    return Noise(f'{ROOT}/data/noise_v1.npz', damper_char=True, big_offsets=True)


def load_bank(name):
    return dict(np.load(f'{BANKS}/{name}.npz', allow_pickle=True))


class Index:
    """(window, day, run) -> row of a v5 bank."""
    def __init__(self, b):
        self.w = b['window'].astype(np.int64)
        self.A = -np.ones((int(self.w.max()) + 1, 12, 2), np.int64)
        self.A[self.w, b['day'].astype(np.int64), b['run'].astype(np.int64)] = np.arange(len(self.w))
        self.fd = b['fault_day'].astype(int)
        self.day = b['day'].astype(int)
        self.run = b['run'].astype(int)

    def refs(self, i, rng, p_none=P_NONE):
        w = self.w[i]
        return draw_refs(int(self.day[i]), int(self.run[i]), int(self.fd[i]), lambda k, r: int(self.A[w, k, r]), rng,
                         p_none)


def featurize_rows(b, rows, nz, rng, states=None):
    """Noise v5 + featurize the given bank rows (one window state per window, drawn in window order if not given).
    Returns ({row: F[35,96] float16}, states)."""
    rows = np.asarray(sorted(set(int(r) for r in rows)))
    if states is None:
        states = {}
        for w in np.unique(b['window'][rows]):
            i0 = rows[b['window'][rows] == w][0]
            states[int(w)] = nz.window_state(rng, str(b['system'][i0]))
    F = np.empty((len(rows), NF1, 96), np.float16)
    for k, i in enumerate(rows):
        F[k] = featurize(nz.apply_window(b['x'][i].astype(np.float32), rng, states[int(b['window'][i])]))
    return dict(zip(rows.tolist(), F)), states


def build_frozen(n_max=12000):
    if os.path.exists(FROZEN5):
        print('frozen v5 exists; not rebuilt', FROZEN5)
        return dict(np.load(FROZEN5, allow_pickle=True))
    b = load_bank('val_v5')
    ix = Index(b)
    rng = np.random.default_rng(FROZEN_SEED)
    cand = np.where(b['sample'] & ((b['cls'] == 0) | (b['eff'] >= EFF_MIN)))[0]
    idx = np.sort(rng.choice(cand, n_max, replace=False)) if len(cand) > n_max else cand
    refs = [ix.refs(i, rng) for i in idx]
    need = set(idx.tolist()) | {r for rr in refs for r in rr}
    nz = noise5()
    Fd, _ = featurize_rows(b, need, nz, rng)
    nref = np.array([len(r) for r in refs], np.int8)
    ft = np.stack([Fd[int(i)] for i in idx]).astype(np.float32)
    fr = np.zeros((len(idx), MAX_REF, NF1, 96), np.float32)
    for k, rr in enumerate(refs):
        for j, r in enumerate(rr):
            fr[k, j] = Fd[r]
    F = context_batch(ft, fr, nref).astype(np.float16)
    out = dict(F=F, cls=b['cls'][idx], par=b['par'][idx], waste=b['waste'][idx], eff=b['eff'][idx],
               system=b['system'][idx], month=b['month'][idx], ftype=b['ftype'][idx], nref=nref,
               window=b['window'][idx], day=b['day'][idx], run=b['run'][idx], fault_day=b['fault_day'][idx],
               row=idx, refs=np.array([(rr + [-1] * MAX_REF)[:MAX_REF] for rr in refs], np.int64))
    os.makedirs(os.path.dirname(FROZEN5), exist_ok=True)
    np.savez_compressed(FROZEN5, **out)
    print('frozen v5:', len(idx), 'samples, faulty', int((out['cls'] > 0).sum()), 'with reference',
          int((nref > 0).sum()), flush=True)
    return out


# ------------------------------------------------------------------ dev days with causal references
def runs_with_refs(runs, max_gap=7):
    """runs [(row, x)] of one split -> [(row, x, [ref x ...], [ref run_id ...])]: references are the same dataset and
    unit's days of the same list with an earlier date within max_gap days, the most recent 3, chosen by date only
    (labels are not read)."""
    meta = pd.DataFrame([dict(i=i, ds=r.dataset, unit=r.unit, date=pd.Timestamp(r.date)) for i, (r, _) in enumerate(runs)])
    out = []
    for i, (r, x) in enumerate(runs):
        m = meta[(meta.ds == r.dataset) & (meta.unit == r.unit) & (meta.date < pd.Timestamp(r.date)) &
                 (meta.date >= pd.Timestamp(r.date) - pd.Timedelta(days=max_gap))].sort_values(['date', 'i'])
        sel = m.i.to_numpy()[-MAX_REF:]
        out.append((r, x, [runs[j][1] for j in sel], [runs[j][0].run_id for j in sel]))
    return out


def dev_with_refs(max_gap=7):
    """The 20 dev days with their causal references (runs_with_refs on the dev split)."""
    return runs_with_refs(load_real_runs('dev'), max_gap)


def dev_inputs(dev, ctx=True):
    F = []
    for r, x, xr, _ in dev:
        f = featurize(x)
        F.append(context(f, [featurize(z) for z in xr]) if ctx else f)
    return np.stack(F).astype(np.float32)


# ------------------------------------------------------------------ models
def load_rung(name):
    if name == 'r6':
        return ev.NpzModel(WEIGHTS), 35
    if os.path.exists(f'{ROOT}/models/{name}.npz'):          # numpy-format rungs (r8)
        m = ev.NpzModel(f'{ROOT}/models/{name}.npz')
        return m, int(m.W['c1_W'].shape[1])
    import torch
    from model import FFNet
    torch.set_num_threads(1)
    sd = torch.load(f'{ROOT}/models/{name}.pt')
    nf = sd['c1.0.weight'].shape[1]
    m = FFNet(nf=nf, w=sd['c1.0.weight'].shape[0])
    m.load_state_dict(sd)
    m.eval()
    return m, nf


def skill_on(model, nf, val, mask=None, single_day=False):
    F = val['F'][:, :nf].astype(np.float32)
    if single_day and nf > NF1:
        F = F.copy()
        F[:, NF1:] = 0
    v = dict(F=F, cls=val['cls'], eff=val['eff'], waste=val['waste'])
    if mask is not None:
        v = {k: a[mask] for k, a in v.items()}
    return ev.skill(model, v)


def dev_scores(model, nf, dev):
    F = dev_inputs(dev, ctx=nf > NF1)
    lo, _, _ = ev.predict(model, F)
    d = ev.det_logit(lo)
    t = np.array([r.fault_type != 'none' for r, *_ in dev])
    p = 1 / (1 + np.exp(-d))
    has = np.array([len(xr) > 0 for _, _, xr, _ in dev])
    return dict(dev_auc=ev.auc(t, d), dev_fa=int((p[~t] > 0.5).sum()), dev_n_healthy=int((~t).sum()),
                dev_fpr_at_0p5=float((p[~t] > 0.5).mean()), dev_tpr_at_0p5=float((p[t] > 0.5).mean()),
                dev_n_with_ref=int(has.sum()),
                dev_d={r.run_id: round(float(q), 3) for (r, *_), q in zip(dev, d)},
                dev_refs={r.run_id: rid for r, _, _, rid in dev if rid}), d, t


# ------------------------------------------------------------------ r7b: Platt on r6's dev log-odds
def platt_fit(d, y):
    from scipy.optimize import minimize

    def f(w):
        q = w[0] * d + w[1]
        return np.sum(np.logaddexp(0, q) - y * q)
    return minimize(f, np.array([1.0, 0.0]), method='Nelder-Mead', options=dict(xatol=1e-8, fatol=1e-10,
                                                                                maxiter=4000)).x


def r7b(d, t):
    y = t.astype(float)
    a, b = platt_fit(d, y)
    p_in = 1 / (1 + np.exp(-(a * d + b)))
    p_loo = np.empty(len(d))
    for i in range(len(d)):
        k = np.arange(len(d)) != i
        aa, bb = platt_fit(d[k], y[k])
        p_loo[i] = 1 / (1 + np.exp(-(aa * d[i] + bb)))
    import exam
    T = exam.fit_temperature(d, y)
    p_T = 1 / (1 + np.exp(-d / T))
    p_T_loo = np.array([1 / (1 + np.exp(-d[i] / exam.fit_temperature(d[np.arange(len(d)) != i],
                                                                     y[np.arange(len(d)) != i])))
                        for i in range(len(d))])
    brier = lambda p: float(np.mean((p - y) ** 2))
    fa = lambda p: int((p[~t] > 0.5).sum())
    det = lambda p: int((p[t] > 0.5).sum())
    return dict(a=float(a), b=float(b), T=T, threshold_logodds=float(-b / a) if a > 0 else None,
                dev_fa_T=fa(p_T), dev_fa_platt_in=fa(p_in), dev_fa_platt_loo=fa(p_loo),
                dev_det_T=det(p_T), dev_det_platt_in=det(p_in), dev_det_platt_loo=det(p_loo),
                n_healthy=int((~t).sum()), n_faulty=int(t.sum()),
                brier_T=brier(p_T), brier_T_loo=brier(p_T_loo), brier_platt_in=brier(p_in), brier_platt_loo=brier(p_loo),
                p_platt_in=[round(float(q), 3) for q in p_in], p_platt_loo=[round(float(q), 3) for q in p_loo],
                p_T=[round(float(q), 3) for q in p_T])


def score(names):
    os.makedirs(OUT, exist_ok=True)
    fn = f'{OUT}/scores.json'
    S = json.load(open(fn)) if os.path.exists(fn) else {}
    val = build_frozen()
    dev = dev_with_refs()
    has = val['nref'] > 0
    for name in names:
        m, nf = load_rung(name)
        s = dict(nf=int(nf), v5=skill_on(m, nf, val), v5_with_ref=skill_on(m, nf, val, has),
                 v5_no_ref=skill_on(m, nf, val, ~has))
        if nf > NF1:
            s['v5_single_day_mode'] = skill_on(m, nf, val, single_day=True)
            s['v5_with_ref_single_day_mode'] = skill_on(m, nf, val, has, single_day=True)
        dv, d, t = dev_scores(m, nf, dev)
        s['dev'] = dv
        if name == 'r6':
            S['r7b'] = dict(base='r6', v5=s['v5'], note='monotone map of the detection log-odds: frozen skill and dev '
                            'AUC are r6\'s by construction', dev_auc=dv['dev_auc'], platt=r7b(d, t))
        S[name] = s
        json.dump(S, open(fn, 'w'), indent=1)
        print(name, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                                         if kk not in ('dev_d', 'dev_refs')})
                                for k, v in s.items()}), flush=True)
    return S


# ------------------------------------------------------------------ check bank v5 / noise v5 against v3 code / v4
def _bank_stats(b, rows):
    x = b['x'][rows].astype(np.float32)
    on = np.nan_to_num(x[:, RI['fan_on']]) > 0.5
    cls = b['cls'][rows]
    h = cls == 0
    sa = x[:, RI['T_sa']] - x[:, RI['T_sa_sp']]
    o = dict(n=int(len(rows)), faulty_share=float((cls > 0).mean()),
             visible_share_of_faulty=float((b['eff'][rows][cls > 0] >= 1).mean()),
             median_effect_faulty=float(np.median(b['eff'][rows][cls > 0])),
             fan_on_hours_healthy=float(on[h].sum(1).mean() / 4),
             healthy_sat_minus_sp_median=float(np.nanmedian(np.where(on[h], sa[h], np.nan))),
             healthy_sat_minus_sp_iqr=[float(v) for v in np.nanpercentile(np.where(on[h], sa[h], np.nan), [25, 75])])
    ft = b['ftype'][rows]
    for t, ch in (('oa_damper_stuck', 'damper_oa'), ('cooling_valve_stuck', 'valve_cc'),
                  ('heating_valve_stuck', 'valve_hc'), ('cooling_valve_leak', 'valve_cc'),
                  ('heating_valve_leak', 'valve_hc'), ('oa_damper_leak', 'damper_oa')):
        s = ft == t
        if s.any():
            v = np.where(on[s], x[s][:, RI[ch]], np.nan)
            o[f'logged_{t}'] = dict(n=int(s.sum()), mean_fan_on=float(np.nanmean(v)))
    o['class_counts'] = {int(k): int(v) for k, v in zip(*np.unique(cls, return_counts=True))}
    return o


def check():
    os.makedirs(OUT, exist_ok=True)
    R = {}
    ref = load_bank('ref_v3code')
    v5 = load_bank('val_v5')
    W = 300
    s5 = np.where(v5['sample'] & (v5['window'] < W))[0]
    R['bank_v3code_300'] = _bank_stats(ref, np.arange(len(ref['cls'])))
    R['bank_v5_first300_samples'] = _bank_stats(v5, s5)
    R['bank_v5_first300_samples']['rows_per_window'] = float(len(s5) / len(np.unique(v5['window'][s5])))
    R['bank_v3code_300']['rows_per_window'] = float(len(ref['cls']) / W)
    R['bank_v5_fault_day1_share'] = float((v5['fault_day'][v5['day'] == 1] == 1).mean())
    # paired: windows whose fault starts on day 5 have the same draws as the shipped code up to the operator seed,
    # so healthy days must match exactly (same buildings, weather, controls, operator jitter)
    w5 = np.unique(v5['window'][(v5['window'] < W) & (v5['fault_day'] == 5)])
    # ref bank rows per window: days 1-4 healthy, then (faulted day, [healthy day if k >= 10]) for k = 5..11
    ref_w = []
    for w in range(W):
        ref_w += [(w, k, 0) for k in range(1, 5)]
        for k in range(5, 12):
            ref_w.append((w, k, 1))
            if k >= 10:
                ref_w.append((w, k, 0))
    if len(ref_w) == len(ref['cls']):
        ri = {t: i for i, t in enumerate(ref_w)}
        ix = Index(v5)
        dh, df, nf_same, nf_tot = [], [], 0, 0
        w5s = set(w5.tolist())
        for (w, k, r), i in ri.items():
            if w not in w5s:
                continue
            j = ix.A[w, k, r]
            if j < 0:
                continue
            a, c = ref['x'][i].astype(np.float32), v5['x'][j].astype(np.float32)
            dd = float(np.nanmax(np.abs(a - c))) if np.isfinite(a - c).any() else 0.0
            if r == 0:
                dh.append(dd)
            else:
                nf_tot += 1
                nf_same += dd == 0.0
                df.append(dd)
        R['paired_day5_windows'] = dict(n_windows=int(len(w5)), healthy_days=len(dh),
                                        healthy_max_abs_diff=float(max(dh) if dh else -1),
                                        faulted_days=nf_tot, faulted_identical=int(nf_same),
                                        note='faulted days differ only where log_commands drew feedback differently '
                                             '(v3 per day, v5 per window)')
    else:
        R['paired_day5_windows'] = dict(skipped=f'ref bank rows {len(ref["cls"])} != {len(ref_w)} expected')
    # noise v4 vs v5 on the same val v5 sample rows (2,000 drawn with seed 1)
    from evaluate import noise_for
    rng = np.random.default_rng(1)
    cand = np.where(v5['sample'] & ((v5['cls'] == 0) | (v5['eff'] >= EFF_MIN)))[0]
    rows = np.sort(rng.choice(cand, 2000, replace=False))
    n4 = noise_for('v4')
    F4 = np.stack([featurize(n4.apply(v5['x'][i].astype(np.float32), rng, str(v5['system'][i]))) for i in rows])
    Fd, _ = featurize_rows(v5, rows, noise5(), rng)
    F5 = np.stack([Fd[int(i)] for i in rows]).astype(np.float32)
    on = np.nan_to_num(v5['x'][rows][:, RI['fan_on']].astype(np.float32)) > 0.5
    rowsum = {}
    for k in range(NF1):
        a = np.where(on, F4[:, k], np.nan)
        c = np.where(on, F5[:, k], np.nan)
        rowsum[k] = dict(mean_v4=float(np.nanmean(a)), mean_v5=float(np.nanmean(c)), sd_v4=float(np.nanstd(a)),
                         sd_v5=float(np.nanstd(c)))
    R['noise_v4_vs_v5_rows'] = rowsum
    R['noise_max_abs_mean_diff'] = float(max(abs(v['mean_v4'] - v['mean_v5']) for v in rowsum.values()))
    R['noise_max_rel_sd_diff'] = float(max(abs(v['sd_v4'] - v['sd_v5']) / max(v['sd_v4'], 1e-6)
                                           for v in rowsum.values()))
    # r6 on the same 2,000 rows under each noise (single-day network: should not care which)
    m = ev.NpzModel(WEIGHTS)
    t = v5['cls'][rows]
    for nm, F in (('v4', F4), ('v5', F5)):
        R[f'r6_on_2000_rows_noise_{nm}'] = ev.skill(m, dict(F=F, cls=t, eff=v5['eff'][rows], waste=v5['waste'][rows]))
    json.dump(R, open(f'{OUT}/check_v4.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != 'noise_v4_vs_v5_rows'}, indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'frozen':
        build_frozen()
    elif cmd == 'check':
        check()
    elif cmd == 'score':
        score(sys.argv[2:])
