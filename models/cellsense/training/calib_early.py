"""Set the early-life coupling's strength: simulate cells with the v2 aging model, read Severson's feature (log variance of
dQ_100-10(V), on curves smoothed over 24 mV) and regress log life on it. The six calibration cells give the target slope,
intercept and scatter; Severson's training cells are printed for reference only."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, time, numpy as np
sys.path.insert(0, _P.SRC); import realdata as RD, calibrate as CA, agesim as AS, simbank as SB, lifeexam as LE, dq_early as DQ
from scipy.ndimage import gaussian_filter1d
sm = lambda x: gaussian_filter1d(x, 4, axis=-1, mode='nearest')

def fill(Q):
    Q = Q.copy()
    for r in Q:
        ok = np.isfinite(r); last = np.where(ok)[0].max(); r[last + 1:] = r[last]
    return Q

def real_rel():
    cells, V = RD.load(); SP = LE.split(cells); out = {}
    for name in ('dev', 'train'):
        ks = [k for k in cells if (k in CA.CAL if name == 'dev' else (SP[k] == 'train' and k not in CA.CAL))]; lv, L = [], []
        for k in ks:
            d = DQ.real_dq(cells[k])
            if d is None: continue
            lv.append(np.log10(np.var(sm(d)))); L.append(np.log10(cells[k]['cycle_life_pub']))
        out[name] = rel(np.array(lv), np.array(L))
    return out

def rel(lv, L):
    A = np.polyfit(lv, L, 1); return dict(n=len(lv), corr=float(np.corrcoef(lv, L)[0, 1]), slope=float(A[0]), icpt=float(A[1]), resid=float(np.std(L - np.polyval(A, lv))), lv_mid=float(np.median(lv)), L_mid=float(np.median(L)))

def sim_rel(seed=0, B=600):
    rng = np.random.default_rng(seed); P0 = SB.fresh(rng, B); R = AS.rates(rng, B)
    Q0, T0, cc0, tot0, ir0 = SB.curves(P0); q0, _ = AS.window(P0['Q_p'], P0['Q_n'], P0['Q_Li'], P0['sig_p'], P0['sig_n'])
    tr = AS.trajectory(P0, R, kappa=tot0 / q0); L = tr['life']; k = np.where(np.isfinite(L) & (L >= 150))[0]
    sub = lambda d: {a: (v[k] if isinstance(v, np.ndarray) else v) for a, v in d.items()}; P0k, Rk = sub(P0), sub(R); trk = {a: (v[k] if a != 'n' else v) for a, v in tr.items()}
    Qa = fill(SB.curves(AS.state_at(P0k, Rk, trk, np.full(len(k), 1)))[0]); Qb = fill(SB.curves(AS.state_at(P0k, Rk, trk, np.full(len(k), 10)))[0])
    lv = np.log10(np.var(sm(Qb - Qa), 1)); m = Rk['c'] < 5.4; out = rel(lv[m], np.log10(L[k][m])); out['all_c'] = {a: round(b, 3) for a, b in rel(lv, np.log10(L[k])).items()}; out['life_pct'] = np.percentile(L[k], [5, 25, 50, 75, 95]).round(0).tolist(); out['frac_kept'] = len(k) / B
    return out

def anchor_lv(Fa):
    """simulated log var dQ_100-10 at each calibration cell's own parameters (central values), slow-graphite share Fa"""
    import fit_life as FL, calibrate as CA2
    F = json.load(open(_P.WORK + '/aging_calibration.json')); E = json.load(open(_P.WORK + '/early_calibration.json')); out = []
    for j, k in enumerate(CA2.CAL):
        f = F[k]; P0 = CA2.params(np.array(FL.CA_FITS[k]['theta']), 30.0); P0['Q_Li'] = P0['Q_Li'] * SB.Q_LI_TRIM
        R = FL.rates_from(np.array(f['z'])[None], f['c']); R.update(F=np.array([Fa[j]]), gam=np.array([np.mean(AS.GAM)]), F_cap=np.array([0.11]), R_s=np.array([np.exp(E[k]['log_R_s'])]), k_e=np.array([0.2]), p_e=2.0, k_f=np.zeros(1), k_f2=np.array([0.05]), k_e2=np.array([0.16]), ge_drv='lli')
        tr = AS.trajectory(P0, R, kappa=np.array([f['kappa']]))
        Qa = fill(SB.curves(AS.state_at(P0, R, tr, np.array([1])))[0]); Qb = fill(SB.curves(AS.state_at(P0, R, tr, np.array([10])))[0])
        out.append(float(np.log10(np.var(sm(Qb - Qa)))))
    return np.array(out)

def match_anchors(iters=4):
    """adjust each calibration cell's slow-graphite share so its simulated dQ_100-10 has the real cell's variance"""
    cells, V = RD.load(); E = json.load(open(_P.WORK + '/early_calibration.json'))
    real = np.array([np.log10(np.var(sm(DQ.real_dq(cells[k])))) for k in CA.CAL]); Fa = np.array([E[k]['f_s'] for k in CA.CAL])
    for it in range(iters):
        lv = anchor_lv(Fa); print('iter', it, 'F', Fa.round(4), 'sim lv', lv.round(3), 'real', real.round(3), flush=True)
        Fa = np.clip(Fa * 10 ** ((real - lv) / 2.2), 0.003, 0.3)
    for k, f in zip(CA.CAL, Fa): E[k]['F_adj'] = float(f)
    json.dump(E, open(_P.WORK + '/early_calibration.json', 'w'), indent=1)

if __name__ == '__main__':
    if sys.argv[1] == 'anchors': match_anchors(); sys.exit()
    print('real', json.dumps(real_rel()), flush=True)
    import itertools
    for pc, sk, sf, h in itertools.product(*[list(map(float, a.split(','))) for a in sys.argv[1:5]]):
            AS.P_C, AS.SIG_KN, AS.SIG_F, AS.SIG_HIDDEN = pc, sk, sf, h; t = time.time(); r = sim_rel()
            print('P_C %.1f SIG_KN %.2f SIG_F %.2f H %.2f' % (pc, sk, sf, h), json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()}), '%.0fs' % (time.time() - t), flush=True)
