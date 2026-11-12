"""Early-life signal: Severson's dQ_100-10(V), the change in the 4C discharge curve between cycles 10 and 100, real vs
simulated. Used to calibrate the early-life physics on the six calibration cells."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import realdata as RD, calibrate as CA, agesim as AS, simbank as SB, cellsim as CS, fit_life as FL
STEP = 4

def real_dq(c, a=10, b=100):
    idx = c['cyc_idx']; ra = int(np.argmin(np.abs(idx + 1 - a))); rb = int(np.argmin(np.abs(idx + 1 - b)))
    if abs(idx[ra] + 1 - a) > 2 or abs(idx[rb] + 1 - b) > 2: return None
    qa, qb = c['Qdlin'][ra].astype(float), c['Qdlin'][rb].astype(float)
    if not (np.isfinite(qa).all() and np.isfinite(qb).all()): return None
    return (qb - qa)[::STEP]

def sim_dq(P0, R, kappa, a=10, b=100):
    """dQ_b-a for simulated cells with rates R (dicts of (B,) arrays)"""
    B = len(P0['Q_p']); tr = AS.trajectory(P0, R, n_max=b + 10, kappa=kappa)
    Pa = AS.state_at(P0, R, tr, np.full(B, a // 10)); Pb = AS.state_at(P0, R, tr, np.full(B, b // 10))
    Qa = SB.curves(Pa)[0]; Qb = SB.curves(Pb)[0]
    def fill(Q):   # q_of_v leaves NaN below the cut-off voltage reached: carry the last value
        Q = Q.copy()
        for r in Q:
            ok = np.isfinite(r); last = np.where(ok)[0].max(); r[last + 1:] = r[last]
        return Q
    return fill(Qb) - fill(Qa), tr

if __name__ == '__main__':
    cells, V = RD.load(); rows = []
    for k, c in cells.items():
        d = real_dq(c)
        if d is None: continue
        rows.append((k, c['cycle_life_pub'], np.log10(np.var(d)), d.min(), d.mean()))
    L = np.array([r[1] for r in rows]); lv = np.array([r[2] for r in rows]); ok = np.isfinite(L)
    print('real cells', ok.sum(), 'corr(log var dQ, log life) = %.3f' % np.corrcoef(lv[ok], np.log10(L[ok]))[0, 1])
    A = np.polyfit(lv[ok], np.log10(L[ok]), 1); print('fit log10 life = %.3f * log10 var + %.3f' % tuple(A))
    print('real log10 var range', np.percentile(lv[ok], [5, 50, 95]).round(2))
    F = json.load(open(_P.WORK + '/aging_calibration.json'))
    for k in CA.CAL:
        c = cells[k]; d = real_dq(c); f = F[k]
        P0 = CA.params(np.array(FL.CA_FITS[k]['theta']), 30.0); P0['Q_Li'] = P0['Q_Li'] * SB.Q_LI_TRIM
        R = FL.rates_from(np.array(f['z']), f['c']); ds, tr = sim_dq(P0, R, np.array([f['kappa']]))
        print(k, 'life', f['life'], 'real log var %.2f min %.4f | sim log var %.2f min %.4f' % (np.log10(np.var(d)), d.min(), np.log10(np.var(ds[0])), ds[0].min()))
