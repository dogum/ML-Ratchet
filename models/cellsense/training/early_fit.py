"""How the slow graphite grows through life on the six calibration cells. At each of several cycles n, the change in the
4C discharge curve since cycle 10, dQ_n-10(V), is fitted with the cell's aging trajectory (fit_life) plus three free
knobs: the slow (poorly connected) graphite share f_s and its resistance R_s, and a multiplier on the slow transport
resistance R_e, plus a small lithium correction. Curves are smoothed over 24 mV before comparing: the simulator's 4C
curve has sharper staging features than real cells, which a fit should not chase."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import realdata as RD, calibrate as CA, agesim as AS, simbank as SB, cellsim as CS, fit_life as FL, dq_early as DQ
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
sm = lambda x: gaussian_filter1d(x, 4, axis=-1, mode='nearest')
NAMES = ['f_s', 'log_R_s', 'log_m_e', 'dlli']; LO = np.array([0, np.log(0.02), -0.3, -0.02]); HI = np.array([0.4, np.log(2.0), 1.2, 0.05])
R_S_FIX = np.log(0.3)      # fixed resistance for the slow graphite so its share is comparable cell to cell (the two trade off)

def fill(Q):
    Q = Q.copy()
    for r in Q:
        ok = np.isfinite(r); last = np.where(ok)[0].max(); r[last + 1:] = r[last]
    return Q

def fit_cell(k, cells, F, targets):
    c = cells[k]; f = F[k]; life = f['life']; out = []
    P0 = CA.params(np.array(FL.CA_FITS[k]['theta']), 30.0); P0['Q_Li'] = P0['Q_Li'] * SB.Q_LI_TRIM
    R = FL.rates_from(np.array(f['z']), f['c']); tr = AS.trajectory(P0, R, n_max=int(life * 1.1) + 20, kappa=np.array([f['kappa']]))
    Pa = AS.state_at(P0, R, tr, np.array([1])); Pa['f_s'][:] = 0.0; Q10 = fill(SB.curves(Pa)[0]); prev = None
    for n in targets(life):
        d = DQ.real_dq(c, 10, n)
        if d is None: continue
        d = sm(d); Pn = AS.state_at(P0, R, tr, np.array([n // 10]))
        def sim(TH):
            TH = np.atleast_2d(TH); B = len(TH); PB = {a: np.repeat(v, B) for a, v in Pn.items()}
            PB['f_s'] = TH[:, 0]; PB['R_s'] = np.exp(TH[:, 1]) if FREE_RS else np.full(B, np.exp(R_S_FIX)); PB['R_e'] = PB['R_e'] * np.exp(TH[:, 2]); PB['Q_Li'] = PB['Q_Li'] * (1 - TH[:, 3])
            return sm(fill(SB.curves(PB)[0]) - Q10)
        def jac(th, h=0.01):
            TT = np.clip(np.repeat(th[None], 5, 0) + np.vstack([np.zeros(4), h * np.eye(4)]), LO, HI); D = sim(TT) - d
            return ((D[1:] - D[0]) / np.maximum(np.diag(TT[1:] - th[None]), 1e-9)[:, None]).T
        rng = np.random.default_rng(n); S = LO + (HI - LO) * rng.random((30, 4))
        if prev is not None: S[0] = prev
        E = np.sqrt(np.mean((sim(S) - d) ** 2, 1)); s0 = S[np.argmin(E)]
        sol = least_squares(lambda th: sim(th)[0] - d, s0, jac=jac, bounds=(LO, HI), max_nfev=25); prev = sol.x
        r = dict(cell=k, n=int(n), life=life, frac=n / life, rms=float(np.sqrt(np.mean(sol.fun ** 2))), real_rms=float(np.sqrt(np.mean(d ** 2))), **dict(zip(NAMES, sol.x.tolist())))
        out.append(r); print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
    return out

FREE_RS = True
if __name__ == '__main__':
    FREE_RS = len(sys.argv) < 4 or sys.argv[3] != 'fixRs'
    cells, V = RD.load(); F = json.load(open(_P.WORK + '/aging_calibration.json'))
    which = sys.argv[1].split(','); tg = (lambda L: sorted({50, 100, 200, int(round(0.5 * L / 10) * 10), int(round(0.8 * L / 10) * 10)})) if FREE_RS else (lambda L: [50, 100, 200])
    res = sum((fit_cell(k, cells, F, tg) for k in which), [])
    json.dump(res, open(f'{_P.WORK}/early_fit_{sys.argv[2]}.json', 'w'), indent=1)
