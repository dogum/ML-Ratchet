"""Fit the simulator's beginning-of-life parameters to the real A123 APR18650M1A: six calibration cells (two per batch,
excluded from every exam), each at cycles 5-10. Fit targets: the 4C discharge voltage curve V(Q), the can temperature T(Q),
the constant-current capacity, the total capacity after the 2.0 V hold, and the 30 ms pulse resistance at 80% SOC.
A vectorized random search finds the basin, then least squares refines each cell. The spread across the six sets the
fresh-cell ranges the training simulator samples from."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import cellsim as CS, realdata as RD
from scipy.optimize import least_squares
CAL = ['b1c7', 'b1c30', 'b2c12', 'b2c30', 'b3c10', 'b3c25']
NAMES = ['Q_p', 'Q_n', 'Q_Li', 'tau_p', 'tau_n', 'I0_p', 'I0_n', 'R_ohm', 'R_e', 'tau_e', 'C_th', 'hA', 'sig_p', 'sig_n']
LO = np.log([1.05, 1.05, 1.0, 1, 20, 0.5, 0.5, 0.002, 0.0005, 1, 20, 0.03, 0.002, 0.002]); HI = np.log([2.0, 2.0, 1.3, 3000, 30000, 300, 300, 0.03, 0.04, 600, 150, 1.0, 0.1, 0.12])
FIX = dict(V_full=3.6, Ea_kp=39570., Ea_kn=35000., Ea_dp=35000., Ea_dn=35000., Ea_e=15000., T_amb=30.)
QG = np.linspace(0.01, 1.0, 120)

def target(c, V):
    rows = [i for i, j in enumerate(c['cyc_idx']) if 4 <= j <= 9]
    q = np.nanmean(c['Qdlin'][rows], 0); t = np.nanmean(c['Tdlin'][rows], 0); cyc = c['cyc_idx'][rows] + 1
    S = c['summary']; idx = [int(np.where(S['cycle'] == k)[0][0]) for k in cyc if (S['cycle'] == k).any()]
    qcc = float(np.nanmax(q)); qg = QG[QG < qcc - 0.004]
    vq = np.interp(qg, q[::-1] if q[0] > q[-1] else q, V[::-1] if q[0] > q[-1] else V)     # V(Q): Qdlin rises as V falls
    tq = np.interp(qg, np.sort(q), t[np.argsort(q)])
    return dict(qg=qg, v=vq, t=tq, q_cc=qcc, q_tot=float(np.mean(S['QDischarge'][idx])), ir=float(np.mean(S['IR'][idx])), T0=float(t[np.isfinite(t)][0]))

def params(theta, T0):
    th = np.atleast_2d(theta); B = len(th); P = {n: np.exp(th[:, i]) for i, n in enumerate(NAMES)}
    P['Q_Li'] = np.minimum(P['Q_Li'], 0.995 * P['Q_n'] + 0.005 * P['Q_p'])
    P.update({k: np.full(B, v) for k, v in FIX.items()}); P['T0'] = np.full(B, T0); return P

def residuals(theta, tg, dt=3.0):
    P = params(theta, tg['T0']); o = CS.simulate(P, dt=dt, N=10)
    out = []
    for b in range(len(P['Q_p'])):
        m = o['cc'][b] & np.isfinite(o['V'][b]); q, v, t = o['Q'][b][m], o['V'][b][m], o['T'][b][m]
        if len(q) < 5: out.append(np.full(len(tg['qg']) * 2 + 3, 1e3)); continue
        vs = np.interp(tg['qg'], q, v, right=2.0); ts = np.interp(tg['qg'], q, t)
        rp = CS.pulse_resistance({k: x[b:b + 1] for k, x in P.items()})[0]
        out.append(np.concatenate([(vs - tg['v']) * 1000 / 5, (ts - tg['t']) / 0.25, [(o['Q_cc'][b] - tg['q_cc']) * 1000 / 3, (o['Q_total'][b] - tg['q_tot']) * 1000 / 3, (rp - tg['ir']) * 1e4 / 2]]))
    return np.array(out)

def warm_starts():
    import ast, re; out = {}
    for line in open(_P.WORK + '/logs_cal2.txt'):
        m = re.match(r"^(b\dc\d+) \{.*?\} (\{.*\})$", line.strip())
        if m: p = ast.literal_eval(m.group(2)); out[m.group(1)] = np.log([p[n] for n in NAMES])
    return out

if __name__ == '__main__':
    cells, V = RD.load(); rng = np.random.default_rng(1); fits = {}; WS = warm_starts() if '--refine' in sys.argv else {}
    for k in CAL:
        tg = target(cells[k], V)
        th = LO + (HI - LO) * rng.random((1000 if WS else 3000, len(NAMES))); best = []
        for i in range(0, len(th), 1000):
            r = residuals(th[i:i + 1000], tg, dt=6.0); best.append((np.sum(r ** 2, 1), th[i:i + 1000]))
        sc = np.concatenate([b[0] for b in best]); TH = np.concatenate([b[1] for b in best]); start = TH[np.argsort(sc)[:1 if WS else 3]]
        if k in WS: start = np.concatenate([np.clip(WS[k], LO + 1e-6, HI - 1e-6)[None], start])
        def jac(z, h=0.01):   # all finite differences in one vectorized simulation
            Z = np.repeat(z[None], len(z) + 1, 0); Z[1:] += h * np.eye(len(z)); Z = np.clip(Z, LO, HI); R = residuals(Z, tg)
            return ((R[1:] - R[0]) / np.maximum(np.diag(Z[1:] - z[None]), 1e-9)[:, None]).T
        sols = [least_squares(lambda z: residuals(z, tg)[0], s, jac=jac, bounds=(LO, HI), x_scale=0.3, max_nfev=60) for s in start]
        sol = min(sols, key=lambda s: s.cost); r = residuals(sol.x, tg)[0]; n = len(tg['qg'])
        fits[k] = dict(theta=sol.x.tolist(), params={n_: float(np.exp(v)) for n_, v in zip(NAMES, sol.x)}, v_rms_mV=float(np.sqrt(np.mean((r[:n] * 5) ** 2))), t_rms_C=float(np.sqrt(np.mean((r[n:2 * n] * 0.25) ** 2))),
                       q_cc_err_mAh=float(r[2 * n] * 3), q_tot_err_mAh=float(r[2 * n + 1] * 3), ir_err_mohm=float(r[2 * n + 2] * 2 / 10), target={a: (b if np.isscalar(b) else None) for a, b in tg.items()})
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in fits[k].items() if a not in ('theta', 'target', 'params')}, {a: round(b, 4) for a, b in fits[k]['params'].items()}, flush=True)
        json.dump(fits, open(_P.WORK + '/calibration.json', 'w'), indent=1)
