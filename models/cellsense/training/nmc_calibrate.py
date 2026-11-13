"""CellSense NMC811: fit the simulator to the three DEV cells only (W3, W4, W8; never the exam cells).

  fresh  each dev cell's first capacity test (diagnostic 1): electrode capacities, lithium inventory, charge voltage,
         diffusion times, exchange currents, resistance and OCP smearing, by least squares on the C/20 curve V(Q) and the
         capacity to 2.5 V.  -> work/nmc/cal_fresh.json
  modes  every later test of each dev cell, with the fresh fit held: lithium lost, negative lost, positive lost and
         resistance growth (the classical "voltage fitting" diagnosis of Dubarry / Birkl).  -> work/nmc/cal_modes.json
These set the new-cell spread and the aging priors of the training simulator (nmcbank.py), and the mismatch left by the
fits is what the next rungs build in.

    python nmc_calibrate.py fresh|modes [tag]     (tag '_ocp': with the dev-fitted OCP correction, nmc_ocpfit.py)
"""
import _paths as _P
import os, sys, json, numpy as np
sys.path.insert(0, _P.SRC); import nmcsim as NS, nmc_data as D
from scipy.optimize import least_squares
OUT = os.path.join(_P.WORK, 'nmc'); os.makedirs(OUT, exist_ok=True)
NAMES = ['Q_p', 'Q_n', 'Q_Li', 'tau_p', 'tau_n', 'I0_p', 'I0_n', 'R_ohm', 'sig_p', 'sig_n']
LO = np.log([6.0, 4.8, 5.5, 300, 10, 0.5, 0.1, 0.001, 1e-4, 1e-4]); HI = np.log([11.0, 8.5, 9.0, 1e5, 2e4, 300, 100, 0.08, 0.05, 0.05])
VF_LO, VF_HI = 4.17, 4.26


def target(row, n=160):
    q_end = row['q_total']; qg = np.linspace(0.03, q_end - 0.03, n)
    return dict(qg=qg, v=np.interp(qg, row['Q'], row['V']), q_tot=q_end)


def params(z, fixed=None):
    z = np.atleast_2d(z); B = len(z); P = {n: np.exp(z[:, i]) for i, n in enumerate(NAMES)}
    P['V_full'] = z[:, len(NAMES)]; P['R_e'] = np.full(B, 1e-4); P['tau_e'] = np.full(B, 100.0)
    P['Q_Li'] = np.minimum(P['Q_Li'], 0.999 * P['Q_n'] + 0.02 * P['Q_p'])
    return P


CORR = None     # OCP correction (nmc_ocpfit.py), used when the tag is '_ocp'


def curve_res(P, tg, ocp=None):
    o = NS.simulate(P, dt=60.0, ocp=ocp, corr=CORR); B = len(P['Q_p']); R = []
    for b in range(B):
        m = np.isfinite(o['V'][b]); q, v = np.append(o['Q'][b][m], o['Q_total'][b]), np.append(o['V'][b][m], 2.5)
        if len(q) < 5: R.append(np.full(len(tg['qg']) + 1, 1e3)); continue
        R.append(np.concatenate([(np.interp(tg['qg'], q, v, right=2.5) - tg['v']) * 1000 / 2.0, [(o['Q_total'][b] - tg['q_tot']) * 1000 / 2.0]]))
    return np.array(R)


def fit_fresh(row, rng):
    tg = target(row); lo = np.append(LO, VF_LO); hi = np.append(HI, VF_HI); res = lambda Z: curve_res(params(Z), tg)
    Z = lo + (hi - lo) * rng.random((4000, len(lo))); c = Chen = NS.chen2020()
    Z[0, :len(NAMES)] = np.clip(np.log([c['Q_p'], c['Q_n'], c['Q_Li'], c['tau_p'], c['tau_n'], c['I0_p'], c['I0_n'], 0.02, 1e-3, 1e-3]), LO, HI); Z[0, -1] = 4.19
    sc = np.concatenate([np.sum(res(Z[i:i + 1000]) ** 2, 1) for i in range(0, len(Z), 1000)]); starts = Z[np.argsort(sc)[:4]]

    def jac(z, h=0.01):
        Zs = np.repeat(z[None], len(z) + 1, 0); Zs[1:] += h * np.eye(len(z)) * np.where(np.arange(len(z)) < len(NAMES), 1, 0.05)[None]
        Zs = np.clip(Zs, lo, hi); Rr = res(Zs); return ((Rr[1:] - Rr[0]) / np.maximum(np.diag(Zs[1:] - z[None]), 1e-9)[:, None]).T
    sols = [least_squares(lambda z: res(z)[0], s, jac=jac, bounds=(lo, hi), max_nfev=80) for s in starts]
    sol = min(sols, key=lambda s: s.cost); r = res(sol.x)[0]
    return sol.x, dict(v_rms_mV=float(np.sqrt(np.mean((r[:-1] * 2) ** 2))), v_max_mV=float(np.abs(r[:-1] * 2).max()), q_err_mAh=float(r[-1] * 2), resid_mV=(r[:-1] * 2).tolist(), qg=tg['qg'].tolist())


# ---- aging: degradation modes on top of a fresh fit ----------------------------------------------------------------
MODES = ['lli', 'lam_ne', 'lam_pe', 'r_growth']


def aged(P0, m, f_li=0.9, x0y0=None, corr=None):
    """apply modes m (B, 4) to fresh parameters P0 (dict of (B,) arrays): lithium lost (fraction of the new cell's cyclable
    lithium), negative and positive electrode lost (fractions; f_li of the lost negative was lithiated, as at the top of
    charge where the cell spends its rest), and resistance growth (ratio - 1)"""
    P = {k: np.array(v, float).copy() for k, v in P0.items()}; lli, ln, lp, rg = m.T
    TP, TN = NS.tables(P0, len(lli), corr); x0, y0 = NS.full_state(P0['Q_p'], P0['Q_n'], P0['Q_Li'], P0['V_full'], TP, TN) if x0y0 is None else x0y0
    P['Q_n'] = P0['Q_n'] * (1 - ln); P['Q_p'] = P0['Q_p'] * (1 - lp); P['I0_n'] = P0['I0_n'] * (1 - ln); P['I0_p'] = P0['I0_p'] * (1 - lp)
    P['Q_Li'] = np.clip(P0['Q_Li'] * (1 - lli) - ln * P0['Q_n'] * y0 * f_li, 0.5, 0.999 * P['Q_n'] + 0.02 * P['Q_p'])
    P['R_ohm'] = P0['R_ohm'] * (1 + rg); return P


def fit_modes(P0, row):
    tg = target(row); lo, hi = np.array([-0.05, -0.05, -0.05, -0.5]), np.array([0.4, 0.4, 0.4, 5.0])
    rep = lambda B: {k: np.repeat(v, B) for k, v in P0.items()}
    res = lambda M: curve_res(aged(rep(len(np.atleast_2d(M))), np.atleast_2d(M), corr=CORR), tg)
    grid = np.random.default_rng(0).uniform(lo, hi * 0.4, (600, 4)); grid[0] = 0
    sc = np.sum(res(grid) ** 2, 1); start = grid[np.argmin(sc)]

    def jac(z, h=2e-3):
        Zs = np.repeat(z[None], 5, 0); Zs[1:] += h * np.eye(4) * np.array([1, 1, 1, 20])[None]; Rr = res(Zs)
        return ((Rr[1:] - Rr[0]) / np.diag(Zs[1:] - z[None])[:, None]).T
    sol = least_squares(lambda z: res(z)[0], start, jac=jac, bounds=(lo, hi), max_nfev=60); r = res(sol.x)[0]
    return sol.x, dict(v_rms_mV=float(np.sqrt(np.mean((r[:-1] * 2) ** 2))), q_err_mAh=float(r[-1] * 2), resid_mV=(r[:-1] * 2).tolist())


if __name__ == '__main__':
    what = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else ''
    if tag == '_ocp': CORR = json.load(open(os.path.join(OUT, 'ocp_corr.json')))
    rows = D.readings(D.DEV); rng = np.random.default_rng(1)
    if what == 'fresh':
        fits = {}
        for r in rows:
            if r['diag'] != 1: continue
            z, info = fit_fresh(r, rng); P = params(z)
            fits[r['cell']] = dict(z=z.tolist(), params={k: float(v[0]) for k, v in P.items()}, **info)
            print(r['cell'], {k: round(v, 3) for k, v in info.items() if not isinstance(v, list)}, {k: round(float(v[0]), 4) for k, v in P.items()}, flush=True)
        json.dump(fits, open(os.path.join(OUT, f'cal_fresh{tag}.json'), 'w'), indent=1)
    elif what == 'modes':
        F = json.load(open(os.path.join(OUT, f'cal_fresh{tag}.json'))); out = {}
        for r in rows:
            P0 = {k: np.array([v]) for k, v in F[r['cell']]['params'].items()}
            m, info = fit_modes(P0, r); out[f"{r['cell']}_{r['diag']}"] = dict(cell=r['cell'], diag=r['diag'], cycle=r['cycle'], q_total=r['q_total'], modes=dict(zip(MODES, m.tolist())), **info)
            print(r['cell'], r['diag'], 'Q %.3f' % r['q_total'], {k: round(float(v), 4) for k, v in zip(MODES, m)}, 'rms %.1f mV' % info['v_rms_mV'], 'dQ %.1f mAh' % info['q_err_mAh'], flush=True)
        json.dump(out, open(os.path.join(OUT, f'cal_modes{tag}.json'), 'w'), indent=1)
