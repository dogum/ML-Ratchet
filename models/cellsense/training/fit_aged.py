"""Which physical aging reproduces a real worn-out curve? Start from a calibration (dev) cell's own new-cell fit and fit
only the aging knobs to its curve near end of life: lost lithium, lost graphite, lost LFP, ohmic, kinetic, slow-polarization
and diffusion growth."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import cellsim as CS, realdata as RD, calibrate as CA
from scipy.optimize import least_squares
AG = ['lli', 'lam_n', 'lam_p', 'g_r', 'g_k', 'g_e', 'g_dn', 'g_dp', 'g_te', 'f_s', 'lnR_s']; LO = np.array([0, 0, 0, 0, 0, 0, 0, 0, -1.0, 0, -3.5]); HI = np.array([0.4, 0.5, 0.4, 1.5, 2.5, 3.0, 3.0, 3.0, 1.5, 0.5, 1.5])

def aged(P0, z):
    P = {k: v.copy() for k, v in P0.items()}; B = len(z)
    x0, y0 = CS.full_state(P0['Q_p'], P0['Q_n'], P0['Q_Li'], P0['V_full'], P0['sig_p'], P0['sig_n'])
    P['Q_Li'] = P0['Q_Li'] * (1 - z[:, 0]); P['Q_n'] = P0['Q_n'] * (1 - z[:, 1]); P['Q_p'] = P0['Q_p'] * (1 - z[:, 2]); P['Q_Li'] = np.minimum(P['Q_Li'], 0.995 * P['Q_n'] + 0.005 * P['Q_p'])
    P['R_ohm'] = P0['R_ohm'] * np.exp(z[:, 3]); P['I0_n'] = P0['I0_n'] * np.exp(-z[:, 4]); P['I0_p'] = P0['I0_p'] * np.exp(-z[:, 4]); P['R_e'] = P0['R_e'] * np.exp(z[:, 5])
    P['tau_n'] = P0['tau_n'] * np.exp(z[:, 6]); P['tau_p'] = P0['tau_p'] * np.exp(z[:, 7]); P['tau_e'] = P0['tau_e'] * np.exp(z[:, 8]); P['f_s'] = z[:, 9]; P['R_s'] = np.exp(z[:, 10]); return P

def target_at(c, V, j0, j1):
    rows = [i for i, j in enumerate(c['cyc_idx']) if j0 <= j <= j1]; q = np.nanmean(c['Qdlin'][rows], 0); t = np.nanmean(c['Tdlin'][rows], 0); S = c['summary']
    idx = c['cyc_idx'][rows]; qcc = float(np.nanmax(q)); qg = np.linspace(0.01, qcc - 0.004, 120)
    return dict(qg=qg, v=np.interp(qg, q, V), t=np.interp(qg, np.sort(q), t[np.argsort(q)]), q_cc=qcc, q_tot=float(np.mean(S['QDischarge'][idx])), ir=float(np.mean(S['IR'][idx])), T0=float(t[0]))

def resid(Z, P0, tg):
    Z = np.atleast_2d(Z); B = len(Z); PB = {k: np.repeat(v, B) for k, v in P0.items()}; PB['T0'] = np.full(B, tg['T0']); P = aged(PB, Z); o = CS.simulate(P, dt=3.0); out = []
    rp = CS.pulse_resistance(P)
    for b in range(B):
        m = o['cc'][b] & np.isfinite(o['V'][b]); q, v, t = o['Q'][b][m], o['V'][b][m], o['T'][b][m]
        if len(q) < 5: out.append(np.full(2 * len(tg['qg']) + 3, 1e3)); continue
        out.append(np.concatenate([(np.interp(tg['qg'], q, v, right=2.0) - tg['v']) * 1000 / 5, (np.interp(tg['qg'], q, t) - tg['t']) / 0.25,
                                   [(o['Q_cc'][b] - tg['q_cc']) * 1000 / 2, (o['Q_total'][b] - tg['q_tot']) * 1000 / 2, (rp[b] - tg['ir']) * 1e4 / 2]]))
    return np.array(out)

if __name__ == '__main__':
    cells, V = RD.load(); fits = json.load(open(_P.WORK + '/calibration.json')); res = {}
    for k in sys.argv[1:]:
        c = cells[k]; L = int(c['cycle_life_pub']); P0 = CA.params(np.array(fits[k]['theta']), 30.0); P0['T0'] = np.array([30.0])
        for frac in (0.6, 0.97):
            j = int(frac * L); tg = target_at(c, V, j - 6, j + 4); rng = np.random.default_rng(0)
            Zs = LO + (HI - LO) * rng.random((600, len(AG))); r = resid(Zs, P0, tg); start = Zs[np.argmin(np.sum(r ** 2, 1))]
            jac = lambda z, h=0.01: ((lambda Z, R: ((R[1:] - R[0]) / np.maximum(np.diag(Z[1:] - z[None]), 1e-9)[:, None]).T)(*(lambda Z: (Z, resid(Z, P0, tg)))(np.clip(np.repeat(z[None], len(z) + 1, 0) + np.vstack([np.zeros(len(z)), h * np.eye(len(z))]), LO, HI))))
            sol = least_squares(lambda z: resid(z, P0, tg)[0], start, jac=jac, bounds=(LO, HI), max_nfev=40); rr = sol.fun; n = len(tg['qg'])
            res[f'{k}@{j}'] = dict(z={a: round(float(b), 3) for a, b in zip(AG, sol.x)}, v_rms=float(np.sqrt(np.mean((rr[:n] * 5) ** 2))), q_cc_err=float(rr[2 * n] * 2), q_tot_err=float(rr[2 * n + 1] * 2), ir_err=float(rr[2 * n + 2] * 0.2),
                                   target_hold=1000 * (tg['q_tot'] - tg['q_cc']), q_tot=tg['q_tot'], ir=tg['ir'])
            print(f'{k} cycle {j}', res[f'{k}@{j}'], flush=True)
    json.dump(res, open(_P.WORK + '/aged_fits.json', 'w'), indent=1)
