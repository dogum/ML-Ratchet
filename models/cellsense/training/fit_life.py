"""Calibrate the aging-rate priors on the six calibration cells: fit each one's whole measured capacity trajectory (total
discharge capacity every 10 cycles, to end of life) with the trajectory model, starting from its own new-cell fit. The
lifetime data is then drawn around these six fits, as the new-cell parameters are. The calibration cells are in no exam."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import realdata as RD, calibrate as CA, agesim as AS, simbank as SB, cellsim as CS
from scipy.optimize import least_squares
KN = ['k_s', 'k_n', 'beta', 'k_p', 'k_pl', 'delta', 'f_li']
# v2: SEI loss linear in cycles (k_s = fraction of lithium per 1000 cycles), and the lithium lost between cycles 10 and 100
# held near what the early curves show (dq_early / early_fit: within a few tenths of a percent)
LO = np.array([np.log(1e-3), np.log(1e-3), 1.0, 0.0, np.log(1e-3), np.log(0.003), 0.0]); HI = np.array([np.log(0.2), np.log(2.0), 3.5, 0.08, np.log(20.0), np.log(0.3), 1.0])
EARLY_LLI_SD = 0.002

def rates_from(z, c):
    z = np.atleast_2d(z); B = len(z)
    return dict(c=np.full(B, c), k_s=np.exp(z[:, 0]), k_n=np.exp(z[:, 1]), beta=z[:, 2], k_p=z[:, 3], k_pl=np.exp(z[:, 4]), delta=np.exp(z[:, 5]), f_li=z[:, 6],
                k_r=np.full(B, 0.3), k_k=np.full(B, 0.3), k_d=np.full(B, 0.3), k_e=np.full(B, 0.4), k_e2=np.full(B, 1.0), te=np.ones(B), k_f=np.full(B, 0.02), k_f2=np.full(B, 0.2), R_s=np.full(B, 0.2), sei='lin')

def main():
    cells, V = RD.load(); fits = {}
    for k in CA.CAL:
        c = cells[k]; life = int(c['cycle_life_pub']); S = c['summary']; qd = S['QDischarge']; ok = (qd > 0.5) & (qd < 1.2)
        n = np.arange(10, life + 1, 10); y = np.array([np.nanmedian(np.where(ok, qd, np.nan)[max(0, m - 3):m + 2]) for m in n])
        ct = S['chargetime'][1:51]; ct = float(np.nanmedian(ct[(ct > 5) & (ct < 30)])); crate = 48.0 / ct
        P0 = CA.params(np.array(CA_FITS[k]['theta']), 30.0); P0['Q_Li'] = P0['Q_Li'] * SB.Q_LI_TRIM
        o = CS.simulate(P0, dt=3.0); q0, _ = AS.window(P0['Q_p'], P0['Q_n'], P0['Q_Li'], P0['sig_p'], P0['sig_n']); kappa = float(o['Q_total'][0] / q0[0])
        def resid(Z):
            Z = np.atleast_2d(Z); B = len(Z); PB = {a: np.repeat(v, B) for a, v in P0.items()}
            tr = AS.trajectory(PB, rates_from(Z, crate), n_max=max(int(n[-1]), 100) + 10, kappa=np.full(B, kappa)); qs = tr['q_total'][:, (n // 10)]
            early = np.repeat((tr['lli'][:, 10] - tr['lli'][:, 1])[:, None] / EARLY_LLI_SD, 5, 1)
            return np.concatenate([(qs - y[None]) * 1000, early], 1)
        rng = np.random.default_rng(0); Z = LO + (HI - LO) * rng.random((4000, len(KN))); best = []
        for i in range(0, len(Z), 500): r = resid(Z[i:i + 500]); best.append(np.sqrt(np.mean(r ** 2, 1)))
        sc = np.concatenate(best); starts = Z[np.argsort(sc)[:3]]
        def jac(z, h=0.01):
            ZZ = np.clip(np.repeat(z[None], len(z) + 1, 0) + np.vstack([np.zeros(len(z)), h * np.eye(len(z))]), LO, HI); R = resid(ZZ)
            return ((R[1:] - R[0]) / np.maximum(np.diag(ZZ[1:] - z[None]), 1e-9)[:, None]).T
        sols = [least_squares(lambda z: resid(z)[0], s, jac=jac, bounds=(LO, HI), max_nfev=60) for s in starts]; sol = min(sols, key=lambda s: s.cost)
        PB = {a: v for a, v in P0.items()}; tr = AS.trajectory(PB, rates_from(sol.x, crate), n_max=3000, kappa=np.array([kappa]))
        fits[k] = dict(z=sol.x.tolist(), c=crate, life=life, life_fit=float(tr['life'][0]), rms_mAh=float(np.sqrt(np.mean(sol.fun[:-5] ** 2))), early_lli=float(tr['lli'][0, 10] - tr['lli'][0, 1]), kappa=kappa)
        print(k, 'life %d fitted %s  rms %.1f mAh' % (life, fits[k]['life_fit'], fits[k]['rms_mAh']), {a: round(float(b), 4) for a, b in zip(KN, sol.x)}, flush=True)
    json.dump(fits, open(_P.WORK + '/aging_calibration.json', 'w'), indent=1)

CA_FITS = json.load(open(_P.WORK + '/calibration.json'))
if __name__ == '__main__':
    main()
