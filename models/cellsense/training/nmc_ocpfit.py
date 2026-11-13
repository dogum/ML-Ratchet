"""CellSense NMC811: measure and fix the open-circuit-curve mismatch, on the three DEV cells only.

The Chen2020 curves were measured on material harvested from one M50T cell, by GITT on half cells. Fitted to the dev
cells' C/20 discharges they leave 11-20 mV rms with a clear, repeatable shape. This script fits ONE additive correction
per electrode (a cubic B-spline in stoichiometry, 24 coefficients each), shared by every dev test, jointly with each test's
own electrode window (start stoichiometries x0, y0, capacities Q_p, Q_n and a resistance R). Lithium-loss and
electrode-loss shift each test's window differently, which is what lets the two electrodes' corrections be told apart.
The model inside the fit is the equilibrium curve minus I*R (at C/20 the solid-diffusion drop is a few mV and nearly
constant, so R absorbs it); the result is then checked with the full simulator.

    python nmc_ocpfit.py            -> work/nmc/ocp_corr.json, work/nmc/fig/ocp_corr.png
"""
import _paths as _P
import os, sys, json, numpy as np
sys.path.insert(0, _P.SRC); import nmcsim as NS, nmc_data as D
from scipy.optimize import least_squares
OUT = os.path.join(_P.WORK, 'nmc'); K = NS.N_SPL; I = D.I_TEST
LAM_SMOOTH, LAM_SIZE = 3.0, 0.05            # penalties per coefficient (second difference, size), in units of 1 mV residuals


def curves(rows, n=140):
    C = []
    for r in rows:
        qg = np.linspace(0.03, r['q_total'] - 0.02, n); C.append(dict(qg=qg, v=np.interp(qg, r['Q'], r['V']), q_end=r['q_total'], v0=r['V_start']))
    return C


def unpack(z, nc):
    ap, an = z[:K], z[K:2 * K]; w = z[2 * K:].reshape(nc, 5); return ap, an, w


def model_v(ap, an, w, C):
    out = []
    for c, (x0, y0, lqp, lqn, lr) in zip(C, w):
        q = np.append(c['qg'], c['q_end']); x = np.clip(x0 + q / np.exp(lqp), 1e-4, 1 - 1e-4); y = np.clip(y0 - q / np.exp(lqn), 1e-4, 1 - 1e-4)
        v = NS.U_p(x) + NS.spline_basis(x, K) @ ap - NS.U_n(y) - NS.spline_basis(y, K) @ an - I * np.exp(lr)
        ocv0 = NS.U_p(x0) + NS.spline_basis(np.array([x0]), K)[0] @ ap - NS.U_n(y0) - NS.spline_basis(np.array([y0]), K)[0] @ an
        out.append((v, ocv0))
    return out


def residuals(z, C, corr=True):
    ap, an, w = unpack(z, len(C))
    if not corr: ap, an = 0 * ap, 0 * an
    R = []
    for c, (v, ocv0) in zip(C, model_v(ap, an, w, C)):
        R += [(v[:-1] - c['v']) * 1000 / 1.0, [(v[-1] - 2.5) * 1000 / 3.0, (ocv0 - c['v0'] - 0.003) * 1000 / 3.0]]   # rest after a 50 mA CV sits ~3 mV above OCV
    d2 = lambda a: np.diff(a, 2)
    R += [LAM_SMOOTH * 1000 * d2(ap), LAM_SMOOTH * 1000 * d2(an), LAM_SIZE * 1000 * ap, LAM_SIZE * 1000 * an]
    return np.concatenate(R)


def fit(rows, corr=True, z0=None):
    C = curves(rows); nc = len(C); ch = NS.chen2020()
    if z0 is None:
        w0 = np.tile([0.26, 0.72, np.log(7.4), np.log(6.9), np.log(0.03)], (nc, 1)); z0 = np.concatenate([np.zeros(2 * K), w0.ravel()])
    lo = np.concatenate([np.full(2 * K, -0.3), np.tile([0.05, 0.6, np.log(5.0), np.log(4.5), np.log(0.015)], nc)])
    hi = np.concatenate([np.full(2 * K, 0.3), np.tile([0.5, 0.999, np.log(11.0), np.log(9.0), np.log(0.06)], nc)])
    from scipy.sparse import lil_matrix
    n_r = len(residuals(z0, C)); S = lil_matrix((n_r, len(z0)), dtype=int); i = 0
    for k, c in enumerate(C):
        m = len(c['qg']) + 2; S[i:i + m, :2 * K] = 1; S[i:i + m, 2 * K + 5 * k: 2 * K + 5 * k + 5] = 1; i += m
    S[i:, :2 * K] = 1
    f = lambda z: residuals(z, C, corr)
    sol = least_squares(f, np.clip(z0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), jac_sparsity=S, x_scale='jac', max_nfev=400, verbose=0)
    ap, an, w = unpack(sol.x, nc); r = f(sol.x); n_c = [len(c['qg']) for c in C]; rms = []; i = 0
    for n in n_c: rms.append(float(np.sqrt(np.mean(r[i:i + n] ** 2)))); i += n + 2
    return sol.x, ap, an, w, rms, C


if __name__ == '__main__':
    rows = D.readings(D.DEV)
    z_b, _, _, w_b, rms_b, C = fit(rows, corr=False)
    print('no correction: per-test rms (mV) median %.1f, max %.1f' % (np.median(rms_b), np.max(rms_b)), flush=True)
    z, ap, an, w, rms, C = fit(rows, corr=True, z0=z_b)
    print('with correction: per-test rms (mV) median %.1f, max %.1f' % (np.median(rms), np.max(rms)), flush=True)
    xs = np.linspace(0, 1, 201); used = dict(x=[float(w[:, 0].min()), float((w[:, 0] + np.array([c['q_end'] for c in C]) / np.exp(w[:, 2])).max())],
                                              y=[float((w[:, 1] - np.array([c['q_end'] for c in C]) / np.exp(w[:, 3])).min()), float(w[:, 1].max())])
    json.dump(dict(a_p=ap.tolist(), a_n=an.tolist(), K=K, lam_smooth=LAM_SMOOTH, lam_size=LAM_SIZE, rms_before=rms_b, rms_after=rms, used=used,
                   windows=[dict(cell=r['cell'], diag=r['diag'], x0=float(a), y0=float(b), Q_p=float(np.exp(c)), Q_n=float(np.exp(d)), R=float(np.exp(e))) for r, (a, b, c, d, e) in zip(rows, w)]),
              open(os.path.join(OUT, 'ocp_corr.json'), 'w'), indent=1)
    print('stoichiometry used: x %.3f-%.3f, y %.3f-%.3f' % (*used['x'], *used['y']))
    for r, ww, a, b in zip(rows, w, rms_b, rms):
        print(r['cell'], r['diag'], 'x0 %.3f y0 %.3f Qp %.3f Qn %.3f R %.1f mOhm' % (ww[0], ww[1], np.exp(ww[2]), np.exp(ww[3]), 1000 * np.exp(ww[4])), 'rms %.1f -> %.1f mV' % (a, b))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(xs, 1000 * NS.spline_basis(xs, K) @ ap); ax[0].axvspan(*used['x'], alpha=0.15); ax[0].set_title('NMC811 OCP correction (mV) vs x')
    ax[1].plot(xs, 1000 * NS.spline_basis(xs, K) @ an); ax[1].axvspan(*used['y'], alpha=0.15); ax[1].set_title('graphite-SiOx OCP correction (mV) vs y')
    mv = model_v(ap, an, w, C); mb = model_v(0 * ap, 0 * an, w_b, C)
    for k in (0, len(C) // 2, len(C) - 1):
        ax[2].plot(C[k]['qg'], 1000 * (mb[k][0][:-1] - C[k]['v']), '--', lw=1); ax[2].plot(C[k]['qg'], 1000 * (mv[k][0][:-1] - C[k]['v']), lw=1, label=f"{rows[k]['cell']} {rows[k]['diag']}")
    ax[2].set_title('residual (mV): dashed Chen2020, solid corrected'); ax[2].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig', 'ocp_corr.png'), dpi=70)
