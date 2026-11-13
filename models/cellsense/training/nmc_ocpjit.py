"""CellSense NMC811: measure how much the open-circuit curves vary from test to test, on the three DEV cells only.

nmc_ocpfit.py fitted ONE correction per electrode shared by all 26 dev tests. Here every dev test is fitted on its own,
with that shared correction scaled by a per-test factor s and its own spline wiggle on each electrode (24 coefficients
each, the same smoothness and size penalties as the shared fit), plus its own electrode window and resistance. The spread
of s and of the wiggle coefficients over the 26 tests is the jitter the simulator should draw (physics version 'ocpd').

The scale's mean and spread are taken from these per-test fits. The wiggle coefficients are not: a single full-cell
curve cannot say which electrode a wiggle belongs to, and wiggles trade against the test's own window, so their size
overstates the change. The wiggle size is instead set by matching what is left after fitting only a window (5 numbers)
with the nominal curves: real dev tests vs simulated jittered curves along the same 26 trajectories.

    python nmc_ocpjit.py      -> work/nmc/ocp_jitter.json (copied to training/nmc_ocp_jitter.json)
"""
import _paths as _P
import os, sys, json, numpy as np
sys.path.insert(0, _P.SRC); import nmcsim as NS, nmc_data as D, nmc_ocpfit as OF
from scipy.optimize import least_squares
OUT = os.path.join(_P.WORK, 'nmc'); K = NS.N_SPL; I = D.I_TEST


def fit_test(c, ap0, an0, w0):
    """one test: z = [s, wiggle_p (K), wiggle_n (K), x0, y0, log Q_p, log Q_n, log R]"""
    def res(z):
        s, wp, wn, w = z[0], z[1:1 + K], z[1 + K:1 + 2 * K], z[1 + 2 * K:]
        (v, ocv0), = OF.model_v(s * ap0 + wp, s * an0 + wn, w[None], [c])
        d2 = lambda a: np.diff(a, 2)
        return np.concatenate([(v[:-1] - c['v']) * 1000, [(v[-1] - 2.5) * 1000 / 3.0, (ocv0 - c['v0'] - 0.003) * 1000 / 3.0],
                               OF.LAM_SMOOTH * 1000 * d2(wp), OF.LAM_SMOOTH * 1000 * d2(wn), OF.LAM_SIZE * 1000 * wp, OF.LAM_SIZE * 1000 * wn])
    z0 = np.concatenate([[1.0], np.zeros(2 * K), w0])
    lo = np.concatenate([[-1.0], np.full(2 * K, -0.3), [0.05, 0.6, np.log(5.0), np.log(4.5), np.log(0.015)]])
    hi = np.concatenate([[3.0], np.full(2 * K, 0.3), [0.5, 0.999, np.log(11.0), np.log(9.0), np.log(0.06)]])
    sol = least_squares(res, np.clip(z0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), x_scale='jac', max_nfev=300)
    r = res(sol.x); n = len(c['qg']); return sol.x, float(np.sqrt(np.mean(r[:n] ** 2)))


if __name__ == '__main__':
    corr = json.load(open(os.path.join(OUT, 'ocp_corr.json'))); ap0, an0 = np.array(corr['a_p']), np.array(corr['a_n'])
    rows = D.readings(D.DEV); C = OF.curves(rows); W = {(w['cell'], w['diag']): w for w in corr['windows']}
    out = []
    for r, c in zip(rows, C):
        w = W[(r['cell'], r['diag'])]; w0 = np.array([w['x0'], w['y0'], np.log(w['Q_p']), np.log(w['Q_n']), np.log(min(max(w['R'], 0.0151), 0.0599))])
        (v_sh, _), = OF.model_v(ap0, an0, w0[None], [c]); rms_shared = float(np.sqrt(np.mean((v_sh[:-1] - c['v']) ** 2)) * 1000)
        z, rms = fit_test(c, ap0, an0, w0); out.append(dict(cell=r['cell'], diag=r['diag'], s=float(z[0]), wp=z[1:1 + K].tolist(), wn=z[1 + K:1 + 2 * K].tolist(),
                                                              rms_shared_mV=rms_shared, rms_mV=rms))
        print(r['cell'], r['diag'], 's %.2f' % z[0], 'rms %.1f -> %.1f mV' % (rms_shared, rms), flush=True)
    # coefficients whose cubic B-spline overlaps the used stoichiometry (support of basis j: knots j-3 .. j+1 on n-2 points)
    t = np.linspace(0, 1, K - 2); sup = lambda j: (t[max(j - 3, 0)], t[min(j, K - 3)])
    use = lambda rng: [j for j in range(K) if sup(j)[1] > rng[0] and sup(j)[0] < rng[1]]
    jp, jn = use(corr['used']['x']), use(corr['used']['y'])
    WP = np.array([o['wp'] for o in out])[:, jp]; WN = np.array([o['wn'] for o in out])[:, jn]; S = np.array([o['s'] for o in out])
    J = dict(scale_mu=float(S.mean()), scale_sd=float(S.std(ddof=1)), sd_p=float(np.sqrt(np.mean(WP ** 2))), sd_n=float(np.sqrt(np.mean(WN ** 2))),
             sd=float(np.sqrt(np.mean(np.concatenate([WP.ravel(), WN.ravel()]) ** 2))), n_tests=len(out),
             rms_shared_mV=float(np.median([o['rms_shared_mV'] for o in out])), rms_pertest_mV=float(np.median([o['rms_mV'] for o in out])), tests=out)
    Wt = [np.array([W[(r['cell'], r['diag'])][k] for k in ('x0', 'y0')] + [np.log(W[(r['cell'], r['diag'])][k]) for k in ('Q_p', 'Q_n', 'R')]) for r in rows]
    # The coefficients above trade against each test's window, so they overstate the change. The jitter size is set instead
    # by matching what a window fit leaves: refit every real dev test's window alone (5 numbers) with the nominal curves
    # (mu x the shared correction), and do the same to simulated jittered curves along the same 26 trajectories, for a
    # range of wiggle sizes. sigma is the size whose median leftover matches the real one.
    mu = J['scale_mu']; z0 = 0 * ap0

    def fit_window(c, ap, an, w0):
        f = lambda w: (OF.model_v(ap, an, w[None], [c])[0][0][:-1] - c['v']) * 1000
        lo = [0.05, 0.6, np.log(5.0), np.log(4.5), np.log(0.015)]; hi = [0.5, 0.999, np.log(11.0), np.log(9.0), np.log(0.06)]
        sol = least_squares(f, np.clip(w0, np.array(lo) + 1e-9, np.array(hi) - 1e-9), bounds=(lo, hi), x_scale='jac', max_nfev=200)
        return float(np.sqrt(np.mean(f(sol.x) ** 2)))
    real = np.array([fit_window(c, mu * ap0, mu * an0, w) for w, c in zip(Wt, C)]); J['window_fit_rms_mV_real'] = real.round(2).tolist()
    rng = np.random.default_rng(0); scan = {}
    for sig in (0.002, 0.004, 0.006, 0.008, 0.010, 0.012):
        rr = []
        for w, c in zip(Wt, C):
            for _ in range(3):
                sc = rng.normal(mu, J['scale_sd']); ap, an = sc * ap0 + rng.normal(0, sig, K), sc * an0 + rng.normal(0, sig, K)
                (v, ocv0), = OF.model_v(ap, an, w[None], [c]); cs = dict(c, v=v[:-1] + rng.normal(0, 2e-4, len(c['qg'])))
                rr.append(fit_window(cs, mu * ap0, mu * an0, w))
        scan[sig] = float(np.median(rr)); print('wiggle %.0f mV per coefficient -> median leftover %.1f mV (real %.1f)' % (1000 * sig, scan[sig], np.median(real)), flush=True)
    sg = np.array(sorted(scan)); lv = np.array([scan[k] for k in sg]); J['scan'] = {f'{1000 * k:.0f}': v for k, v in scan.items()}
    J['sigma'] = float(np.interp(np.median(real), lv, sg)); J['window_fit_rms_mV_real_median'] = float(np.median(real))
    print('real dev tests, window fit with the nominal curves: median %.1f mV (%.1f-%.1f); matching wiggle sigma %.1f mV per coefficient' %
          (np.median(real), real.min(), real.max(), 1000 * J['sigma']))
    json.dump(J, open(os.path.join(OUT, 'ocp_jitter.json'), 'w'), indent=1)
    print('scale %.2f +- %.2f; wiggle rms per coefficient: positive %.1f mV, negative %.1f mV, both %.1f mV; per-test rms %.1f -> %.1f mV (median)' %
          (J['scale_mu'], J['scale_sd'], 1000 * J['sd_p'], 1000 * J['sd_n'], 1000 * J['sd'], J['rms_shared_mV'], J['rms_pertest_mV']))
