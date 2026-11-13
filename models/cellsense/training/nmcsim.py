"""CellSense NMC811 physics: a vectorized single-particle model of the LG INR21700-M50T (NMC811 / graphite-SiOx), the cell
Chen et al. 2020 (J. Electrochem. Soc. 167, 080534) measured for PyBaMM's Chen2020 parameter set. Thousands of cells at once
in numpy.

Same open-circuit curves as PyBaMM's Chen2020 set (both measured on electrode material harvested from this cell), Fickian
diffusion in one spherical particle per electrode (cellsim.Sphere: finite volumes solved exactly between steps), Butler-
Volmer kinetics with the surface-dependent exchange current, an ohmic resistance and an electrolyte polarization with a
first-order lag. Isothermal (a C/20 test heats a 21700 cell by well under 0.1 K). Checked against PyBaMM's SPM and SPMe in
nmc_check.py.

State variables are in capacity units, as in cellsim: an electrode holds Q_p or Q_n amp-hours between stoichiometry 0 and
1, and the cell's cyclable lithium is Q_Li = x Q_p + y Q_n (x: NMC lithiation, y: graphite-SiOx lithiation).

Aging enters as Dubarry's degradation modes: lithium lost (LLI), negative electrode lost (LAM_NE, part of it lithiated),
positive electrode lost (LAM_PE), and resistance growth."""
import numpy as np
import cellsim as CS

F, Rg = CS.F, CS.Rg


def U_p(x):   # NMC811, Chen 2020 (PyBaMM nmc_LGM50_ocp_Chen2020)
    return (-0.8090 * x + 4.4875 - 0.0428 * np.tanh(18.5138 * (x - 0.5542)) - 17.7326 * np.tanh(15.7890 * (x - 0.3117))
            + 17.5842 * np.tanh(15.9308 * (x - 0.3120)))


def U_n(y):   # graphite-SiOx, Chen 2020 (PyBaMM graphite_LGM50_ocp_Chen2020)
    return (1.9793 * np.exp(-39.3631 * y) + 0.2482 - 0.0909 * np.tanh(29.8538 * (y - 0.1234)) - 0.04478 * np.tanh(14.9159 * (y - 0.2769))
            - 0.0205 * np.tanh(30.4444 * (y - 0.6103)))


def full_state(Q_p, Q_n, Q_Li, V_full, TP, TN):
    """stoichiometries at rest at open-circuit voltage V_full, conserving lithium (bisection on x; less x -> higher voltage)"""
    lo = np.full_like(Q_p, 1e-6); hi = np.minimum(1 - 1e-6, Q_Li / Q_p)
    for _ in range(50):
        x = 0.5 * (lo + hi); y = np.clip((Q_Li - x * Q_p) / Q_n, 1e-6, 1 - 1e-6); ocv = TP(x) - TN(y)
        hi = np.where(ocv < V_full, x, hi); lo = np.where(ocv < V_full, lo, x)
    x = 0.5 * (lo + hi); return x, np.clip((Q_Li - x * Q_p) / Q_n, 1e-6, 1 - 1e-6)


def spline_basis(s, n=None):
    """cubic B-spline basis on stoichiometry [0, 1] with n functions (open uniform knots): the OCP correction's shape"""
    from scipy.interpolate import BSpline
    n = n or N_SPL; t = np.concatenate([[0.0] * 3, np.linspace(0, 1, n - 2), [1.0] * 3])
    return BSpline.design_matrix(np.clip(np.ravel(s), 0, 1), t, 3).toarray().reshape(np.shape(s) + (n,))


N_SPL = 24


def ocp_funcs(corr=None):
    """the open-circuit curves, optionally with the additive corrections fitted on the dev cells (nmc_ocpfit.py):
    corr = {'a_p': [...], 'a_n': [...]} spline coefficients in volts"""
    if not corr: return U_p, U_n
    ap, an = np.asarray(corr['a_p'], float), np.asarray(corr['a_n'], float)
    return (lambda x: U_p(x) + spline_basis(x, len(ap)) @ ap), (lambda y: U_n(y) + spline_basis(y, len(an)) @ an)


SIG_LEVELS = np.concatenate([[0.0], np.exp(np.linspace(np.log(1e-5), np.log(0.1), 240))])
_BANKS = {}


class FastTable(CS.Table):
    """cellsim.Table, but the smeared curve for each cell is interpolated between curves precomputed once at 241 smearing
    widths (0 to 0.1 in stoichiometry), instead of being recomputed for every cell: within 0.02 mV on the positive and 1.7 mV
    on the negative (1.2 mV away from the nearly empty end, y > 0.01), 10-40x faster"""
    def __init__(self, U, key, sig, B, n=2001):
        if key not in _BANKS:
            g = np.linspace(0, 1, n); _BANKS[key] = np.stack([CS.smooth(U, g, s) if s > 0 else U(np.clip(g, 1e-6, 1 - 1e-6)) for s in SIG_LEVELS])
        Tb = _BANKS[key]; self.n = n; self.rows = np.arange(B)
        s = np.zeros(B) if sig is None else np.clip(np.broadcast_to(np.asarray(sig, float), (B,)), 0, SIG_LEVELS[-1])
        i = np.clip(np.searchsorted(SIG_LEVELS, s) - 1, 0, len(SIG_LEVELS) - 2); w = ((s - SIG_LEVELS[i]) / (SIG_LEVELS[i + 1] - SIG_LEVELS[i]))[:, None]
        self.T = Tb[i] * (1 - w) + Tb[i + 1] * w


def tables(P, B, corr=None):
    """per-cell OCP lookup tables. With P['ocp_scale'] (B,) and P['dU_p'], P['dU_n'] (B, N_SPL) spline coefficients (V), each
    cell gets its own curve: Chen2020 + ocp_scale x the dev-fitted correction + its own small spline wiggle (OCP jitter)"""
    if 'ocp_scale' not in P:
        up, un = ocp_funcs(corr); key = None if not corr else (tuple(corr['a_p']), tuple(corr['a_n']))
        return FastTable(up, ('p', key), P.get('sig_p'), B), FastTable(un, ('n', key), P.get('sig_n'), B)
    TP, TN = FastTable(U_p, ('p', None), P.get('sig_p'), B), FastTable(U_n, ('n', None), P.get('sig_n'), B)
    g = np.linspace(0, 1, TP.n); Bs = spline_basis(g, N_SPL); s = np.asarray(P['ocp_scale'], float)[:, None]
    cp = Bs @ np.asarray(corr['a_p'], float) if corr else 0 * g; cn = Bs @ np.asarray(corr['a_n'], float) if corr else 0 * g
    TP.T = TP.T + s * cp[None] + P['dU_p'] @ Bs.T; TN.T = TN.T + s * cn[None] + P['dU_n'] @ Bs.T
    return TP, TN


def simulate(P, I=0.24, dt=30.0, V_min=2.5, N=10, T=23.0, t_max=None, ocp=None, corr=None):
    """Constant-current discharge at I (A) from rest at open-circuit voltage V_full down to V_min.
    P: dict of (B,) arrays: Q_p, Q_n, Q_Li (Ah); V_full (V); tau_p, tau_n (R^2/D, s, at 25 C); I0_p, I0_n (A, exchange current
    at half stoichiometry, 25 C); R_ohm (ohm); R_e (ohm), tau_e (s); optional sig_p, sig_n (OCP smearing) and ocp_scale,
    dU_p, dU_n (per-cell OCP jitter, see tables). corr: the dev-fitted OCP correction (nmc_ocpfit.py). Returns t, V, Q
    (B, steps; NaN when done) and Q_total (B,): the capacity delivered to V_min."""
    B = len(P['Q_p']); sp, sn = CS.Sphere(N), CS.Sphere(N); TP, TN = ocp if ocp is not None else tables(P, B, corr)
    x0, y0 = full_state(P['Q_p'], P['Q_n'], P['Q_Li'], P['V_full'], TP, TN)
    cp, cn = sp.init(x0), sn.init(y0); TK = T + 273.15
    arr = lambda Ea: np.exp(Ea / Rg * (1 / 298.15 - 1 / TK))
    tp, tn = P['tau_p'] / arr(P.get('Ea_dp', 0.0)), P['tau_n'] / arr(P.get('Ea_dn', 0.0))
    i0p, i0n = P['I0_p'] * arr(P.get('Ea_kp', 17800.0)), P['I0_n'] * arr(P.get('Ea_kn', 35000.0)); c2 = 2 * Rg * TK / F
    steps = int(np.ceil((t_max or 3600.0 * 1.15 * np.max(np.minimum(P['Q_Li'], P['Q_p'])) / I) / dt)) + 1
    out = {k: np.full((B, steps), np.nan) for k in ('t', 'V', 'Q')}
    alive = np.ones(B, bool); Q = np.zeros(B); ve = np.zeros(B); Qend = np.full(B, np.nan); V_prev = np.full(B, 9.9)
    Iv = np.full(B, float(I)) if np.isscalar(I) else np.asarray(I, float)
    for k in range(steps):
        xs, ys = np.clip(sp.surface(cp), 1e-6, 1 - 1e-6), np.clip(sn.surface(cn), 1e-6, 1 - 1e-6)
        eta = c2 * (np.arcsinh(Iv / (4 * i0p * np.sqrt(xs * (1 - xs)))) + np.arcsinh(Iv / (4 * i0n * np.sqrt(ys * (1 - ys)))))
        V = TP(xs) - TN(ys) - eta - Iv * P['R_ohm'] - ve
        newly = alive & (V <= V_min) & (k > 0)
        f = np.clip((V_prev - V_min) / np.maximum(V_prev - V, 1e-9), 0, 1)
        Qend = np.where(newly, Q - (1 - f) * Iv * dt / 3600, Qend); alive &= ~newly; V_prev = V
        out['t'][alive, k] = k * dt; out['V'][alive, k] = V[alive]; out['Q'][alive, k] = Q[alive]
        if not alive.any(): break
        Ik = np.where(alive, Iv, 0.0); Q = Q + Ik * dt / 3600
        cp = sp.step(cp, dt / tp, Ik / 3600 / P['Q_p'] * tp); cn = sn.step(cn, dt / tn, -Ik / 3600 / P['Q_n'] * tn)
        ve = ve + (Ik * P['R_e'] - ve) * (1 - np.exp(-dt / P['tau_e']))
    out['Q_total'] = np.where(np.isfinite(Qend), Qend, Q); out['x0'], out['y0'] = x0, y0
    return out


def q_of_v(out, vgrid):
    """capacity discharged at each voltage of a descending grid; NaN below the end of the curve"""
    B = out['V'].shape[0]; Qg = np.full((B, len(vgrid)), np.nan)
    for b in range(B):
        v, q = out['V'][b], out['Q'][b]; m = np.isfinite(v)
        if m.sum() < 3: continue
        v, q = np.minimum.accumulate(v[m]), q[m]; v = np.append(v, 2.5 - 1e-6); q = np.append(q, out['Q_total'][b])
        ok = (vgrid <= v[0]) & (vgrid >= v[-1]); Qg[b, ok] = np.interp(-vgrid[ok], -v, q); Qg[b, vgrid > v[0]] = 0.0
    return Qg


def ocv_window(Q_p, Q_n, Q_Li, V_hi, V_lo, TP, TN):
    """equilibrium (open-circuit) capacity between V_hi and V_lo"""
    xf, yf = full_state(Q_p, Q_n, Q_Li, V_hi, TP, TN); lo = np.maximum(xf, 1e-6); hi = np.full_like(Q_p, 1 - 1e-6)
    for _ in range(50):
        x = 0.5 * (lo + hi); y = np.clip((Q_Li - x * Q_p) / Q_n, 1e-7, 1 - 1e-6); ocv = TP(x) - TN(y)
        lo = np.where(ocv > V_lo, x, lo); hi = np.where(ocv > V_lo, hi, x)
    return (0.5 * (lo + hi) - xf) * Q_p


def chen2020():
    """lumped parameters of the fresh Chen2020 cell (PyBaMM's values; see nmc_check.py for the derivation)"""
    A = 0.065 * 1.58; Fh = F / 3600
    Qn = 33133 * 0.75 * 85.2e-6 * A * Fh; Qp = 63104 * 0.665 * 75.6e-6 * A * Fh
    I0n = 6.48e-7 * 1000 ** 0.5 * 33133 * 0.5 * (3 * 0.75 / 5.86e-6) * 85.2e-6 * A
    I0p = 3.42e-6 * 1000 ** 0.5 * 63104 * 0.5 * (3 * 0.665 / 5.22e-6) * 75.6e-6 * A
    return dict(Q_p=Qp, Q_n=Qn, Q_Li=29866 / 33133 * Qn + 17038 / 63104 * Qp, tau_n=5.86e-6 ** 2 / 3.3e-14, tau_p=5.22e-6 ** 2 / 4e-15, I0_n=I0n, I0_p=I0p)
