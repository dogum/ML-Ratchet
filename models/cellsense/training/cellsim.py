"""CellSense physics: a vectorized single-particle model with electrolyte polarization and a lumped thermal model, for an
LFP / graphite cell discharging at constant current. Thousands of cells at once in numpy.

Same open-circuit curves as PyBaMM's Prada2013 set (LFP: Afshar 2017; graphite: Chen 2020), Fickian diffusion in a
spherical particle per electrode (finite volumes), Butler-Volmer kinetics, an electrolyte that polarizes with a first-order
lag (the SPMe's quasi-steady drop), ohmic resistance, and Arrhenius temperature dependence. Checked against PyBaMM's SPM
and SPMe in test_cellsim.py.

State variables are in capacity units: an electrode holds Q_p or Q_n amp-hours between stoichiometry 0 and 1, and the
cell's cyclable lithium is Q_Li = x Q_p + y Q_n (x: LFP lithiation, y: graphite lithiation)."""
import numpy as np
F, Rg = 96485.33212, 8.314462618

def U_p(x):   # LFP, Afshar 2017 (as PyBaMM Prada2013)
    return 3.4077 - 0.020269 * x + 0.5 * np.exp(-150 * x) - 0.9 * np.exp(-30 * (1 - x))

def U_n(y):   # graphite, Chen 2020 (as PyBaMM Prada2013)
    return 1.9793 * np.exp(-39.3631 * y) + 0.2482 - 0.0909 * np.tanh(29.8538 * (y - 0.1234)) - 0.04478 * np.tanh(14.9159 * (y - 0.2769)) - 0.0205 * np.tanh(30.4444 * (y - 0.6103))

GH_Z, GH_W = np.polynomial.hermite_e.hermegauss(7); GH_W = GH_W / GH_W.sum()

def smooth(U, s, sig):
    """open-circuit potential averaged over a spread of local states of charge (sig, in stoichiometry): real electrodes
    hold particles of many sizes, which blurs the sharp staging steps of a single-particle curve"""
    if sig is None: return U(s)
    sig = np.asarray(sig, float)
    if sig.ndim and sig.ndim < np.ndim(s): sig = sig.reshape(sig.shape + (1,) * (np.ndim(s) - sig.ndim))
    return sum(w * U(np.clip(s + sig * z, 1e-6, 1 - 1e-6)) for z, w in zip(GH_Z, GH_W))

class Table:
    """a cell-by-cell lookup table of a (smoothed) open-circuit curve on a uniform stoichiometry grid: fast linear interpolation"""
    def __init__(self, U, sig, B, n=2001):
        self.n = n; g = np.linspace(0, 1, n)
        if sig is None: self.T = np.broadcast_to(U(np.clip(g, 1e-6, 1 - 1e-6)), (B, n)).copy()
        else: self.T = smooth(U, np.broadcast_to(g, (B, n)), np.asarray(sig, float)[:, None])
        self.rows = np.arange(B)
    def __call__(self, s):
        f = np.clip(s, 0, 1) * (self.n - 1); i = np.minimum(f.astype(int), self.n - 2); w = f - i
        return self.T[self.rows, i] * (1 - w) + self.T[self.rows, i + 1] * w

def full_state(Q_p, Q_n, Q_Li, V_full, sig_p=None, sig_n=None):
    """stoichiometries at rest after charging to V_full (open circuit), conserving lithium: bisection on x"""
    lo = np.full_like(Q_p, 1e-6); hi = np.minimum(1 - 1e-6, Q_Li / Q_p)
    for _ in range(60):
        x = 0.5 * (lo + hi); y = np.clip((Q_Li - x * Q_p) / Q_n, 1e-6, 1 - 1e-6)
        ocv = smooth(U_p, x, sig_p) - smooth(U_n, y, sig_n); hi = np.where(ocv < V_full, x, hi); lo = np.where(ocv < V_full, lo, x)   # less x -> higher LFP potential, more Li in graphite
    x = 0.5 * (lo + hi); return x, np.clip((Q_Li - x * Q_p) / Q_n, 1e-6, 1 - 1e-6)

class Sphere:
    """finite-volume diffusion in N equal-thickness shells of a unit sphere (c: stoichiometry; flux at r = 1 set by the
    current), solved exactly between time steps in the operator's eigenbasis, so any step size is stable and accurate.
    State is kept in modal coordinates a, with c = Phi a."""
    def __init__(self, N=10):
        from scipy.linalg import eigh
        self.N = N; r = np.linspace(0, 1, N + 1); self.rc = 0.5 * (r[1:] + r[:-1]); vol = (r[1:] ** 3 - r[:-1] ** 3) / 3; A = r[1:-1] ** 2; dr = 1.0 / N
        K = np.zeros((N, N))
        for i in range(N - 1):
            g = A[i] / dr; K[i, i] -= g; K[i + 1, i + 1] -= g; K[i, i + 1] += g; K[i + 1, i] += g
        lam, Phi = eigh(K, np.diag(vol)); self.lam = np.minimum(lam, 0.0); self.lam[np.argmax(lam)] = 0.0; self.Phi = Phi; self.W = vol
        self.b = Phi[-1, :] / 3.0                                          # boundary injection (mean c changes by `rate`)
        w = (1 - self.rc[-1]) / (self.rc[-1] - self.rc[-2]); self.s = Phi[-1, :] * (1 + w) - Phi[-2, :] * w
    def init(self, c0):   # uniform stoichiometry c0 (B,)
        return (self.Phi.T @ (self.W * np.ones(self.N)))[None, :] * c0[:, None]
    def step(self, a, dt_over_tau, rate):
        h = np.asarray(dt_over_tau, float)[:, None]; E = np.exp(self.lam[None] * h)
        G = np.where(self.lam[None] < 0, (E - 1) / np.where(self.lam[None] < 0, self.lam[None], 1.0), h)
        return E * a + G * self.b[None] * np.asarray(rate, float)[:, None]
    def surface(self, a):   # linear extrapolation to r = 1 from the outer two shells (as PyBaMM's finite volumes)
        return a @ self.s
    def mean(self, a):
        return 3 * (a @ (self.Phi.T @ self.W) if False else (a @ self.Phi.T) @ self.W)

def overpotential(I, xs, ys, P, TK):
    arr = lambda Ea: np.exp(Ea / Rg * (1 / 298.15 - 1 / TK))
    i0p = P['I0_p'] * arr(P['Ea_kp']) * 2 * np.sqrt(xs * (1 - xs)); i0n = P['I0_n'] * arr(P['Ea_kn']) * 2 * np.sqrt(ys * (1 - ys))
    return 2 * Rg * TK / F * (np.arcsinh(I / (2 * i0p)) + np.arcsinh(I / (2 * i0n)))

def simulate(P, I=4.4, dt=2.0, V_min=2.0, I_cut=0.022, cv=True, t_max=None, N=10, isothermal=False):
    """Constant-current discharge at I to V_min, then (cv=True) a hold at V_min until the current falls to I_cut, as in
    Severson et al. P: dict of (B,) arrays. Capacities in Ah; Q_Li cyclable lithium (Ah); V_full charge voltage; tau_p,
    tau_n diffusion times R^2/D (s, at 25 C); I0_p, I0_n exchange currents (A, at 25 C, at half stoichiometry); R_ohm (ohm);
    R_e and tau_e electrolyte polarization (ohm, s); Ea_* activation energies (J/mol); C_th heat capacity (J/K), hA (W/K),
    T_amb, T0 (C). Returns per-step arrays t, V, Q, T, I (B, steps), NaN when finished; 'cc' marks the constant-current
    part; and Q_cc, Q_total (B,)."""
    B = len(P['Q_p']); sp, sn = Sphere(N), Sphere(N)
    sgp, sgn = P.get('sig_p'), P.get('sig_n'); x0, y0 = full_state(P['Q_p'], P['Q_n'], P['Q_Li'], P['V_full'], sgp, sgn)
    cp = sp.init(x0); cn = sn.init(y0); TP, TN = Table(U_p, sgp, B), Table(U_n, sgn, B)
    # a slow population of graphite (fraction f_s of it, behind extra resistance R_s): poorly connected material that sits out
    # a fast discharge and gives up its lithium during the hold. f_s = 0 is the plain single-particle model.
    fs = np.asarray(P.get('f_s', np.zeros(B)), float); Rs = np.asarray(P.get('R_s', np.ones(B)), float); slow = fs > 1e-6; cs = sn.init(y0)
    Qnf = P['Q_n'] * (1 - fs); Qns = np.maximum(P['Q_n'] * fs, 1e-9); I0f = P['I0_n'] * (1 - fs); I0s = np.maximum(P['I0_n'] * fs, 1e-9)
    T = np.asarray(P.get('T0', P['T_amb']), float).copy(); ve = np.zeros(B); Q = np.zeros(B)
    steps = int(np.ceil((t_max or 3600.0 * 1.2 * np.max(np.minimum(P['Q_Li'], P['Q_p'])) / I + (2400 if cv else 0)) / dt)) + 1
    out = {k: np.full((B, steps), np.nan) for k in ('t', 'V', 'Q', 'T', 'I')}; out['cc'] = np.zeros((B, steps), bool)
    alive = np.ones(B, bool); in_cv = np.zeros(B, bool); Q_cc = np.full(B, np.nan); V_prev = np.full(B, 9.9); I_prev = np.full(B, float(I))
    arr = lambda Ea, TK: np.exp(Ea / Rg * (1 / 298.15 - 1 / TK))
    for k in range(steps):
        TK = T + 273.15
        R_e = P['R_e'] * arr(P['Ea_e'], TK); R_o = P['R_ohm'] * arr(P.get('Ea_o', np.zeros(B)), TK)
        tp, tn = P['tau_p'] / arr(P['Ea_dp'], TK), P['tau_n'] / arr(P['Ea_dn'], TK)
        akp, akn = arr(P['Ea_kp'], TK), arr(P['Ea_kn'], TK); c2 = 2 * Rg * TK / F
        def split(Ii, ys, yss):
            """current carried by the slow graphite, from both branches linearised about an even split"""
            i0f = I0f * akn * 2 * np.sqrt(ys * (1 - ys)); i0s = I0s * akn * 2 * np.sqrt(yss * (1 - yss))
            rf = c2 / np.sqrt(4 * i0f ** 2 + (Ii * (1 - fs)) ** 2); rs = Rs + c2 / (2 * i0s)
            Uf, Us = TN(ys), TN(yss); phi = (Ii + Uf / rf + Us / rs) / (1 / rf + 1 / rs)
            return np.where(slow, (phi - Us) / rs, 0.0), i0f
        def volt(Ii, xs, ys, yss):
            xs, ys, yss = np.clip(xs, 1e-6, 1 - 1e-6), np.clip(ys, 1e-6, 1 - 1e-6), np.clip(yss, 1e-6, 1 - 1e-6)
            Is, i0f = split(Ii, ys, yss); If = Ii - Is
            i0p = P['I0_p'] * akp * 2 * np.sqrt(xs * (1 - xs)); eta = c2 * (np.arcsinh(Ii / (2 * i0p)) + np.arcsinh(If / (2 * i0f)))
            o = TP(xs) - TN(ys); return o - eta - Ii * R_o - ve, o, Is
        Ik = np.full(B, float(I)); Vcc, ocv, Is_k = volt(Ik, sp.surface(cp), sn.surface(cn), sn.surface(cs))
        newly = alive & ~in_cv & (Vcc <= V_min)
        f = np.clip((V_prev - V_min) / np.maximum(V_prev - Vcc, 1e-9), 0, 1) if k else np.ones(B)          # where V crossed V_min within the last step
        Q_cc = np.where(newly, Q - (1 - f) * I * dt / 3600, Q_cc); V_prev = Vcc
        if cv: in_cv |= newly
        else: alive &= ~newly
        if in_cv.any():   # current that holds V_min: solved implicitly, with the surface concentrations at the end of the step
            x0s = sp.surface(sp.step(cp, dt / tp, 0 * tp)); xd = sp.surface(sp.step(cp, dt / tp, 1 / 3600 / P['Q_p'] * tp)) - x0s
            y0s = sn.surface(sn.step(cn, dt / tn, 0 * tn)); yd = sn.surface(sn.step(cn, dt / tn, -1 / 3600 / Qnf * tn)) - y0s
            ys_s = sn.surface(cs)                                                    # slow population: explicit (its time constant is long)
            lo, hi = np.zeros(B), np.minimum(float(I), 1.2 * I_prev + 0.01)          # the hold current only falls: bracket from the last one
            for _ in range(17):
                mid = 0.5 * (lo + hi); Is_m = split(mid, np.clip(y0s + mid * yd, 1e-6, 1 - 1e-6), np.clip(ys_s, 1e-6, 1 - 1e-6))[0]
                f = volt(mid, x0s + mid * xd, y0s + (mid - Is_m) * yd, ys_s)[0] - V_min
                lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
            Ik = np.where(in_cv, 0.5 * (lo + hi), Ik); alive &= ~(in_cv & (Ik < I_cut)); I_prev = np.where(in_cv, Ik, I_prev)
            Is_cv = split(Ik, np.clip(y0s + Ik * yd, 1e-6, 1 - 1e-6), np.clip(ys_s, 1e-6, 1 - 1e-6))[0]
            o_cv = volt(Ik, x0s + Ik * xd, y0s + (Ik - Is_cv) * yd, ys_s)
            ocv = np.where(in_cv, o_cv[1], ocv); Is_k = np.where(in_cv, o_cv[2], Is_k)
        V = np.where(in_cv, V_min, Vcc)
        out['t'][alive, k] = k * dt; out['V'][alive, k] = V[alive]; out['Q'][alive, k] = Q[alive]; out['T'][alive, k] = T[alive]; out['I'][alive, k] = Ik[alive]
        out['cc'][:, k] = alive & ~in_cv
        if not alive.any(): break
        Ik = np.where(alive, Ik, 0.0); Q = Q + Ik * dt / 3600
        Is_k = np.where(alive, Is_k, 0.0)
        cp = sp.step(cp, dt / tp, Ik / 3600 / P['Q_p'] * tp); cn = sn.step(cn, dt / tn, -(Ik - Is_k) / 3600 / Qnf * tn); cs = sn.step(cs, dt / tn, -Is_k / 3600 / Qns * tn)
        ve = ve + (Ik * R_e - ve) * (1 - np.exp(-dt / P['tau_e']))
        if not isothermal:
            heat = Ik * (ocv - V); T = T + dt * (heat - P['hA'] * (T - P['T_amb'])) / P['C_th']
    out['Q_cc'] = np.where(np.isfinite(Q_cc), Q_cc, Q); out['Q_total'] = Q
    return out

def pulse_resistance(P, soc=0.8, Ip=3.96, T=30.0):
    """the cell's 30 ms pulse resistance at a state of charge (as Severson's IR): ohmic plus charge transfer, no diffusion"""
    x0, y0 = full_state(P['Q_p'], P['Q_n'], P['Q_Li'], P['V_full'], P.get('sig_p'), P.get('sig_n'))
    usable = np.minimum(P['Q_Li'] - 1e-3 * P['Q_n'] - x0 * P['Q_p'], (1 - x0) * P['Q_p'])     # rough: lithium that can move before either electrode runs out
    dq = (1 - soc) * usable; x = np.clip(x0 + dq / P['Q_p'], 1e-4, 1 - 1e-4); y = np.clip(y0 - dq / P['Q_n'], 1e-4, 1 - 1e-4); TK = T + 273.15
    R_o = P['R_ohm'] * np.exp(P.get('Ea_o', 0 * P['R_ohm']) / Rg * (1 / 298.15 - 1 / TK))
    return R_o + overpotential(Ip, x, y, P, TK) / Ip

def q_of_v(out, vgrid):
    """discharge capacity at each voltage of a descending grid (as Severson's Qdlin); NaN above the first or below the last point"""
    B = out['V'].shape[0]; Q = np.full((B, len(vgrid)), np.nan); Tg = np.full((B, len(vgrid)), np.nan)
    for b in range(B):
        v, q, t = out['V'][b], out['Q'][b], out['T'][b]; m = np.isfinite(v) & out['cc'][b]       # the constant-current part only
        if m.sum() < 3: continue
        v, q, t = v[m], q[m], t[m]; v = np.minimum.accumulate(v)                  # monotone for inversion
        ok = (vgrid <= v[0]) & (vgrid >= v[-1])
        Q[b, ok] = np.interp(-vgrid[ok], -v, q); Tg[b, ok] = np.interp(-vgrid[ok], -v, t)
        Q[b, vgrid > v[0]] = 0.0; Tg[b, vgrid > v[0]] = t[0]
    return Q, Tg
