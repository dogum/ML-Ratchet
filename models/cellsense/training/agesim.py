"""Aging trajectories for CellSense's lifetime rung: how the four degradation modes grow cycle by cycle under fast charging,
with the two mechanisms behind the capacity "knee" in LFP/graphite cells (Attia et al. 2022):
  1. lost graphite (LAM_NE) accelerates until the negative electrode, not the lithium, limits the capacity;
  2. as the graphite shrinks, it is filled closer to the brim at the top of each fast charge; past a threshold that falls
     with charging rate, lithium plates, and plating eats lithium (LLI) fast.
Plus SEI growth (LLI ~ sqrt(cycles)), slow LFP loss, and resistance that grows with SEI and lost material.
The capacity after the 2.0 V hold is tracked from the electrodes' equilibrium (open-circuit) window every 10 cycles; end of
life is the first cycle below 0.88 Ah, as in Severson et al. Only two cycles per cell are then simulated in full (4C
curves): an early reference (cycles 5-10) and the cycle being read."""
import _paths as _P  # portable paths (see _paths.py)
import sys, numpy as np
sys.path.insert(0, _P.SRC); import cellsim as CS, simbank as SB

def window(Q_p, Q_n, Q_Li, sp, sn, V_lo=2.0, V_hi=3.6, tabs=None):
    """equilibrium capacity between V_hi and V_lo, and the graphite filling at V_hi. tabs: per-cell OCV lookup tables"""
    TP, TN = tabs if tabs is not None else (CS.Table(CS.U_p, sp, len(Q_p)), CS.Table(CS.U_n, sn, len(Q_p)))
    lo = np.full_like(Q_p, 1e-6); hi = np.minimum(1 - 1e-6, Q_Li / Q_p)
    for _ in range(36):                                                  # full end: OCV = V_hi
        x = 0.5 * (lo + hi); y = np.clip((Q_Li - x * Q_p) / Q_n, 1e-6, 1 - 1e-6); ocv = TP(x) - TN(y)
        hi = np.where(ocv < V_hi, x, hi); lo = np.where(ocv < V_hi, lo, x)
    xf = 0.5 * (lo + hi); yf = np.clip((Q_Li - xf * Q_p) / Q_n, 1e-6, 1 - 1e-6)
    lo = np.maximum(xf, 1e-6); hi = np.full_like(Q_p, 1 - 1e-6)
    for _ in range(36):                                                  # empty end: OCV = V_lo, more lithium in LFP
        x = 0.5 * (lo + hi); y = np.clip((Q_Li - x * Q_p) / Q_n, 1e-7, 1 - 1e-6); ocv = TP(x) - TN(y)
        lo = np.where(ocv > V_lo, x, lo); hi = np.where(ocv > V_lo, hi, x)
    return (0.5 * (lo + hi) - xf) * Q_p, yf

AGE_CAL = None; EARLY = None
# Early-life coupling (v2), calibrated on the six calibration cells (dq_early.py, early_fit.py, calib_early.py): the change in
# the 4C curve over the first 100 cycles is a growing share of slow (poorly connected) graphite, and cells that grow more of
# it lose their graphite sooner. k_n scales as (F / F_anchor)^P_C around each calibration cell.
# v2.1: late-life knobs from a whole-life curve check on the same six cells (late_fit.py): the slow share levels off near
# 8-14%, and the slow transport polarization grows late (~(n/1000)^2) and with lost lithium and graphite, not from the start.
P_C, SIG_F, SIG_KN, SIG_HIDDEN, SIG_RS, GAM = 3.0, 0.4, 0.3, 0.3, 0.15, (0.7, 0.9)

def rates(rng, B, v2=True):
    """aging rates drawn around the six calibration cells' fitted trajectories (fit_life.py), then moved to a random
    charging rate: faster charging speeds graphite loss and lowers the plating threshold. The spread is wide, so lives run
    from a few hundred to a few thousand cycles."""
    global AGE_CAL
    if AGE_CAL is None:
        import json; F = json.load(open(_P.WORK + '/aging_calibration%s.json' % ('' if v2 else '_v1'))); AGE_CAL = (np.array([f['z'] for f in F.values()]), np.array([f['c'] for f in F.values()]))
    Z, C = AGE_CAL; i = rng.integers(0, len(Z), B); z = Z[i].copy(); ci = C[i]
    z[:, 0] += rng.normal(0, 0.6, B); z[:, 1] += rng.normal(0, 1.6, B); z[:, 2] = np.clip(z[:, 2] + rng.normal(0, 0.5, B), 1.0, 4.5)
    z[:, 3] = np.clip(z[:, 3] + rng.normal(0, 0.015, B), 0, 0.1); z[:, 4] += rng.normal(0, 0.8, B); z[:, 5] += rng.normal(0, 0.5, B); z[:, 6] = np.clip(z[:, 6] + rng.normal(0, 0.15, B), 0, 1)
    c = rng.uniform(3.6, 8.0, B); u = lambda a, b: np.exp(rng.uniform(np.log(a), np.log(b), B))
    if not v2:
        return dict(c=c, k_s=np.exp(z[:, 0]) * (1 + 0.08 * (c - ci)), k_n=np.exp(z[:, 1]) * (c / ci) ** rng.uniform(0.5, 2.0, B), beta=z[:, 2], k_p=z[:, 3], k_pl=np.exp(z[:, 4]),
                    delta=np.exp(z[:, 5]) * (ci / c) ** rng.uniform(0.5, 1.5, B), f_li=z[:, 6],
                    k_r=u(0.05, 1.0), k_k=u(0.05, 0.8), k_d=u(0.05, 0.8), k_e=u(0.5, 5.0), k_e2=u(0.3, 2.5), te=rng.uniform(0.8, 1.6, B), k_f=u(0.005, 0.1), k_f2=u(0.05, 0.6), R_s=u(0.03, 1.0))
    global EARLY
    if EARLY is None:
        import json; E = json.load(open(_P.WORK + '/early_calibration.json')); EARLY = (np.array([e.get('F_adj', e['f_s']) for e in E.values()]), np.array([np.exp(e['log_R_s']) for e in E.values()]))
    Fi, Rsi = EARLY[0][i], EARLY[1][i]; z = Z[i].copy(); H = SIG_HIDDEN      # life factors not visible early, scaled down so early change carries most of it
    z[:, 0] += rng.normal(0, 0.6 * H, B); z[:, 2] = np.clip(z[:, 2] + rng.normal(0, 0.5 * H, B), 1.0, 4.5)
    z[:, 3] = np.clip(z[:, 3] + rng.normal(0, 0.015 * H, B), 0, 0.1); z[:, 4] += rng.normal(0, 0.8 * H, B); z[:, 5] += rng.normal(0, 0.5 * H, B); z[:, 6] = np.clip(z[:, 6] + rng.normal(0, 0.15 * H, B), 0, 1)
    F = Fi * np.exp(rng.normal(0, SIG_F, B)) * (c / ci) ** rng.uniform(0.3, 1.0, B)
    z[:, 1] = Z[i, 1] + rng.normal(0, SIG_KN, B) + P_C * np.log(F / Fi)
    return dict(c=c, k_s=np.exp(z[:, 0]) * (1 + 0.08 * (c - ci)), k_n=np.exp(z[:, 1]) * (c / ci) ** rng.uniform(0.7, 1.5, B), beta=z[:, 2], k_p=z[:, 3], k_pl=np.exp(z[:, 4]),
                delta=np.exp(z[:, 5]) * (ci / c) ** rng.uniform(0.7, 1.3, B), f_li=z[:, 6],
                k_r=u(0.05, 1.0), k_k=u(0.05, 0.8), k_d=u(0.05, 0.8), k_e=rng.uniform(0.0, 0.4, B), p_e=2.0, k_e2=u(0.05, 0.5), ge_drv='lli', te=rng.uniform(0.8, 1.6, B), k_f=np.zeros(B), k_f2=rng.uniform(0.0, 0.1, B),
                R_s=Rsi * np.exp(rng.normal(0, SIG_RS, B)), sei='lin', anchor=i, F=F, gam=rng.uniform(*GAM, B), F_cap=rng.uniform(0.08, 0.14, B))

def trajectory(P0, R, n_max=3000, dn=10, eol=0.88, kappa=None):
    """modes and capacity every dn cycles; kappa: delivered (4C + hold) over equilibrium capacity for the new cell"""
    B = len(P0['Q_p']); steps = n_max // dn + 1; n = np.arange(steps) * dn
    Qp0, Qn0, Li0 = P0['Q_p'], P0['Q_n'], P0['Q_Li']; sp, sn = P0['sig_p'], P0['sig_n']
    tabs = (CS.Table(CS.U_p, sp, B), CS.Table(CS.U_n, sn, B)); q_eq0, y0 = window(Qp0, Qn0, Li0, sp, sn, tabs=tabs); kappa = np.ones(B) if kappa is None else kappa
    L_pl = np.zeros(B); out = {k: np.zeros((B, steps)) for k in ('lli', 'lam_ne', 'lam_pe', 'q_total', 'g_r', 'g_e', 'y_top', 'f_s')}
    for i, ni in enumerate(n):
        lam_n = np.clip(R['k_n'] * (ni / 1000) ** R['beta'], 0, 0.9); lam_p = np.clip(R['k_p'] * ni / 1000, 0, 0.5)
        sei = R['k_s'] * ni / 1000 if R.get('sei') == 'lin' else R['k_s'] * np.sqrt(ni)    # v2: linear (Attia et al. 2020: these cells do not follow sqrt(t))
        lli = np.clip(sei + L_pl, 0, 0.9)
        Q_n = Qn0 * (1 - lam_n); Q_p = Qp0 * (1 - lam_p); Q_Li = np.clip(Li0 * (1 - lli) - lam_n * Qn0 * y0 * R['f_li'], 0.05, 0.995 * Q_n + 0.005 * Q_p)
        q_eq, y_top = window(Q_p, Q_n, Q_Li, sp, sn, tabs=tabs)
        excess = np.maximum(0, y_top - (1 - R['delta'])); L_pl = L_pl + dn * R['k_pl'] * excess ** 2   # plating, quadratic onset
        out['lli'][:, i] = 1 - Q_Li / Li0; out['lam_ne'][:, i] = lam_n; out['lam_pe'][:, i] = lam_p; out['y_top'][:, i] = y_top
        out['g_r'][:, i] = 1 + R['k_r'] * (out['lli'][:, i] + 0.5 * lam_n); out["g_e"][:, i] = np.minimum(12, 1 + R['k_e'] * (ni / 1000) ** R.get('p_e', 1.0) + 6 * R['k_e2'] * ((L_pl if R.get('ge_drv') == 'pl' else out['lli'][:, i]) + lam_n))
        out['q_total'][:, i] = q_eq * kappa
        early = R['F_cap'] * np.tanh(R['F'] * (ni / 100) ** R['gam'] / R['F_cap']) if 'F' in R else R['k_f'] * ni / 1000
        out['f_s'][:, i] = np.minimum(0.4, early + R['k_f2'] * (out['lli'][:, i] + lam_n))
    below = out['q_total'] < eol; life = np.where(below.any(1), n[np.argmax(below, 1)], np.nan)
    out['n'] = n; out['life'] = life; return out

def state_at(P0, R, tr, i):
    """simulator parameters for each cell at trajectory index i (B,)"""
    b = np.arange(len(i)); P = {k: v.copy() for k, v in P0.items()}; x0, y0 = CS.full_state(P0['Q_p'], P0['Q_n'], P0['Q_Li'], P0['V_full'], P0['sig_p'], P0['sig_n'])
    lam_n, lam_p, lli = tr['lam_ne'][b, i], tr['lam_pe'][b, i], tr['lli'][b, i]; g = tr['g_r'][b, i]
    P['Q_n'] = P0['Q_n'] * (1 - lam_n); P['Q_p'] = P0['Q_p'] * (1 - lam_p); P['Q_Li'] = np.clip(P0['Q_Li'] * (1 - lli), 0.05, 0.995 * P['Q_n'] + 0.005 * P['Q_p'])
    P['R_ohm'] = P0['R_ohm'] * g; P['R_e'] = P0['R_e'] * tr['g_e'][b, i]; P['tau_e'] = P0['tau_e'] * R['te']; P['I0_n'] = P0['I0_n'] / (1 + R['k_k'] * (lli + lam_n))
    P['tau_n'] = P0['tau_n'] * (1 + R['k_d'] * lam_n * 2)
    P['f_s'] = tr['f_s'][b, i]; P['R_s'] = R['R_s']
    return P

if __name__ == '__main__':
    import time
    rng = np.random.default_rng(0); B = 400; P0 = SB.fresh(rng, B); R = rates(rng, B); t = time.time(); tr = trajectory(P0, R); print('%.1f s' % (time.time() - t))
    L = tr['life']; print('life percentiles', np.nanpercentile(L, [5, 25, 50, 75, 95]), 'never reached', np.isnan(L).mean())
    print('q_total fresh', np.percentile(tr['q_total'][:, 0], [5, 50, 95]).round(3))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for b in range(40): ax[0].plot(tr['n'], tr['q_total'][b], lw=0.8); ax[1].plot(tr['n'], tr['lli'][b], lw=0.8)
    ax[0].axhline(0.88, color='k', ls='--'); ax[0].set_ylim(0.7, 1.15); ax[0].set_xlabel('cycle'); ax[0].set_ylabel('capacity (Ah)'); ax[1].set_ylabel('LLI')
    plt.savefig(_P.WORK + '/scratch/traj.png', dpi=70)
