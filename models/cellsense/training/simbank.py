"""CellSense training data: pairs of 4C discharges of the same simulated cell, once new and once aged, on Severson's
voltage grid (Q(V) and can temperature T(V), constant-current part only), with the aging written down exactly.

New cells are drawn around the six calibration fits (a mixture in log-parameter space, so combinations stay realistic).
Aging moves four physical knobs (Dubarry's degradation modes plus resistance):
  LLI     cyclable lithium lost (SEI growth, plating), as a fraction of the new cell's lithium
  LAM_NE  graphite lost, part of it while lithiated (taking its lithium with it)
  LAM_PE  LFP lost
  R       resistance growth: ohmic and electrolyte resistance up, exchange current and graphite diffusion down
Labels: total capacity after the 2.0 V hold (what Severson's 80% rule uses), constant-current capacity, 30 ms pulse
resistance at 80% SOC, the four modes, and capacity fade relative to new."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import cellsim as CS
CAL = json.load(open(_P.WORK + '/calibration.json')); NAMES = ['Q_p', 'Q_n', 'Q_Li', 'tau_p', 'tau_n', 'I0_p', 'I0_n', 'R_ohm', 'R_e', 'tau_e', 'C_th', 'hA', 'sig_p', 'sig_n']
TH = np.array([CAL[k]['theta'] for k in CAL])
VG = np.linspace(3.5, 2.0, 1000)[::4].copy()                        # Severson's Vdlin grid, every 4th point (250 levels, 6 mV)
FIX = dict(V_full=3.6, Ea_kp=39570., Ea_kn=35000., Ea_dp=35000., Ea_dn=35000., Ea_e=15000.)
LABELS = ['q_total', 'q_cc', 'ir', 'lli', 'lam_ne', 'lam_pe', 'r_growth', 'fade']
# The calibrated simulator reads 11-27 mAh (1-2.5%) high on the six calibration cells' capacities, because the fits favour
# the voltage curve's shape. Lithium inventory is trimmed by the six cells' own mean error so new cells hold the right charge.
Q_LI_TRIM = 1 - np.mean([CAL[k]['q_tot_err_mAh'] for k in CAL]) / 1000 / np.mean([CAL[k]['target']['q_tot'] for k in CAL])

def fresh(rng, B):
    i = rng.integers(0, len(TH), B); j = rng.integers(0, len(TH), B); a = np.where(rng.random(B) < 0.35, rng.random(B), 1.0)
    sd = np.where(np.isin(NAMES, ['Q_p', 'Q_n', 'Q_Li']), 0.025, 0.1)        # capacities vary little cell to cell; kinetics and transport more
    th = a[:, None] * TH[i] + (1 - a[:, None]) * TH[j] + rng.normal(0, 1, (B, len(NAMES))) * sd
    P = {n: np.exp(th[:, k]) for k, n in enumerate(NAMES)}
    P['Q_Li'] = np.minimum(P['Q_Li'] * Q_LI_TRIM, 0.995 * P['Q_n'] + 0.005 * P['Q_p'])
    P.update({k: np.full(B, v) for k, v in FIX.items()}); P['T_amb'] = rng.uniform(29.5, 30.5, B); P['T0'] = P['T_amb'] + rng.uniform(-1.0, 2.0, B)
    return P

def age(rng, P0, scale=1.0):
    B = len(P0['Q_p']); P = {k: v.copy() for k, v in P0.items()}
    young = rng.random(B) < 0.2; sk = lambda hi, p=1.6: np.where(young, 0.03, 1.0) * scale * hi * rng.random(B) ** p
    lli, lam_n, lam_p = sk(0.28), sk(0.30), sk(0.20, 2.0); f_li = rng.random(B)
    # resistance: the 30 ms pulse resistance of real cells grows only 10-40% by end of life, but the slow (seconds-to-minutes)
    # transport polarization grows several-fold: the six calibration cells leave 16 mAh to the 2.0 V hold when new and
    # 63-118 mAh at end of life. So ohmic and kinetic growth are mild and the slow electrolyte/transport term is not.
    g_r, g_k, g_d, g_e = 1 + sk(0.5), 1 + sk(0.5), 1 + sk(1.0), 1 + sk(9.0, 1.2)
    x0, y0 = CS.full_state(P0['Q_p'], P0['Q_n'], P0['Q_Li'], P0['V_full'], P0['sig_p'], P0['sig_n'])
    P['Q_n'] = P0['Q_n'] * (1 - lam_n); P['Q_p'] = P0['Q_p'] * (1 - lam_p)
    P['Q_Li'] = P0['Q_Li'] * (1 - lli) - lam_n * P0['Q_n'] * y0 * f_li; P['Q_Li'] = np.clip(P['Q_Li'], 0.3, 0.995 * P['Q_n'] + 0.005 * P['Q_p'])
    P['R_ohm'] = P0['R_ohm'] * g_r; P['R_e'] = P0['R_e'] * g_e; P['tau_e'] = P0['tau_e'] * rng.uniform(0.8, 1.6, B); P['I0_n'] = P0['I0_n'] / g_k; P['I0_p'] = P0['I0_p'] / np.sqrt(g_k)
    P['tau_n'] = P0['tau_n'] * g_d; P['tau_p'] = P0['tau_p'] * (1 + (g_d - 1) * rng.random(B))
    # inhomogeneous aging: a growing share of the graphite becomes poorly connected (slow population behind R_s). Fits to the
    # calibration cells' worn-out curves put it at 4-11% behind 0.05-0.3 ohm; it sits out a 4C discharge and drains in the hold.
    P['f_s'] = sk(0.3, 1.4); P['R_s'] = np.exp(rng.uniform(np.log(0.03), np.log(1.0), B))
    P['T0'] = P0['T_amb'] + rng.uniform(-1.0, 2.0, B)
    return P

def curves(P, rng=None, dt=3.0):
    o = CS.simulate(P, dt=dt, N=10); Q, T = CS.q_of_v(o, VG)
    return Q, T, o['Q_cc'], o['Q_total'], CS.pulse_resistance(P)

def make(seed, B, out=None, chunk=500):
    rng = np.random.default_rng(seed); R = {k: [] for k in ('Q0', 'T0c', 'Q1', 'T1c', 'y')}
    for s in range(0, B, chunk):
        n = min(chunk, B - s); P0 = fresh(rng, n); P1 = age(rng, P0)
        Q0, T0, cc0, tot0, ir0 = curves(P0); Q1, T1, cc1, tot1, ir1 = curves(P1)
        y = np.stack([tot1, cc1, ir1 * 1000, 1 - P1['Q_Li'] / P0['Q_Li'], 1 - P1['Q_n'] / P0['Q_n'], 1 - P1['Q_p'] / P0['Q_p'], ir1 / ir0 - 1, 1 - tot1 / tot0], 1)
        for k, v in (('Q0', Q0), ('T0c', T0), ('Q1', Q1), ('T1c', T1), ('y', y)): R[k].append(v.astype(np.float32))
        print(seed, s + n, flush=True)
    R = {k: np.concatenate(v) for k, v in R.items()}
    if out: np.savez_compressed(out, **R, labels=np.array(LABELS), vgrid=VG)
    return R

if __name__ == '__main__':
    make(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
