"""Check nmcsim against PyBaMM (Chen2020 parameter set, the LG M50T): the single-particle model at C/20, C/2 and 1C for a
fresh cell and for each degradation mode, then the SPMe's electrolyte polarization (two lag constants fitted on the fresh cell
at 1C, then held fixed). Writes work/nmc_physics_check.json.

    python nmc_check.py
"""
import _paths as _P
import sys, json, numpy as np, pybamm
sys.path.insert(0, _P.SRC); import nmcsim as NS

p0 = pybamm.ParameterValues('Chen2020')
CASES = [('fresh', (0, 0, 0)), ('lost lithium 10%', (0.10, 0, 0)), ('lost negative 10%', (0, 0.10, 0)), ('lost positive 10%', (0, 0, 0.10))]


def ours(deg, V_full=4.19):
    """our lumped parameters, degradation applied as volume fractions (LAM) and lithium inventory (LLI)"""
    L = NS.chen2020(); lli, lam_n, lam_p = deg
    P = {k: np.array([v]) for k, v in L.items()}
    P['Q_n'] = P['Q_n'] * (1 - lam_n); P['Q_p'] = P['Q_p'] * (1 - lam_p)
    P['I0_n'] = P['I0_n'] * (1 - lam_n); P['I0_p'] = P['I0_p'] * (1 - lam_p)          # less active surface area
    P['Q_Li'] = P['Q_Li'] * (1 - lli) - lam_n * L['Q_n'] * 0.9                          # negative lost while charged takes its lithium
    P.update(V_full=np.array([V_full]), R_ohm=np.zeros(1), R_e=np.zeros(1), tau_e=np.ones(1))
    return P


def run_pybamm(model, P, x0, y0, I):
    q = p0.copy(); q['Negative electrode active material volume fraction'] = 0.75 * float(P['Q_n'][0] / NS.chen2020()['Q_n'])
    q['Positive electrode active material volume fraction'] = 0.665 * float(P['Q_p'][0] / NS.chen2020()['Q_p'])
    q['Initial concentration in negative electrode [mol.m-3]'] = y0 * 33133; q['Initial concentration in positive electrode [mol.m-3]'] = x0 * 63104
    q['Current function [A]'] = I; q['Lower voltage cut-off [V]'] = 2.5; q['Upper voltage cut-off [V]'] = 4.4
    sim = pybamm.Simulation(model, parameter_values=q, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-10))
    T_end = 3600 * 1.3 * 5.0 / I; s = sim.solve([0, T_end], t_interp=np.linspace(0, T_end, 6000))      # dense output: the adaptive steps alone are too sparse to interpolate
    return s['Discharge capacity [A.h]'].entries, s['Voltage [V]'].entries


def compare(q, v, qb, vb):
    m = np.isfinite(v); q, v = q[m], v[m]
    qq = np.linspace(0.01, 0.97 * min(q[-1], qb[-1]), 500); dv = np.abs(np.interp(qq, q, v) - np.interp(qq, qb, vb))
    return dict(cap_err_pct=float(100 * abs(q[-1] / qb[-1] - 1)), V_maxerr_mV=float(1000 * dv.max()), V_mederr_mV=float(1000 * np.median(dv)))


def spm():
    rows = []
    for name, deg in CASES:
        P = ours(deg); TP, TN = NS.tables(P, 1); x0, y0 = NS.full_state(P['Q_p'], P['Q_n'], P['Q_Li'], P['V_full'], TP, TN)
        for C in (0.05, 0.5, 1.0):
            I = C * 5.0; o = NS.simulate(P, I=I, dt=min(30.0, 3.0 / C), N=20, T=25.0, V_min=2.5)
            q, v = np.append(o['Q'][0], o['Q_total'][0]), np.append(o['V'][0], 2.5)
            qb, vb = run_pybamm(pybamm.lithium_ion.SPM(), P, float(x0[0]), float(y0[0]), I)
            r = dict(case=name, C=C, cap_ours=float(o['Q_total'][0]), cap_pybamm=float(qb[-1])) | compare(q, v, qb, vb); rows.append(r); print(r, flush=True)
    return rows


def spme():
    from scipy.optimize import least_squares
    P = ours((0, 0, 0)); TP, TN = NS.tables(P, 1); x0, y0 = NS.full_state(P['Q_p'], P['Q_n'], P['Q_Li'], P['V_full'], TP, TN)
    ref = {C: run_pybamm(pybamm.lithium_ion.SPMe(), P, float(x0[0]), float(y0[0]), C * 5.0) for C in (1.0,)}

    def res(z, C=1.0):
        Q = dict(P); Q['R_e'] = np.array([np.exp(z[0])]); Q['tau_e'] = np.array([np.exp(z[1])]); o = NS.simulate(Q, I=C * 5.0, dt=3.0 / C, N=20, T=25.0)
        qb, vb = ref[C]; m = np.isfinite(o['V'][0]); qq = np.linspace(0.05, 0.95 * min(o['Q_total'][0], qb[-1]), 200)
        return 1000 * (np.interp(qq, o['Q'][0][m], o['V'][0][m]) - np.interp(qq, qb, vb))
    z = least_squares(res, [np.log(0.01), np.log(100.0)], x_scale=1.0).x; R_e, tau_e = float(np.exp(z[0])), float(np.exp(z[1])); rows = []
    for name, deg in CASES:
        for C in (1.0, 0.5, 0.05):
            Q = ours(deg); Q['R_e'] = np.array([R_e]); Q['tau_e'] = np.array([tau_e]); TP, TN = NS.tables(Q, 1); x0, y0 = NS.full_state(Q['Q_p'], Q['Q_n'], Q['Q_Li'], Q['V_full'], TP, TN)
            o = NS.simulate(Q, I=C * 5.0, dt=min(30.0, 3.0 / C), N=20, T=25.0); qb, vb = run_pybamm(pybamm.lithium_ion.SPMe(), Q, float(x0[0]), float(y0[0]), C * 5.0)
            r = dict(case=name, C=C, cap_ours=float(o['Q_total'][0]), cap_pybamm=float(qb[-1])) | compare(np.append(o['Q'][0], o['Q_total'][0]), np.append(o['V'][0], 2.5), qb, vb)
            rows.append(r); print('SPMe', r, flush=True)
    return dict(R_e=R_e, tau_e=tau_e, rows=rows)


if __name__ == '__main__':
    out = dict(pybamm=pybamm.__version__, parameter_set='Chen2020', spm=spm(), spme=spme())
    json.dump(out, open(_P.WORK + '/nmc_physics_check.json', 'w'), indent=1)
