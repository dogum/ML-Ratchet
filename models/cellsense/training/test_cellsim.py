"""Check cellsim against PyBaMM (Prada2013 parameter set: an A123 LFP/graphite cell): the single-particle model at 1C and
4C for fresh and degraded electrodes, and the SPMe's electrolyte polarization."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, pybamm
sys.path.insert(0, _P.SRC); import cellsim as CS

def pb_params(p, lli=0.0, lam_n=0.0, lam_p=0.0):
    """our lumped parameters from a PyBaMM parameter set, with degradation applied the same way to both"""
    A = p['Electrode height [m]'] * p['Electrode width [m]']; F = CS.F
    eps_n = p['Negative electrode active material volume fraction'] * (1 - lam_n); eps_p = p['Positive electrode active material volume fraction'] * (1 - lam_p)
    Qn = p['Maximum concentration in negative electrode [mol.m-3]'] * eps_n * p['Negative electrode thickness [m]'] * A * F / 3600
    Qp = p['Maximum concentration in positive electrode [mol.m-3]'] * eps_p * p['Positive electrode thickness [m]'] * A * F / 3600
    ce = p['Initial concentration in electrolyte [mol.m-3]']
    def I0(side, eps, L, R, cmax, m):
        a = 3 * eps / R; return m * ce ** 0.5 * cmax * 0.5 * a * L * A
    I0n = I0('n', eps_n, p['Negative electrode thickness [m]'], p['Negative particle radius [m]'], p['Maximum concentration in negative electrode [mol.m-3]'], 6.48e-7)
    I0p = I0('p', eps_p, p['Positive electrode thickness [m]'], p['Positive particle radius [m]'], p['Maximum concentration in positive electrode [mol.m-3]'], 6e-7)
    tau_n = p['Negative particle radius [m]'] ** 2 / p['Negative particle diffusivity [m2.s-1]']; tau_p = p['Positive particle radius [m]'] ** 2 / p['Positive particle diffusivity [m2.s-1]']
    return dict(Q_p=Qp, Q_n=Qn, eps_n=eps_n, eps_p=eps_p, I0_n=I0n, I0_p=I0p, tau_n=tau_n, tau_p=tau_p)

def run_pybamm(model, p, L, x0, y0, I, T=298.15):
    q = p.copy(); q['Negative electrode active material volume fraction'] = L['eps_n']; q['Positive electrode active material volume fraction'] = L['eps_p']
    q['Initial concentration in negative electrode [mol.m-3]'] = y0 * q['Maximum concentration in negative electrode [mol.m-3]']
    q['Initial concentration in positive electrode [mol.m-3]'] = x0 * q['Maximum concentration in positive electrode [mol.m-3]']
    q['Current function [A]'] = I; q['Lower voltage cut-off [V]'] = 2.0; q['Upper voltage cut-off [V]'] = 4.0
    sim = pybamm.Simulation(model, parameter_values=q, solver=pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-9))
    s = sim.solve([0, 3600 * 1.3 * L['Q_p'] / I])
    return s['Discharge capacity [A.h]'].entries, s['Voltage [V]'].entries

if __name__ == '__main__' and '--spme' not in sys.argv and '--cv' not in sys.argv:
    p = pybamm.ParameterValues('Prada2013'); res = {}; rows = []
    base = pb_params(p); Q_Li_fresh = None
    for name, deg in [('fresh', (0, 0, 0)), ('lost lithium 15%', (0.15, 0, 0)), ('lost graphite 20%', (0, 0.2, 0)), ('lost LFP 15%', (0, 0, 0.15))]:
        L = pb_params(p, lam_n=deg[1], lam_p=deg[2])
        if Q_Li_fresh is None:
            x0f, y0f = CS.full_state(np.array([L['Q_p']]), np.array([L['Q_n']]), np.array([0.02 * L['Q_p'] + 0.95 * L['Q_n']]), np.array([3.6]))
            Q_Li_fresh = float(0.02 * L['Q_p'] + 0.95 * L['Q_n'])
        Q_Li = Q_Li_fresh * (1 - deg[0]) - deg[1] * 0.95 * base['Q_n']          # graphite lost while charged takes its lithium with it
        for C in (1, 4):
            I = C * 2.3
            P = {k: np.array([v]) for k, v in L.items()}; P.update(Q_Li=np.array([Q_Li]), V_full=np.array([3.6]), R_ohm=np.zeros(1), R_e=np.zeros(1), tau_e=np.ones(1),
                     Ea_kp=np.array([39570.0]), Ea_kn=np.array([35000.0]), Ea_e=np.zeros(1), Ea_dp=np.zeros(1), Ea_dn=np.zeros(1), C_th=np.ones(1), hA=np.ones(1), T_amb=np.array([25.0]))
            x0, y0 = CS.full_state(P['Q_p'], P['Q_n'], P['Q_Li'], P['V_full'])
            o = CS.simulate(P, I=I, dt=1.0, N=20, isothermal=True, cv=False)
            q, v = o['Q'][0], o['V'][0]; m = np.isfinite(v); q, v = q[m], v[m]
            qb, vb = run_pybamm(pybamm.lithium_ion.SPM(), p, L, float(x0[0]), float(y0[0]), I)
            qq = np.linspace(0.01, 0.97 * min(q[-1], qb[-1]), 400); dv = np.abs(np.interp(qq, q, v) - np.interp(qq, qb, vb))
            qq2 = np.linspace(0.02, 0.9 * min(q[-1], qb[-1]), 400); dv2 = np.abs(np.interp(qq2, q, v) - np.interp(qq2, qb, vb))
            rows.append(dict(case=name, C=C, cap_ours=float(q[-1]), cap_pybamm=float(qb[-1]), cap_err_pct=float(100 * abs(q[-1] / qb[-1] - 1)), V_maxerr_mV=float(1000 * dv.max()),
                             V_maxerr_at_Q=float(qq[dv.argmax()]), V_maxerr_2_90pct_mV=float(1000 * dv2.max()), V_mederr_mV=float(1000 * np.median(dv))))
            print(rows[-1], flush=True)
    res['spm'] = rows
    json.dump(res, open(_P.WORK + '/physics_check.json', 'w'), indent=1)

def spme_check():
    """SPMe vs SPM: fit our electrolyte resistance and time constant at 4C on the fresh cell, then test them unchanged at
    1C and 2C and on degraded cells"""
    from scipy.optimize import least_squares
    p = pybamm.ParameterValues('Prada2013'); out = []; base = pb_params(p); Q_Li0 = 0.02 * base['Q_p'] + 0.95 * base['Q_n']
    cases = [('fresh', (0, 0, 0), 4), ('fresh', (0, 0, 0), 1), ('fresh', (0, 0, 0), 2), ('lost lithium 15%', (0.15, 0, 0), 4), ('lost graphite 20%', (0, 0.2, 0), 4)]
    curves = {}
    for name, deg, C in cases:
        L = pb_params(p, lam_n=deg[1], lam_p=deg[2]); Q_Li = Q_Li0 * (1 - deg[0]) - deg[1] * 0.95 * base['Q_n']
        x0, y0 = CS.full_state(np.array([L['Q_p']]), np.array([L['Q_n']]), np.array([Q_Li]), np.array([3.6])); I = C * 2.3
        qs, vs = run_pybamm(pybamm.lithium_ion.SPM(), p, L, float(x0[0]), float(y0[0]), I); qe, ve = run_pybamm(pybamm.lithium_ion.SPMe(), p, L, float(x0[0]), float(y0[0]), I)
        curves[(name, C)] = (L, Q_Li, I, qs, vs, qe, ve)
    L, Q_Li, I, qs, vs, qe, ve = curves[('fresh', 4)]
    def ours(L, Q_Li, I, Re, te):
        P = {k: np.array([v]) for k, v in L.items()}; P.update(Q_Li=np.array([Q_Li]), V_full=np.array([3.6]), R_ohm=np.zeros(1), R_e=np.array([Re]), tau_e=np.array([te]),
                 Ea_kp=np.array([39570.0]), Ea_kn=np.array([35000.0]), Ea_e=np.zeros(1), Ea_dp=np.zeros(1), Ea_dn=np.zeros(1), C_th=np.ones(1), hA=np.ones(1), T_amb=np.array([25.0]))
        o = CS.simulate(P, I=I, dt=1.0, N=20, isothermal=True, cv=False); m = np.isfinite(o['V'][0]); return o['Q'][0][m], o['V'][0][m]
    qq = np.linspace(0.02, 0.9 * qe[-1], 200)
    fit = least_squares(lambda z: (np.interp(qq, *ours(L, Q_Li, I, z[0], z[1])) - np.interp(qq, qe, ve)) * 1000, [0.01, 60], bounds=([0, 1], [0.2, 2000]))
    Re, te = fit.x; print('fitted electrolyte R %.4f ohm, tau %.0f s' % (Re, te))
    for (name, C), (L, Q_Li, I, qs, vs, qe, ve) in curves.items():
        q, v = ours(L, Q_Li, I, Re, te); qq = np.linspace(0.02, 0.9 * min(q[-1], qe[-1]), 400); dv = np.abs(np.interp(qq, q, v) - np.interp(qq, qe, ve))
        out.append(dict(case=name, C=C, vs='SPMe', cap_err_pct=float(100 * abs(q[-1] / qe[-1] - 1)), V_maxerr_2_90pct_mV=float(1000 * dv.max()), V_mederr_mV=float(1000 * np.median(dv)),
                        electrolyte_drop_mV=float(1000 * np.median(np.abs(np.interp(qq, qs, vs) - np.interp(qq, qe, ve))))))
        print(out[-1], flush=True)
    return {'R_e_ohm': float(Re), 'tau_e_s': float(te), 'rows': out}

if __name__ == '__main__' and '--spme' in sys.argv:
    r = json.load(open(_P.WORK + '/physics_check.json')); r['spme'] = spme_check(); json.dump(r, open(_P.WORK + '/physics_check.json', 'w'), indent=1)

def cv_check():
    """the constant-voltage hold at 2.0 V after a 4C discharge, against PyBaMM's experiment runner (SPM)"""
    p = pybamm.ParameterValues('Prada2013'); out = []; base = pb_params(p); Q_Li0 = 0.02 * base['Q_p'] + 0.95 * base['Q_n']
    for name, deg in [('fresh', (0, 0, 0)), ('lost lithium 15%', (0.15, 0, 0)), ('lost graphite 20%', (0, 0.2, 0))]:
        L = pb_params(p, lam_n=deg[1], lam_p=deg[2]); Q_Li = Q_Li0 * (1 - deg[0]) - deg[1] * 0.95 * base['Q_n']
        x0, y0 = CS.full_state(np.array([L['Q_p']]), np.array([L['Q_n']]), np.array([Q_Li]), np.array([3.6]))
        q = p.copy(); q['Negative electrode active material volume fraction'] = L['eps_n']; q['Positive electrode active material volume fraction'] = L['eps_p']
        q['Initial concentration in negative electrode [mol.m-3]'] = float(y0[0]) * q['Maximum concentration in negative electrode [mol.m-3]']
        q['Initial concentration in positive electrode [mol.m-3]'] = float(x0[0]) * q['Maximum concentration in positive electrode [mol.m-3]']
        q['Lower voltage cut-off [V]'] = 1.9; q['Upper voltage cut-off [V]'] = 4.0
        exp = pybamm.Experiment([("Discharge at 9.2 A until 2.0 V", "Hold at 2.0 V until 46 mA")])
        s = pybamm.Simulation(pybamm.lithium_ion.SPM(), parameter_values=q, experiment=exp).solve()
        qb = s['Discharge capacity [A.h]'].entries; cc_b = s.cycles[0].steps[0]['Discharge capacity [A.h]'].entries[-1]
        P = {k: np.array([v]) for k, v in L.items()}; P.update(Q_Li=np.array([Q_Li]), V_full=np.array([3.6]), R_ohm=np.zeros(1), R_e=np.zeros(1), tau_e=np.ones(1),
                 Ea_kp=np.array([39570.0]), Ea_kn=np.array([35000.0]), Ea_e=np.zeros(1), Ea_dp=np.zeros(1), Ea_dn=np.zeros(1), C_th=np.ones(1), hA=np.ones(1), T_amb=np.array([25.0]))
        o = CS.simulate(P, I=9.2, dt=1.0, N=20, isothermal=True, I_cut=0.046)
        out.append(dict(case=name, cc_ours=float(o['Q_cc'][0]), cc_pybamm=float(cc_b), total_ours=float(o['Q_total'][0]), total_pybamm=float(qb[-1]),
                        cv_part_ours=float(o['Q_total'][0] - o['Q_cc'][0]), cv_part_pybamm=float(qb[-1] - cc_b)))
        print(out[-1], flush=True)
    return out

if __name__ == '__main__' and '--cv' in sys.argv:
    r = json.load(open(_P.WORK + '/physics_check.json')); r['cv_hold'] = cv_check(); json.dump(r, open(_P.WORK + '/physics_check.json', 'w'), indent=1)
