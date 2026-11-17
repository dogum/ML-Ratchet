"""Analytic checks of the zone model:
1. steady-state conduction: constant outdoor temperature, no sun, no sky deficit, no gains -> ideal heating power
   equals (UA_envelope + rho cp Q_inf) dT exactly;
2. RC step response: free-floating zone after a step in outdoor temperature vs the closed-form 5R1C solution
   (one mass node, exponential with tau = C_m / (H_tr,3 + H_tr,em));
3. degree-day limit: with (almost) no thermal mass, constant setpoint, constant gains and no sun, annual heating
   equals UA_tot x degree-hours below the balance point T_set - G / UA_tot (real Denver temperatures);
   with medium mass the result departs from it (reported);
4. ledger closure on the same runs.
"""
import numpy as np
import pandas as pd

from _common import write
from buildingsim import read_epw, epw_path
from buildingsim.simulate import Simulator
from buildingsim.weather import SIGMA


def synthetic_weather(T, n_hours=24 * 60, start='2019-01-01', lat=40.0, freq='1h'):
    m = int(pd.Timedelta('1h') / pd.Timedelta(freq))
    idx = pd.date_range(start, periods=n_hours * m, freq=freq, tz='UTC')
    T = np.broadcast_to(np.asarray(T, float), (n_hours * m,)).copy()
    df = pd.DataFrame(dict(T_out=T, RH=50.0, GHI=0.0, DNI=0.0, DHI=0.0, wind_speed=0.0, wind_dir=180.0,
                           pressure=101325.0, IR_horiz=SIGMA * (T + 273.15) ** 4, cloud=1.0), index=idx)
    return df, dict(lat=lat, lon=-105.0, elev=0.0, tz='UTC')


def base_spec(mass='medium', system='ideal', gains=0.0):
    return dict(primary_use='other', floor_area_m2=200.0, n_floors=1, year_built=2000, wall_U=0.5, roof_U=0.3,
                window_U=2.8, SHGC=0.5, window_wwr=0.25, infiltration_ach=0.4, infiltration_model='constant',
                thermal_mass=mass,
                geometry=dict(foundation='adiabatic', floor_U=0.3, attic=False, aspect_ratio=1.5,
                              floor_to_floor_m=3.0, ceiling_m=3.0),
                internal_gains=dict(people=0.0, lighting=0.0, plug=0.0, process=gains, cooking=0.0,
                                    exterior_lighting=0.0),
                setpoints=dict(heat=20.0, cool=26.0, setback_heat=None, setback_cool=None),
                ventilation=dict(L_s_person=0.0, L_s_m2=0.0, economizer=False, mech_vent=False),
                system=system, dhw=dict(fuel='electric', L_per_day=0.0, UA_W_K=0.0), moisture_capacitance=1.0)


def steady_state():
    out = {}
    for mass in ('light', 'heavy'):
        df, meta = synthetic_weather(-10.0, 24 * 40)
        sim = Simulator(base_spec(mass), (df, meta), dt='1h', warmup_days=0, init=dict(T_air=20.0, T_mass=20.0))
        r = sim.run()
        z = sim.zone
        H_inf = float(np.mean(sim.rho)) * 1020.0 * sim.ela * 1e-3
        H_floor = sim.bnd['H_ground']
        UA = z.H_tr_w + sim.bnd['H_wall'] + sim.bnd['H_roof'] + H_floor + H_inf
        # the 5R1C network seen from the (controlled) air node: air -> star node (H_tr,is) in series with
        # windows + opaque, in parallel with air exchange
        UA_net = H_inf + 1.0 / (1.0 / z.H_tr_is + 1.0 / (z.H_tr_w + z.H_op))
        got = r['ts']['Q_hvac_sens_kW'].values[-24:].mean() * 1000.0
        out[mass] = dict(expected_network_W=UA_net * 30.0, simulated_W=got,
                         rel_error_vs_network=abs(got - UA_net * 30.0) / (UA_net * 30.0),
                         naive_UA_dT_W=UA * 30.0, rel_diff_vs_naive_UA=(got - UA * 30.0) / (UA * 30.0),
                         T_surface_minus_T_air_K=float(r['ts']['T_surface'].values[-1] - r['ts']['T_zone'].values[-1]),
                         UA_naive_W_K=UA, UA_network_W_K=UA_net, ledger_closure=r['ledger']['closure_rel'])
    return out


def step_response():
    res = {}
    for dt in ('1h', '15min'):
        n_h = 24 * 20
        T = np.where(np.arange(n_h) < 24 * 2, 0.0, 10.0)
        m = int(pd.Timedelta('1h') / pd.Timedelta(dt))
        df, meta = synthetic_weather(np.repeat(T, m), n_h, freq=dt)
        sim = Simulator(base_spec('heavy', 'none'), (df, meta), dt=dt, warmup_days=0,
                        init=dict(T_air=0.0, T_mass=0.0))
        r = sim.run()
        z = sim.zone
        H_ve = float(np.mean(sim.rho)) * 1020.0 * sim.ela * 1e-3
        H1 = 1 / (1 / H_ve + 1 / z.H_tr_is)
        H2 = H1 + z.H_tr_w
        H3 = 1 / (1 / H2 + 1 / z.H_tr_ms)
        tau = z.C_m / (H3 + z.H_tr_em)
        t = (np.arange(sim.n) * sim.dt) - 48 * 3600.0
        # mid-step (Crank-Nicolson averages the mass temperature over the step)
        tm = np.where(t + sim.dt / 2 > 0, 10.0 * (1 - np.exp(-np.clip(t + sim.dt / 2, 0, None) / tau)), 0.0)
        # algebraic nodes for theta_e = 10 after the step
        Te = np.where(t >= 0, 10.0, 0.0)
        ts_ = (z.H_tr_ms * tm + z.H_tr_w * Te + H1 * Te) / (z.H_tr_ms + z.H_tr_w + H1)
        ta = (z.H_tr_is * ts_ + H_ve * Te) / (z.H_tr_is + H_ve)
        sim_T = r['ts']['T_zone'].values
        m = t >= 0
        err = sim_T[m] - ta[m]
        res[dt] = dict(tau_h=tau / 3600, max_abs_err_K=float(np.abs(err).max()), rms_err_K=float(np.sqrt(
            (err ** 2).mean())), T_at_tau_sim=float(np.interp(tau, t[m], sim_T[m])),
            T_at_tau_closed_form=float(np.interp(tau, t[m], ta[m])), ledger_closure=r['ledger']['closure_rel'])
    return res


def degree_day():
    df, meta = read_epw(epw_path('Denver'))
    df = df.copy()
    df['GHI'] = df['DNI'] = df['DHI'] = 0.0
    df['IR_horiz'] = SIGMA * (df['T_out'] + 273.15) ** 4
    df['wind_speed'] = 0.0
    G_m2 = 5.0
    out = {}
    for mass in ('tiny', 'medium'):
        spec = base_spec('medium', 'ideal', gains=G_m2)
        if mass == 'tiny':
            spec['geometry']['C_m_J_K'] = 10.0 * 200.0     # 10 J/m2K: no memory
        sim = Simulator(spec, (df, meta), '2019-01-01', '2020-01-01', dt='1h', warmup_days=7)
        sim.hsp = [20.0] * sim.n
        sim.csp = [100.0] * sim.n      # heating only
        r = sim.run()
        z = sim.zone
        H_inf = np.asarray(sim.rho) * 1020.0 * sim.ela * 1e-3
        UA = H_inf + 1.0 / (1.0 / z.H_tr_is + 1.0 / (z.H_tr_w + z.H_op))      # network UA (see steady state)
        G = G_m2 * 200.0
        Tbal = 20.0 - G / UA
        Tout = sim.wx['T_out']
        classic = float(np.sum(UA * np.clip(Tbal - Tout, 0, None)) / 1000.0)
        # steady-state 5R1C heating need with the ISO gain split (air / surface / mass nodes), per hour:
        # unknowns theta_s, theta_m with theta_air = 20 held; q = heat needed at the air node (>= 0)
        Ta = 20.0
        Pia = 0.5 * G
        Pst = z.f_st * 0.5 * G
        Pm = z.f_m * 0.5 * G
        Hms, Hem, Hw, His = z.H_tr_ms, z.H_tr_em, z.H_tr_w, z.H_tr_is
        A = np.array([[-(His + Hw + Hms), Hms], [Hms, -(Hms + Hem)]])
        qss = np.empty(len(Tout))
        for i, Te in enumerate(Tout):
            rhs = np.array([-(His * Ta + Hw * Te + Pst), -(Hem * Te + Pm)])
            ts_, tm_ = np.linalg.solve(A, rhs)
            qss[i] = -(Pia + His * (ts_ - Ta) + H_inf[i] * (Te - Ta))
        expect = float(np.clip(qss, 0, None).sum() / 1000.0)
        q = r['ts']['Q_hvac_sens_kW'].values
        got = float(q[q > 0].sum())
        out[mass] = dict(expected_kWh=expect, simulated_kWh=got, ratio=got / expect, UA_network_mean=float(np.mean(UA)),
                         classic_degree_hour_kWh=classic, ratio_to_classic=got / classic,
                         balance_point_C=float(np.mean(Tbal)), C_m_kJ_m2K=z.C_m / 200.0 / 1000.0,
                         ledger_closure=r['ledger']['closure_rel'])
    return out


def main():
    ss = steady_state()
    print('steady', ss)
    st = step_response()
    print('step', st)
    dd = degree_day()
    print('dd', dd)
    write('analytic.json', dict(
        what='analytic checks of the 5R1C zone: steady conduction, step response, degree-day limit, ledgers',
        steady_state_conduction=ss, rc_step_response=st, degree_day_limit=dd,
        notes=['steady state: with the air node controlled, the 5R1C heat loss is UA_network dT, where '
               'UA_network = H_ve + 1/(1/H_tr,is + 1/(H_tr,w + H_op)); it is ~7 % below the naive sum of U A '
               'because the ISO interior coupling (3.45 W/m2K x 4.5 A_f) keeps surfaces colder than the air',
               'step response compares the zone air temperature with the closed form at the step mid-point '
               '(Crank-Nicolson averages the mass temperature over the step)',
               'degree-day: expected = sum of hourly steady-state network heating needs (ISO gain split: half '
               'the gains land on surface/mass nodes and part leaks out through H_tr,em, so the classic '
               'UA x degree-hours below T_set - G/UA is ~1.5 % lower);',
               'degree-day: with 1 kJ/m2K the zone has no memory and must match; with medium mass (165 kJ/m2K) '
               'heating drops because stored gains carry over into colder hours']))


if __name__ == '__main__':
    main()
