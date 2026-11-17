"""Multi-node zone (zone_model='multinode') checks:
1. steady state: heating = (sum U A + windows + air exchange) dT exactly (no ISO interior coupling here);
2. periodic wall conduction: a 0.2 m concrete wall (R_si 0.13, R_se 0.04) under a 24 h sinusoidal outdoor
   temperature, air held at 20 C: the heat flow into the room vs the ISO 13786 periodic thermal transmittance
   (exact solution of the 1-D heat equation via transfer matrices); frequency domain (spatial discretisation) and
   time domain at 1 h and 15 min (time integration);
3. BESTEST 600 / 900 with the BESTEST layer tables, vs the reference ranges (and vs the 5R1C zone)."""
import numpy as np
import pandas as pd

from _common import write
from buildingsim.simulate import Simulator
from buildingsim.zone_multinode import _discretise
from buildingsim import read_epw
from check_analytic import synthetic_weather, base_spec
import check_bestest as cb


def iso13786_Y12(layers, rsi, rse, T=86400.0):
    def Z_layer(t, k, rho, cp):
        d = np.sqrt(k * T / (np.pi * rho * cp))
        xi = t / d
        a = xi * (1 + 1j)
        R = t / k
        return np.array([[np.cosh(a), R * np.sinh(a) / a], [a / R * np.sinh(a), np.cosh(a)]])
    Z = np.array([[1, -rse], [0, 1]], complex)     # ISO 13786 sign convention: Z = Z_se ... Z_si
    Z = np.array([[1, rse], [0, 1]], complex)
    for L in layers[::-1]:          # outside -> inside
        Z = Z @ Z_layer(*L)
    Z = Z @ np.array([[1, rsi], [0, 1]], complex)
    return -1.0 / Z[0, 1]          # complex: heat flux into the room per K of outdoor amplitude


def ladder_Y12(layers, rsi, rse, T=86400.0):
    C, Rb, r_in, r_out = _discretise(layers)
    n = len(C)
    w = 2 * np.pi / T
    G = np.zeros((n, n), complex)
    for i, g in enumerate(1 / Rb):
        G[i, i] += g; G[i + 1, i + 1] += g; G[i, i + 1] -= g; G[i + 1, i] -= g
    gi, go = 1 / (rsi + r_in), 1 / (rse + r_out)
    G[0, 0] += gi
    G[-1, -1] += go
    A = G + 1j * w * np.diag(C)
    b = np.zeros(n, complex)
    b[-1] = go                      # outdoor amplitude 1, room 0
    x = np.linalg.solve(A, b)
    return gi * x[0]                # flux into the room


def main():
    res = {}
    # 1. steady state
    df, meta = synthetic_weather(-10.0, 24 * 60)
    sp = dict(base_spec('heavy'), zone_model='multinode')
    sim = Simulator(sp, (df, meta), dt='1h', warmup_days=0, init=dict(T_air=20.0, T_mass=20.0))
    r = sim.run()
    z = sim.zone
    H_inf = float(np.mean(sim.rho)) * 1020.0 * sim.ela * 1e-3
    UA = z.H_tr_w + sim.bnd['H_wall'] + sim.bnd['H_roof'] + sim.bnd['H_ground'] + H_inf
    got = r['ts']['Q_hvac_sens_kW'].values[-24:].mean() * 1000
    res['steady_state'] = dict(expected_W=UA * 30, simulated_W=got, rel_error=abs(got - UA * 30) / (UA * 30),
                               n_nodes=z.n, ledger_closure=r['ledger']['closure_rel'])
    print(res['steady_state'])
    # 2. periodic wall
    wall = [(0.2, 1.13, 2000.0, 1000.0)]
    Y = iso13786_Y12(wall, 0.13, 0.04)
    Yd = ladder_Y12(wall, 0.13, 0.04)
    U = 1 / (0.13 + 0.2 / 1.13 + 0.04)
    lagY = float(abs(np.angle(Y)) / (2 * np.pi) * 24)
    lagD = float(abs(np.angle(Yd)) / (2 * np.pi) * 24)
    fd = dict(U=U, iso13786_abs=abs(Y), iso13786_lag_h=lagY, ladder_abs=abs(Yd), ladder_lag_h=lagD,
              abs_rel_err=abs(abs(Yd) - abs(Y)) / abs(Y), lag_err_h=lagD - lagY)
    print('freq', fd)
    td = {}
    for dt in ('1h', '15min'):
        m = int(pd.Timedelta('1h') / pd.Timedelta(dt))
        nh = 24 * 30
        t = np.arange(nh * m) / m
        Tout = 5.0 + 10.0 * np.sin(2 * np.pi * t / 24.0)
        df, meta = synthetic_weather(Tout, nh, freq=dt)
        spec = dict(base_spec('heavy'), zone_model='multinode', window_wwr=0.0, infiltration_ach=1e-6)
        spec['geometry'] = dict(spec['geometry'], foundation='adiabatic', floor_U=0.0, roof_exposed=False,
                                constructions=dict(wall=wall))
        sim = Simulator(spec, (df, meta), dt=dt, warmup_days=0, init=dict(T_air=20.0, T_mass=20.0))
        sim.hsp = [20.0] * sim.n
        sim.csp = [20.000001] * sim.n
        out = sim.run()
        q = out['ts']['Q_hvac_sens_kW'].values * 1000.0
        A = sum(sim.geo['opaque'].values())
        last = slice(len(q) - 24 * 10 * m, len(q))
        qq = q[last]
        tt = (np.arange(len(q)) / m)[last]
        # least-squares fit q = c0 + a sin + b cos  (24 h harmonic); expected flux into room = -(heating)
        X = np.c_[np.ones_like(tt), np.sin(2 * np.pi * tt / 24), np.cos(2 * np.pi * tt / 24)]
        c0, a, b = np.linalg.lstsq(X, qq, rcond=None)[0]
        # heating = -(wall flux into room); wall flux = A U (Tmean - 20) + A Y * 10 sin(...) (phase of Y)
        amp = np.hypot(a, b) / (A * 10.0)
        phase = np.arctan2(b, a)                  # q ~ amp sin(wt + phase)
        lag_h = float(((np.pi - phase) % (2 * np.pi)) / (2 * np.pi) * 24) if True else None
        # time-domain shift is relative to the step-mean convention: the q of step k is the mean over the step
        td[dt] = dict(mean_W=float(c0), expected_mean_W=float(-A * U * (5.0 - 20.0)), amp_per_m2K=float(amp),
                      expected_amp_per_m2K=abs(Y), lag_h=lag_h, expected_lag_h=fd['iso13786_lag_h'],
                      ledger_closure=out['ledger']['closure_rel'])
        print(dt, td[dt])
    res['periodic_wall'] = dict(frequency_domain=fd, time_domain=td,
                                note='lag of the time-domain fit includes half a step of the step-mean convention')
    # 3. BESTEST 600 / 900 with layers
    df, meta = read_epw(cb.EPW)
    meta = dict(meta, tz='Etc/GMT+7')
    LAY = {'600': dict(wall=[(0.012, 0.16, 950, 840), (0.066, 0.04, 12, 840), (0.009, 0.14, 530, 900)],
                       roof=[(0.010, 0.16, 950, 840), (0.1118, 0.04, 12, 840), (0.019, 0.14, 530, 900)],
                       floor=[(0.025, 0.14, 650, 1200), (1.003, 0.04, 10, 1)]),
           '900': dict(wall=[(0.100, 0.51, 1400, 1000), (0.0615, 0.04, 10, 1400), (0.009, 0.14, 530, 900)],
                       roof=[(0.010, 0.16, 950, 840), (0.1118, 0.04, 12, 840), (0.019, 0.14, 530, 900)],
                       floor=[(0.080, 1.13, 1400, 1000), (1.007, 0.04, 10, 1)])}
    bt = {}
    for case in ('600', '900', '600FF', '900FF'):
        spec = cb.spec_for(case)
        spec['zone_model'] = 'multinode'
        spec['thermal_mass'] = 'very_light'
        spec['geometry'] = dict(spec['geometry'], constructions=LAY[case[:3]], solar_floor_frac=0.9,
                                internal_mass_kJ_m2K=0.0)
        orig = cb.spec_for
        cb.spec_for = lambda c, s=spec: s
        r = cb.run_case(case, df, meta)
        cb.spec_for = orig
        rng = cb.RANGES[case]
        bt[case] = {k: dict(sim=round(r[k], 3), lo=v[0], hi=v[1], inside=bool(v[0] <= r[k] <= v[1])) for k, v in rng.items()}
        print(case, bt[case])
    res['bestest_multinode'] = bt
    write('multinode.json', dict(what='multi-node zone checks: steady state, ISO 13786 periodic wall, BESTEST', **res))


if __name__ == '__main__':
    main()
