"""Emulate EPA chassis-dynamometer procedures with roadsim.

FTP-75: cold-start UDDS (bags 1+2) after a soak at 24 C, 10-minute hot soak, hot-start bag 3 (first 505 s of UDDS).
        FE = 1 / (0.43 (g1+g2)/(d1+d2) + 0.57 (g3+g2)/(d3+d2))  [40 CFR 600.113]
HWFET, US06: hot start (preceded by a preconditioning run).
Hybrids: test starts at the SOC reached after preconditioning runs; residual net battery energy converted to fuel with a
          fixed 1/(0.33 x 0.90) equivalence (SAE J1711-style correction).
BEV: DC battery energy over a warm cycle, divided by an assumed 0.88 wall-to-battery charging efficiency to compare
     with EPA's AC recharge energy (SAE J1634).  HVAC off for FTP/HWFET/US06 (as on the test).
"""
import os, sys, math, copy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'roadsim'))
from roadsim import simulate, make_trip, cycle
from roadsim.sim import get_vehicle
from roadsim.constants import MILE, GAL, FUELS, EPA_KWH_PER_GAL, KWH

T_TEST = 23.9
ETA_CHARGE = 0.88
ETA_BATT_EQ = 0.33 * 0.90
RHO_REF = 1.1767


def test_spec(spec, hvac=False):
    s = copy.deepcopy(spec)
    s.setdefault('aux', {}).setdefault('HVAC', {})['enabled'] = hvac
    return s


def _gal(fuel_g, fuel='gasoline'):
    return fuel_g / FUELS[fuel]['density'] / GAL


def _bag(out, i0, i1):
    """fuel (g) and distance (mi) for samples (i0, i1]  (interval values are stored at the end index)."""
    fr = out['fuel_rate'][i0 + 1:i1 + 1].astype(float)
    v = out['v'].astype(float)
    d = np.sum(0.5 * (v[i0:i1] + v[i0 + 1:i1 + 1])) / MILE
    eb = np.sum(out['batt_current'][i0 + 1:i1 + 1].astype(float) * 0)  # placeholder
    return fr.sum(), d


def _batt_out_J(out, i0, i1, veh):
    """net chemical energy out of the battery over (i0, i1] (J)."""
    if veh.battery is None:
        return 0.0
    I = out['batt_current'][i0 + 1:i1 + 1].astype(float)
    soc = out['soc'][i0:i1].astype(float)
    return float(np.sum(veh.battery.ocv.vec(soc) * I))


def _fuel_equiv_g(fuel_g, batt_J, fuel='gasoline'):
    return fuel_g + batt_J / ETA_BATT_EQ / FUELS[fuel]['lhv'] * 1e3


def trip_for(name, T=T_TEST, **kw):
    return make_trip(name, T_amb=T, rho_air=RHO_REF, **kw)


def precondition(spec, name, strategy=None, n=2, init=None, T=T_TEST):
    """run the cycle n times warm to settle SOC/temperatures; returns final state."""
    st = init or dict(engine=88.0, cabin=T, battery=T, tire_warm=1.0, driveline=80.0)
    soc = None
    for _ in range(n):
        o = simulate(spec, trip_for(name, T), strategy=strategy, init_soc=soc, init_temps=st, record=False)
        st = o['final_state']; soc = st['soc']
    return st


def ftp(spec, strategy=None, T=T_TEST, cold=True):
    veh = get_vehicle(spec)
    fuel = veh.fuel
    init = dict(engine=T, cabin=T, battery=T, tire_warm=1.0)
    soc0 = None
    if veh.battery is not None:
        stp = precondition(spec, 'udds', strategy, n=2, T=T)
        soc0 = stp['soc']
    if not cold:
        init = dict(engine=88.0, cabin=T, battery=T, tire_warm=1.0, driveline=80.0)
    o1 = simulate(spec, trip_for('udds', T), strategy=strategy, init_soc=soc0, init_temps=init)
    f1, d1 = _bag(o1, 0, 505); f2, d2 = _bag(o1, 505, 1369)
    b1 = _batt_out_J(o1, 0, 505, veh); b2 = _batt_out_J(o1, 505, 1369, veh)
    fs = o1['final_state']
    if veh.engine is not None:
        e = veh.engine
        fs = dict(fs, engine=T + (fs['engine'] - T) * math.exp(-600 * e.hA / e.C_th),
                  driveline=T + (fs['driveline'] - T) * math.exp(-600 * 6.0 / 25e3))
    v3 = cycle('udds')[:506]
    o3 = simulate(spec, make_trip(v3, T_amb=T, rho_air=RHO_REF), strategy=strategy, init_soc=fs['soc'], init_temps=fs)
    f3, d3 = _bag(o3, 0, 505); b3 = _batt_out_J(o3, 0, 505, veh)
    g = [_fuel_equiv_g(f, b, fuel) for f, b in ((f1, b1), (f2, b2), (f3, b3))]
    G = [_gal(x, fuel) for x in g]
    fe = 1.0 / (0.43 * (G[0] + G[1]) / (d1 + d2) + 0.57 * (G[2] + G[1]) / (d3 + d2))
    bags = [d1 / G[0], d2 / G[1], d3 / G[2]]
    led = max(o1['ledger']['closure_rel'], o3['ledger']['closure_rel'])
    trace = max(o1['summary']['trace_rms_mps'], o3['summary']['trace_rms_mps'])
    return dict(FE=fe, bags=bags, closure=led, trace_rms=trace, trace_max=max(o1['summary']['trace_max_mps'], o3['summary']['trace_max_mps']),
                soc_net=[b1 + b2 + b3], raw_fuel_g=f1 + f2 + f3)


def hot_cycle(spec, name, strategy=None, T=T_TEST):
    veh = get_vehicle(spec)
    st = precondition(spec, name, strategy, n=2 if veh.battery is not None else 1, T=T)
    o = simulate(spec, trip_for(name, T), strategy=strategy, init_soc=st['soc'] if veh.battery is not None else None, init_temps=st)
    N = len(o['v'])
    if veh.arch == 'bev':
        E_dc = o['ledger']['batt_chem_out'] * 1e3       # J
        d = o['summary']['distance_m'] / MILE
        kwh_mi_ac = E_dc / KWH / d / ETA_CHARGE
        return dict(kWh_100mi_ac=100 * kwh_mi_ac, MPGe=EPA_KWH_PER_GAL / kwh_mi_ac, closure=o['ledger']['closure_rel'],
                    trace_rms=o['summary']['trace_rms_mps'], trace_max=o['summary']['trace_max_mps'], o=o)
    f, d = _bag(o, 0, N - 1)
    b = _batt_out_J(o, 0, N - 1, veh)
    g = _fuel_equiv_g(f, b, veh.fuel)
    return dict(FE=d / _gal(g, veh.fuel), closure=o['ledger']['closure_rel'], trace_rms=o['summary']['trace_rms_mps'],
                trace_max=o['summary']['trace_max_mps'], batt_J=b, o=o)


def sc03(spec, strategy=None):
    """SC03: 35 C, 850 W/m2 solar, A/C on, after a 10-minute soak (cabin hot-soaked); hot engine."""
    s = test_spec(spec, hvac=True)
    veh = get_vehicle(s)
    T = 35.0
    st = dict(engine=88.0, cabin=50.0, battery=T, tire_warm=1.0, driveline=80.0)
    soc = None
    if veh.battery is not None:
        stp = precondition(s, 'sc03', strategy, n=2, init=dict(st, cabin=24.0), T=T); soc = stp['soc']
    o = simulate(s, make_trip('sc03', T_amb=T, rho_air=RHO_REF, solar=850.0), strategy=strategy, init_soc=soc, init_temps=st)
    N = len(o['v'])
    f, d = _bag(o, 0, N - 1); b = _batt_out_J(o, 0, N - 1, veh)
    return dict(FE=d / _gal(_fuel_equiv_g(f, b, veh.fuel), veh.fuel), closure=o['ledger']['closure_rel'],
                hvac_W=float(np.mean(o['hvac_power'][1:])))
