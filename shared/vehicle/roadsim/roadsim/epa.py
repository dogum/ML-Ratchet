"""Emulate EPA chassis-dynamometer procedures with roadsim (roadsim.epa; moved here from shared/vehicle/checks/
epa_procedures.py on 14 Nov 2026, which keeps working as a thin wrapper over this module).

Stable interface: every procedure takes `options=` (a preset name, an options dict, or None for physics v1; omitted =
roadsim's default physics, as simulate()). The check chain's wrapper passes its current options (checks/_opts.py).

    from roadsim import epa
    epa.ftp(spec, options='roadsim-physics-v3')['FE']           # FTP-75 fuel economy (mpg), J1711-corrected
    epa.hot_cycle(spec, 'hwfet', options=...)['FE']; epa.sc03(spec, options=...)['FE']
    epa.ftp(spec, T=epa.T_COLD, options=...)                     # 20 F cold FTP

FTP-75: cold-start UDDS (bags 1+2) after a soak at 24 C, 10-minute hot soak, hot-start bag 3 (first 505 s of UDDS).
        FE = 1 / (0.43 (g1+g2)/(d1+d2) + 0.57 (g3+g2)/(d3+d2))  [40 CFR 600.113]
HWFET, US06: hot start (preceded by a preconditioning run).
Hybrids: test starts at the SOC reached after preconditioning runs; residual net battery energy converted to fuel with a
          fixed 1/(0.33 x 0.90) equivalence (SAE J1711-style correction).
BEV: DC battery energy over a warm cycle, divided by an assumed 0.88 wall-to-battery charging efficiency to compare
     with EPA's AC recharge energy (SAE J1634).  HVAC off for FTP/HWFET/US06 (as on the test).

Procedure corrections of 13 Nov 2026 (applied to every physics version; logged in roadsim/CALIBRATION.md):
- 20 F cold FTP (40 CFR 1066.710(c)): heat on, A/C off, outside air ((c)(1)(i)(D)), full automatic climate control
  at 72 F ((c)(2); the D3 cold tests are logged "HVAC ON-AUTO-72F"). Before: HVAC off. The vehicle is soaked 12-36 h at
  -7 C ((e)), so the tyres start at the soak temperature (before: warm tyres).
- Every soak (FTP 10-minute hot soak, SC03 soak) cools every thermal state RoadSim has: coolant (and oil, coupled),
  transmission, axle oil, tyres, PTC loop, v1 tyre warm-up state (before: coolant and transmission only).
- SC03 (40 CFR 1066.835): 35 C, 850 W/m2, preconditioning drive over the first 505 s of the UDDS ((c)(5)), 9-11 min
  soak with the engine off ((c)(6)), A/C "maximum": recirculated air, automatic 72 F ((c)(4)). Cabin air and interior
  at 45 C at the start of SC03 (D3 dev SC03 tests: 44-46 C). Before: no preconditioning drive, cabin 50 C, hot engine.
- Options: every simulate() call takes the procedure's `options` argument (the check chain passes checks/_opts.py's).
"""
import math, copy
import numpy as np

from .sim import simulate, get_vehicle
from .trips import make_trip, cycle
from .constants import MILE, GAL, FUELS, EPA_KWH_PER_GAL, KWH
from . import thermal as _thermal
from .options import resolve as _resolve, DEFAULT_PHYSICS

DEFAULT = object()      # options not given: roadsim's default physics (as simulate())


def _o(options):
    return DEFAULT_PHYSICS if options is DEFAULT else options


def merged(extra=None, options=DEFAULT):
    """options plus `extra` (per-test procedure settings), as an options argument for simulate()."""
    o = _o(options)
    if not extra:
        return o
    if o is None:
        return dict(extra)
    if isinstance(o, str):
        return dict(extra, preset=o)
    return dict(o, **extra)

T_TEST = 23.9
T_COLD = -6.7
T_SET_72F = 22.2
COLD_HVAC = {'hvac_T_set': T_SET_72F, 'hvac_recirc_heat': 0.0}          # 1066.710(c)(2) auto 72 F, (c)(1)(i)(D) outside air
SC03_HVAC = {'hvac_T_set': T_SET_72F, 'hvac_recirc_cool': 1.0}          # 1066.835(c)(4) A/C maximum: recirculation
SC03_CABIN_SOAK_C = 45.0   # cabin air at the end of the SC03 soak: D3 dev SC03 tests start at 44.2 (CX9), 45.5 (Accord LX),
                           # 46.0 C (Camry XLE). The v1 cabin cannot reach it in a simulated 10-min soak (32 C), so the
                           # measured value is the procedure input for every physics version (interior mass at the same value)
ETA_CHARGE = 0.88
ETA_BATT_EQ = 0.33 * 0.90
RHO_REF = 1.1767


def test_spec(spec, hvac=False, T_set=None):
    s = copy.deepcopy(spec)
    h = s.setdefault('aux', {}).setdefault('HVAC', {})
    h['enabled'] = hvac
    if T_set is not None:
        h['T_heat_set'] = T_set; h['T_cool_set'] = T_set
    return s


def _sim(spec, trip, extra=None, options=DEFAULT, **kw):
    return simulate(spec, trip, options=merged(extra, options), **kw)


def soak(spec, fs, T, secs, extra=None, options=DEFAULT):
    """state after `secs` s parked with the engine off at ambient T (no sun on the powertrain): each thermal state of the
    run decays by its own Newton law (coolant + oil coupled exactly). The cabin is left to the caller."""
    from scipy.linalg import expm
    veh = get_vehicle(spec)
    opt = _resolve(merged(extra, options), spec)
    P = dict(_thermal.PARAMS); P.update(opt.get('thermal_params') or {})
    fs = dict(fs)
    mr = veh.mass / 1600.0
    e = veh.engine
    if e is not None and fs.get('engine') is not None:
        if 'oil' in fs:
            Cc = e.C_th; Co = P['oil_C0_J_K'] + P['oil_C_J_K_per_L'] * e.disp
            UA = P['oil_UA_W_K_per_L'] * e.disp; hAo = P['oil_hA_W_K_per_L'] * e.disp
            A = np.array([[-(e.hA + UA) / Cc, UA / Cc], [UA / Co, -(UA + hAo) / Co]])
            x = expm(A * secs) @ np.array([fs['engine'] - T, fs['oil'] - T])
            fs['engine'] = T + float(x[0]); fs['oil'] = T + float(x[1])
        else:
            fs['engine'] = T + (fs['engine'] - T) * math.exp(-secs * e.hA / e.C_th)
    if fs.get('driveline') is not None:
        fs['driveline'] = T + (fs['driveline'] - T) * math.exp(-secs * 6.0 / 25e3)
    if 'axle' in fs:
        C = P['axle_C_J_K'] * mr ** 0.8
        fs['axle'] = T + (fs['axle'] - T) * math.exp(-secs * P['axle_hA_W_K'] / C)
    tau_ty = P['tire_C_J_K'] * mr ** 0.85 / (P['tire_h0'] * P['tire_area_m2'] * mr ** 0.6)
    if 'tire_C' in fs:
        fs['tire_C'] = T + (fs['tire_C'] - T) * math.exp(-secs / tau_ty)
    fs['tire_warm'] = fs.get('tire_warm', 0.0) * math.exp(-secs / tau_ty)
    if 'ptc_loop' in fs:
        C = P['ptc_loop_C_J_K'] * (veh.cabin.C / 9.0e4) ** 0.5
        fs['ptc_loop'] = T + (fs['ptc_loop'] - T) * math.exp(-secs * P['ptc_loop_hA_W_K'] / C)
    fs['eng_on'] = False
    return fs


def cabin_soak(spec, fs, T, secs, solar, extra=None, options=DEFAULT):
    """cabin (air and interior) after `secs` s parked with the HVAC off under `solar` W/m2, simulated with the run's own
    cabin model (a standstill trip; only the cabin states are kept)."""
    s = test_spec(spec, hvac=False)
    n = int(secs) + 1
    o = _sim(s, make_trip(np.zeros(n), T_amb=T, rho_air=RHO_REF, solar=solar), extra, options, init_temps=dict(fs),
             init_soc=fs.get('soc'), record=False, check=False)
    f = o['final_state']
    out = {'cabin': f['cabin']}
    if 'cabin_mass' in f:
        out['cabin_mass'] = f['cabin_mass']
    return out


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


def precondition(spec, name, strategy=None, n=2, init=None, T=T_TEST, extra=None, solar=0.0, soc0=None, options=DEFAULT):
    """run the cycle n times warm to settle SOC/temperatures; returns final state. soc0: start SOC (None = the
    simulator's default; plug-in charge-sustaining tests pass their CS level)."""
    st = init or dict(engine=88.0, cabin=T, battery=T, tire_warm=1.0, driveline=80.0)
    soc = soc0
    for _ in range(n):
        o = _sim(spec, trip_for(name, T, solar=solar), extra, options, strategy=strategy, init_soc=soc, init_temps=st, record=False)
        st = o['final_state']; soc = st['soc']
    return st


def ftp(spec, strategy=None, T=T_TEST, cold=True, soc0=None, options=DEFAULT):
    """FTP-75. At T <= 0 C this is the cold FTP (1066.710): heater on in automatic 72 F with outside air, soaked tyres."""
    veh = get_vehicle(spec)
    fuel = veh.fuel
    cold_test = T <= 0.0
    extra = None
    if cold_test:
        spec = test_spec(spec, hvac=True, T_set=T_SET_72F)
        extra = COLD_HVAC
    init = dict(engine=T, cabin=T, battery=T, tire_warm=0.0 if cold_test else 1.0)
    soc_given, soc0 = soc0, None
    if veh.battery is not None:
        stp = precondition(spec, 'udds', strategy, n=2, T=T, extra=extra, soc0=soc_given, options=options)
        soc0 = stp['soc'] if soc_given is None else soc_given     # a given soc0 (plug-in CS test) is also the test's start SOC
    if not cold:
        init = dict(engine=88.0, cabin=T, battery=T, tire_warm=1.0, driveline=80.0)
    o1 = _sim(spec, trip_for('udds', T), extra, options, strategy=strategy, init_soc=soc0, init_temps=init)
    f1, d1 = _bag(o1, 0, 505); f2, d2 = _bag(o1, 505, 1369)
    b1 = _batt_out_J(o1, 0, 505, veh); b2 = _batt_out_J(o1, 505, 1369, veh)
    fs = soak(spec, o1['final_state'], T, 600, extra, options)
    v3 = cycle('udds')[:506]
    o3 = _sim(spec, make_trip(v3, T_amb=T, rho_air=RHO_REF), extra, options, strategy=strategy, init_soc=fs['soc'], init_temps=fs)
    f3, d3 = _bag(o3, 0, 505); b3 = _batt_out_J(o3, 0, 505, veh)
    g = [_fuel_equiv_g(f, b, fuel) for f, b in ((f1, b1), (f2, b2), (f3, b3))]
    G = [_gal(x, fuel) for x in g]
    fe = 1.0 / (0.43 * (G[0] + G[1]) / (d1 + d2) + 0.57 * (G[2] + G[1]) / (d3 + d2))
    bags = [d1 / G[0], d2 / G[1], d3 / G[2]]
    led = max(o1['ledger']['closure_rel'], o3['ledger']['closure_rel'])
    trace = max(o1['summary']['trace_rms_mps'], o3['summary']['trace_rms_mps'])
    return dict(FE=fe, bags=bags, closure=led, trace_rms=trace, trace_max=max(o1['summary']['trace_max_mps'], o3['summary']['trace_max_mps']),
                soc_net=[b1 + b2 + b3], raw_fuel_g=f1 + f2 + f3)


def hot_cycle(spec, name, strategy=None, T=T_TEST, soc0=None, options=DEFAULT):
    veh = get_vehicle(spec)
    st = precondition(spec, name, strategy, n=2 if veh.battery is not None else 1, T=T, soc0=soc0, options=options)
    soc = (st['soc'] if soc0 is None else soc0) if veh.battery is not None else None
    o = _sim(spec, trip_for(name, T), None, options, strategy=strategy, init_soc=soc, init_temps=st)
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


def sc03(spec, strategy=None, return_run=False, soc0=None, options=DEFAULT):
    """SC03 (1066.835): 35 C, 850 W/m2, A/C maximum (recirculation, automatic 72 F). Preconditioning drive over the first
    505 s of the UDDS (A/C on, lamps on), then a 10-minute soak with the engine off under the lamps, then SC03. Hybrids
    first settle their SOC over two SC03 runs (as before)."""
    s = test_spec(spec, hvac=True, T_set=T_SET_72F)
    veh = get_vehicle(s)
    T = 35.0; SOL = 850.0
    st = dict(engine=88.0, cabin=T, battery=T, tire_warm=1.0, driveline=80.0)
    soc = None
    if veh.battery is not None:
        stp = precondition(s, 'sc03', strategy, n=2, init=dict(st, cabin=24.0), T=T, extra=SC03_HVAC, solar=SOL, soc0=soc0,
                           options=options)
        soc = stp['soc'] if soc0 is None else soc0; st = dict(stp, cabin=T)
        st.pop('cabin_mass', None)
    op = _sim(s, make_trip(cycle('udds')[:506], T_amb=T, rho_air=RHO_REF, solar=SOL), SC03_HVAC, options, strategy=strategy,
              init_soc=soc, init_temps=st, record=False)
    fs = soak(s, op['final_state'], T, 600, SC03_HVAC, options)
    fs['cabin'] = fs['cabin_mass'] = SC03_CABIN_SOAK_C
    o = _sim(s, make_trip('sc03', T_amb=T, rho_air=RHO_REF, solar=SOL), SC03_HVAC, options, strategy=strategy,
             init_soc=fs.get('soc') if veh.battery is not None else None, init_temps=fs)
    N = len(o['v'])
    f, d = _bag(o, 0, N - 1); b = _batt_out_J(o, 0, N - 1, veh)
    r = dict(FE=d / _gal(_fuel_equiv_g(f, b, veh.fuel), veh.fuel), closure=o['ledger']['closure_rel'],
             hvac_W=float(np.mean(o['hvac_power'][1:])), cabin_start_C=float(fs['cabin']),
             cabin_C_120_300_595=[float(o['T_cabin'][i]) for i in (120, 300, 595)])
    if return_run:
        r['o'] = o
    return r
