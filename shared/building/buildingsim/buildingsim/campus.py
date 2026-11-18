"""Physics versions 1.1 and 1.2 (opt-in; the default is 1.0). Plan and results: shared/building/NOTEBOOK.md, 17 Nov 2026.

    simulate(..., options={'physics': '1.1'})      # or the environment variable BUILDINGSIM_PHYSICS=1.1

1.1  campus and district thermal plants (buildings with a district plant only):
     - loop operation: hot-water heating loops run while the 7-day running-mean outdoor temperature is below
       T_hw_off; chilled-water loops while it is above T_chw_off (24x7 uses keep it on: process cooling); steam loops
       run all year;
     - a constant loop base while a loop runs (distribution and standby losses, process steam and cooling),
       q_steam / q_hw / q_chw in W per m2 of floor, booked as end uses 'plant_heating_district' and
       'plant_cooling_district';
     - summer reheat: VAV with district heat reheats a share f_rh of the minimum airflow from supply-air temperature
       to the heating setpoint while the air handler cools and the heat loop runs (in the time loop: the heat enters
       the zone and is removed again with chilled water).
1.2  1.1 + weather-insensitive electricity (non-residential uses only):
     - a share s_on of scheduled interior lighting and plug energy runs flat around the clock (energy-preserving);
     - a daily activity multiplier on occupancy, lighting and plug: lognormal AR(1), log-sd sigma_d, lag-1 rho_d,
       mean-preserving, seeded by options['seed'] (default 0).

1.3  1.2 + the heat-pump signature (residential heat_pump_air / heat_pump_ground only):
     - emitter store: heat-pump heating electricity passes a first-order store (floor slab, buffer tank), time
       constant tau_e h, energy-preserving over the run;
     - heating limit: no space heating while the 24-h running-mean outdoor temperature is above T_limit;
     - field COP factor f_cop on air-source compressor power (installed / rated).
1.4  1.3 + generator intensity: generate_buildings(..., physics='1.4') multiplies each generated building's per-area
     intensities by a factor per use type (INTENSITY in data/physics.json; the W1 mechanism of phase2).

The parameters are fitted on dev data only (BDG2 dev, RECS / CBECS prior and dev, HEAPO dev; tools/calibrate_*.py)
and stored in data/physics.json. options={'campus': {...}} overrides them (calibration only).
"""
import json
import os

import numpy as np
import pandas as pd

PHYSICS_VERSIONS = ('1.0', '1.1', '1.2', '1.3', '1.4')
DEFAULT_PHYSICS = '1.0'
_PFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'physics.json')
_P = None

#: Uses whose chilled-water loop never locks out (process cooling around the clock).
ALWAYS_COOL_USES = ('healthcare', 'lab')


def resolve(options):
    """The physics version of a run: options['physics'] > $BUILDINGSIM_PHYSICS > '1.0'."""
    v = (options or {}).get('physics')
    if v is None:
        v = os.environ.get('BUILDINGSIM_PHYSICS') or DEFAULT_PHYSICS
    v = str(v)
    if v not in PHYSICS_VERSIONS:
        raise ValueError(f'physics {v!r}: one of {PHYSICS_VERSIONS}')
    return v


def at_least(v, w):
    return tuple(int(x) for x in v.split('.')) >= tuple(int(x) for x in w.split('.'))


def params(overrides=None):
    """Fitted parameters (data/physics.json) with optional overrides."""
    global _P
    if _P is None:
        with open(_PFILE) as f:
            _P = json.load(f)
    p = {k: v for k, v in _P.items() if not k.startswith('_')}
    p.update(overrides or {})
    return p


# ------------------------------------------------------------------------------------------------ 1.1 district plants
def district_kinds(spec):
    """(has district heat, heat meter 'steam'|'hot_water', has district chilled water) from a filled spec."""
    h = spec['hvac']
    sysn = spec['system']
    dhw = spec.get('dhw') if isinstance(spec.get('dhw'), dict) else {}
    heat = sysn == 'district_steam_chw' or h.get('heat_source') == 'district' or dhw.get('fuel') == 'district'
    kind = 'hot_water' if h.get('district_heat') == 'hot_water' else 'steam'
    cool = sysn == 'district_steam_chw'
    return heat, kind, cool


def setup_district(sim, P):
    """Loop-enable arrays (1 = loop running) for the heat and chilled-water loops, from the 7-day running mean of the
    outdoor temperature over the simulated period."""
    s = sim.spec
    heat, kind, cool = district_kinds(s)
    n = sim.n
    rm = pd.Series(sim.wx['T_out']).rolling(max(1, int(round(7 * 86400 / sim.dt))), min_periods=1).mean().values
    if kind == 'hot_water' and P.get('T_hw_off') is not None:
        heat_on = (rm < float(P['T_hw_off'])).astype(float)
    else:
        heat_on = np.ones(n)
    if P.get('T_chw_off') is not None and s['primary_use'] not in ALWAYS_COOL_USES:
        cool_on = (rm > float(P['T_chw_off'])).astype(float)
    else:
        cool_on = np.ones(n)
    if not heat:
        heat_on = np.zeros(n)
    if not cool:
        cool_on = np.zeros(n)
    A = sim.geo['A_f']
    q_heat = float(P['q_hw'] if kind == 'hot_water' else P['q_steam'])
    return dict(heat=heat, kind=kind, cool=cool, heat_on=heat_on, cool_on=cool_on,
                base_heat_W=q_heat * A * heat_on, base_cool_W=float(P['q_chw']) * A * cool_on,
                f_rh=float(P['f_rh']), heat_on_list=heat_on.tolist())


# ------------------------------------------------------------------------------------------------ 1.2 electricity
def day_index(cal):
    """0, 1, 2, ... per local calendar day of the simulated period."""
    doy = np.asarray(cal['doy'])
    return np.concatenate([[0], np.cumsum(doy[1:] != doy[:-1])]).astype(int)


def activity(cal, sigma, rho, seed):
    """Daily lognormal AR(1) multiplier (stationary log-sd sigma, lag-1 rho), mean 1, one value per step."""
    d = day_index(cal)
    nd = int(d[-1]) + 1 if len(d) else 0
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(nd)
    x = np.empty(nd)
    if nd:
        x[0] = z[0]
        a = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
        for j in range(1, nd):
            x[j] = rho * x[j - 1] + a * z[j]
    m = np.exp(sigma * x - 0.5 * sigma * sigma)
    return m[d]


def electricity(sc, light, cal, P, seed):
    """1.2 schedules: always-on share s_on of lighting and plug (energy-preserving over the run), then the daily
    activity multiplier on occupancy, lighting and plug. Returns (sched dict copy, light)."""
    s_on = float(P['s_on'])
    sc = dict(sc)
    light = np.asarray(light, float)
    plug = np.asarray(sc['plug'], float)
    light = (1.0 - s_on) * light + s_on * light.mean()
    plug = (1.0 - s_on) * plug + s_on * plug.mean()
    if float(P['sigma_d']) > 0:
        m = activity(cal, float(P['sigma_d']), float(P['rho_d']), seed)
        sc['occ'] = np.asarray(sc['occ'], float) * m
        light = light * m
        plug = plug * m
    sc['plug'] = plug
    return sc, light


# ------------------------------------------------------------------------------------------------ 1.3 heat pumps
def heating_limit(sim, P):
    """Space heating off (heating setpoint -100 C) while the 24-h running-mean outdoor temperature is above T_limit."""
    if P.get('T_limit') is None:
        return
    rm = pd.Series(sim.wx['T_out']).rolling(max(1, int(round(86400 / sim.dt))), min_periods=1).mean().values
    off = rm > float(P['T_limit'])
    h = np.where(off, -100.0, np.asarray(sim.hsp_sched, float))
    sim.hsp_sched = h
    sim.hsp = h.tolist()


def field_cop(sim, P):
    """Installed / rated COP of an air-source heat pump's heating coil (compressor power / f_cop)."""
    coil = (getattr(sim.system, 'heat', None) or {}).get('coil') if isinstance(getattr(sim.system, 'heat', None),
                                                                              dict) else None
    if coil is not None and hasattr(coil, 'f_cop') and sim.spec['system'] == 'heat_pump_air':
        coil.f_cop = float(P['f_cop'])


def emitter_store(x, dt, tau_h):
    """First-order store between the heat pump and the zone: y_k = y_(k-1) + a (x_k - y_(k-1)), a = 1 - exp(-dt/tau),
    rescaled so the run's total is unchanged (energy-preserving). tau_h <= 0 returns x."""
    x = np.asarray(x, float)
    if tau_h <= 0 or len(x) == 0:
        return x
    a = 1.0 - np.exp(-dt / (tau_h * 3600.0))
    from scipy.signal import lfilter
    y = lfilter([a], [1.0, -(1.0 - a)], x, zi=[(1.0 - a) * x[0]])[0]
    s = y.sum()
    return y * (x.sum() / s) if s > 0 else y
