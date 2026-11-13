"""Dev-calibrated wrappers around roadsim (roadsim itself is frozen; nothing here edits it). Each is a ratchet rung.

maf : 'fuel_rate_maf' = the fuel an OBD MAF-based estimate reports (VED label model). During fuel cut-off the MAF
      still reads the motoring air flow, so the estimate shows ~idle-level fuel. Dev fit (7 MAF dev cars, 10,273
      decel seconds): f_cut = exp(-1.578) D^0.342 (rpm/1000)^0.226 g/s (resid sd 0.29 in log).
rw  : on-road extra loads the simulator misses (A/C too cheap, real-world hybrid operation, accessories), fitted on
      dev twins (label-matched dev cars on their own logged trips, after maf). Extra power, per second:
        P = c + k_ac * clip(T - 15, 0, 15) + k_cold * clip(5 - T, 0, 25)   [W]
        conv (fuel power) c 1380, k_ac 133, k_cold 19.7  (8 dev cars, 645 trips)
        hybrid (fuel power) c 2831, k_ac 204, k_cold 24.6 (5 dev HEVs, 275 trips)
        bev (battery power) c 246, k_ac 0, k_cold -42.7 (1 dev Leaf, 258 trips), never below -0.8 x the HVAC power
      scaled by cabin size (HVAC UA / 120 W/K) for other classes. PHEV: hybrid terms while the engine runs, bev terms
      otherwise. Not applied to dynamometer tests (lab: A/C off, no accessories beyond the test).
"""
from common import *
from roadsim import simulate
from roadsim.vehicle import fill_spec

CAL = {'maf': dict(a=-1.578, b=0.342, c=0.226),
       'rw': {'ice': (1380.0, 133.3, 19.7), 'hev': (2830.9, 203.9, 24.6), 'bev': (246.0, 0.0, -42.7)}}


def maf_fuel(out, spec):
    D = ((spec.get('engine') or {}).get('displacement_L')) or 2.0
    rpm = np.asarray(out['eng_rpm'], float); f = np.asarray(out['fuel_rate'], float)
    c = CAL['maf']
    fcut = np.exp(c['a']) * D ** c['b'] * np.maximum(rpm / 1000, 0.3) ** c['c']
    return np.where(rpm > 300, np.maximum(f, fcut), f)


def rw_power(T, key, scale):
    c, kac, kc = CAL['rw'][key]
    return scale * (c + kac * np.clip(T - 15, 0, 15) + kc * np.clip(5 - T, 0, 25))


def targets(out, spec, T_sim, dyno=False):
    """-> dict of per-trip targets (MJ): raw / rw x true / maf fuel. Battery power + = discharge."""
    arch = spec['architecture']
    full = fill_spec(spec)
    scale = float(((full.get('aux') or {}).get('HVAC') or {}).get('UA_W_K', 120.0)) / 120.0
    fr = np.nan_to_num(np.asarray(out['fuel_rate'], float))
    fm = maf_fuel(out, spec) if full.get('engine') else fr
    pb = np.nan_to_num(np.asarray(out['batt_current'], float) * np.asarray(out['batt_voltage'], float))
    T = np.asarray(T_sim, float)
    g = arch_group(arch)
    on = np.asarray(out['eng_on'], float) if full.get('engine') else np.zeros(len(T))
    xf = np.zeros(len(T)); xb = np.zeros(len(T))
    if not dyno:
        if g == 'ICE': xf = rw_power(T, 'ice', scale)
        elif g == 'HEV': xf = rw_power(T, 'hev', scale)
        elif g == 'EV': xb = rw_power(T, 'bev', scale)
        else:
            xf = rw_power(T, 'hev', scale) * on; xb = rw_power(T, 'bev', scale) * (1 - on)
        if g in ('EV', 'PHEV'):
            hv = np.nan_to_num(np.asarray(out.get('hvac_power', np.zeros(len(T))), float))
            xb = np.maximum(xb, -0.8 * hv)
    def E(f, b):
        if g in ('ICE', 'HEV'): return float(np.sum(f) * LHV)
        if g == 'EV': return float(np.sum(b) / 1e6)
        return float(np.sum(f) * LHV + np.sum(b) / 1e6)
    xg = xf / (LHV * 1e6)
    return {'E_raw': E(fr, pb), 'E_rw': E(fr + xg, pb + xb), 'E_maf_raw': E(fm, pb), 'E_maf_rw': E(fm + xg, pb + xb)}


def run(spec, trip, flags='none', init_soc=None, init_temps=None, meta=None, strategy=None, record=True, rng=None):
    fl = set(flags.split('+'))
    out = simulate(spec, trip, strategy=strategy, init_soc=init_soc, init_temps=init_temps, record=True, rng=rng)
    if (spec.get('engine') or {}) and out.get('eng_rpm') is not None:
        out['fuel_rate_maf'] = maf_fuel(out, spec) if 'maf' in fl else np.asarray(out['fuel_rate'], float)
    return out


def phev_patch(spec):
    """Rung r5 (dev twin evidence): real dev PHEVs (Volt, Ford Energi) burned no fuel on 95% of trips with >30% displayed
    SOC above 5 C, while simulated generator PHEVs took ~31% of their energy from fuel in that state. Cause in the sim:
    generator PHEV packs limited to ~3C (energy-cell resistance) and power-split MG2 torque limits, so ordinary launches
    start the engine. Patch (hardware rule of thumb for plug-in power cells and traction motors):
    pack power >= 1.2 x traction motor power, pack resistance / 2.5, power-split MG2 torque x 1.8."""
    import copy
    from roadsim.sim import get_vehicle
    from roadsim.vehicle import fill_spec
    if not spec['architecture'].startswith('phev'): return spec
    s = copy.deepcopy(spec); full = fill_spec(s); veh = get_vehicle(s)
    Pt = max(m['max_power_kW'] for m in full['motors'] if m['role'] in ('traction', 'MG2'))
    b = s.setdefault('battery', {})
    b['max_power_kW'] = max(3.0 * full['battery']['capacity_kWh'], 1.2 * Pt)
    b['R0_ohm'] = veh.battery.R0_25 / 2.5
    if spec['architecture'] == 'phev_powersplit':
        M = veh.motors
        mg2 = M.get('MG2') if isinstance(M, dict) else [m for m in M if getattr(m, 'role', '') == 'MG2'][0]
        T0 = float(mg2.t_lim(10.0))
        if T0:
            s['motors'] = [dict(m, max_torque_Nm=1.8 * T0) if m['role'] == 'MG2' else m for m in full['motors']]
    return s
