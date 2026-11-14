"""The one place EnergyNet's training code calls RoadSim (Phase D, 14 Nov 2026).

PHYSICS is pinned explicitly, so a later change of RoadSim's default never moves the model silently. Physics v2 holds
what v1 kept in this folder's wrappers.py: on-road extra loads (options 'onroad_aux_W', road runs only) and the OBD MAF
label model (roadsim.obd.label_fuel). wrappers.py stays as v1 ran it (pinned to v1) so v1 reproduces.

dyno_spec(): chassis-dynamometer emulation as the RoadSim checks do it (epa_vehicles.py): road load from the spec without
tyre-temperature effects (so no on-road load either), HVAC off unless the test needs it (SC03 A/C, 20 F cold FTP heat)."""
import copy
from common import *
from roadsim import simulate
from roadsim import obd

PHYSICS = 'roadsim-physics-v2'
DONOR_MODEL = 'v2'          # the v2 preset's generator donors (hybrid bodies from real hybrid records)


def dyno_spec(spec, hvac=False):
    s = copy.deepcopy(spec)
    s['tire'] = dict(s.get('tire') or {}, K_t=0.0, temp_effects=False)
    h = s.setdefault('aux', {}).setdefault('HVAC', {})
    h['enabled'] = bool(hvac)
    return s


def sim(spec, trip, dyno=False, hvac=False, **kw):
    s = dyno_spec(spec, hvac) if dyno else spec
    return simulate(s, trip, options=PHYSICS, **kw)


def label_fuel(out, spec):
    """per-second fuel (g/s) as VED's MAF-based estimate would show it (upstream label model, with missed enrichment)."""
    return obd.label_fuel(out, spec, 'maf')


def targets(out, spec, T_sim, dyno=False):
    """per-trip energy targets (MJ) from one physics-v2 run:
      E_raw    simulated energy (fuel LHV / battery DC / both)
      E_rw     + v1's dev-fitted on-road loads wrapper (wrappers.rw_power) ON TOP of v2 (rung r4 re-taken; v2 already
               carries its own on-road load, so this tests whether v1's extra still helps)
      E_lab    fuel through the OBD MAF label model (roadsim.obd.label_fuel), battery as is
      E_lab_rw both"""
    import wrappers as W
    arch = spec['architecture']
    from roadsim.vehicle import fill_spec
    full = fill_spec(spec)
    g = arch_group(arch)
    fr = np.nan_to_num(np.asarray(out['fuel_rate'], float))
    fm = np.nan_to_num(label_fuel(out, spec)) if full.get('engine') else fr
    pb = np.nan_to_num(np.asarray(out['batt_current'], float) * np.asarray(out['batt_voltage'], float))
    T = np.asarray(T_sim, float)
    on = np.asarray(out['eng_on'], float) if full.get('engine') else np.zeros(len(T))
    scale = float(((full.get('aux') or {}).get('HVAC') or {}).get('UA_W_K', 120.0)) / 120.0
    xf = np.zeros(len(T)); xb = np.zeros(len(T))
    if not dyno:                         # as v1: every class, scaled by cabin size
        if g == 'ICE': xf = W.rw_power(T, 'ice', scale)
        elif g == 'HEV': xf = W.rw_power(T, 'hev', scale)
        elif g == 'EV': xb = W.rw_power(T, 'bev', scale)
        else: xf = W.rw_power(T, 'hev', scale) * on; xb = W.rw_power(T, 'bev', scale) * (1 - on)
        if g in ('EV', 'PHEV'):
            hv = np.nan_to_num(np.asarray(out.get('hvac_power', np.zeros(len(T))), float))
            xb = np.maximum(xb, -0.8 * hv)
    def E(f, b):
        if g in ('ICE', 'HEV'): return float(np.sum(f) * LHV)
        if g == 'EV': return float(np.sum(b) / 1e6)
        return float(np.sum(f) * LHV + np.sum(b) / 1e6)
    xg = xf / (LHV * 1e6)
    return [E(fr, pb), E(fr + xg, pb + xb), E(fm, pb), E(fm + xg, pb + xb)]
