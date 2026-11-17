"""Calibration wrappers around BuildingSim (Phase 1 code is never edited; these are upstream candidates).

W1 intensity: generated site EUIs are off by use type (Phase 1 generator.json: schools 1.8x, healthcare 1.9x, labs 2.0x,
   food 1.5x, single-family 1.4x too high; lodging 0.66x too low). We scale the per-area intensities that drive both
   base load and HVAC load (lighting, plug, process, cooking, people, outdoor air per m2 and per person, DHW volume)
   by one factor per use type, fitted so the median generated site EUI matches the weighted median of real
   CBECS 2018 / RECS 2020 respondents of the prior + dev splits (calib_intensity.py, never exam).
   Residential homes: the excess is mostly heating (LIMITS 4), so their factor also scales infiltration by sqrt(f).
W2 campus: real BDG2 campus district meters read ~3.5x the generic twins (Phase 1 bdg2_dev_meters.json). A share of
   commercial buildings is converted to campus district service (steam/hot water and/or chilled water) with long
   air-handler hours, higher VAV minimum flow and outdoor air, and a distribution factor (district_eff < 1:
   meter = delivered / eff) drawn around the value fitted on the BDG2 dev twins (calib_campus.py, dev only).
"""
import copy
import json
import os

import numpy as np

from common import ROOT

CAL = os.path.join(ROOT, 'models', 'calibration.json')
# starting values = 1 / Phase 1 generated-vs-survey ratio; replaced by calib_intensity.py output when present
INTENSITY0 = dict(office=1 / 0.91, warehouse=1 / 0.98, retail=1 / 1.11, residential_multi=1 / 1.12, other=1 / 1.17,
                  assembly=1 / 1.26, residential_single=1 / 1.40, food=1 / 1.53, education=1 / 1.81,
                  healthcare=1 / 1.91, lab=1 / 2.0, lodging=1 / 0.66)
CAMPUS0 = dict(eff_chw_median=1.0, eff_heat_median=1.0, eff_sd=0.35, p_campus=0.3)


def load_cal():
    if os.path.exists(CAL):
        c = json.load(open(CAL))
    else:
        c = {}
    c.setdefault('intensity', dict(INTENSITY0))
    c.setdefault('campus', dict(CAMPUS0))
    return c


def apply_intensity(f, fac):
    """f: filled spec (modified in place). fac: multiplicative factor on per-area intensities."""
    ig = f['internal_gains']
    for k in ('lighting', 'plug', 'process', 'cooking', 'people_W_m2', 'exterior_lighting'):
        if ig.get(k) is not None:
            ig[k] = float(ig[k]) * fac
    if ig.get('people') is not None and not f['primary_use'].startswith('residential'):
        pass                                  # 'people' is W per person (heat), not a count
    v = f['ventilation']
    for k in ('L_s_m2', 'L_s_person'):
        if v.get(k) is not None:
            v[k] = float(v[k]) * fac
    d = f.get('dhw')
    if isinstance(d, dict) and d.get('L_per_day') is not None:
        d['L_per_day'] = float(d['L_per_day']) * fac
    if f['primary_use'].startswith('residential'):
        if f.get('ach50') is not None:
            f['ach50'] = float(f['ach50']) * np.sqrt(fac)
        if f.get('infiltration_ach') is not None:
            f['infiltration_ach'] = float(f['infiltration_ach']) * np.sqrt(fac)
    f['_w1_intensity'] = fac
    return f


def make_campus(f, rng, cal, kind=None):
    """Convert a filled commercial spec to campus district service. kind: 'both' | 'heat' | 'cool'."""
    kind = kind or rng.choice(['both', 'heat', 'cool'], p=[0.55, 0.25, 0.20])
    h = f['hvac']
    c = cal['campus']
    if kind == 'both':
        f['system'] = 'district_steam_chw'
        h['district_heat'] = 'hot_water' if rng.random() < 0.3 else 'steam'
    elif kind == 'heat':
        f['system'] = 'vav_chiller_boiler'
        h['heat_source'] = 'district'
        h.pop('cooling_source', None)
    else:
        f['system'] = 'district_steam_chw'          # chilled water metered; heat from the campus too but unmetered
    # one district factor per building (the plant/distribution); heat and chw medians fitted separately on dev
    z = rng.normal(0, c['eff_sd'])
    h['district_eff'] = float(np.clip(c['eff_heat_median'] * np.exp(z), 0.05, 1.0))
    h['_eff_chw'] = float(np.clip(c['eff_chw_median'] * np.exp(z), 0.05, 1.0))
    h['vav_min_frac'] = float(rng.uniform(0.3, 0.6))
    v = f['ventilation']
    v['L_s_m2'] = float(v.get('L_s_m2') or 0.3) * float(rng.uniform(1.0, 2.0))
    v['economizer'] = bool(rng.random() < 0.5)
    sch = f.get('occupancy_schedule')
    if isinstance(sch, dict) and sch.get('type') == 'commercial' and rng.random() < 0.35:
        sch['weekday'] = (0.0, 24.0)               # air handlers run around the clock (labs, residences, hospitals)
        sch['saturday'] = (0.0, 24.0)
        sch['sunday'] = (0.0, 24.0)
    f['_w2_campus'] = kind
    return f, kind


def campus_meters(ts, f):
    """Separate chw distribution factor: the simulator has one district_eff for both; we rescale the chw meter."""
    h = f['hvac']
    if '_eff_chw' in h and 'meter_chilled_water_kW' in ts:
        r = h['district_eff'] / h['_eff_chw']
        ts['meter_chilled_water_kW'] = ts['meter_chilled_water_kW'] * r
    return ts


# ----------------------------------------------------------------------------- W3 (physics v2): always-on base loads
# Twin test on the 52 BDG2 DEV buildings (devtwin_v1 vs dev_bdg2 features, 2017):
#   district heat: real summer level = 0.72 x annual mean vs twin 0.06; normalised heating slope real/twin 0.11
#   chilled water: night/day real 0.74 vs twin 0.45; normalised cooling slope real/twin 0.56
#   electricity  : night/day real 0.77 vs twin 0.56 (implied extra flat load ~0.45 x mean); weather slopes ~0.4 x twin
# Real campus meters carry large temperature-independent flows (distribution losses, reheat, DHW, process, 24/7 air
# handlers and servers) that the single-zone model does not. W3 moves a share phi of each district meter's annual
# energy into a flat base (annual total unchanged, so W2's fitted distribution factor still holds) and adds a flat
# unmodelled electricity load. phi is drawn per building around the dev-fitted value.
W3 = dict(phi_heat=(0.70, 0.15), phi_chw=(0.50, 0.15), elec_base=(0.50, 0.35), p_elec_base=0.85,
          note='dev twin test (devtwin_v1 vs dev_bdg2), see STATUS.md')


def w3_base_loads(meters, rng, commercial=True):
    """meters: channel -> hourly kW (clean simulated). Returns a new dict. Residential: electricity base only, smaller."""
    out = {}
    for ch, x in meters.items():
        x = np.asarray(x, float)
        mu = float(np.nanmean(x))
        if ch in ('dheat', 'dcool') and mu > 0:
            m, s = W3['phi_heat'] if ch == 'dheat' else W3['phi_chw']
            phi = float(np.clip(rng.normal(m, s), 0.05, 0.95))
            x = (1 - phi) * x + phi * mu
        elif ch == 'elec' and mu > 0 and rng.random() < W3['p_elec_base']:
            m, s = W3['elec_base']
            b = float(np.clip(rng.normal(m, s), 0.0, 1.5)) * (1.0 if commercial else 0.3)
            x = x + b * mu
        out[ch] = x
    return out
