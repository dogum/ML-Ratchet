"""FaultFinder shared code: paths, the channel mapping (BuildingSim trend <-> BuildingData contract), day tensors
for real and simulated trend logs, the fault-class list and the exam taxonomy mapping.

Channel mapping (documented in MODEL_CARD.md):
  BuildingSim trend      BuildingData contract      FaultFinder raw channel
  OAT                    T_oa                       T_oa
  RAT                    T_ra                       T_ra
  MAT                    T_ma                       T_ma
  SAT                    T_sa                       T_sa
  (VAV reset rule)       T_sa_sp (T_sa_sp_cool/heat for single-zone units)   T_sa_sp
  ZAT                    T_zone                     T_zone
  OA_damper_pct / 100    damper_oa                  damper_oa
  cool_cmd_pct / 100     valve_cc ; RTU: (comp1_status+comp2_status)/2     valve_cc
  heat_cmd_pct / 100     valve_hc                   valve_hc
  fan_speed_pct / 100    fan_sa_speed               fan_speed
  fan_kW                 fan_sa_power_kW ; Nesbitt: x_fan_sa_current_A     fan_power (scaled per day)
  fan_status             fan_sa_status ; else fan_sa_speed > 0.05          fan_on
  occupied               occ                        occ
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORK as ROOT, BLDG  # noqa: E402,F401  (ML Ratchet paths; puts BLDG and BLDG/buildingsim on sys.path)

RAW = ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_zone', 'damper_oa', 'valve_cc', 'valve_hc', 'fan_speed',
       'fan_power', 'fan_on', 'occ']
RI = {c: i for i, c in enumerate(RAW)}
TEMPS = ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_zone']      # measured temperatures (flatline rule applies)
NT = 96                                                   # 15-min bins per local day

# ------------------------------------------------------------------ fault classes (what the network diagnoses)
CLASSES = ['none', 'oa_damper_stuck', 'oa_damper_leak', 'economizer', 'cc_valve_stuck', 'cc_valve_leak',
           'hc_valve_stuck', 'hc_valve_leak', 'reheat_valve_leak', 'cooling_capacity', 'low_airflow',
           'sensor_bias_T_sa', 'sensor_bias_T_oa', 'sensor_bias_T_ma', 'sensor_bias_T_ra', 'zone_setpoint',
           'schedule', 'vav_box_stuck', 'plant_efficiency']
CI = {c: i for i, c in enumerate(CLASSES)}
NC = len(CLASSES)

# BuildingSim fault type (+ sensor) -> class
SIM2CLASS = {'oa_damper_stuck': 'oa_damper_stuck', 'oa_damper_leak': 'oa_damper_leak',
             'economizer_disabled': 'economizer', 'economizer_high_limit': 'economizer',
             'cooling_valve_stuck': 'cc_valve_stuck', 'cooling_valve_leak': 'cc_valve_leak',
             'heating_valve_stuck': 'hc_valve_stuck', 'heating_valve_leak': 'hc_valve_leak',
             'reheat_valve_leak': 'reheat_valve_leak', 'coil_fouling': 'cooling_capacity',
             'refrigerant_charge': 'cooling_capacity', 'condenser_fouling': 'cooling_capacity',
             'chiller_fouling': 'plant_efficiency', 'boiler_fouling': 'plant_efficiency',
             'low_airflow': 'low_airflow', 'sensor_bias:T_sup': 'sensor_bias_T_sa',
             'sat_setpoint_offset': 'sensor_bias_T_sa', 'sensor_bias:T_out': 'sensor_bias_T_oa',
             'sensor_bias:T_mix': 'sensor_bias_T_ma', 'sensor_bias:T_ret': 'sensor_bias_T_ra',
             'sensor_bias:T_zone': 'zone_setpoint', 'setpoint_override': 'zone_setpoint',
             'schedule_override': 'schedule', 'vav_box_stuck': 'vav_box_stuck'}

# exam taxonomy (bd.taxonomy()) -> class; None = "other" (BuildingSim cannot simulate it: graded for detection only)
EXAM2CLASS = {'none': 'none', 'oa_damper_stuck': 'oa_damper_stuck', 'oa_damper_leak': 'oa_damper_leak',
              'cc_valve_stuck': 'cc_valve_stuck', 'cc_valve_leak': 'cc_valve_leak',
              'hc_valve_stuck': 'hc_valve_stuck', 'hc_valve_leak': 'hc_valve_leak',
              'cc_fouling': 'cooling_capacity', 'condenser_fouling': 'cooling_capacity',
              'air_filter_blockage': 'low_airflow', 'sensor_bias_T_sa': 'sensor_bias_T_sa',
              'sensor_bias_T_oa': 'sensor_bias_T_oa', 'sensor_bias_T_zone': 'zone_setpoint',
              'schedule_fault': 'schedule',
              # not simulable -> other
              'ea_damper_stuck': None, 'ea_damper_leak': None, 'supply_duct_leak': None, 'hc_fouling': None,
              'hc_reduced_capacity': None, 'return_fan_failure': None, 'return_fan_fixed_speed': None,
              'supply_fan_fault': None, 'control_unstable': None, 'cc_valve_reverse_action': None,
              'sensor_bias_T_chw': None, 'sensor_bias_dp_chw': None, 'sensor_bias_flow_oa': None,
              'chiller_off': None, 'lighting_schedule_fault': None, 'excessive_infiltration': None}
OTHER_WHY = {
    'ea_damper_stuck': 'BuildingSim has no exhaust/relief damper', 'ea_damper_leak': 'no exhaust damper',
    'supply_duct_leak': 'no duct model', 'hc_fouling': 'no heating-coil capacity fault (boiler_fouling is efficiency only)',
    'hc_reduced_capacity': 'no heating-coil capacity fault', 'return_fan_failure': 'no return fan',
    'return_fan_fixed_speed': 'no return fan', 'control_unstable': 'no control dynamics (ideal controls)',
    'cc_valve_reverse_action': 'no valve control dynamics', 'sensor_bias_T_chw': 'chilled-water loop not in the trend',
    'sensor_bias_dp_chw': 'no hydronic pressure model', 'chiller_off': 'no plant-availability fault',
    'lighting_schedule_fault': 'lights not in the AHU/RTU trend', 'excessive_infiltration': 'no envelope fault'}


def season_of(date):
    m = pd.Timestamp(date).month
    return 'winter' if m in (12, 1, 2) else ('summer' if m in (6, 7, 8) else 'transition')


def severity_bucket(ftype, sev):
    """Exam severity bucket from the dataset's severity string (documented rule): stuck at an extreme (0/100 %, min)
    vs partial; leaks low <= 25 % vs high; sensor bias low < 3.5 (deg) vs high; graded levels 1 = low, >= 2 high;
    percentage degradations <= 25 % low, else high; anything else 'unspecified'."""
    if ftype == 'none':
        return 'none'
    if sev is None or (isinstance(sev, float) and np.isnan(sev)):
        return 'unspecified'
    s = str(sev).strip().lower()
    if s == 'min':
        return 'high'
    num = ''.join(ch for ch in s.split()[0] if ch in '+-.0123456789')
    try:
        v = float(num)
    except ValueError:
        return 'unspecified'
    if ftype.endswith('_stuck'):
        return 'high' if v in (0.0, 100.0) else 'low'
    if ftype.endswith('_leak'):
        if '%' not in s and v <= 3 and ftype == 'hc_valve_leak':     # ERS graded leak levels 0.4 / 1 / 2
            return 'low' if v < 1 else 'high'
        return 'low' if v <= 25 else 'high'
    if ftype.startswith('sensor_bias'):
        return 'low' if abs(v) < 3.5 else 'high'
    if ftype in ('hc_fouling', 'hc_reduced_capacity'):
        return 'low' if v <= 1 else 'high'
    if ftype in ('air_filter_blockage', 'condenser_fouling', 'excessive_infiltration', 'return_fan_fixed_speed'):
        return 'low' if abs(v) <= 25 else 'high'
    return 'unspecified'


# ------------------------------------------------------------------ real trend logs -> raw day arrays
def _flatline(x):
    v = x[np.isfinite(x)]
    return len(v) > 10 and (np.nanmax(v) - np.nanmin(v)) < 1e-3


def real_raw(d, dataset):
    """Contract trend log of one run (bd.fault_data rows) -> DataFrame of the 13 raw channels on its own time grid
    (local time index). Data-quality rules: impossible values -> NaN; a measured temperature that is constant all
    day (e.g. ERS AHU-A outdoor air reads exactly 0 F in summer 2007) is treated as missing."""
    g = lambda c: d[c].to_numpy(float) if c in d and d[c].notna().any() else np.full(len(d), np.nan)
    o = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(d['time_local'].to_numpy())))
    for c in ('T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_zone'):
        o[c] = g(c)
    sp = g('T_sa_sp')
    if not np.isfinite(sp).any():
        spc, sph, hc = g('T_sa_sp_cool'), g('T_sa_sp_heat'), np.nan_to_num(g('valve_hc'))
        sp = np.where((hc > 0.02) & np.isfinite(sph), sph, spc)
    sp = np.where(sp < 1.0, np.nan, sp)          # Korea logs 0 when the AHU is off
    o['T_sa_sp'] = sp
    o['damper_oa'] = g('damper_oa')
    if 'comp1_status' in d and d['comp1_status'].notna().any():
        o['valve_cc'] = 0.5 * (np.nan_to_num(g('comp1_status')) + np.nan_to_num(g('comp2_status')))
    else:
        o['valve_cc'] = g('valve_cc')
    hc = g('valve_hc')
    if dataset == 'korea_office':
        hc = np.full(len(d), np.nan)            # units / meaning unclear in the public file
    o['valve_hc'] = hc
    o['fan_speed'] = g('fan_sa_speed')
    fp = g('fan_sa_power_kW')
    if not np.isfinite(fp).any():
        fp = g('x_fan_sa_current_A')
    o['fan_power'] = fp
    fs = g('fan_sa_status')
    if not np.isfinite(fs).any():
        spd = g('fan_sa_speed')
        fs = np.where(np.isfinite(spd), (spd > 0.05).astype(float), np.nan)
    o['fan_on'] = fs
    o['occ'] = g('occ')
    for c in TEMPS + ['T_sa_sp']:
        x = o[c].to_numpy()
        x = np.where((x < -45) | (x > 70), np.nan, x)
        if c in TEMPS and _flatline(x):
            x[:] = np.nan
        o[c] = x
    for c in ('damper_oa', 'valve_cc', 'valve_hc', 'fan_speed'):
        o[c] = o[c].clip(-0.05, 1.05)
    return o[RAW]


def to_day(raw, day=None):
    """Raw-channel frame (local-time index) -> (x [13, 96] float32 with NaN where missing). Bins are 15-min means
    over the local calendar day `day` (default: the day of the first sample)."""
    if day is None:
        day = raw.index[0].normalize()
    day = pd.Timestamp(day)
    sel = raw[(raw.index >= day) & (raw.index < day + pd.Timedelta(days=1))]
    b = ((sel.index - day).total_seconds() // 900).astype(int)
    x = np.full((len(RAW), NT), np.nan, np.float32)
    if len(sel) == 0:
        return x
    g = sel.groupby(b).mean()
    g = g[(g.index >= 0) & (g.index < NT)]
    x[:, g.index.to_numpy()] = g.to_numpy(np.float32).T
    return x


def load_real_runs(split, datasets=None):
    """[(run row, x [13,96])] for every gradable run of a split ('dev' | 'exam') or the Korea background."""
    import buildingdata as bd
    r = bd.faults(split=split)
    if datasets is not None:
        r = r[r.dataset.isin(datasets)]
    out = []
    for ds, g in r.groupby('dataset', sort=False):
        data = bd.fault_data(ds)
        for _, row in g.iterrows():
            d = data[data.run_id == row.run_id]
            if len(d) == 0:
                continue
            raw = real_raw(d, ds)
            out.append((row, to_day(raw)))
    return out


# ------------------------------------------------------------------ simulated trend logs -> raw day arrays
def sat_setpoint(sim):
    """Supply-air setpoint a VAV controller would log (BuildingSim's reset rule; sat_setpoint_offset included,
    a supply sensor bias not: the setpoint is unchanged, the sensor is wrong)."""
    sy = sim.system
    n = sim.n
    if not hasattr(sy, 'sat'):
        return np.full(n, np.nan)
    To = np.asarray(sim.T_out[:n], float)
    s = np.full(n, float(sy.sat))
    if sy.sat_reset:
        s = np.where(To <= 15.6, sy.sat + 2.8, np.where(To < 21.1, sy.sat + 2.8 * (21.1 - To) / 5.5, sy.sat))
    off = sim.fault_arrays.get('sat_setpoint_offset')
    if off is not None:
        s = s + np.nan_to_num(off)
    return s


def sim_raw(out, tz):
    """BuildingSim output -> raw-channel frame in local time."""
    tr = out['trend']
    sim = out['sim']
    o = pd.DataFrame(index=tr.index.tz_convert(tz).tz_localize(None))
    o['T_oa'] = tr.OAT.to_numpy()
    o['T_ra'] = tr.RAT.to_numpy()
    o['T_ma'] = tr.MAT.to_numpy()
    o['T_sa'] = tr.SAT.to_numpy()
    sp = sat_setpoint(sim)
    o['T_sa_sp'] = sp[-len(tr):] if len(sp) >= len(tr) else np.nan
    o['T_zone'] = tr.ZAT.to_numpy()
    o['damper_oa'] = tr.OA_damper_pct.to_numpy() / 100
    o['valve_cc'] = tr.cool_cmd_pct.to_numpy() / 100
    o['valve_hc'] = tr.heat_cmd_pct.to_numpy() / 100
    o['fan_speed'] = tr.fan_speed_pct.to_numpy() / 100
    o['fan_power'] = tr.fan_kW.to_numpy()
    o['fan_on'] = tr.fan_status.to_numpy().astype(float)
    o['occ'] = tr.occupied.to_numpy().astype(float)
    return o[RAW]


def hvac_energy(out, tz):
    """HVAC site energy per step (kW): electricity for HVAC + heating gas/district + cooling district, local index."""
    ts = out['ts']
    e = out['trend'].hvac_kW.to_numpy() + ts.heating_gas_kW.to_numpy() + ts.heating_district_kW.to_numpy() + \
        ts.cooling_district_kW.to_numpy()
    return pd.Series(e, index=ts.index.tz_convert(tz).tz_localize(None))


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
