"""HVAC fault data -> unified AHU/RTU trend-log contract (proc/faults/).

Output: proc/faults/runs.parquet  - one row per run (a unit-day, or a whole file for multi-day sims): dataset, facility,
                                    unit, date, kind (experimental_lab | real_building | simulated | hil), fault_type,
                                    severity, fault_scope, n_rows, dt_s, fault_frac, split (added by make_splits.py)
        proc/faults/data/<dataset>.parquet - time series in the contract (SI; signals as fractions 0..1), with columns
                                    run_id, time_local, <channels>, fault (0/1 per row), plus x_* extras
        proc/faults/taxonomy.json, channel_map.json
Channels (BuildingSim trend-log contract proposed by B; A emits the same names or maps to them):
  T_oa T_ra T_ma T_sa T_sa_sp [degC]; RH_oa RH_ra RH_sa [%]; damper_oa damper_ra damper_ea valve_cc valve_hc [0..1];
  fan_sa_speed fan_ra_speed [0..1]; fan_sa_status fan_ra_status [0/1]; fan_sa_power_kW fan_ra_power_kW; flow_sa_m3s
  flow_oa_m3s flow_ra_m3s; p_duct_Pa p_duct_sp_Pa; T_chw_in T_chw_out T_hw_in T_hw_out [degC]; T_zone [degC]; occ [0/1];
  power_kW (unit electric power); comp1_status comp2_status [0/1]
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import os, re, glob, json
import numpy as np, pandas as pd

D = f'{BLDG_ROOT}/data'; R = f'{D}/raw/faults'; O = f'{D}/proc/faults'
os.makedirs(f'{O}/data', exist_ok=True)
F2C = lambda f: (f - 32) / 1.8
CFM = 0.000471947; INWC = 249.089

TAXONOMY = {
    'none': 'fault-free operation',
    'oa_damper_stuck': 'outdoor-air damper stuck at a fixed position', 'oa_damper_leak': 'outdoor-air damper leaking (cannot close below X%)',
    'ea_damper_stuck': 'exhaust damper stuck', 'ea_damper_leak': 'exhaust damper leaking',
    'cc_valve_stuck': 'cooling-coil valve stuck', 'cc_valve_leak': 'cooling-coil valve leaking (bypass opened)',
    'hc_valve_stuck': 'heating-coil valve stuck', 'hc_valve_leak': 'heating-coil valve leaking (bypass opened)',
    'supply_duct_leak': 'supply duct leak', 'hc_fouling': 'heating coil fouled (air side)', 'hc_reduced_capacity': 'heating coil capacity reduced (water side)',
    'cc_fouling': 'cooling coil fouled', 'return_fan_failure': 'return fan stopped', 'return_fan_fixed_speed': 'return fan stuck at a fixed speed',
    'supply_fan_fault': 'supply fan degraded / stuck', 'air_filter_blockage': 'air filter partially blocked',
    'control_unstable': 'unstable control loop (fan, valve, damper or sequencing)', 'cc_valve_reverse_action': 'cooling valve reverse acting',
    'sensor_bias_T_sa': 'supply-air temperature sensor bias / setpoint offset', 'sensor_bias_T_oa': 'outdoor-air temperature sensor bias',
    'sensor_bias_T_chw': 'chilled-water supply temperature bias / setpoint offset', 'sensor_bias_dp_chw': 'chilled-water differential pressure sensor bias',
    'sensor_bias_flow_oa': 'outdoor airflow sensor bias', 'sensor_bias_T_zone': 'thermostat (zone temperature) bias',
    'chiller_off': 'chiller stopped / unavailable', 'schedule_fault': 'HVAC operating schedule wrong (early start, late stop, no setback)',
    'lighting_schedule_fault': 'lighting schedule wrong', 'condenser_fouling': 'RTU condenser air flow reduced', 'excessive_infiltration': 'envelope infiltration increased',
    'cyber_attack': 'cyber attack on BAS (simulated / HIL)', 'degradation': 'gradual degradation (simulated)', 'other': 'other'}

PATTERNS = [(r'^OADamStuck|^Phy_OADam', 'oa_damper_stuck'), (r'^OADamLea', 'oa_damper_leak'), (r'^EADamStuck', 'ea_damper_stuck'),
            (r'^EADamLea', 'ea_damper_leak'), (r'^(Phy_)?CooCoi?l?Val_?Stuck', 'cc_valve_stuck'), (r'^CooCoiValLea', 'cc_valve_leak'),
            (r'^HeaCoiValStuck', 'hc_valve_stuck'), (r'^HeaCoiValLe?a', 'hc_valve_leak'), (r'^SupDucLea', 'supply_duct_leak'),
            (r'^HeaCoiFou', 'hc_fouling'), (r'^HeatCoiReduCapa|^HeaCoiReduCapa', 'hc_reduced_capacity'), (r'^CooCoiFou', 'cc_fouling'),
            (r'^RetFan_Fail', 'return_fan_failure'), (r'^RetFanFixed', 'return_fan_fixed_speed'), (r'^SupFan', 'supply_fan_fault'),
            (r'^AirFilBlock', 'air_filter_blockage'), (r'Unst', 'control_unstable'), (r'^CooCoiVal_Reverse', 'cc_valve_reverse_action'),
            (r'^TSup|^Phy_TSup', 'sensor_bias_T_sa'), (r'^TOut', 'sensor_bias_T_oa'), (r'^TCHWSup', 'sensor_bias_T_chw'),
            (r'^DifPreCHW|^RelPreCHW', 'sensor_bias_dp_chw'), (r'^Vout', 'sensor_bias_flow_oa'), (r'^Chiller_Off', 'chiller_off'),
            (r'^Schedule', 'schedule_fault'), (r'^Cyber|^DoS|^Dos|^Intru', 'cyber_attack'), (r'^Infiltration', 'excessive_infiltration')]


def ftype(stem):
    for p, t in PATTERNS:
        if re.search(p, stem): return t
    return 'other'


def sev(stem):
    m = re.search(r'_([pm]?[0-9.]+)$', stem)
    return None if not m else m.group(1).replace('p', '+').replace('m', '-')


ERS_MAP = {'T_oa': ('OA-TEMP', 'F'), 'T_ra': ('RA-TEMP', 'F'), 'T_ma': ('MA-TEMP', 'F'), 'T_sa': ('SA-TEMP', 'F'), 'T_sa_sp': ('SAT_SPT', 'F'),
           'RH_ra': ('RA-HUMD', 1), 'RH_sa': ('SA-HUMD', 1), 'damper_oa': ('OA-DMPR', 0.01), 'damper_ea': ('EA-DMPR', 0.01),
           'damper_ra': ('RA-DMPR', 'inv%'), 'valve_cc': ('CHWC-VLV', 0.01), 'valve_hc': ('HWC-VLV', 'inv%'),
           'fan_sa_speed': ('SF-SPD', 0.01), 'fan_ra_speed': ('RF-SPD', 0.01), 'fan_sa_status': ('SF-SST', 1), 'fan_ra_status': ('RF-SST', 1),
           'fan_sa_power_kW': ('SF-WAT', 0.001), 'fan_ra_power_kW': ('RF-WAT', 0.001), 'flow_sa_m3s': ('SA-CFM', CFM), 'flow_oa_m3s': ('OA-CFM', CFM),
           'flow_ra_m3s': ('RA-CFM', CFM), 'p_duct_Pa': ('SA-SP', INWC), 'p_duct_sp_Pa': ('SA_SPSPT', INWC), 'T_chw_in': ('CHWC-EWT', 'F'),
           'T_chw_out': ('CHWC-LWT', 'F'), 'T_hw_in': ('HWC-EWT', 'F'), 'T_hw_out': ('HWC-LWT', 'F'), 'occ': ('SYS-CTL', 1)}
LBNL_MAP = {'T_oa': ('AHU: Outdoor Air Temperature', 'F'), 'T_ra': ('AHU: Return Air Temperature', 'F'), 'T_ma': ('AHU: Mixed Air Temperature', 'F'),
            'T_sa': ('AHU: Supply Air Temperature', 'F'), 'T_sa_sp': ('AHU: Supply Air Temperature Set Point', 'F'),
            'T_sa_sp_heat': ('AHU: Supply Air Temperature Heating Set Point', 'F'), 'T_sa_sp_cool': ('AHU: Supply Air Temperature Cooling Set Point', 'F'),
            'fan_sa_status': ('AHU: Supply Air Fan Status', 1), 'fan_ra_status': ('AHU: Return Air Fan Status', 1),
            'fan_sa_speed': ('AHU: Supply Air Fan Speed Control Signal', 1), 'fan_ra_speed': ('AHU: Return Air Fan Speed Control Signal', 1),
            'damper_oa': ('AHU: Outdoor Air Damper Control Signal', 1), 'damper_ra': ('AHU: Return Air Damper Control Signal', 1),
            'damper_ea': ('AHU: Exhaust Air Damper Control Signal', 1), 'valve_cc': ('AHU: Cooling Coil Valve Control Signal', 1),
            'valve_hc': ('AHU: Heating Coil Valve Control Signal', 1), 'p_duct_Pa': ('AHU: Supply Air Duct Static Pressure', INWC),
            'p_duct_sp_Pa': ('AHU: Supply Air Duct Static Pressure Set Point', INWC), 'occ': ('Occupancy Mode Indicator', 1)}
RTU_MAP = {'T_sa': ('RTU: Supply Air Temperature', 'F'), 'T_ra': ('RTU: Return Air Temperature', 'F'), 'fan_sa_status': ('RTU: Supply Air Fan Status', 1),
           'flow_sa_m3s': ('RTU: Supply Air Volumetric Flow Rate', CFM), 'power_kW': ('RTU: Electricity', 0.06),   # Wh per minute -> kW
           'fan_sa_power_kW': ('RTU: Fan Electricity', 0.06), 'occ': ('Occupancy Mode Indicator', 1),
           'comp1_status': ('RTU: Compressor 1 On/Off Status', 1), 'comp2_status': ('RTU: Compressor 2 On/Off Status', 1)}
# Nesbitt Hall: positional blocks (AHU-1 = columns 31.., AHU-2 = 52..) - names repeat with .1 suffixes
NES_AHU = {'AHU-1': {'T_ra': (31, 'F'), 'T_ma': (32, 'F'), 'T_sa': (33, 'F'), 'fan_sa_speed': (34, 0.01), 'damper_oa': (35, 0.01),
                     'valve_cc': (36, 0.01), 'fan_ra_speed': (38, 0.01), 'p_duct_Pa': (41, INWC), 'T_sa_sp': (42, 'F'), 'p_duct_sp_Pa': (43, INWC),
                     'RH_ra': (44, 1), 'x_fan_sa_current_A': (47, 1), 'x_fan_ra_current_A': (48, 1), 'valve_hc': (50, 0.01)},
           'AHU-2': {'T_ra': (52, 'F'), 'T_ma': (53, 'F'), 'T_sa': (54, 'F'), 'fan_sa_speed': (55, 0.01), 'damper_oa': (56, 0.01),
                     'valve_cc': (57, 0.01), 'fan_ra_speed': (59, 0.01), 'p_duct_Pa': (62, INWC), 'T_sa_sp': (63, 'F'), 'p_duct_sp_Pa': (64, INWC),
                     'RH_ra': (65, 1), 'RH_sa': (68, 1), 'x_fan_sa_current_A': (69, 1), 'x_fan_ra_current_A': (70, 1), 'valve_hc': (71, 0.01),
                     'damper_ea': (73, 0.01), 'damper_ra': (74, 0.01)}}
NES_COMMON = {'T_oa': (2, 'F'), 'RH_oa': (3, 1), 'x_building_kWh_h': (5, 1), 'x_T_chw_supply': (8, 'F'), 'x_T_chw_return': (7, 'F'), 'x_chiller_on': (20, 1)}
# (fault window local time, target unit) - RBC-Nesbitt explanations Table 2
NES_FAULTS = {'20160706': ('16:00', '23:00', 'plant'), '20160808': ('10:22', '21:16', 'AHU-1'), '20160907': ('10:30', '22:09', 'AHU-1'),
              '20160911': ('18:30', '20:30', 'plant'), '20161201': ('10:00', '20:45', 'AHU-2'), '20170103': ('10:00', '20:30', 'AHU-2'),
              '20170114': ('01:30', '07:00', 'system'), '20170709': ('04:00', '15:30', 'plant'), '20170711': ('10:00', '20:01', 'AHU-2'),
              '20170718': ('11:00', '20:01', 'AHU-2'), '20170803': ('10:00', '21:27', 'plant'), '20170805': ('04:00', '08:00', 'system'),
              '20170811': ('10:05', '20:06', 'AHU-2'), '20170915': ('10:30', '15:21', 'plant'), '20180709': ('10:15', '20:15', 'AHU-2'),
              '20180710': ('10:30', '20:30', 'AHU-2'), '20180711': ('10:00', '20:00', 'AHU-2'), '20180718': ('09:30', '19:30', 'AHU-2'),
              '20180722': ('20:20', '21:40', 'system'), '20180723': ('08:00', '18:00', 'plant')}
NES_NATURAL = {'20160706', '20170114', '20170709'}
# LBNL synthesis experimental scenario tables (inventory PDF, Tables 2-2, 3-3, 3-4, 4-3): date -> (fault_type, severity)
SZCAV = {'2017-03-18': ('oa_damper_stuck', '100%'), '2017-03-19': ('oa_damper_stuck', '50%'), '2017-03-20': ('oa_damper_leak', '20%'),
         '2017-03-21': ('oa_damper_leak', '50%'), '2017-03-24': ('hc_valve_stuck', '0%'), '2017-03-25': ('hc_valve_stuck', '100%'),
         '2017-03-26': ('hc_valve_stuck', '50%'), '2017-03-22': ('hc_valve_leak', '5%'), '2017-03-23': ('hc_valve_leak', '40%'),
         '2017-03-11': ('cc_valve_stuck', '0%'), '2017-03-12': ('cc_valve_stuck', '100%'), '2017-03-31': ('cc_valve_stuck', '50%'),
         '2017-03-15': ('cc_valve_leak', '5%'), '2017-03-16': ('cc_valve_leak', '50%'), '2017-04-01': ('none', None)}
SZVAV = {'2017-09-18': ('oa_damper_stuck', 'min'), '2017-09-19': ('oa_damper_stuck', '100%'), '2017-09-14': ('hc_valve_stuck', '100%'),
         '2017-09-15': ('hc_valve_stuck', '50%'), '2017-09-12': ('hc_valve_leak', '40%'), '2017-09-22': ('cc_valve_stuck', '100%'),
         '2017-09-11': ('cc_valve_leak', '50%'), '2017-09-20': ('none', None), '2017-09-21': ('none', None), '2017-09-23': ('none', None),
         '2017-09-24': ('none', None)}
RTU_T = {'2017-08-27': ('condenser_fouling', '25%'), '2017-08-29': ('condenser_fouling', '50%'), '2017-12-01': ('schedule_fault', 'HVAC setback onset +3h'),
         '2017-12-03': ('AMBIGUOUS', 'listed as both early-termination fault and unfaulted'), '2017-12-07': ('excessive_infiltration', '+20%'),
         '2017-12-14': ('excessive_infiltration', '+40%'), '2018-02-07': ('lighting_schedule_fault', 'onset +3h'),
         '2018-02-09': ('lighting_schedule_fault', 'early termination 3h'), '2017-12-20': ('schedule_fault', 'no overnight HVAC setback'),
         '2018-02-18': ('lighting_schedule_fault', 'no overnight setback'), '2018-02-01': ('sensor_bias_T_zone', '+4F core'),
         '2018-01-31': ('sensor_bias_T_zone', '-4F core'), '2018-01-28': ('sensor_bias_T_zone', '+4F perimeter'),
         '2018-01-27': ('sensor_bias_T_zone', '-4F perimeter'), '2017-09-01': ('none', None), '2017-11-30': ('none', None),
         '2017-12-09': ('none', None), '2017-12-24': ('none', None), '2018-02-04': ('none', None), '2018-02-05': ('none', None)}


def conv(s, how):
    s = pd.to_numeric(s, errors='coerce')
    if how == 'F': return F2C(s)
    if how == 'inv%': return 1 - s / 100.0
    return s * how


def apply_map(df, mp, by_pos=False):
    out = {}
    cols = list(df.columns)
    norm = {re.sub(r'\s+', ' ', c).strip(): c for c in cols}
    for k, (src, how) in mp.items():
        if by_pos:
            if src < len(cols): out[k] = conv(df.iloc[:, src], how)
        else:
            c = norm.get(re.sub(r'\s+', ' ', src).strip())
            if c is not None: out[k] = conv(df[c], how)
    return pd.DataFrame(out, index=df.index)


RUNS = []


def add_run(store, rid, df, **meta):
    df = df.copy(); df.insert(0, 'run_id', rid)
    store.append(df)
    RUNS.append(dict(run_id=rid, n_rows=len(df), fault_frac=float(df.fault.mean()), **meta))


def write(name, store):
    d = pd.concat(store, ignore_index=True)
    for c in d.columns:
        if d[c].dtype == 'float64': d[c] = d[c].astype('float32')
    d.to_parquet(f'{O}/data/{name}.parquet', index=False, compression='zstd')
    print(name, d.shape, d.run_id.nunique(), 'runs', flush=True)


def ers(kind_dir, real):
    """ASHRAE RP-1312 ERS (Iowa Energy Center), Ghalamsiah 2025 folder 01. Real: AHU-A faulted, baseline.csv = AHU-B same day."""
    store = []
    base = f'{R}/ghalamsiah2025/01_RBC-ASHRAE1312/{kind_dir}'
    for season in sorted(os.listdir(base)):
        for day in sorted(os.listdir(f'{base}/{season}')):
            dd = f'{base}/{season}/{day}'
            for f in sorted(glob.glob(f'{dd}/*.csv')):
                stem = os.path.basename(f)[:-4]
                if stem == 'weather': continue
                x = pd.read_csv(f, encoding='latin-1')
                y = apply_map(x, ERS_MAP)
                if 'TIME' in x:
                    t = pd.to_datetime(day, format='%Y%m%d') + pd.to_timedelta((x.TIME // 100) * 60 + x.TIME % 100, unit='m')
                else:
                    t = pd.to_datetime(day, format='%Y%m%d') + pd.to_timedelta(np.arange(len(x)), unit='m')
                y.insert(0, 'time_local', t.values)
                zone = [c for c in x.columns if c.startswith('RM-TEMP')]
                if zone: y['T_zone'] = F2C(x[zone].apply(pd.to_numeric, errors='coerce').mean(axis=1))
                is_base = stem.lower() == 'baseline'
                ft = 'none' if is_base else ftype(stem)
                y['fault'] = np.int8(0 if is_base else 1)
                unit = ('AHU-B' if is_base else 'AHU-A') if real else ('sim-AHU-B' if is_base else 'sim-AHU-A')
                rid = f"ERS{'' if real else 'sim'}_{day}_{unit}"
                add_run(store, rid, y, dataset='ers_rp1312' if real else 'ers_rp1312_sim', facility='Iowa Energy Resource Station (ERS)',
                        unit=unit, date=pd.Timestamp(day).date().isoformat(), season=season.lower(),
                        kind='experimental_lab' if real else 'simulated', fault_type=ft, severity=None if is_base else sev(stem),
                        fault_label=stem, fault_scope='unit', dt_s=60, source='Ghalamsiah 2025 figshare 29297999 (01_RBC-ASHRAE1312)')
    write('ers_rp1312' if real else 'ers_rp1312_sim', store)


def lbnl(fname, dataset, table, kind, facility, mp, unit):
    store = []
    x = pd.read_csv(f'{R}/lbnl_synthesis/{fname}', encoding='latin-1')
    tc = x.columns[0]; t = pd.to_datetime(x[tc], format='mixed')
    y = apply_map(x, mp); y.insert(0, 'time_local', t.values)
    if fname == 'RTU.csv':
        z = [c for c in x.columns if c.startswith('Terminal: Room') and c.endswith('Air Temperature')]
        y['T_zone'] = F2C(x[z].apply(pd.to_numeric, errors='coerce').mean(axis=1))
        y['x_hvac_power_kW'] = pd.to_numeric(x['HVAC System: Electricity'], errors='coerce') * 0.06
    gt = pd.to_numeric(x['Fault Detection Ground Truth'], errors='coerce')
    days = t.dt.date
    for dday, idx in y.groupby(days).groups.items():
        key = dday.isoformat()
        if table is not None:
            ft, sv = table.get(key, ('UNLISTED', None))
        else:
            ft, sv = ('unknown_label_only', None)
        yy = y.loc[idx].copy()
        if fname == 'RTU.csv' or table is not None and fname != 'MZVAV-2-1.csv':
            yy['fault'] = np.int8(0 if ft == 'none' else 1)           # day-level scenario table (RTU ground-truth column is all 1)
        else:
            yy['fault'] = gt.loc[idx].astype('int8').values
        yy['x_ground_truth_file'] = gt.loc[idx].astype('float32').values
        add_run(store, f'{dataset}_{key}', yy, dataset=dataset, facility=facility, unit=unit, date=key, season=None, kind=kind,
                fault_type=ft, severity=sv, fault_label=None, fault_scope='unit', dt_s=60,
                source='LBNL FDD data synthesis, figshare 11752740 (' + fname + ')')
    write(dataset, store)


def lbnl_sim(fname, dataset, facility):
    """Simulated LBNL sets: fault type not given per day in the file (ground truth 0/1 only); kept for simulator checks."""
    store = []
    x = pd.read_csv(f'{R}/lbnl_synthesis/{fname}', encoding='latin-1')
    t = pd.to_datetime(x[x.columns[0]], format='mixed')
    y = apply_map(x, LBNL_MAP); y.insert(0, 'time_local', t.values)
    y['fault'] = pd.to_numeric(x['Fault Detection Ground Truth'], errors='coerce').astype('int8')
    for dday, idx in y.groupby(t.dt.date).groups.items():
        add_run(store, f'{dataset}_{dday.isoformat()}', y.loc[idx], dataset=dataset, facility=facility, unit='AHU', date=dday.isoformat(),
                season=None, kind='simulated', fault_type='see_inventory' if y.loc[idx, 'fault'].max() else 'none', severity=None,
                fault_label=None, fault_scope='unit', dt_s=60, source='LBNL FDD data synthesis, figshare 11752740 (' + fname + ')')
    write(dataset, store)


def nesbitt():
    store = []
    base = f'{R}/ghalamsiah2025/02_RBC-Nesbitt'
    files = [(f, False) for f in sorted(glob.glob(f'{base}/2*.csv'))] + [(f, True) for f in sorted(glob.glob(f'{base}/Baselines/*.csv'))]
    for f, is_base in files:
        stem = os.path.basename(f)[:-4]; day = stem[:8]; label = re.sub(r'^\d{8}_+', '', stem).strip()
        x = pd.read_csv(f, encoding='latin-1')
        t = pd.to_datetime(day, format='%Y%m%d') + pd.to_timedelta(pd.to_numeric(x.iloc[:, 1], errors='coerce'), unit='m')
        common = apply_map(x, NES_COMMON, by_pos=True)
        for unit, mp in NES_AHU.items():
            y = apply_map(x, mp, by_pos=True); y = pd.concat([common, y], axis=1); y.insert(0, 'time_local', t.values)
            if is_base:
                ft, scope, flag = 'none', 'none', np.zeros(len(y), np.int8)
            else:
                a, b, target = NES_FAULTS[day]; ft = ftype(label)
                w = (t >= pd.Timestamp(f'{day} {a}')) & (t <= pd.Timestamp(f'{day} {b}'))
                hit = target in ('plant', 'system') or target == unit
                flag = (w & hit).astype('int8').values; scope = target
                if not hit: ft = 'none'
            y['fault'] = flag
            add_run(store, f'NES_{day}_{unit}' + ('_base' if is_base else ''), y, dataset='nesbitt', facility='Nesbitt Hall, Drexel University (Philadelphia)',
                    unit=unit, date=pd.Timestamp(day).date().isoformat(), season=None, kind='real_building', fault_type=ft,
                    severity=None if is_base else sev(label), fault_label=None if is_base else label, fault_scope=scope, dt_s=300,
                    natural_fault=(day in NES_NATURAL) and not is_base, source='Ghalamsiah 2025 figshare 29297999 (02_RBC-Nesbitt)')
    write('nesbitt', store)


def ghal_sim(folder, dataset, dt_s):
    store = []
    base = f'{R}/ghalamsiah2025/{folder}'
    for f in sorted(glob.glob(f'{base}/*.csv')):
        stem = os.path.basename(f)[:-4]
        x = pd.read_csv(f, encoding='latin-1', nrows=3)
        store.append(None)
        RUNS.append(dict(run_id=f'{dataset}_{stem}', dataset=dataset, facility=folder, unit='AHU', date=None, season=None,
                         kind='hil' if 'HIL' in folder else 'simulated', fault_type='none' if 'Baseline' in stem else ftype(stem),
                         severity=sev(stem), fault_label=stem, fault_scope='unit', dt_s=dt_s, n_rows=None, fault_frac=None,
                         raw_file=os.path.relpath(f, D), n_raw_columns=x.shape[1],
                         source=f'Ghalamsiah 2025 figshare 29297999 ({folder}); kept raw (Modelica point names), not converted'))
    print(dataset, 'indexed', len(store))


def korea():
    x = pd.read_csv(f'{R}/korea_office/x/FDD_dataset/combined_data.csv')
    y = pd.DataFrame({'run_id': 'KOR_' + x['AHU name'].astype(str), 'time_local': pd.to_datetime(x.Time, format='%d/%m/%Y %H:%M'),
                      'T_ra': x['Return temperature'], 'T_sa': x['Supply air temperature'], 'T_sa_sp': x['Set point temperature'],
                      'valve_hc': x['Valve position'], 'fan_sa_status': x['Supply fan'], 'x_T_hw_supply_1': x['Heating supply temperature 1'],
                      'x_T_hw_supply_2': x['Heating supply temperature 2'], 'x_heating_pump_total': x['Total heating pump']})
    y['fault'] = np.int8(-1)      # no labels in the public files
    for c in y.columns:
        if y[c].dtype == 'float64': y[c] = y[c].astype('float32')
    y.to_parquet(f'{O}/data/korea_office.parquet', index=False, compression='zstd')
    for rid, g in y.groupby('run_id'):
        RUNS.append(dict(run_id=rid, dataset='korea_office', facility='large office, South Korea (BMS)', unit=rid[4:], date=None, season='heating',
                         kind='real_building', fault_type='unlabelled', severity=None, fault_label=None, fault_scope='unit', dt_s=3600,
                         n_rows=len(g), fault_frac=None, source='Wang 2024 figshare 25932409 (labels described but not in the public files)'))
    print('korea', y.shape, y.run_id.nunique())


if __name__ == '__main__':
    ers('Real', True); ers('Simulation', False)
    lbnl('MZVAV-2-1.csv', 'lbnl_mzvav2_ers_exp', {**{d: ('hc_valve_leak', s) for d, s in [('2007-08-28', '0.4GPM'), ('2007-08-29', '1.0GPM'), ('2007-08-30', '2.0GPM')]},
         **{d: ('none', None) for d in ['2008-08-19', '2008-08-25', '2008-09-04', '2009-01-19', '2009-02-16', '2009-02-17', '2009-05-03', '2009-05-04', '2009-05-05', '2009-05-06']}},
         'experimental_lab', 'Iowa Energy Resource Station (ERS)', LBNL_MAP, 'AHU-A')
    lbnl('SZCAV.csv', 'lbnl_szcav_flexlab', SZCAV, 'experimental_lab', 'LBNL FLEXLAB X3A (Berkeley)', LBNL_MAP, 'SZCAV-AHU')
    lbnl('SZVAV.csv', 'lbnl_szvav_flexlab', SZVAV, 'experimental_lab', 'LBNL FLEXLAB X3A (Berkeley)', LBNL_MAP, 'SZVAV-AHU')
    lbnl('RTU.csv', 'lbnl_rtu_ornl', RTU_T, 'experimental_lab', 'ORNL Flexible Research Platform 2 (Oak Ridge)', RTU_MAP, 'RTU')
    lbnl_sim('MZVAV-1.csv', 'lbnl_mzvav1_sim', 'PNNL large office model (Chicago TMY3)')
    lbnl_sim('MZVAV-2-2.csv', 'lbnl_mzvav2_ers_sim', 'ERS model (HVACSIM+)')
    nesbitt()
    for fo, ds, dt in [('03_RBC-5wk', 'ghal_rbc5wk_sim', 300), ('05_G36-5wk', 'ghal_g36_5wk_sim', 300), ('07_G36-Cyber', 'ghal_g36_cyber_sim', 300),
                       ('08_G36-HIL', 'ghal_g36_hil', 60)]:
        ghal_sim(fo, ds, dt)
    korea()
    runs = pd.DataFrame(RUNS)
    # de-duplicate: LBNL MZVAV-2-1 fault days are byte-identical to ERS real AHU-A days (checked 2007-08-28) -> drop them
    ers_days = set(runs[(runs.dataset == 'ers_rp1312') & (runs.unit == 'AHU-A')].date)
    dup = (runs.dataset == 'lbnl_mzvav2_ers_exp') & runs.date.isin(ers_days)
    runs['duplicate_of'] = np.where(dup, 'ERS_' + runs.date.str.replace('-', '') + '_AHU-A', None)
    runs['gradable'] = runs.kind.isin(['experimental_lab', 'real_building']) & ~dup & ~runs.fault_type.isin(['AMBIGUOUS', 'UNLISTED', 'unlabelled'])
    runs.to_parquet(f'{O}/runs.parquet', index=False)
    json.dump(TAXONOMY, open(f'{O}/taxonomy.json', 'w'), indent=1)
    json.dump({'ERS': ERS_MAP, 'LBNL': LBNL_MAP, 'RTU': RTU_MAP, 'Nesbitt_AHU_positional': NES_AHU, 'Nesbitt_common_positional': NES_COMMON},
              open(f'{O}/channel_map.json', 'w'), indent=1, default=str)
    g = runs[runs.gradable]
    print(g.groupby(['dataset', 'kind']).agg(runs=('run_id', 'size'), faulty=('fault_type', lambda s: (s != 'none').sum())).to_string())
    print(g.fault_type.value_counts().to_string())
