"""realdata: frozen real-world data for the ML Ratchet vehicle project (numpy/pandas only).

    import sys; sys.path.insert(0, 'path/to/ML-Ratchet/shared/vehicle')   # or pip install -e . in the repo
    import realdata as rd
    idx = rd.ved_index()                                  # one row per VED trip (meta + quality flags)
    for meta, arr in rd.load_trips('dev', arch='hev_powersplit', label_ok=True, limit=5): ...
    spec = rd.load_vehicle('VED_455')                     # vehicle spec dict (shared contract)
    epa = rd.epa_testcars(split='dev')                    # one row per EPA test vehicle configuration
    cyc = rd.cycles()                                     # dict name -> (meta, arrays) of dynamometer schedules
    hd = rd.hd_data()                                     # ZTBus legs/missions, Mersin bus trips, ICCT truck results
    sp = rd.splits()                                      # the frozen split file

Arrays follow the trip contract (1 Hz, SI, float32): t, v, grade, elev, lat, lon, heading, T_amb, wind_speed,
wind_dir, rho_air, surface, fuel_rate [g/s], engine_rpm, batt_current [A, + = discharge], batt_voltage, soc [0-1],
aux_power [W]; extras: oat (car's own outside-air sensor), valid (1 = within 5 s of a logged row or a held stop).
"""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import os, json, functools
import numpy as np, pandas as pd

DATA = _VEH + '/data'
PROC = f'{DATA}/proc'
_VED_ARCH = None


@functools.lru_cache(maxsize=1)
def splits():
    return json.load(open(f'{PROC}/splits.json'))


@functools.lru_cache(maxsize=1)
def ved_index():
    """All VED trips (meta + quality flags). Use q_label_ok for graded trips, q_route_ok for route inputs."""
    return pd.read_parquet(f'{PROC}/ved/index.parquet')


def _load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_trip(trip_id_or_path):
    """Return (meta, arrays) for one trip id (VED_*, ZTBus_*, or a cycle name) or an explicit .npz path."""
    p = trip_id_or_path
    if not str(p).endswith('.npz'):
        tid = str(p)
        if tid.startswith('VED_'):
            veh = '_'.join(tid.split('_')[:2]); p = f'{PROC}/ved/trips/{veh}/{tid}.npz'
        elif tid.startswith('ZTBus_'):
            p = f'{PROC}/hd/ztbus/{tid}.npz'
        else:
            p = f'{PROC}/cycles/{tid}.npz'
    meta = json.load(open(p[:-4] + '.json'))
    return meta, _load_npz(p)


def load_trips(split, arch=None, source='VED', label_ok=None, route_ok=None, vehicles=None, min_km=None,
               limit=None, seed=None, meta_only=False):
    """Iterate (meta, arrays) for trips in a split ('dev' | 'exam' | 'train_noise').
    arch: contract architecture ('conv', 'hev_powersplit', 'hev_parallel_p2', 'phev_series', 'phev_powersplit', 'bev')
          or a VED type ('ICE', 'HEV', 'PHEV', 'EV'). label_ok / route_ok filter on quality flags.
    source: 'VED' or 'ZTBus'. limit + seed: a fixed random draw (never cherry-pick).
    meta_only=True returns the filtered index DataFrame instead of an iterator."""
    if source == 'VED':
        d = ved_index()
    elif source == 'ZTBus':
        d = pd.read_parquet(f'{PROC}/hd/ztbus_legs.parquet')
        d['path'] = 'ztbus/' + d.trip_id + '.npz'; d['architecture'] = 'bev'; d['ved_type'] = 'BUS'
    else:
        raise ValueError('source must be VED or ZTBus (cycles: use cycles(); tables: hd_data())')
    d = d[d.split == split]
    if arch is not None:
        arch = [arch] if isinstance(arch, str) else list(arch)
        d = d[d.architecture.isin(arch) | d.ved_type.isin(arch)]
    if label_ok is not None: d = d[d.q_label_ok == label_ok]
    if route_ok is not None and 'q_route_ok' in d: d = d[d.q_route_ok == route_ok]
    if vehicles is not None: d = d[d.vehicle_id.isin(list(vehicles))]
    if min_km is not None: d = d[d.distance_m >= 1000 * min_km]
    if limit is not None and len(d) > limit:
        d = d.sample(limit, random_state=seed if seed is not None else 0)
    if meta_only:
        return d
    base = f'{PROC}/ved' if source == 'VED' else f'{PROC}/hd'

    def gen():
        for r in d.itertuples():
            p = f'{base}/{r.path}'
            yield json.load(open(p[:-4] + '.json')), _load_npz(p)
    return gen()


# ---------------- vehicles ----------------
@functools.lru_cache(maxsize=1)
def _ved_vehicles():
    from .vedspec import ved_vehicle_table
    t = ved_vehicle_table()
    m = pd.read_parquet(f'{PROC}/fe/ved_label_match.parquet')
    t = t.join(m, how='left')
    t['split'] = t.index.map(splits()['VED']['vehicles'])
    return t


def ved_vehicles():
    """One row per VED vehicle: static data, architecture guess, fueleconomy.gov label match, split."""
    return _ved_vehicles().copy()


_TRANS = {'A': 'AT', 'L': 'AT', 'SA': 'AT', 'S': 'AT', 'AM': 'AMT', 'AMS': 'AMT', 'AV': 'CVT', 'AVS': 'CVT', 'SCV': 'CVT', 'M': 'AMT', 'OT': None}


def load_vehicle(vehicle_id):
    """Vehicle spec dict (shared contract). Unknown fields are None. Supports VED_*, EPA*, ZTBus_*, ICCT2018_*, Mersin33M_*."""
    vid = str(vehicle_id)
    if vid.startswith('VED_'):
        from .vedspec import spec_from_row
        r = _ved_vehicles().loc[vid]
        s = spec_from_row(r)
        s['labels_fueleconomy'] = {k: (None if pd.isna(r.get(k)) else r.get(k)) for k in
                                   ['label_city_mpg', 'label_hwy_mpg', 'label_comb_mpg', 'unadj_city_mpg', 'unadj_hwy_mpg',
                                    'label_city_kWh_per_100mi', 'label_hwy_kWh_per_100mi', 'match_confidence', 'n_candidates', 'example_models']}
        s['split'] = r.split
        return s
    if vid.startswith('EPA'):
        r = epa_testcars().loc[vid]
        cls = 'car' if r.veh_type == 'Car' else ('pickup' if 'PICKUP' in str(r.model).upper() else 'suv')
        LBF, MPH = 4.4482216, 0.44704
        return {'id': vid, 'class': cls, 'architecture': {'conv': 'conv', 'hev': 'hev_powersplit', 'phev': 'phev_powersplit', 'bev': 'bev', 'fcev': 'bev'}[r.architecture],
                'source': 'EPA test car list', 'make': r.make, 'model': r.model, 'model_year': int(r.model_year),
                'mass_kg': float(r.test_mass_kg) if pd.notna(r.test_mass_kg) else None, 'payload_kg': 0.0,
                'mass_note': 'Equivalent test weight (ETW) = curb + 300 lb, rounded to the ETW class',
                'CdA_m2': None, 'Crr': None, 'Crr_speed_coef': None,
                'road_load_ABC': {'A_lbf': r.A_lbf, 'B_lbf_per_mph': r.B_lbf_mph, 'C_lbf_per_mph2': r.C_lbf_mph2,
                                  'A_N': r.A_N, 'B_N_per_mps': r.B_N_per_mps, 'C_N_per_mps2': r.C_N_per_mps2,
                                  'set_A_lbf': r.setA_lbf, 'set_B_lbf_per_mph': r.setB_lbf_mph, 'set_C_lbf_per_mph2': r.setC_lbf_mph2},
                'wheel_radius_m': None, 'final_drive': float(r.axle_ratio) if pd.notna(r.axle_ratio) else None, 'driveline_eff': None,
                'n_v_ratio': float(r.nv_ratio) if pd.notna(r.nv_ratio) else None,
                'engine': None if r.architecture == 'bev' else {'fuel': r.fuel, 'displacement_L': None if pd.isna(r.displacement_L) else float(r.displacement_L),
                                                               'cylinders': None if pd.isna(r.cylinders) else int(r.cylinders),
                                                               'max_power_kW': None if pd.isna(r.rated_hp) else float(r.rated_hp) * 0.7457,
                                                               'bsfc_map': None, 'idle_rpm': None, 'max_rpm': None, 'warmup_model': None},
                'transmission': {'type': 'single' if r.architecture == 'bev' else _TRANS.get(str(r.trans_code), None), 'raw_code': r.trans_code,
                                 'gears': None if pd.isna(r.gears) else int(r.gears)},
                'drive': r.drive, 'motors': None, 'battery': None, 'aux': None, 'regen': None, 'control': None,
                'arch_source': r.arch_source, 'split': r.split,
                'results': {c: (None if pd.isna(r.get(c)) else float(r.get(c))) for c in r.index if c.endswith(('_mpg', '_co2_g_mi', '_kWh_per_mi'))}}
    if vid.startswith('ZTBus_'):
        return {'id': vid, 'class': 'bus', 'architecture': 'bev', 'source': 'ZTBus (ETH Zurich / VBZ)',
                'note': 'Articulated trolleybus with traction battery (in-motion charging under the overhead line). Mass, CdA, Crr not published in the dataset; passenger counts are logged per second.',
                'mass_kg': None, 'payload_kg': None, 'CdA_m2': None, 'Crr': None, 'engine': None, 'transmission': {'type': 'single'},
                'split': 'mixed (by mission)'}
    if vid.startswith('ICCT2018_'):
        t = hd_data()['icct_trucks']; r = t[t.vehicle_id == vid].iloc[0]
        return {'id': vid, 'class': 'truck_hd', 'architecture': 'conv', 'source': r.source,
                'mass_kg_curb_with_trailer_and_500L_fuel': float(r.curb_mass_kg), 'payload_kg': None,
                'mass_note': 'test mass = curb + payload of each cycle (see hd_data()["icct_trucks"].test_mass_kg)',
                'Cd_US_coastdown': None if pd.isna(r.Cd_US_coastdown) else float(r.Cd_US_coastdown), 'Cd_EU_constant_speed': float(r.Cd_EU_const),
                'CdA_m2': None, 'Crr': None, 'final_drive': float(r.axle_ratio), 'axles': r.axles,
                'engine': {'fuel': 'diesel', 'displacement_L': r.engine_L, 'max_power_kW': '300-330', 'cylinders': 6},
                'transmission': {'type': 'AMT', 'gears': 12}, 'split': r.split}
    if vid.startswith('Mersin33M_'):
        fuel = 'diesel' if 'diesel' in vid else 'cng'
        return {'id': vid, 'class': 'bus', 'architecture': 'conv', 'source': 'Mersin route 33M (Zenodo 22716221)',
                'engine': {'fuel': fuel}, 'note': 'Bus model/mass not published; trip-level records only.'}
    raise KeyError(vid)


# ---------------- tables ----------------
@functools.lru_cache(maxsize=1)
def _epa():
    v = pd.read_parquet(f'{PROC}/epa/testcars_vehicles.parquet')
    s = pd.read_parquet(f'{PROC}/epa/testcars_split.parquet')
    return v.join(s)


def epa_testcars(split=None, arch=None, long=False):
    """EPA test car list 2008-2026. Wide (default): one row per test vehicle configuration (epa_vehicle_id index) with
    ETW, road-load target ABC (lbf and SI), engine/transmission, per-cycle results (<cycle>_mpg, <cycle>_co2_g_mi,
    <cycle>_kWh_per_mi; cycles FTP, HWY, US06, SC03, FTP_cold, CD_*). long=True: one row per test."""
    if long:
        t = pd.read_parquet(f'{PROC}/epa/testcars_tests.parquet')
        t = t.join(_epa()[['split', 'family']], on='epa_vehicle_id')
    else:
        t = _epa().copy()
    if split: t = t[t.split == split]
    if arch: t = t[t.architecture == arch]
    return t


def fueleconomy(split=None):
    """fueleconomy.gov vehicles.csv (all model years), cleaned, with split by model family."""
    f = pd.read_parquet(f'{PROC}/fe/fe_vehicles.parquet').merge(pd.read_parquet(f'{PROC}/fe/fe_split.parquet'), on='id')
    return f[f.split == split] if split else f


def cycles(name=None, duty=None):
    """Dynamometer schedules in trip format. name -> (meta, arrays); otherwise dict of all (optionally filtered by duty
    'light' | 'heavy' | 'motorcycle')."""
    idx = pd.read_parquet(f'{PROC}/cycles/index.parquet')
    if name is not None:
        return load_trip(f'{PROC}/cycles/{name}.npz')
    if duty is not None: idx = idx[idx.duty == duty]
    return {n: load_trip(f'{PROC}/cycles/{n}.npz') for n in idx.trip_id}


@functools.lru_cache(maxsize=1)
def hd_data():
    """Heavy-duty real data:
    ztbus_legs     : 1 Hz bus legs (trip format + elec_power, passengers, traction_force), load with load_trips(..., source='ZTBus')
    ztbus_missions : 1409 mission-level records (distance, energy, passengers, temperature) for 2 buses, with split
    mersin_trips   : 1456 trip-level diesel/CNG bus records on route 33M (fuel, duration, stops, passengers, AC)
    icct_trucks    : measured chassis-dyno fuel consumption of 3 tractor-trailers on 5 cycles (ICCT 2018)"""
    return {'ztbus_legs': pd.read_parquet(f'{PROC}/hd/ztbus_legs.parquet'),
            'ztbus_missions': pd.read_parquet(f'{PROC}/hd/ztbus_missions.parquet'),
            'mersin_trips': pd.read_parquet(f'{PROC}/hd/mersin_trips.parquet'),
            'icct_trucks': pd.read_csv(f'{PROC}/hd/published_truck_results.csv')}


def baselines(table='ved_exam'):
    """Classical baseline predictions and errors (see data/DATA.md). table: 'ved_exam' | 'ved_dev' | 'summary'."""
    if table == 'summary':
        return json.load(open(f'{PROC}/baselines/summary.json'))
    return pd.read_parquet(f'{PROC}/baselines/{table}.parquet')
