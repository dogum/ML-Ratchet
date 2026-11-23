"""Law 5 config: RoadSim's trip energy per km for light-duty petrol cars, hybrids and battery-electric cars.

Target: energy per km at the tank or battery (MJ/km): fuel at LHV 0.042 MJ/g (E10 petrol, RoadData's convention) or DC
battery energy, the same definition as RoadData's VED labels and EnergyNet's targets. Plug-in hybrids are left out
(their energy splits between two sources and depends on the start charge, which the law does not read).

Inputs, computed the same way from a simulated trip and a real one (`trip_inputs`), 1 Hz speed v and grade:
    v  mean speed over the trip, stops included (m/s)                      = distance / duration
    K  positive kinetic energy per metre (m/s2)                            = sum max(0, v_i^2 - v_(i-1)^2) / 2 / distance
    G  net grade (%)                                                       = 100 x (sum grade_i v_i) / distance
    U  absolute grade (%)                                                  = 100 x (sum |grade_i| v_i) / distance
    M  test mass with driver (t); D  CdA (m2); R  rolling resistance Crr x 1000; T  ambient temperature (C)
    h  1 for a hybrid, else 0; e  1 for a battery-electric car, else 0
Vehicle inputs for real trips: VED vehicles carry RoadData's mass (the class default where it has none) and
class-default CdA and Crr (EnergyNet's realfeat.py convention: CdA = CdA_class (m / m_class)^0.5); EPA test cars carry the test mass and CdA and Crr from the
EPA target road-load coefficients (C = 0.5 rho_ref CdA, A = m g Crr), as RoadSim's EPA checks do.
"""
import glob, json, os
import numpy as np, pandas as pd
from _paths import ROOT, other_work

NAME = 'road'
TITLE = 'Trip energy per km'
VERSIONS = ['p1', 'p2']           # RoadSim physics v1 (as first released) and v2 (dev-calibrated, the default since 13 Nov 2026)
PHYSICS = {'p1': 'roadsim-physics-v1', 'p2': 'roadsim-physics-v2'}
VARS = ['v', 'K', 'G', 'U', 'M', 'D', 'R', 'T', 'h', 'e']
VAR_INFO = {'v': {'label': 'mean speed, stops included', 'symbol': 'v̄', 'unit': 'm/s', 'range': [3, 30]},
            'K': {'label': 'positive kinetic energy per metre', 'symbol': 'PKE', 'unit': 'm/s²', 'range': [0.05, 0.8]},
            'G': {'label': 'net grade', 'symbol': 'g_net', 'unit': '%', 'range': [-2, 2]},
            'U': {'label': 'absolute grade', 'symbol': 'g_abs', 'unit': '%', 'range': [0, 4]},
            'M': {'label': 'mass with driver', 'symbol': 'm', 'unit': 't', 'range': [1.1, 3.2]},
            'D': {'label': 'drag area', 'symbol': 'CdA', 'unit': 'm²', 'range': [0.45, 1.4]},
            'R': {'label': 'rolling resistance', 'symbol': 'Crr', 'unit': '×10⁻³', 'range': [5, 13]},
            'T': {'label': 'ambient temperature', 'symbol': 'T', 'unit': '°C', 'range': [-15, 35]},
            'h': {'label': 'hybrid (1) or not (0)', 'symbol': 'hyb', 'unit': '', 'range': [0, 1]},
            'e': {'label': 'battery-electric (1) or not (0)', 'symbol': 'EV', 'unit': '', 'range': [0, 1]}}
TARGET = {'name': 'E', 'label': 'energy per km (fuel at LHV or DC battery)', 'symbol': 'E', 'unit': 'MJ/km'}
SEARCH_SECONDS = 600
SEARCH_KW = {}
LHV = 0.042                       # MJ/g
MI = 1609.344; GAL_L = 3.785411784
GROUPS = ('ICE', 'HEV', 'EV')
CLS = {'car': (1650, 0.62, 0.0085), 'suv': (2000, 0.85, 0.0090), 'pickup': (2450, 1.10, 0.0095), 'van': (2600, 1.25, 0.0090)}
G0 = 9.81


def group_of(arch):
    return 'ICE' if arch == 'conv' else ('EV' if arch == 'bev' else ('PHEV' if str(arch).startswith('phev') else 'HEV'))


def trip_inputs(v, grade=None, T=None):
    """v: 1 Hz speed (m/s); grade: fraction (rise / run); T: ambient (C). -> dict(v, K, G, U, T, dist_m, dur_s)"""
    v = np.nan_to_num(np.asarray(v, float)).clip(0, None)
    g = np.zeros_like(v) if grade is None else np.nan_to_num(np.asarray(grade, float))
    d = float(v.sum())
    if d <= 0:
        return None
    dv2 = np.diff(v ** 2, prepend=v[0] ** 2)
    return dict(v=d / len(v), K=float(np.clip(dv2, 0, None).sum() / 2 / d), G=float(100 * np.sum(g * v) / d),
                U=float(100 * np.sum(np.abs(g) * v) / d), T=float(np.nanmean(T)) if T is not None else 24.0,
                dist_m=d, dur_s=float(len(v)))


def vehicle_inputs(m_kg, cda, crr, arch):
    grp = group_of(arch)
    return dict(M=float(m_kg) / 1e3, D=float(cda), R=float(crr) * 1e3, h=float(grp == 'HEV'), e=float(grp == 'EV'), group=grp)


def class_cda_crr(cls, m_kg):
    mc, Cc, Rc = CLS.get(cls, CLS['car'])
    return Cc * (m_kg / mc) ** 0.5, Rc


def bank_ready(ver):
    return (len(glob.glob(f'{ROOT}/banks/road_{ver}/train_*.parquet')) >= 10
            and len(glob.glob(f'{ROOT}/banks/road_{ver}/val_*.parquet')) >= 3)


def X_of(df):
    return np.c_[[df[k].values.astype(float) for k in VARS]].T


def load_bank(ver, split):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{ROOT}/banks/road_{ver}/{split}_*.parquet'))],
                   ignore_index=True)
    ok = np.isfinite(X_of(df)).all(1) & np.isfinite(df.E) & (df.E > 0) & (df.dist_m >= 500)
    return df[ok].reset_index(drop=True)


def target(df):
    return df['E'].values.astype(float)


# ------------------------------------------------------------------ real data (RoadData; VED and EPA)
def _rd():
    import realdata as rd
    return rd


def ved_rows(split, trip_ids=None):
    """VED trips of one split (label_ok; petrol, hybrid and electric cars) with the law's inputs and E (MJ/km) from
    RoadData's per-second label (valid seconds only), as EnergyNet's realfeat.py builds it."""
    rd = _rd()
    floor = {'ICE': 1 * LHV, 'HEV': 1 * LHV, 'EV': 0.01 * 3.6}          # EnergyNet's per-trip floors (MJ)
    rows = []
    specs = {}
    for meta, arr in rd.load_trips(split, label_ok=True):
        grp = group_of(meta['architecture'])
        if grp not in GROUPS or (trip_ids is not None and meta['trip_id'] not in trip_ids):
            continue
        vid = meta['vehicle_id']
        if vid not in specs:
            s = rd.load_vehicle(vid)
            # no recorded mass: the class default, as EnergyNet's realfeat.py does (m_class)
            m = float(s['mass_kg']) + float(s.get('payload_kg') or 0) if s.get('mass_kg') else CLS.get(s['class'], CLS['car'])[0]
            cda, crr = class_cda_crr(s['class'], m)
            specs[vid] = (m, cda, crr, s['class'])
        m, cda, crr, cls = specs[vid]
        ok = arr['valid'] == 1
        if grp == 'EV':
            E = np.nan_to_num(arr['batt_current'].astype(float) * arr['batt_voltage'].astype(float)) / 1e6
        else:
            E = np.nan_to_num(arr['fuel_rate'].astype(float)) * LHV
        Etot = float(np.where(ok, E, 0.0).sum())
        if Etot <= floor[grp]:
            continue
        ti = trip_inputs(arr['v'], arr['grade'], arr['T_amb'])
        if ti is None:
            continue
        rows.append(dict(id=meta['trip_id'], vehicle_id=vid, src='VED', test='', cls=cls, arch=meta['architecture'],
                         E=Etot / (meta['distance_m'] / 1e3), E_total_MJ=Etot, dist_km=meta['distance_m'] / 1e3,
                         **{k: ti[k] for k in ('v', 'K', 'G', 'U', 'T')}, **vehicle_inputs(m, cda, crr, meta['architecture'])))
    return pd.DataFrame(rows)


EPA_TESTS = {'FTP': ('UDDS', 24.0), 'HWY': ('HWFET', 24.0), 'US06': ('US06', 24.0), 'FTPcold': ('UDDS', -6.7),
             'SC03': ('SC03', 35.0)}
EPA_COL = {'FTP': 'FTP_mpg_fix', 'HWY': 'HWY_mpg_fix', 'US06': 'US06_mpg_fix', 'FTPcold': 'FTP_cold_mpg', 'SC03': 'SC03_mpg'}
EPA_BEV_COL = {'FTP': 'CD_UDDS_kWh_per_mi', 'HWY': 'CD_HWY_kWh_per_mi'}


def epa_cycle_inputs():
    rd = _rd(); cyc = rd.cycles(); out = {}
    for test, (n, T) in EPA_TESTS.items():
        out[test] = trip_inputs(cyc[n][1]['v'], None, np.full(len(cyc[n][1]['v']), T))
    return out


def epa_rows(split):
    """EPA test cars of one split (gasoline petrol and hybrids, battery-electric), one row per vehicle and test.
    Truth as EnergyNet's exam scores it: 1/mpg (FTP = the 0.43 cold / 0.57 hot UDDS composite as published) or the
    BEV's wall kWh/mi, turned into tank or DC energy per km (petrol 745 g/L; charger 0.88)."""
    rd = _rd()
    fx = f"{other_work('energynet')}/data/epa_fixed.parquet"         # EnergyNet's corrected cycle labels (epa_fix.py)
    e = rd.epa_testcars(split=split)
    if os.path.exists(fx):
        e = e.join(pd.read_parquet(fx))
    else:
        raise FileNotFoundError(f'{fx}: EnergyNet\'s corrected EPA cycle labels (run its epa_fix.py first)')
    ci = epa_cycle_inputs(); rows = []
    for vid, r in e.iterrows():
        if r.architecture not in ('conv', 'hev', 'bev'):
            continue
        if not (r.test_mass_kg > 0 and r.A_N > 0 and r.C_N_per_mps2 > 0):
            continue
        if r.architecture != 'bev' and r.fuel != 'gasoline':
            continue
        arch = {'conv': 'conv', 'hev': 'hev_powersplit', 'bev': 'bev'}[r.architecture]
        m = float(r.test_mass_kg); cda = float(2 * r.C_N_per_mps2 / 1.1767); crr = float(r.A_N / (m * G0))
        for test in EPA_TESTS:
            col = EPA_BEV_COL.get(test) if arch == 'bev' else EPA_COL[test]
            if col is None or col not in r or not (pd.notna(r[col]) and r[col] > 0):
                continue
            E = (r[col] * 0.88 * 3.6 / (MI / 1e3)) if arch == 'bev' else (745.0 * GAL_L * LHV / r[col] / (MI / 1e3))
            rows.append(dict(id=f'{vid}|{test}', vehicle_id=vid, src='EPA', test=test, cls=str(r.veh_type), arch=arch,
                             E=float(E), family=r.family, **{k: ci[test][k] for k in ('v', 'K', 'G', 'U', 'T')},
                             **vehicle_inputs(m, cda, crr, arch)))
    return pd.DataFrame(rows)


def load_dev():
    """Dev set: every VED dev trip (8 petrol cars, 5 hybrids, 1 Leaf; plug-ins left out) plus the same number of EPA
    dev tests (a fixed draw, seed 5), so the gate weighs road trips and dynamometer tests equally."""
    p = f'{ROOT}/results/road_dev_rows.parquet'
    if os.path.exists(p):
        return pd.read_parquet(p)
    v = ved_rows('dev'); ep = epa_rows('dev')
    ep = ep.sample(min(len(ep), len(v)), random_state=5)
    d = pd.concat([v, ep], ignore_index=True)
    d = d[np.isfinite(X_of(d)).all(1) & np.isfinite(d.E) & (d.E > 0)].reset_index(drop=True)
    d.to_parquet(p)
    return d


def dev_error(y, yh):
    return float(np.median(np.abs(yh / y - 1)))      # MdAPE as a fraction


def rho(T):
    return 101325.0 / (287.05 * (np.asarray(T, float) + 273.15))


def wheel_energy(df):
    """Road-load energy at the wheels per km (MJ/km) from the law's own inputs:
    (m g Crr + 0.5 rho CdA v^2 + m a_pos + m g grade) x 1000 m, v the mean speed, a_pos the positive kinetic energy per
    metre, grade the net grade; floored at 5 % of the rolling term (a trip cannot give energy back on net)."""
    m = df['M'].values * 1e3
    roll = m * G0 * df['R'].values / 1e3
    aero = 0.5 * rho(df['T'].values) * df['D'].values * df['v'].values ** 2
    kin = m * df['K'].values
    grd = m * G0 * df['G'].values / 100
    return np.maximum(roll + aero + kin + grd, 0.05 * roll) * 1e3 / 1e6


def road_load(df, eta):
    """Textbook rival: wheel energy / eta, one eta per powertrain group (ICE, HEV, EV), fitted on dev."""
    grp = np.where(df['e'].values > 0.5, 'EV', np.where(df['h'].values > 0.5, 'HEV', 'ICE'))
    return wheel_energy(df) / np.array([eta[g] for g in grp])


def fit_eta(df):
    """eta per group: the median ratio of wheel energy to real (or simulated) energy (robust to the MdAPE metric)."""
    grp = np.where(df['e'].values > 0.5, 'EV', np.where(df['h'].values > 0.5, 'HEV', 'ICE'))
    r = wheel_energy(df) / df['E'].values
    return {g: float(np.median(r[grp == g])) for g in GROUPS if (grp == g).any()}
