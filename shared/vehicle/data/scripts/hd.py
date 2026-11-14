"""Heavy-duty real data -> trip format + tables.
1) ZTBus (ETH Zurich, CC-BY-4.0; Zenodo mirror 10.5281/zenodo.17470816 of 10.3929/ethz-b-000626723):
   10 complete 1 Hz missions of 2 VBZ Zurich electric (battery-trolley) buses + mission-level metadata for all 1409 missions.
   Missions are split into legs at stationary periods > 300 s (layovers); each leg is one trip with its electrical energy.
2) Mersin route 33M (Zenodo 10.5281/zenodo.22716221, CC-BY-4.0): 1456 trip-level records (diesel and CNG buses), no traces.
"""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, json, glob, hashlib, numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
ROOT = _VEH + '/data'
OUT = f'{ROOT}/proc/hd'; os.makedirs(f'{OUT}/ztbus', exist_ok=True)
SEED = 20261112
RD = 287.058


def leg_bounds(v, min_stop=300):
    still = v < 0.3
    n = len(v); cuts = []
    i = 0
    while i < n:
        if still[i]:
            j = i
            while j < n and still[j]: j += 1
            if j - i >= min_stop: cuts.append((i, j))
            i = j
        else:
            i += 1
    legs = []; start = 0
    for a, b in cuts:
        if a > start: legs.append((start, a + 30))  # keep 30 s of the stop at the end
        start = max(b - 30, 0)
    if start < n - 1: legs.append((start, n))
    return [(max(a, 0), min(b, n)) for a, b in legs if b - a >= 120]


def ztbus():
    meta_all = pd.read_csv(f'{ROOT}/raw/hd/ztbus/metaData.csv')
    files = sorted(glob.glob(f'{ROOT}/raw/hd/ztbus/samples/*.csv'))
    rng = np.random.default_rng(SEED)
    names = [os.path.basename(f)[:-4] for f in files]
    dev = [n for n in names if n.startswith('B183')]; dev = [str(rng.choice(dev))]
    d208 = [n for n in names if n.startswith('B208')]; dev.append(str(rng.choice(d208)))
    rows = []
    for f in files:
        name = os.path.basename(f)[:-4]
        d = pd.read_csv(f, low_memory=False)
        d = d.sort_values('time_unix').drop_duplicates('time_unix')
        t_abs = d.time_unix.values.astype(float)
        grid = np.arange(t_abs[0], t_abs[-1] + 1)
        I = lambda c: np.interp(grid, t_abs[d[c].notna().values], d[c].values[d[c].notna().values].astype(float)) if d[c].notna().any() else np.full(len(grid), np.nan)
        v = np.clip(I('odometry_vehicleSpeed'), 0, None)
        P = I('electric_powerDemand')
        lat = np.degrees(I('gnss_latitude')); lon = np.degrees(I('gnss_longitude'))
        alt = I('gnss_altitude'); Tk = I('temperature_ambient'); pax = I('itcs_numberOfPassengers')
        course = (np.degrees(I('gnss_course')) + 360) % 360
        gridok = np.isin(grid, t_abs)
        split = 'dev' if name in dev else 'exam'
        for k, (a, b) in enumerate(leg_bounds(v)):
            sl = slice(a, b); vv = v[sl]; n = b - a
            s = np.r_[0, np.cumsum(0.5 * (vv[1:] + vv[:-1]))]
            dist = float(s[-1])
            # grade from GNSS altitude: resample on distance (5 m), smooth sigma 50 m, differentiate
            z = alt[sl]
            if dist > 200 and np.isfinite(z).all():
                mv = np.r_[True, np.diff(s) > 0.01]
                sg = np.arange(0, dist, 5.0)
                zg = gaussian_filter1d(np.interp(sg, s[mv], z[mv]), 10, mode='nearest')
                gg = np.clip(np.gradient(zg, sg), -0.12, 0.12)
                grade = np.interp(s, sg, gg); elev = np.interp(s, sg, zg)
            else:
                grade = np.zeros(n); elev = z
            T = Tk[sl] - 273.15
            p = 101325 * (1 - 2.25577e-5 * np.nan_to_num(elev, nan=430)) ** 5.25588
            arr = dict(t=np.arange(n, dtype=float), v=vv, grade=grade, elev=elev, lat=lat[sl], lon=lon[sl], heading=course[sl],
                       T_amb=T, wind_speed=np.zeros(n), wind_dir=np.zeros(n), rho_air=p / (RD * (T + 273.15)),
                       surface=np.zeros(n, np.int8), fuel_rate=np.full(n, np.nan), engine_rpm=np.full(n, np.nan),
                       batt_current=np.full(n, np.nan), batt_voltage=np.full(n, np.nan), soc=np.full(n, np.nan),
                       aux_power=np.full(n, np.nan), elec_power=P[sl], passengers=pax[sl],
                       traction_force=I('traction_tractionForce')[sl], valid=gridok[sl].astype(np.uint8))
            arr = {kk: (x.astype(np.float32) if kk not in ('surface', 'valid') else x) for kk, x in arr.items()}
            tid = f'ZTBus_{name}_leg{k:02d}'
            np.savez_compressed(f'{OUT}/ztbus/{tid}.npz', **arr)
            E = float(np.nansum(P[sl]) / 3.6e6)
            m = dict(trip_id=tid, vehicle_id=f'ZTBus_B{name[1:4]}', source='ZTBus', date=pd.Timestamp(grid[a], unit='s').isoformat() + 'Z',
                     distance_m=dist, duration_s=float(n), T_amb_mean=float(np.nanmean(T)), split=split, mission=name,
                     elec_kWh=E, kWh_per_km=E / (dist / 1000) if dist > 500 else None, passengers_mean=float(np.nanmean(pax[sl])) if np.isfinite(pax[sl]).any() else None,
                     grid_available_frac=None, valid_frac=float(gridok[sl].mean()),
                     notes=('Measured outputs: elec_power [W] = total electrical power demand of the bus (traction + auxiliaries), '
                            'regardless of whether it came from the trolley grid or the battery; negative = regeneration. '
                            'Grade from smoothed GNSS altitude (no lidar DEM for Zurich here). No wind data (set 0).'),
                     q_label_ok=bool(dist > 1000 and gridok[sl].mean() > 0.95))
            json.dump({kk: (None if isinstance(x, float) and not np.isfinite(x) else x) for kk, x in m.items()}, open(f'{OUT}/ztbus/{tid}.json', 'w'), indent=1)
            rows.append(m)
    legs = pd.DataFrame(rows)
    legs.to_parquet(f'{OUT}/ztbus_legs.parquet', index=False)
    # mission-level table (all 1409 missions; labels without traces)
    meta_all['split'] = np.where(meta_all.name.isin(dev), 'dev', 'exam')
    meta_all['traced'] = meta_all.name.isin(names)
    meta_all['kWh'] = meta_all.energyConsumption / 3.6e6
    meta_all['kWh_per_km'] = meta_all.kWh / (meta_all.drivenDistance / 1000)
    meta_all['T_amb_mean_C'] = meta_all.temperature_ambient_mean - 273.15
    meta_all.to_parquet(f'{OUT}/ztbus_missions.parquet', index=False)
    # sanity: integrated power vs metadata energy for the traced missions
    chk = []
    for f in files:
        name = os.path.basename(f)[:-4]
        d = pd.read_csv(f, usecols=['time_unix', 'electric_powerDemand', 'odometry_vehicleSpeed'])
        dt = np.diff(d.time_unix.values, append=d.time_unix.values[-1] + 1)
        E = float(np.nansum(d.electric_powerDemand.values * np.clip(dt, 0, 5)))
        Dist = float(np.nansum(d.odometry_vehicleSpeed.values * np.clip(dt, 0, 5)))
        mm = meta_all.set_index('name').loc[name]
        chk.append(dict(mission=name, E_trace_MJ=E / 1e6, E_meta_MJ=mm.energyConsumption / 1e6, dist_trace_km=Dist / 1000, dist_meta_km=mm.drivenDistance / 1000))
    chk = pd.DataFrame(chk); chk['E_ratio'] = chk.E_trace_MJ / chk.E_meta_MJ; chk['d_ratio'] = chk.dist_trace_km / chk.dist_meta_km
    chk.to_csv(f'{ROOT}/proc/checks/ztbus_trace_vs_meta.csv', index=False)
    print(chk.round(3).to_string())
    print(legs.groupby('split').agg(legs=('trip_id', 'size'), km=('distance_m', lambda s: s.sum() / 1000), kWh=('elec_kWh', 'sum')))
    print(legs.kWh_per_km.describe())
    return dev


def mersin():
    x = pd.read_excel(f'{ROOT}/raw/hd/mersin/mersin_33M_bus_dataset.xlsx', sheet_name=None)
    out = []
    for fuel, d in (('diesel', x['Diesel']), ('cng', x['CNG'])):
        d = d.copy()
        d['fuel'] = fuel
        d['vehicle_id'] = f'Mersin33M_{fuel}_bus'
        d['date'] = pd.to_datetime(d.Date)
        d['distance_m'] = d.Distance_km * 1000
        d['duration_s'] = d.Trip_duration_h * 3600
        d['v_mean'] = d.distance_m / d.duration_s
        d['ac_on'] = d.AC_use.astype(str).str.contains('EVET')
        if fuel == 'diesel':
            d['fuel_L'] = d.Fuel_consumed_L; d['fuel_g'] = d.fuel_L * 832.0
        else:
            d['fuel_m3'] = d.Fuel_consumed_m3; d['fuel_g'] = d.fuel_m3 * 717.0 / 1.0  # m3 at ~0 C, 1 atm: ~0.717 kg/m3 (assumed)
        d['L_or_m3_per_100km'] = (d.get('fuel_L', d.get('fuel_m3'))) / (d.Distance_km / 100)
        d['split'] = np.where((d.date >= '2026-02-16') & (d.date <= '2026-02-22'), 'dev', 'exam')
        d['q_label_ok'] = (d.Stops_served >= 40) & (d.Trip_duration_h.between(0.5, 2.5))
        out.append(d)
    m = pd.concat(out, ignore_index=True)
    m['trip_id'] = 'Mersin33M_' + m.fuel + '_' + m.date.dt.strftime('%Y%m%d') + '_' + m.Trip_No.astype(str)
    m.to_parquet(f'{OUT}/mersin_trips.parquet', index=False)
    print(m.groupby(['fuel', 'split']).agg(n=('trip_id', 'size'), ok=('q_label_ok', 'sum'), per100=('L_or_m3_per_100km', 'median')))


def published_trucks():
    """Measured chassis-dynamometer fuel consumption of 3 tractor-trailers (ICCT briefing, May 2018, Table 3 + Table 4 + Fig. 8):
    F. Rodriguez, O. Delgado, R. Muncrief, 'Fuel consumption testing of tractor-trailers in the European Union and the
    United States', ICCT Briefing, May 2018. https://theicct.org/publication/fuel-consumption-testing-of-tractor-trailers-in-the-european-union-and-the-united-states/
    Testing by WVU (US truck) and TU Graz (EU trucks). Same tyre rolling resistance assumed for all three on the dyno."""
    trucks = {'BIC-US': dict(curb_mass_kg=17208, axles='6x4', engine_L='14-15', axle_ratio=2.41, Cd_EU_const=0.481, Cd_US_coastdown=0.438, reg_year=2015, std='US 2010'),
              'BIC-EU': dict(curb_mass_kg=15256, axles='4x2', engine_L='12-13', axle_ratio=2.53, Cd_EU_const=0.542, Cd_US_coastdown=0.471, reg_year=2015, std='Euro VI'),
              'AVG-EU': dict(curb_mass_kg=14865, axles='4x2', engine_L='12-13', axle_ratio=2.53, Cd_EU_const=0.511, Cd_US_coastdown=None, reg_year=2014, std='Euro VI')}
    res = [('EU_LongHaul', None, 19.3, (30.1, 29.9, 32.6)), ('EU_RegionalDelivery', None, 12.9, (31.7, 31.6, 34.3)),
           ('US_55mph_grade', 'HD_cruise_55mph_1037D', 17.2, (28.5, 28.6, 30.7)), ('US_55mph_flat', 'HD_cruise_55mph_flat', 17.2, (24.5, 25.6, 27.9)),
           ('US_ARB_transient', 'HD_transient_1037A', 17.2, (64.4, 56.4, 60.3))]
    rows = []
    for cyc, cid, pay, vals in res:
        for (tk, spec), L in zip(trucks.items(), vals):
            rows.append(dict(vehicle_id=f'ICCT2018_{tk}', truck=tk, cycle=cyc, cycle_id=cid, cycle_available=cid is not None, payload_t=pay,
                             test_mass_kg=spec['curb_mass_kg'] + pay * 1000, fuel='diesel', L_per_100km=L, g_per_km=L * 8.32,
                             **spec, engine_power_kW='300-330', transmission='AMT 12-speed (direct 12th)', cab='high-roof sleeper',
                             source='ICCT Briefing May 2018 (Rodriguez, Delgado, Muncrief), Table 3', measured='chassis dynamometer'))
    d = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    dev_truck = str(rng.choice(sorted(trucks)))
    d['split'] = np.where(d.truck == dev_truck, 'dev', 'exam')
    d.to_csv(f'{OUT}/published_truck_results.csv', index=False)
    print(d[['truck', 'cycle', 'payload_t', 'L_per_100km', 'split']].to_string())
    return dev_truck


# ---------------------------------------------------------------- ZTBus, all 1,409 missions (added 13 Nov 2026)
# `hd.py ztbus_full --zip raw/hd/ztbus/full/ZTBus_compressed.zip [--jobs N]`. Reads every mission CSV straight out of
# the ETH zip (no extraction) and cuts it into legs exactly as ztbus() does (same leg rule, grade, rho, label), plus
# two extra arrays (grid_available, door_open) and more leg meta. Writes
#   proc/hd/ztbus_full/trips/B<bus>/ZTBusFull_<mission>_legNN.npz/.json   (git-ignored, rebuildable)
#   proc/hd/ztbus_full/legs.parquet, missions.parquet, summary.json       (tracked)
#   proc/hd/ztbus_full_splits.json                                        (frozen; written once, then only checked)
# Nothing ztbus() writes is touched.
FULL = f'{OUT}/ztbus_full'
FULL_VERSION = 3          # bump to force a rebuild of the per-mission outputs
SEED_FULL = 20261113
SPLIT_FULL = f'{OUT}/ztbus_full_splits.json'
_SEASON = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
           6: 'summer', 7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn', 11: 'autumn'}


def canonical_sha256(obj):
    """sha256 of canonical JSON: sorted keys, no whitespace, UTF-8."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()


def ztbus_full_split_content(meta_all):
    """The ZTBus full split, from mission metadata only (no trace, no label). In force: the frozen rule of
    proc/splits.json (HD.ZTBus): its two dev missions are dev, every other mission is exam. Also recorded, NOT in force:
    a proposal for a larger dev side, fixed here before any full-data result was computed."""
    frozen = json.load(open(f'{ROOT}/proc/splits.json'))['HD']['ZTBus']
    dev0 = list(frozen['dev_missions'])
    names = sorted(meta_all.name)
    missions = {n: ('dev' if n in dev0 else 'exam') for n in names}
    # proposal: unit = (bus, calendar month of the mission start, UTC). Within each (bus, meteorological season)
    # stratum, max(1, round(0.2 n)) months are drawn with numpy default_rng(20261113), strata and months in sorted
    # order; the months holding the two frozen dev missions are dev as well.
    t = pd.to_datetime(meta_all.startTime_iso, utc=True)
    blk = pd.DataFrame({'name': meta_all.name, 'bus': meta_all.busNumber.astype(int), 'month': t.dt.strftime('%Y-%m'),
                        'season': t.dt.month.map(_SEASON)})
    rng = np.random.default_rng(SEED_FULL)
    dev_blocks = set()
    for (bus, season), g in sorted(blk.groupby(['bus', 'season']), key=lambda kv: (kv[0][0], kv[0][1])):
        months = sorted(g.month.unique())
        k = max(1, int(round(0.2 * len(months))))
        dev_blocks |= {(bus, m) for m in rng.choice(months, k, replace=False)}
    dev_blocks |= {(int(r.bus), r.month) for r in blk[blk.name.isin(dev0)].itertuples()}
    prop = {n: ('dev' if (int(b), m) in dev_blocks else 'exam') for n, b, m in zip(blk.name, blk.bus, blk.month)}
    prop = {n: prop[n] for n in names}
    moved = sorted(n for n in names if missions[n] == 'exam' and prop[n] == 'dev')
    # Amendment (2026-11-14, approved by the project owner): the by-month proposal is in force, except that the v1
    # sample missions graded as exam traces in v1 (proc/hd/ztbus_legs.parquet) stay exam, so v1's ZTBus trace exam is
    # unchanged. v1 results on ZTBus mission totals are re-scored on the missions that are exam under this split.
    v1_trace_exam = set(pd.read_parquet(f'{ROOT}/proc/hd/ztbus_legs.parquet').query("split == 'exam'").mission)
    pinned = sorted(n for n in names if prop[n] == 'dev' and n in v1_trace_exam)
    adopted = {n: ('exam' if n in v1_trace_exam else prop[n]) for n in names}
    count = lambda d: {s: sum(v == s for v in d.values()) for s in ('dev', 'exam')}
    return {
        'version': 2, 'frozen': True, 'created': '2026-11-13', 'amended': '2026-11-14',
        'dataset': 'ZTBus full (1,409 missions), doi:10.3929/ethz-b-000626723',
        'unit': 'mission',
        'in_force': 'v2_by_month',
        'rule': ('unit = (bus, calendar month UTC); within each (bus, meteorological season) stratum, max(1, round(0.2 n)) '
                 'months drawn with numpy default_rng(20261113) (strata sorted by bus then season name, months sorted); '
                 'months holding the two frozen dev missions are dev too; every mission takes the side of its month, except the '
                 'v1 trace-exam sample missions, which stay exam. Fixed on 2026-11-13 before any full-data trace was '
                 'processed; put in force on 2026-11-14.'),
        'why': ('Under the frozen v1 rule (proc/splits.json HD.ZTBus: one mission per bus = dev, all others = exam) the '
                'simulator could calibrate on 2 bus missions only. The other missions had never been downloaded as traces; '
                'EnergyNet v1 graded the mission totals of all 1,409 as exam, so those v1 numbers are re-scored on the '
                'missions that remain exam here, and the change is logged in the open (proc/splits.json is not edited).'),
        'counts': count(adopted),
        'missions': adopted,
        'v1_rule': ('proc/splits.json HD.ZTBus (sha256 d917df78...): "' + frozen['rule'] + '"'),
        'v1_counts': count(missions),
        'moved_from_v1_exam_to_dev': len([n for n in names if missions[n] == 'exam' and adopted[n] == 'dev']),
        'pinned_exam_v1_trace_missions': pinned,
        'seed': SEED_FULL,
        'dev_blocks': sorted(f'B{b}_{m}' for b, m in dev_blocks),
    }


def ztbus_full_split(meta_all):
    """Write the split file once; afterwards recompute and require equality (never rewrite)."""
    c = ztbus_full_split_content(meta_all)
    if os.path.exists(SPLIT_FULL):
        old = json.load(open(SPLIT_FULL, encoding='utf-8'))
        sha = old.pop('sha256_canonical_content_without_this_field', None)
        ok = (sha == canonical_sha256(old)) and old == c
        if not ok:
            print('WARNING: ztbus_full_splits.json differs from the recomputed split; the file on disk is used', flush=True)
        return old, ok
    out = dict(c); out['sha256_canonical_content_without_this_field'] = canonical_sha256(c)
    with open(SPLIT_FULL, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    return c, True


def _mission_legs(name, d, prefix):
    """ztbus()'s per-mission body (same legs, arrays and label) returning [(tid, arrays, meta)], plus extra meta."""
    d = d.sort_values('time_unix').drop_duplicates('time_unix')
    t_abs = d.time_unix.values.astype(float)
    grid = np.arange(t_abs[0], t_abs[-1] + 1)
    I = lambda c: np.interp(grid, t_abs[d[c].notna().values], d[c].values[d[c].notna().values].astype(float)) if c in d and d[c].notna().any() else np.full(len(grid), np.nan)
    v = np.clip(I('odometry_vehicleSpeed'), 0, None)
    P = I('electric_powerDemand')
    lat = np.degrees(I('gnss_latitude')); lon = np.degrees(I('gnss_longitude'))
    alt = I('gnss_altitude'); Tk = I('temperature_ambient'); pax = I('itcs_numberOfPassengers')
    course = (np.degrees(I('gnss_course')) + 360) % 360
    F = I('traction_tractionForce')
    grid_av = np.round(I('status_gridIsAvailable')); door = np.round(I('status_doorIsOpen'))
    gridok = np.isin(grid, t_abs)
    # logged (not interpolated) coverage of a signal on the 1 Hz grid
    have = lambda c: np.isin(grid, t_abs[d[c].notna().values]) if c in d else np.zeros(len(grid), bool)
    gnss_ok, pax_ok, T_ok = have('gnss_latitude'), have('itcs_numberOfPassengers'), have('temperature_ambient')
    route = pd.Series((d.itcs_busRoute if 'itcs_busRoute' in d else pd.Series('-', index=d.index)).astype(str).values,
                      index=d.time_unix.values.astype(float))
    out = []
    for k, (a, b) in enumerate(leg_bounds(v)):
        sl = slice(a, b); vv = v[sl]; n = b - a
        s = np.r_[0, np.cumsum(0.5 * (vv[1:] + vv[:-1]))]
        dist = float(s[-1])
        z = alt[sl]
        if dist > 200 and np.isfinite(z).all():
            mv = np.r_[True, np.diff(s) > 0.01]
            sg = np.arange(0, dist, 5.0)
            zg = gaussian_filter1d(np.interp(sg, s[mv], z[mv]), 10, mode='nearest')
            gg = np.clip(np.gradient(zg, sg), -0.12, 0.12)
            grade = np.interp(s, sg, gg); elev = np.interp(s, sg, zg)
        else:
            grade = np.zeros(n); elev = z
        T = Tk[sl] - 273.15
        p = 101325 * (1 - 2.25577e-5 * np.nan_to_num(elev, nan=430)) ** 5.25588
        arr = dict(t=np.arange(n, dtype=float), v=vv, grade=grade, elev=elev, lat=lat[sl], lon=lon[sl], heading=course[sl],
                   T_amb=T, wind_speed=np.zeros(n), wind_dir=np.zeros(n), rho_air=p / (RD * (T + 273.15)),
                   surface=np.zeros(n, np.int8), fuel_rate=np.full(n, np.nan), engine_rpm=np.full(n, np.nan),
                   batt_current=np.full(n, np.nan), batt_voltage=np.full(n, np.nan), soc=np.full(n, np.nan),
                   aux_power=np.full(n, np.nan), elec_power=P[sl], passengers=pax[sl],
                   traction_force=F[sl], valid=gridok[sl].astype(np.uint8),
                   grid_available=grid_av[sl], door_open=door[sl])
        arr = {kk: (x.astype(np.float32) if kk not in ('surface', 'valid') else x) for kk, x in arr.items()}
        tid = f'{prefix}_{name}_leg{k:02d}'
        E = float(np.nansum(P[sl]) / 3.6e6)
        Pl = P[sl]
        t0 = pd.Timestamp(grid[a], unit='s', tz='UTC')
        r = route[(route.index >= grid[a]) & (route.index < grid[b - 1] + 1)]
        r = r[~r.isin(['-', 'nan', ''])]
        gl = grid_av[sl]; dl = door[sl]; pl = pax[sl]
        m = dict(trip_id=tid, vehicle_id=f'ZTBus_B{name[1:4]}', source='ZTBus_full', date=pd.Timestamp(grid[a], unit='s').isoformat() + 'Z',
                 distance_m=dist, duration_s=float(n), T_amb_mean=float(np.nanmean(T)) if np.isfinite(T).any() else None,
                 mission=name, leg=k,
                 elec_kWh=E, kWh_per_km=E / (dist / 1000) if dist > 500 else None,
                 regen_kWh=float(np.nansum(np.minimum(Pl, 0)) / 3.6e6),
                 passengers_mean=float(np.nanmean(pl)) if np.isfinite(pl).any() else None,
                 passengers_max=float(np.nanmax(pl)) if np.isfinite(pl).any() else None,
                 passengers_frac=float(pax_ok[sl].mean()), T_amb_frac=float(T_ok[sl].mean()),
                 grid_available_frac=float(np.nanmean(gl)) if np.isfinite(gl).any() else None,
                 door_open_frac=float(np.nanmean(dl)) if np.isfinite(dl).any() else None,
                 n_door_openings=int(np.sum(np.diff(np.nan_to_num(dl)) > 0.5)),
                 v_mean_moving=float(vv[vv > 0.3].mean()) if (vv > 0.3).any() else None,
                 idle_frac=float((vv < 0.3).mean()),
                 route=(str(r.mode().iloc[0]) if len(r) else None), routes='|'.join(sorted(set(r))) if len(r) else None,
                 month=int(t0.month), season=_SEASON[int(t0.month)], hour_utc=int(t0.hour),
                 gnss_frac=float(gnss_ok[sl].mean()), valid_frac=float(gridok[sl].mean()),
                 notes=('Measured outputs: elec_power [W] = total electrical power demand of the bus (traction + auxiliaries), '
                        'regardless of whether it came from the trolley grid or the battery; negative = regeneration. '
                        'Grade from smoothed GNSS altitude (no lidar DEM for Zurich here). No wind data (set 0). '
                        'Extras: grid_available (1 = trolley line available), door_open.'),
                 q_label_ok=bool(dist > 1000 and gridok[sl].mean() > 0.95),
                 q_route_ok=bool(gnss_ok[sl].mean() >= 0.9))   # grade/elev/lat/lon from >= 90% logged GNSS seconds
        out.append((tid, arr, m))
    info = dict(n_seconds=int(len(grid)), n_rows=int(len(t_abs)), missing_s=int(len(grid) - gridok.sum()),
                dist_trace_km=float(np.nansum(v) / 1000), E_trace_kWh=float(np.nansum(P) / 3.6e6),
                gnss_frac=float(gnss_ok.mean()), passengers_frac=float(pax_ok.mean()), T_amb_frac=float(T_ok.mean()),
                grid_available_frac=float(np.nanmean(grid_av)) if np.isfinite(grid_av).any() else None)
    return out, info


def _full_one(job):
    """Worker: one mission from the zip -> legs on disk + a small result JSON (resume marker)."""
    import zipfile
    zpath, member, split = job
    name = os.path.basename(member)[:-4]
    bus = name[:4]
    os.makedirs(f'{FULL}/trips/{bus}', exist_ok=True)
    done = f'{FULL}/trips/{bus}/{name}.mission.json'
    if os.path.exists(done):
        r = json.load(open(done))
        if r.get('version') == FULL_VERSION and r.get('split') == split:
            return r
    with zipfile.ZipFile(zpath) as zf, zf.open(member) as f:
        d = pd.read_csv(f, low_memory=False)
    legs, info = _mission_legs(name, d, 'ZTBusFull')
    metas = []
    for tid, arr, m in legs:
        m['split'] = split
        np.savez_compressed(f'{FULL}/trips/{bus}/{tid}.npz', **arr)
        m['path'] = f'trips/{bus}/{tid}.npz'
        mm = {kk: (None if isinstance(x, float) and not np.isfinite(x) else x) for kk, x in m.items()}
        json.dump(mm, open(f'{FULL}/trips/{bus}/{tid}.json', 'w'), indent=1)
        metas.append(mm)
    r = dict(version=FULL_VERSION, mission=name, member=member, split=split, n_legs=len(metas), info=info, legs=metas)
    json.dump(r, open(done + '.tmp', 'w')); os.replace(done + '.tmp', done)
    return r


def _compare_samples_to_v1(legs):
    """The 10 sample missions are in the full zip too: their full legs must equal proc/hd/ztbus/ (v1)."""
    v1 = pd.read_parquet(f'{OUT}/ztbus_legs.parquet')
    res = {}
    for mission, g in v1.groupby('mission'):
        f = legs[legs.mission == mission]
        if not len(f):
            res[mission] = 'not in full zip'; continue
        if len(f) != len(g):
            res[mission] = f'legs {len(g)} (v1) vs {len(f)} (full)'; continue
        worst = 0.0
        for tid in g.trip_id:
            k = tid.rsplit('_leg', 1)[1]
            with np.load(f'{OUT}/ztbus/{tid}.npz') as A, np.load(f'{FULL}/trips/{mission[:4]}/ZTBusFull_{mission}_leg{k}.npz') as B:
                for c in A.files:
                    a, b = A[c].astype(float), B[c].astype(float)
                    if a.shape != b.shape:
                        worst = np.inf; break
                    m = np.isfinite(a) & np.isfinite(b)
                    if (np.isfinite(a) != np.isfinite(b)).any(): worst = np.inf
                    if m.any(): worst = max(worst, float(np.max(np.abs(a[m] - b[m]))))
        res[mission] = 'identical' if worst == 0 else f'max abs diff {worst:.3g}'
    return res


def ztbus_full(zpath, jobs=8):
    import zipfile, concurrent.futures as cf, time
    t0 = time.time()
    os.makedirs(f'{FULL}/trips', exist_ok=True)
    gi = f'{FULL}/.gitignore'
    if not os.path.exists(gi):
        open(gi, 'w').write('# per-leg arrays: rebuildable with `mlratchet data vehicle --only ztbus_full` (~GB)\ntrips/\n')
    meta_all = pd.read_csv(f'{ROOT}/raw/hd/ztbus/metaData.csv')
    split, split_ok = ztbus_full_split(meta_all)
    with zipfile.ZipFile(zpath) as zf:
        members = sorted(n for n in zf.namelist() if n.endswith('.csv') and os.path.basename(n) != 'metaData.csv'
                         and not os.path.basename(n).startswith('.') and '__MACOSX' not in n)
    names = [os.path.basename(n)[:-4] for n in members]
    print(f'{len(members)} mission CSVs in the zip; {len(set(names) & set(meta_all.name))} in metaData.csv', flush=True)
    jobs_ = [(str(zpath), mb, split['missions'].get(os.path.basename(mb)[:-4], 'unassigned')) for mb in members]
    res, bad = [], []
    with cf.ProcessPoolExecutor(max(1, int(jobs))) as ex:
        futs = {ex.submit(_full_one, j): j[1] for j in jobs_}
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                res.append(f.result())
            except Exception as e:
                bad.append((futs[f], repr(e))); print('FAILED', futs[f], repr(e), flush=True)
            if i % 100 == 0:
                print(f'{i}/{len(futs)} missions, {time.time() - t0:.0f} s', flush=True)
    res.sort(key=lambda r: r['mission'])
    legs = pd.DataFrame([m for r in res for m in r['legs']])
    legs.to_parquet(f'{FULL}/legs.parquet', index=False)
    # mission table: ETH metadata + what the trace says
    mt = meta_all.copy()
    mt['bus'] = 'B' + mt.busNumber.astype(str)
    mt['split'] = mt.name.map(split['missions'])
    mt['split_v1_rule'] = np.where(mt.name.isin(json.load(open(f'{ROOT}/proc/splits.json'))['HD']['ZTBus']['dev_missions']), 'dev', 'exam')
    mt['kWh'] = mt.energyConsumption / 3.6e6
    mt['kWh_per_km'] = mt.kWh / (mt.drivenDistance / 1000)
    mt['T_amb_mean_C'] = mt.temperature_ambient_mean - 273.15
    st = pd.to_datetime(mt.startTime_iso, utc=True)
    mt['month'] = st.dt.month; mt['season'] = mt.month.map(_SEASON); mt['year'] = st.dt.year
    tr = pd.DataFrame([dict(name=r['mission'], traced=True, n_legs=r['n_legs'],
                            n_legs_label_ok=sum(m['q_label_ok'] for m in r['legs']), **r['info']) for r in res])
    mt = mt.merge(tr, on='name', how='left')
    mt['traced'] = mt.traced.fillna(False).astype(bool)
    mt['E_ratio'] = mt.E_trace_kWh / mt.kWh
    mt['d_ratio'] = mt.dist_trace_km / (mt.drivenDistance / 1000)
    mt.to_parquet(f'{FULL}/missions.parquet', index=False)
    # summary (deterministic: no timings)
    ok = legs[legs.q_label_ok]
    smp = _compare_samples_to_v1(legs) if len(legs) else {}
    summ = dict(
        missions_in_metadata=int(len(meta_all)), missions_in_zip=len(members), missions_processed=len(res),
        missions_failed=bad, missions_not_in_zip=sorted(set(meta_all.name) - set(names)),
        zip_missions_not_in_metadata=sorted(set(names) - set(meta_all.name)),
        legs=int(len(legs)), legs_label_ok=int(len(ok)),
        hours=float(legs.duration_s.sum() / 3600) if len(legs) else 0.0, km=float(legs.distance_m.sum() / 1000) if len(legs) else 0.0,
        legs_by_split=legs.groupby('split').size().to_dict() if len(legs) else {},
        legs_label_ok_by_split=ok.groupby('split').size().to_dict() if len(ok) else {},
        missions_by_split=mt.groupby('split').size().to_dict(),
        split_sha256=split.get('sha256_canonical_content_without_this_field') or canonical_sha256(split),
        split_sha256_ok=bool(split_ok),
        trace_vs_meta=dict(E_ratio=mt.E_ratio.describe().round(5).to_dict(), d_ratio=mt.d_ratio.describe().round(5).to_dict()),
        samples_match_v1=smp)
    json.dump(summ, open(f'{FULL}/summary.json', 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != 'samples_match_v1'}, indent=1, default=str))
    print('samples vs v1:', smp)
    print(f'done in {time.time() - t0:.0f} s', flush=True)
    return summ


if __name__ == '__main__' and len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == 'ztbus_full':
    import argparse
    _p = argparse.ArgumentParser(); _p.add_argument('cmd'); _p.add_argument('--zip', required=True); _p.add_argument('--jobs', type=int, default=8)
    _a = _p.parse_args()
    ztbus_full(_a.zip, _a.jobs)
elif __name__ == '__main__':
    dev = ztbus(); mersin(); dev_truck = published_trucks()
    json.dump({'ZTBus_dev_missions': dev, 'ICCT_dev_truck': dev_truck, 'Mersin_dev': '2026-02-16..22 (7-day subsample)'}, open(f'{OUT}/hd_dev.json', 'w'))
