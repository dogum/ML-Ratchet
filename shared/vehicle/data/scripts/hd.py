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


if __name__ == '__main__':
    dev = ztbus(); mersin(); dev_truck = published_trucks()
    json.dump({'ZTBus_dev_missions': dev, 'ICCT_dev_truck': dev_truck, 'Mersin_dev': '2026-02-16..22 (7-day subsample)'}, open(f'{OUT}/hd_dev.json', 'w'))
