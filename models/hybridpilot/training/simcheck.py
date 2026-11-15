"""Simulator-real gap for hybrids (report only; nothing is tuned on exam): each VED hybrid trip is driven by the car's
calibrated RoadSim twin (hp_env.twin_spec; warm fragment, logged start SOC, real speed / grade / weather) under a
controller, and compared with the log on the trip's valid seconds, as RoadSim's own twin check does
(shared/vehicle/checks/check_ved_twin.py): simulated fuel through the OBD label model (roadsim.obd.label_fuel: MAF
fuel-cut air and missed enrichment) vs VED's estimated fuel; plug-ins on fuel + battery energy; engine-on share and
starts per km.

Gap per trip = sim / real - 1. Reported per architecture: trip median, car median (median over cars of the per-car
median), and the ratio of totals.
usage: simcheck.py dev|exam [CKPT 'json params'] [n_per_car]    -> WORK/data/simcheck_<split>[_<tag>].parquet / .json"""
import sys, os, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from multiprocessing import Pool
from hp_env import *

LHV = 0.042            # MJ/g (RoadData's label definition)
WARM_TWIN = dict(engine=85.0, oil=90.0, driveline=70.0, axle=60.0, tire_warm=1.0, cabin=21.0)
STRATS = ['aecms', 'rule', 'hybridpilot']
RULE = {'hev_powersplit': 'powersplit_rule', 'hev_parallel_p2': 'parallel_rule', 'hev_series': 'series_thermostat',
        'phev_powersplit': 'phev_cdcs', 'phev_series': 'phev_cdcs'}


def starts_per_km(on, d_km):
    on = np.asarray(on, bool)
    st = 0; run_off = 99
    for i in range(1, len(on)):            # a start needs >= 3 s off before it
        if on[i] and not on[i - 1]:
            if run_off >= 3:
                st += 1
            run_off = 0
        if not on[i]:
            run_off += 1
    return st / max(d_km, 1e-3)


def run_trip(job):
    import realdata as rd
    from roadsim import simulate, obd
    from hp_score import strat_arg
    trip_id, ckpt, params = job
    meta, a = rd.load_trip(trip_id)
    tr = {k: np.nan_to_num(np.asarray(a[k], float)) for k in ['t', 'v', 'grade', 'elev', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'rho_air']}
    tr['surface'] = np.nan_to_num(a['surface']).astype(int)
    ok = np.asarray(a['valid']) == 1
    spec = twin_spec(meta['vehicle_id'])
    arch = spec['architecture']
    phev = arch.startswith('phev')
    s = np.asarray(a['soc'], float); s = s[np.isfinite(s)]
    soc0 = float(s[0]) if len(s) and 0.02 < float(s[0]) <= 1.0 else None
    T0 = float(np.nanmean(a['T_amb']))
    init = dict(WARM_TWIN, battery=max(20.0, T0))
    fs = meta.get('fuel_source') or ''
    d_km = float(np.asarray(a['v'], float)[ok].sum()) / 1000
    fr = np.asarray(a['fuel_rate'], float)
    er = np.asarray(a['engine_rpm'], float)
    m_rpm = ok & np.isfinite(er)
    base = dict(trip=trip_id, vehicle=meta['vehicle_id'], arch=arch, mild=is_mild(spec), km=d_km, T=T0, fuel_source=fs,
                fuel_real_g=float(np.nansum(fr[ok])))
    if m_rpm.sum() > 30:
        base['eng_on_real'] = float((er[m_rpm] > 300).mean())
        base['starts_km_real'] = starts_per_km(er[ok] > 300, d_km)
    pr = np.asarray(a['batt_current'], float) * np.asarray(a['batt_voltage'], float)
    if np.isfinite(pr[ok]).mean() > 0.5:
        base['batt_real_kWh'] = float(np.nansum(pr[ok]) / 3.6e6)
    rows = []
    for nm in STRATS:
        if nm == 'hybridpilot':
            if not ckpt:
                continue
            import evaluate as E
            from hp_policy import HybridPilot
            stg = HybridPilot(**E.case_params(dict(nav=False), ckpt, params))
        else:
            stg = strat_arg(spec, RULE[arch] if nm == 'rule' else nm)
        try:
            o = simulate(spec, tr, strategy=stg, init_soc=soc0, init_temps=init, options=PHYS, check=False)
        except Exception as ex:
            rows.append(dict(base, strategy=nm, error=repr(ex)[:200])); continue
        fl = obd.label_fuel(o, spec, fs)
        r = dict(base, strategy=nm, fuel_sim_g=float(fl[ok].sum()), fuel_raw_g=float(np.asarray(o['fuel_rate'], float)[ok].sum()))
        on = np.asarray(o['eng_on'], float) > 0.5
        if m_rpm.sum() > 30:
            r['eng_on_sim'] = float(on[m_rpm].mean())
        r['starts_km_sim'] = starts_per_km(on[ok], d_km)
        pb = np.asarray(o['batt_current'], float) * np.asarray(o['batt_voltage'], float)
        r['batt_sim_kWh'] = float(pb[ok].sum() / 3.6e6)
        if phev and 'batt_real_kWh' in base:
            Er = base['fuel_real_g'] * LHV + base['batt_real_kWh'] * 3.6
            Es = r['fuel_sim_g'] * LHV + r['batt_sim_kWh'] * 3.6
            r['energy_real_MJ'] = Er; r['energy_sim_MJ'] = Es
            if Er > 0.2:
                r['err'] = 100 * (Es / Er - 1)
        elif not phev and base['fuel_real_g'] > 2:
            r['err'] = 100 * (r['fuel_sim_g'] / base['fuel_real_g'] - 1)
        rows.append(r)
    return rows


def summarize(df):
    out = {}
    df = df[df.get('err').notna()] if 'err' in df else df
    for (g, nm), d in df.groupby([np.where(df.mild, 'mild', df.arch), 'strategy']):
        car_med = d.groupby('vehicle').err.median()
        if g.startswith('phev'):
            tot = d.energy_sim_MJ.sum() / d.energy_real_MJ.sum()
        else:
            tot = d.fuel_sim_g.sum() / d.fuel_real_g.sum()
        o = dict(n_trips=int(len(d)), n_cars=int(d.vehicle.nunique()), trip_median=float(d.err.median()),
                 trip_iqr=[float(d.err.quantile(0.25)), float(d.err.quantile(0.75))], car_median=float(car_med.median()),
                 ratio_totals=float(tot))
        if 'eng_on_sim' in d and d.eng_on_real.notna().any():
            o['eng_on_pts'] = float(100 * (d.eng_on_sim - d.eng_on_real).median())
        if 'starts_km_real' in d and d.starts_km_real.notna().any():
            o['starts_km_sim'] = float(d.starts_km_sim.median()); o['starts_km_real'] = float(d.starts_km_real.median())
        out.setdefault(g, {})[nm] = o
    return out


def main(split, ckpt=None, params=None, n_per=None, tag=''):
    import realdata as rd
    idx = rd.load_trips(split, arch=['hev_powersplit', 'hev_parallel_p2', 'phev_series', 'phev_powersplit'], label_ok=True, meta_only=True)
    idx = idx[idx.duration_s >= 180]
    n_per = n_per or (12 if split == 'dev' else 3)
    pick = idx.sort_values('trip_id').sample(frac=1.0, random_state=SEED[split] + 1).groupby('vehicle_id').head(n_per).sort_values('trip_id')
    build_twins(sorted(set(pick.vehicle_id)))
    t0 = time.time()
    rows = []
    with Pool(int(os.environ.get('HP_WORKERS', '5'))) as pool:
        for rr in pool.imap_unordered(run_trip, [(t, ckpt, params or {}) for t in pick.trip_id], chunksize=2):
            rows += rr
    df = pd.DataFrame(rows)
    fn = f'{ROOT}/data/simcheck_{split}{tag}{PTAG}'
    df.to_parquet(fn + '.parquet')
    S = summarize(df)
    json.dump(dict(split=split, ckpt=ckpt, params=params, n_trips=int(df.trip.nunique()), physics=PHYS, summary=S,
                   wall_s=time.time() - t0), open(fn + '.json', 'w'), indent=1)
    for g, d in S.items():
        for nm, o in d.items():
            print(f"{g:16s} {nm:12s} trips {o['n_trips']:4d} cars {o['n_cars']:3d}  trip median {o['trip_median']:+6.1f}  car median "
                  f"{o['car_median']:+6.1f}  totals {o['ratio_totals']:.3f}  eng-on pts {o.get('eng_on_pts', float('nan')):+5.1f}  "
                  f"starts/km {o.get('starts_km_sim', float('nan')):.2f} vs {o.get('starts_km_real', float('nan')):.2f}")
    return S


if __name__ == '__main__':
    split = sys.argv[1]
    ck = sys.argv[2] if len(sys.argv) > 2 else None
    pr = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    n = int(sys.argv[4]) if len(sys.argv) > 4 else None
    tag = sys.argv[5] if len(sys.argv) > 5 else ''
    main(split, ck, pr, n, tag)
