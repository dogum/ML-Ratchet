"""Real-data sanity check of the simulator for hybrids (report only; calibration decisions use dev only).
Each VED hybrid trip is driven by the car's RoadSim twin (warm fragment, real speed/grade/weather) with causal
strategies; simulated vs logged: fuel over valid seconds, engine-on fraction, engine starts per km, and for PHEVs the
SOC change and battery energy.
usage: simcheck.py dev|exam [extra_ckpt]"""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from hp_env import *
from roadsim import simulate
from roadsim.strategies import list_strategies


def starts_onfrac(on, valid, d_km):
    on = on.astype(bool)
    # debounce: an engine "start" needs >= 3 s on after >= 3 s off
    st = 0; run_off = 99; cur = on[0]; run_on = 0
    for i in range(1, len(on)):
        if on[i] and not on[i - 1]:
            if run_off >= 3:
                st += 1
            run_off = 0
        if not on[i]:
            run_off += 1
    return st / max(d_km, 1e-3), float(on[valid].mean()) if valid.any() else np.nan


HP_PARAMS = {}


def main(split, ckpt=None):
    import realdata as rd
    idx = rd.load_trips(split, arch=['hev_powersplit', 'hev_parallel_p2', 'phev_series', 'phev_powersplit'], label_ok=True, meta_only=True)
    idx = idx[idx.duration_s >= 180]
    n_per = 4 if split == 'dev' else 3
    pick = idx.sort_values('trip_id').sample(frac=1.0, random_state=SEED[split] + 1).groupby('vehicle_id').head(n_per).sort_values('trip_id')
    pol = None
    if ckpt:
        from hp_policy import HybridPilot, MLP
        net = MLP.load(ckpt)
    rows = []
    t0 = time.time()
    for r in pick.itertuples():
        meta, a = rd.load_trip(r.trip_id)
        tr = {k: np.nan_to_num(np.asarray(a[k], float)) for k in ['t', 'v', 'grade', 'elev', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'rho_air', 'surface']}
        tr['surface'] = tr['surface'].astype(int)
        valid = a['valid'].astype(bool)
        spec = twin_spec(r.vehicle_id)
        arch = spec['architecture']
        phev = arch.startswith('phev')
        soc_log = np.asarray(a['soc'], float)
        soc0 = float(soc_log[np.isfinite(soc_log)][0]) if phev and np.isfinite(soc_log).any() else None
        if soc0 is not None and soc0 > 1.5: soc0 /= 100.0
        fuel_log = np.nan_to_num(np.asarray(a['fuel_rate'], float))
        on_log = np.nan_to_num(np.asarray(a['engine_rpm'], float)) > 300
        d_km = meta['distance_m'] / 1e3
        F_log = float(fuel_log[valid].sum())
        s_log, of_log = starts_onfrac(on_log, valid, d_km)
        base = dict(trip=r.trip_id, vehicle=r.vehicle_id, arch=arch, dist_km=d_km, T_amb=float(np.mean(tr['T_amb'])), v_mean=float(np.mean(tr['v'])),
                    fuel_log_g=F_log, starts_km_log=s_log, onfrac_log=of_log, fuel_source=meta['fuel_source'])
        if phev and soc0 is not None:
            sl = soc_log[np.isfinite(soc_log)]
            sl = sl / 100.0 if sl.max() > 1.5 else sl
            base.update(soc0_log=float(sl[0]), dsoc_log=float(sl[-1] - sl[0]), kwh_log=float(meta.get('elec_kWh_net_discharge') or np.nan))
        strats = [s for s in list_strategies(arch) if s not in ('dp', 'mpc')]
        for sname in strats + (['hybridpilot'] if ckpt else []):
            try:
                if sname == 'hybridpilot':
                    stg = HybridPilot(net=net, preview_s=0, nav=False, **HP_PARAMS)
                else:
                    stg = sname
                o = simulate(spec, tr, strategy=stg, init_soc=soc0, init_temps=dict(WARM), record=True, options='roadsim-physics-v1')
            except Exception as ex:
                print('fail', r.trip_id, sname, ex); continue
            F_sim = float(o['fuel_rate'][valid].sum())
            s_sim, of_sim = starts_onfrac(o['eng_on'] > 0.5, valid, d_km)
            s_rot, of_rot = starts_onfrac(o['eng_rpm'] > 300, valid, d_km)
            row = dict(base, strategy=sname, fuel_sim_g=F_sim, starts_km_sim=s_sim, onfrac_sim=of_sim, starts_km_rot_sim=s_rot, onfrac_rot_sim=of_rot,
                       dsoc_sim=float(o['soc'][-1] - o['soc'][0]), kwh_sim=o['summary']['batt_kWh_out'])
            rows.append(row)
        if len(rows) % 50 < len(strats):
            print(len(rows), f'{time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(f'{ROOT}/data/simcheck_{split}.parquet')
    df['fuel_ratio'] = df.fuel_sim_g / df.fuel_log_g
    print(df.groupby(['arch', 'strategy']).agg(n=('trip', 'size'), fuel_ratio_med=('fuel_ratio', 'median'),
                                                onfrac_log=('onfrac_log', 'mean'), onfrac_sim=('onfrac_sim', 'mean'), onfrac_rot=('onfrac_rot_sim', 'mean'),
                                                starts_log=('starts_km_log', 'mean'), starts_sim=('starts_km_sim', 'mean'), starts_rot=('starts_km_rot_sim', 'mean')).round(3).to_string())


if __name__ == '__main__':
    if len(sys.argv) > 3:
        HP_PARAMS.update(json.loads(sys.argv[3]))
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
