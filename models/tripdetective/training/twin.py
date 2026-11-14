"""Twin test: simulate each DEV car's real trips with its twin spec, pass the run through the OBD log emulator with the
car's own channel set, and compare log statistics, real vs twin. Only dev cars are touched.
usage: python3 twin.py [n_trips_per_car] [tag]"""
from common import *
import realdata as rd
from roadsim import simulate
import fleet, obdlog

FSRC = {'maf_x_trims_over_afr': 'maf', 'absload_rpm_displacement': 'absload', 'obd_fuel_rate_Lph': 'pid', 'none': None}


def logstats(a, dt_valid=True):
    v = np.asarray(a['v'], float); n = len(v)
    acc = np.gradient(v)
    f = np.asarray(a['fuel_rate'], float); r = np.asarray(a['engine_rpm'], float)
    ok = np.asarray(a.get('valid', np.ones(n)), bool)
    km = max(v[ok].sum() / 1000, 1e-3)
    s = {}
    if np.isfinite(f[ok]).mean() > 0.9:
        s['fuel_g_km'] = np.nansum(f[ok]) / km
        mv = ok & (v > 2)
        s['fuel_p50_moving'] = np.nanmedian(f[mv]) if mv.sum() > 20 else np.nan
        s['fuel_p90_moving'] = np.nanpercentile(f[mv], 90) if mv.sum() > 20 else np.nan
        if np.isfinite(r[ok]).mean() > 0.9:
            idle = ok & (v < 0.3) & (r > 300)
            s['idle_fuel'] = np.nanmedian(f[idle]) if idle.sum() > 10 else np.nan
            dec = ok & (acc < -0.6) & (v > 6) & (r > 300)
            s['decel_fuel'] = np.nanmedian(f[dec]) if dec.sum() > 5 else np.nan
            cr = ok & (np.abs(acc) < 0.15) & (v > 11) & (v < 17) & (r > 300)
            s['cruise_fuel_40_60'] = np.nanmedian(f[cr]) if cr.sum() > 10 else np.nan
            s['fuel_lag1_ac'] = np.corrcoef(np.diff(f[ok])[1:], np.diff(f[ok])[:-1])[0, 1] if ok.sum() > 50 else np.nan
    if np.isfinite(r[ok]).mean() > 0.9:
        idle = ok & (v < 0.3) & (r > 300)
        s['idle_rpm'] = np.nanmedian(r[idle]) if idle.sum() > 10 else np.nan
        cr = ok & (v > 13.9) & (v < 19.4) & (r > 300)
        s['rpm_50_70kmh'] = np.nanmedian(r[cr]) if cr.sum() > 10 else np.nan
        mv = ok & (v > 3)
        s['eng_off_moving'] = np.mean(r[mv] < 100) if mv.sum() > 20 else np.nan
        st = ok & (v < 0.3)
        s['eng_off_stopped'] = np.mean(r[st] < 100) if st.sum() > 10 else np.nan
    I = np.asarray(a['batt_current'], float); V = np.asarray(a['batt_voltage'], float)
    if np.isfinite(I[ok]).mean() > 0.9:
        P = I * V / 1000
        s['batt_kWh_km'] = np.nansum(P[ok]) / 3600 / km
        st = ok & (v < 0.3)
        s['batt_kW_stopped'] = np.nanmedian(P[st]) if st.sum() > 10 else np.nan
        mv = ok & (v > 2)
        s['batt_kW_p90'] = np.nanpercentile(P[mv], 90) if mv.sum() > 20 else np.nan
        s['batt_kW_p10'] = np.nanpercentile(P[mv], 10) if mv.sum() > 20 else np.nan
    return s


def run(ntr=20, tag='v1', seed=0, prior='v1'):
    import priors, copy
    vt = rd.ved_vehicles(); dev = vt[vt.split == 'dev']
    rows = []
    for vid, row in dev.iterrows():
        try:
            spec = fleet.twin_spec(vid, row)
        except Exception as e:
            print(vid, 'spec fail', e); continue
        for m, a in rd.load_trips('dev', vehicles=[vid], label_ok=True, limit=ntr, seed=seed):
            rng = np.random.default_rng(abs(hash(m['trip_id'])) % 2**32)
            soc0 = None
            if np.isfinite(a['soc']).any(): soc0 = float(np.nanmedian(a['soc'][:30])) if np.isfinite(a['soc'][:30]).any() else None
            sp = priors.vehicle_prior(copy.deepcopy(spec), rng, prior)
            out = simulate(sp, a, init_temps=dict(WARM, battery=float(np.nanmean(a['T_amb']) + 10)) if prior != 'v1' else WARM, init_soc=soc0, options='roadsim-physics-v1')
            cfg = obdlog.logger_config(row.architecture, rng, fuel_src=FSRC.get(m['fuel_source'], 'maf'))
            cfg['fuel'] = FSRC.get(m['fuel_source'])
            cfg['rpm'] = bool(np.isfinite(a['engine_rpm']).mean() > 0.5)
            cfg['batt'] = bool(np.isfinite(a['batt_current']).mean() > 0.5)
            cfg['aux'] = bool(np.isfinite(a['aux_power']).mean() > 0.5)
            cfg['p_dropout'] = 0.0
            sim = obdlog.emulate(out, cfg, rng, spec)
            sim['valid'] = a['valid']
            sr, ss = logstats(a), logstats(sim)
            for k in set(sr) | set(ss):
                rows.append(dict(vid=vid, arch=row.architecture, trip=m['trip_id'], stat=k, real=sr.get(k, np.nan), sim=ss.get(k, np.nan),
                                 T=float(m['T_amb_mean'])))
    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_parquet(f'{ROOT}/data/twin_{tag}.parquet')
    g = d.groupby(['vid', 'arch', 'stat'])[['real', 'sim']].median().reset_index()
    g['ratio'] = g.sim / g.real
    pd.set_option('display.width', 200)
    p = g.pivot_table(index=['vid', 'arch'], columns='stat', values='ratio')
    print(p.round(2).to_string())
    pr = g.pivot_table(index=['vid', 'arch'], columns='stat', values='real')
    print('REAL'); print(pr.round(2).to_string())
    return d


if __name__ == '__main__':
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 20, sys.argv[2] if len(sys.argv) > 2 else 'v1', prior=sys.argv[3] if len(sys.argv) > 3 else 'v1')
