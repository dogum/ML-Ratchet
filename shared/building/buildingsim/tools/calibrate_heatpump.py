"""Physics 1.3, the heat-pump signature: fit f_cop on RECS dev fold A and tau_e on HEAPO dev fold A, test on fold B,
and measure the reader AUC (heat pump vs resistance) on RECS dev. Plan: shared/building/NOTEBOOK.md, 17 Nov 2026.

    python buildingsim/tools/calibrate_heatpump.py recs <physics> [f_cop ...]   # twins of RECS dev electric-heat homes
    python buildingsim/tools/calibrate_heatpump.py heapo <physics> [f_cop]      # twins of the 25 HEAPO dev homes
    python buildingsim/tools/calibrate_heatpump.py fit                          # f_cop (RECS A), tau_e (HEAPO A)
    python buildingsim/tools/calibrate_heatpump.py report                       # checks/out/heatpump_fit.json

Dev only: bd.recs(split='dev'); HEAPO homes with role 'dev' in models/buildingdetective/training/
split_exam2_heapo.json (read-only; processed series in models/buildingdetective/work/exam2/heapo/proc). Never a
HEAPO exam home, HUE or WPuQ. tau_e (the emitter store) acts after the time loop on the heat-pump electricity
(campus.emitter_store), so the HEAPO grid over tau_e is applied offline to one run per home, with the same function.
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, BLDG_CODE)
sys.path.insert(0, os.path.join(BLDG_CODE, 'buildingsim'))
sys.path.insert(0, os.path.join(BLDG_CODE, 'checks'))
os.environ.setdefault('OMP_NUM_THREADS', '1')

REPO = os.path.dirname(os.path.dirname(BLDG_CODE))
DET = os.path.join(REPO, 'models', 'buildingdetective')
HEAPO_SPLIT = os.path.join(DET, 'training', 'split_exam2_heapo.json')
HEAPO_WORK = os.path.join(DET, 'work', 'exam2', 'heapo')
OUT = os.path.join(BLDG_ROOT, 'checks', 'out', 'heatpump_fit')
F_COP = (0.6, 0.7, 0.8, 0.9, 1.0)
TAU = (0.0, 3.0, 6.0, 12.0, 24.0)
SEED = 20261117


# ------------------------------------------------------------------------------------------------------------ RECS
def recs_label(row):
    he, fu = row['heating_equipment'], row['heating_fuel']
    if he in ('heat_pump_air', 'heat_pump_ground'):
        return 1
    if he in ('baseboard_resistance', 'electric_furnace') or (fu == 'electricity' and he == 'portable_electric'):
        return 0
    return None


def recs_rows():
    import buildingdata as bd
    r = bd.recs(split='dev')
    rows = []
    for _, x in r.iterrows():
        y = recs_label(x)
        if y is not None and np.isfinite(x['elec_kWh']) and x['elec_kWh'] > 0 and x['floor_area_m2'] > 0:
            rows.append(x)
    rng = np.random.default_rng(SEED)
    fold = rng.permutation(len(rows)) % 2           # 0 = fold A (fit), 1 = fold B (test)
    return rows, fold


def _recs_job(job):
    i, row, physics, fcops = job
    from buildingsim import simulate, read_epw, epw_path
    from buildingsim.stock import recs_to_spec
    from buildingsim.generator import weather_hint
    spec = recs_to_spec(row, seed=0, with_band=False)
    if spec is None:
        return i, None
    city = weather_hint(str(row['climate_zone']))
    w = read_epw(epw_path(city))
    res = {}
    for f in fcops:
        o = {'physics': physics}
        if physics != '1.0':
            o['campus'] = dict(f_cop=f)
        out = simulate(spec, w, '2019-01-01', '2020-01-01', dt='1h', options=o)
        T = out['ts']['T_out']
        Td = T.resample('D').mean()
        res[str(f)] = dict(elec_kWh=out['summary']['kWh_by_meter']['electricity'],
                           heat_elec_kWh=out['summary']['kWh_by_end_use']['heating_electricity'],
                           hdd=float(np.maximum(18.0 - Td, 0).sum()), cdd=float(np.maximum(Td - 18.0, 0).sum()),
                           area=float(out['spec']['floor_area_m2']), system=out['spec']['system'])
        if spec['system'] != 'heat_pump_air':
            break                                     # f_cop acts only on air-source units
    return i, res


def recs_run(physics, fcops):
    os.makedirs(OUT, exist_ok=True)
    rows, fold = recs_rows()
    path = os.path.join(OUT, f'recs_p{physics}.json')
    have = json.load(open(path)) if os.path.exists(path) else {}
    jobs = [(i, r.to_dict(), physics, fcops) for i, r in enumerate(rows)
            if not all(str(f) in have.get(str(r['id']), {}) for f in fcops)]
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        for k, (i, res) in enumerate(ex.map(_recs_job, jobs, chunksize=4)):
            have.setdefault(str(rows[i]['id']), {}).update(res or {'none': None})
            if k % 50 == 0:
                print(k, len(jobs), f'{time.time() - t0:.0f} s', flush=True)
    with open(path, 'w') as f:
        json.dump(have, f)


def recs_table(physics, f_cop):
    """Real and twin rows of every RECS dev electric-heat respondent with a twin."""
    rows, fold = recs_rows()
    sim = json.load(open(os.path.join(OUT, f'recs_p{physics}.json')))
    out = []
    for r, fd in zip(rows, fold):
        s = sim.get(str(r['id']), {})
        s = s.get(str(f_cop)) or s.get('1.0') or (next(iter(s.values())) if s and 'none' not in s else None)
        if not s:
            continue
        out.append(dict(id=r['id'], y=recs_label(r), fold=int(fd), housing=r['housing_type'],
                        real_kwh=float(r['elec_kWh']), real_hdd=float(r['hdd18C_year']),
                        real_cdd=float(r['cdd18C_year']), real_area=float(r['floor_area_m2']),
                        sim_kwh=s['elec_kWh'], sim_hdd=s['hdd'], sim_cdd=s['cdd'], sim_area=s['area'],
                        sim_heat=s['heat_elec_kWh']))
    return pd.DataFrame(out)


def feats(d, side):
    return np.c_[np.log(d[f'{side}_kwh'] / d[f'{side}_area']), np.log(d[f'{side}_hdd'] + 50),
                 np.log(d[f'{side}_cdd'] + 50), np.log(d[f'{side}_area'])]


def logit_fit(X, y, l2=1e-2, it=60):
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + l2 * w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def auc(s, y):
    from scipy.stats import rankdata
    rk = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reader_auc(d, test_mask):
    """Logistic reader fitted on the twins (all rows), applied to the real rows in test_mask."""
    Xs, Xr = feats(d, 'sim'), feats(d, 'real')
    mu, sd = Xs.mean(0), Xs.std(0)
    w = logit_fit((Xs - mu) / sd, d.y.values)
    sc = np.c_[np.ones(len(Xr)), (Xr - mu) / sd] @ w
    ss = np.c_[np.ones(len(Xs)), (Xs - mu) / sd] @ w
    y = d.y.values
    return dict(real_auc=auc(sc[test_mask], y[test_mask]), sim_auc=auc(ss[test_mask], y[test_mask]),
                n=int(test_mask.sum()), n_hp=int(y[test_mask].sum()))


def hp_coef(d, side):
    """Heat-pump coefficient of log(kWh/m2) ~ log HDD + log CDD + log area + hp (least squares)."""
    X = np.c_[np.ones(len(d)), feats(d, side)[:, 1:], d.y.values]
    c, *_ = np.linalg.lstsq(X, feats(d, side)[:, 0], rcond=None)
    return float(c[-1])


# ------------------------------------------------------------------------------------------------------------ HEAPO
def heapo_dev():
    sp = json.load(open(HEAPO_SPLIT))
    dev = sorted((k for k, v in sp['units'].items() if v['role'] == 'dev'), key=int)
    p = pd.read_csv(os.path.join(HEAPO_WORK, 'x', 'heapo_data', 'reports', 'protocols.csv'), sep=';',
                    low_memory=False)
    p['h'] = pd.to_numeric(p.Household_ID, errors='coerce').astype('Int64').astype(str)
    p = p[p.h.isin(dev)].sort_values(['Visit_Year', 'Visit_Date', 'Report_ID']).groupby('h').first()
    homes = []
    for j, h in enumerate(dev):
        r = p.loc[h]
        homes.append(dict(id=h, fold=j % 2, weather_id=sp['units'][h]['weather_id'],
                          area=float(r['Building_FloorAreaHeated_Total']),
                          year=int(r['Building_ConstructionYear']),
                          system='heat_pump_ground' if r['HeatPump_Installation_Type'] == 'ground-source'
                          else 'heat_pump_air', dhw_hp=bool(r['DHW_Production_ByHeatPump'])))
    return homes


def _heapo_job(job):
    h, physics, f_cop = job
    from buildingsim import simulate
    d = pd.read_parquet(os.path.join(HEAPO_WORK, 'proc', f"{h['id']}.parquet"))
    t = pd.DatetimeIndex(d['time_utc'])
    wx = pd.read_csv(os.path.join(HEAPO_WORK, 'x', 'heapo_data', 'weather_data', 'hourly', f"{h['weather_id']}.csv"),
                     sep=';')
    wx.index = pd.DatetimeIndex(pd.to_datetime(wx['Timestamp'], utc=True))
    wx = wx[~wx.index.duplicated()].reindex(t)
    w = pd.DataFrame(dict(T_out=d['T_out'].values, GHI=d['GHI'].values,
                          RH=wx['Humidity_avg_hourly'].interpolate(limit_direction='both').fillna(75.0).values,
                          wind_speed=wx['WindSpeed_hourly'].interpolate(limit_direction='both').fillna(2.0).values,
                          pressure=wx['Pressure_BarometricHeight_avg_hourly'].interpolate(
                              limit_direction='both').fillna(960.0).values * 100.0), index=t)
    meta = dict(lat=47.38, lon=8.54, elev=450.0, tz='Europe/Zurich')
    spec = dict(primary_use='residential_single', floor_area_m2=h['area'], year_built=h['year'], climate_zone='5A',
                system=h['system'], dhw=dict(fuel='heat_pump' if h['dhw_hp'] else 'electric'))
    o = {'physics': physics}
    if physics != '1.0':
        o['campus'] = dict(f_cop=f_cop, tau_e=0.0)
    out = simulate(spec, (w, meta), t[0], t[-1] + pd.Timedelta('1h'), dt='1h', options=o)
    ts = out['ts']
    return h['id'], dict(time=[str(x) for x in ts.index], elec=ts['meter_electricity_kW'].tolist(),
                         heat_elec=ts['heating_electricity_kW'].tolist(), real=d['elec_kW'].tolist(),
                         T=d['T_out'].tolist())


def heapo_run(physics, f_cop):
    os.makedirs(OUT, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    res = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        for hid, r in ex.map(_heapo_job, [(h, physics, f_cop) for h in heapo_dev()]):
            res[hid] = r
            print('heapo', hid, flush=True)
    with open(os.path.join(OUT, f'heapo_p{physics}_f{f_cop}.json'), 'w') as f:
        json.dump(res, f)


def heapo_stats(r, tau):
    """Winter hour-of-day profile r (Dec-Feb, each series / its mean), heating slope ratio (daily mean kW vs daily T
    below 12 C, twin / real), winter night/day (0-5 h vs 10-16 h local), annual ratio."""
    from buildingsim.campus import emitter_store
    t = pd.DatetimeIndex(r['time']).tz_convert('Europe/Zurich')
    real = np.asarray(r['real'], float)
    sim = np.asarray(r['elec'], float)
    if tau > 0:
        he = np.asarray(r['heat_elec'], float)
        sim = sim - he + emitter_store(he, 3600.0, tau)
    ok = np.isfinite(real)
    win = np.isin(t.month, [12, 1, 2]) & ok
    pr = pd.Series(real[win] / real[win].mean()).groupby(t.hour[win]).mean().values
    ps_ = pd.Series(sim[win] / sim[win].mean()).groupby(t.hour[win]).mean().values
    day = t.normalize()
    dr = pd.Series(np.where(ok, real, np.nan)).groupby(day).mean()
    ds = pd.Series(np.where(ok, sim, np.nan)).groupby(day).mean()
    dT = pd.Series(np.asarray(r['T'], float)).groupby(day).mean()
    m = (dT < 12) & dr.notna()
    sr = np.polyfit(dT[m], dr[m], 1)[0]
    ss = np.polyfit(dT[m], ds[m], 1)[0]
    night = win & (t.hour < 5)
    dayh = win & (t.hour >= 10) & (t.hour < 16)
    return dict(profile_r=float(np.corrcoef(pr, ps_)[0, 1]), slope_ratio=float(ss / sr),
                night_day_real=float(real[night].mean() / real[dayh].mean()),
                night_day_sim=float(sim[night].mean() / sim[dayh].mean()),
                annual_ratio=float(sim[ok].mean() / real[ok].mean()))


def heapo_summary(res, homes, tau, fold=None):
    st = [heapo_stats(res[h['id']], tau) for h in homes if fold is None or h['fold'] == fold]
    d = pd.DataFrame(st)
    return dict(n=len(d), profile_r_median=float(d.profile_r.median()), slope_ratio_median=float(d.slope_ratio.median()),
                night_day_real_median=float(d.night_day_real.median()),
                night_day_sim_median=float(d.night_day_sim.median()), annual_ratio_median=float(d.annual_ratio.median()))


def do_fit():
    d = recs_table('1.3', 1.0)
    A = d[d.fold == 0]
    real_c = hp_coef(A, 'real')
    best = None
    for f in F_COP:
        dA = recs_table('1.3', f)
        dA = dA[dA.fold == 0]
        c = hp_coef(dA, 'sim')
        print('f_cop', f, 'sim hp coef', round(c, 3), 'real', round(real_c, 3))
        if best is None or abs(c - real_c) < best[0]:
            best = (abs(c - real_c), f, c)
    f_cop = best[1]
    homes = heapo_dev()
    res = json.load(open(os.path.join(OUT, f'heapo_p1.3_f{f_cop}.json')))
    tb = None
    for tau in TAU:
        s = heapo_summary(res, homes, tau, fold=0)
        print('tau', tau, s)
        if tb is None or s['profile_r_median'] > tb[0]:
            tb = (s['profile_r_median'], tau)
    pf = os.path.join(BLDG_CODE, 'buildingsim', 'buildingsim', 'data', 'physics.json')
    P = json.load(open(pf))
    P.update(f_cop=f_cop, tau_e=tb[1])
    P['_heatpump'] = ('physics 1.3 fitted by tools/calibrate_heatpump.py: f_cop on RECS dev fold A, tau_e on HEAPO dev '
                      'fold A; T_limit = median of 4 HEAPO dev protocol heating limits; report checks/heatpump_fit.json')
    with open(pf, 'w') as f:
        json.dump(P, f, indent=1)
    with open(os.path.join(OUT, 'choice.json'), 'w') as f:
        json.dump(dict(f_cop=f_cop, real_hp_coef_A=real_c, sim_hp_coef_A=best[2], tau_e=tb[1],
                       profile_r_A=tb[0]), f, indent=1)
    print('fit f_cop', f_cop, 'tau_e', tb[1])


def report():
    from _common import write
    ch = json.load(open(os.path.join(OUT, 'choice.json')))
    out = dict(choice=ch)
    for name, phys, f in (('physics_1_0', '1.0', 1.0), ('physics_1_3', '1.3', ch['f_cop'])):
        d = recs_table(phys, f)
        B = (d.fold == 1).values
        out[f'recs_{name}'] = dict(reader_fold_B=reader_auc(d, B), reader_all=reader_auc(d, np.ones(len(d), bool)),
                                   hp_coef_real_B=hp_coef(d[B], 'real'), hp_coef_sim_B=hp_coef(d[B], 'sim'),
                                   hp_coef_real_all=hp_coef(d, 'real'), hp_coef_sim_all=hp_coef(d, 'sim'),
                                   n=len(d))
    # ceiling: the same reader fitted on the REAL rows (5-fold CV), i.e. what annual meters carry
    d = recs_table('1.0', 1.0)
    X = feats(d, 'real')
    X = (X - X.mean(0)) / X.std(0)
    y = d.y.values
    f5 = np.random.default_rng(0).integers(0, 5, len(y))
    s = np.zeros(len(y))
    for k in range(5):
        w = logit_fit(X[f5 != k], y[f5 != k])
        s[f5 == k] = np.c_[np.ones((f5 == k).sum()), X[f5 == k]] @ w
    out['recs_real_fitted_ceiling_cv5'] = auc(s, y)
    homes = heapo_dev()
    r10 = json.load(open(os.path.join(OUT, 'heapo_p1.0_f1.0.json')))
    r13 = json.load(open(os.path.join(OUT, f"heapo_p1.3_f{ch['f_cop']}.json")))
    for fold, nm in ((0, 'A'), (1, 'B'), (None, 'all')):
        out[f'heapo_physics_1_0_{nm}'] = heapo_summary(r10, homes, 0.0, fold)
        out[f'heapo_physics_1_3_{nm}'] = heapo_summary(r13, homes, ch['tau_e'], fold)
    write('heatpump_fit.json', dict(what='physics 1.3 heat-pump signature: RECS dev reader AUC, HEAPO dev profiles',
                                    grid=dict(f_cop=F_COP, tau_e=TAU), **out))
    for k, v in out.items():
        print(k, v)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'recs':
        recs_run(sys.argv[2], [float(x) for x in sys.argv[3:]] or [1.0])
    elif cmd == 'heapo':
        heapo_run(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
    elif cmd == 'fit':
        do_fit()
    else:
        report()
