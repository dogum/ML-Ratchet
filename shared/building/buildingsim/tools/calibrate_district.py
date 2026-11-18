"""Fit the physics 1.1 district plant parameters (campus.py) on BDG2 DEV buildings only, 2016; cross-check on 2017 and
leave-one-site-out. Plan: shared/building/NOTEBOOK.md, 17 Nov 2026.

    python buildingsim/tools/calibrate_district.py run       # simulate the grid (<= 3 workers), cache in checks/out
    python buildingsim/tools/calibrate_district.py fit       # fit, cross-check, write data/physics.json + report

Grid: f_rh (summer reheat share, in the time loop) x T_hw_off (hot-water loop lockout, also gates the reheat) are
simulated with every loop base at 0; T_chw_off and the three loop bases are applied offline, exactly as campus.py
does (base = q x floor area x loop-on, an additive end use), so the offline meters equal the simulator's (checked
against check_bdg2.py --physics 1.1). For each grid point q_steam, q_hw, q_chw are set so the median annual sim/real
ratio of each meter is 1.0; the grid point with the lowest sum of the three medians of daily CV(RMSE) wins.
No exam building is read: the buildings come from bd.bdg2_meta(split='dev').
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE
import itertools
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

OUT = os.path.join(BLDG_ROOT, 'checks', 'out', 'district_fit')
THERMAL = ('chilledwater', 'steam', 'hotwater')
COL = {'chilledwater': 'meter_chilled_water_kW', 'steam': 'meter_steam_kW', 'hotwater': 'meter_hot_water_kW'}
F_RH = (0.0, 0.25, 0.5)
T_HW = (None, 12.8, 15.6, 18.3)
T_CHW = (None, 7.2, 10.0, 12.8)
YEARS = (2016, 2017)
ZERO = dict(q_steam=0.0, q_hw=0.0, q_chw=0.0, T_chw_off=None)


def dev_buildings():
    import buildingdata as bd
    meta = bd.bdg2_meta(split='dev')
    rows = []
    for _, b in meta.iterrows():
        ms = [m for m in THERMAL if b.get(f'q_ok_{m}')]
        if ms:
            rows.append(dict(id=b['id'], site=b['site_id'], use=b['primary_use'], meters=ms))
    return rows


def _weather(site):
    import buildingdata as bd
    w = bd.site_weather(site).copy()
    if 'LW_down' in w:
        w['IR_horiz'] = w['LW_down']
    wm = bd.site_weather_meta(site)
    return w, dict(lat=wm['lat'], lon=wm['lon'], elev=wm.get('elev', 0.0), tz=wm['tz'])


def sim_one(job):
    """One building, one year, every (f_rh, T_hw_off) the building needs; writes one npz."""
    bid, site, year = job
    path = os.path.join(OUT, f'{bid}_{year}.npz')
    if os.path.exists(path):
        return path
    import buildingdata as bd
    from buildingsim import simulate
    from buildingsim.campus import district_kinds
    from buildingsim.spec import fill_spec
    from check_bdg2 import spec_from_bdg2
    w, wm = _weather(site)
    spec = spec_from_bdg2(bd.bdg2_spec(bid))
    start = pd.Timestamp(f'{year}-01-01', tz=wm['tz']).tz_convert('UTC')
    end = pd.Timestamp(f'{year + 1}-01-01', tz=wm['tz']).tz_convert('UTC')
    kind = district_kinds(fill_spec(dict(spec, lat=wm['lat'], lon=wm['lon'])))[1]
    t_hw = T_HW if kind == 'hot_water' else (None,)
    arrays = {}
    out = None
    for f_rh, th in itertools.product(F_RH, t_hw):
        out = simulate(spec, (w, wm), start, end, dt='1h',
                       options={'physics': '1.1', 'campus': dict(ZERO, f_rh=f_rh, T_hw_off=th)})
        ts = out['ts']
        for m, c in COL.items():
            if c in ts:
                arrays[f'{m}|{f_rh}|{th}'] = ts[c].to_numpy(float)
    ts = out['ts']
    loc = ts.index.tz_convert(wm['tz'])
    real = {}
    for m in THERMAL:
        try:
            r = bd.bdg2(m, building=bid)
        except Exception:
            continue
        if r is not None and r.shape[1]:
            real[f'real|{m}'] = r.iloc[:, 0].reindex(ts.index).to_numpy(float)
    np.savez_compressed(path, T_out=ts['T_out'].to_numpy(float), day=loc.normalize().asi8, month=loc.month.values,
                        hour=loc.hour.values, dow=loc.day_of_week.values, area=float(out['spec']['floor_area_m2']),
                        kind=kind, **arrays, **real)
    return path


def run():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(b['id'], b['site'], y) for b in dev_buildings() for y in YEARS]
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        for p in ex.map(sim_one, jobs):
            print('done', os.path.basename(p), f'{time.time() - t0:.0f} s', flush=True)


# ------------------------------------------------------------------------------------------------------- offline fit
def load(year):
    data = []
    for b in dev_buildings():
        z = np.load(os.path.join(OUT, f"{b['id']}_{year}.npz"), allow_pickle=False)
        d = {k: z[k] for k in z.files}
        T = pd.Series(d['T_out'])
        d['rm'] = T.rolling(7 * 24, min_periods=1).mean().to_numpy()      # as campus.setup_district at 1 h
        for m in b['meters']:
            r = d.get(f'real|{m}')
            if r is None:
                continue
            ok = np.isfinite(r)
            if ok.sum() < 4000 or np.nansum(r) <= 0:
                continue
            data.append(dict(b=b, m=m, d=d, ok=ok, real=r))
    return data


def meter(x, th):
    """Twin meter of one building-meter at parameters th (offline twin of campus.setup_district + outputs)."""
    from buildingsim.campus import ALWAYS_COOL_USES
    d, m = x['d'], x['m']
    t_hw = th['T_hw_off'] if str(d['kind']) == 'hot_water' else None
    s = d[f"{m}|{th['f_rh']}|{t_hw}"]
    A = float(d['area'])
    if m == 'chilledwater':
        on = np.ones(len(s)) if (th['T_chw_off'] is None or x['b']['use'] in ALWAYS_COOL_USES) else \
            (d['rm'] > th['T_chw_off']).astype(float)
        q = th['q_chw']
    else:
        on = np.ones(len(s)) if t_hw is None else (d['rm'] < t_hw).astype(float)
        q = th['q_hw'] if str(d['kind']) == 'hot_water' else th['q_steam']
    return s + q * A * on / 1000.0


def stats(x, sv):
    d, ok, rv = x['d'], x['ok'], x['real']
    r, s = rv[ok], sv[ok]
    day = d['day'][ok]
    dr = pd.Series(r).groupby(day).mean()
    ds = pd.Series(s).groupby(day).mean()
    month, hour, dow = d['month'][ok], d['hour'][ok], d['dow'][ok]
    wk = dow < 5
    night = wk & ((hour < 5) | (hour >= 22))
    dayh = wk & (hour >= 10) & (hour < 16)
    summer = np.isin(month, [6, 7, 8])
    winter = np.isin(month, [12, 1, 2])
    lv = lambda v, mk: float(v[mk].mean() / max(v.mean(), 1e-9))  # noqa: E731
    return dict(building=x['b']['id'], site=x['b']['site'], meter=x['m'], ratio=float(s.mean() / r.mean()),
                daily_r=float(np.corrcoef(dr, ds)[0, 1]) if ds.std() > 0 else None,
                daily_cvrmse=float(np.sqrt(np.mean((dr - ds) ** 2)) / dr.mean()),
                summer_real=lv(r, summer), summer_sim=lv(s, summer), winter_real=lv(r, winter),
                winter_sim=lv(s, winter), night_day_real=float(r[night].mean() / max(r[dayh].mean(), 1e-9)),
                night_day_sim=float(s[night].mean() / max(s[dayh].mean(), 1e-9)))


def solve_q(data, th, m):
    """q (W/m2) making the median annual ratio of meter m equal to 1 (bisection; 0 if already >= 1)."""
    xs = [x for x in data if x['m'] == m]
    if not xs:
        return 0.0
    key = {'chilledwater': 'q_chw', 'steam': 'q_steam', 'hotwater': 'q_hw'}[m]

    def med(q):
        t = dict(th, **{key: q})
        return np.median([meter(x, t)[x['ok']].mean() / x['real'][x['ok']].mean() for x in xs])
    if med(0.0) >= 1.0:
        return 0.0
    lo, hi = 0.0, 400.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if med(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit(data):
    """Best grid point and its q's on these building-meters."""
    best = None
    for f_rh, t_hw, t_chw in itertools.product(F_RH, T_HW, T_CHW):
        th = dict(f_rh=f_rh, T_hw_off=t_hw, T_chw_off=t_chw, q_steam=0.0, q_hw=0.0, q_chw=0.0)
        for m in THERMAL:
            th[{'chilledwater': 'q_chw', 'steam': 'q_steam', 'hotwater': 'q_hw'}[m]] = solve_q(data, th, m)
        cv = {m: float(np.median([stats(x, meter(x, th))['daily_cvrmse'] for x in data if x['m'] == m]))
              for m in THERMAL if any(x['m'] == m for x in data)}
        obj = sum(cv.values())
        if best is None or obj < best[0]:
            best = (obj, th, cv)
    return best


def summary(rows):
    d = pd.DataFrame(rows)
    out = {}
    for m, g in d.groupby('meter'):
        out[m] = dict(n=len(g), annual_ratio_median=float(g.ratio.median()),
                      annual_ratio_p25=float(g.ratio.quantile(.25)), annual_ratio_p75=float(g.ratio.quantile(.75)),
                      within_30pct=float((np.abs(g.ratio - 1) <= 0.3).mean()),
                      daily_r_median=float(g.daily_r.median()), daily_cvrmse_median=float(g.daily_cvrmse.median()),
                      **{f'{k}_median': float(g[k].median()) for k in ('summer_real', 'summer_sim', 'winter_real',
                                                                         'winter_sim', 'night_day_real',
                                                                         'night_day_sim')})
    return out


def do_fit():
    from _common import write
    d16, d17 = load(2016), load(2017)
    obj, th, cv = fit(d16)
    print('fit 2016:', th, cv)
    base10 = dict(f_rh=0.0, T_hw_off=None, T_chw_off=None, q_steam=0.0, q_hw=0.0, q_chw=0.0)
    res = dict(fit_2016=dict(params=th, objective=obj, daily_cvrmse_median=cv),
               physics_1_0_equivalent_2016=summary([stats(x, meter(x, base10)) for x in d16]),
               in_sample_2016=summary([stats(x, meter(x, th)) for x in d16]),
               fit2016_scored_2017=summary([stats(x, meter(x, th)) for x in d17]))
    # leave one site out: refit on the other sites (2016), score the held-out site (2016 and 2017)
    loso16, loso17, per_site = [], [], {}
    for s in sorted({x['b']['site'] for x in d16}):
        _, ths, _ = fit([x for x in d16 if x['b']['site'] != s])
        per_site[s] = ths
        loso16 += [stats(x, meter(x, ths)) for x in d16 if x['b']['site'] == s]
        loso17 += [stats(x, meter(x, ths)) for x in d17 if x['b']['site'] == s]
        print('held out', s, ths)
    res.update(loso_params=per_site, loso_2016=summary(loso16), loso_2017=summary(loso17),
               loso_rows_2016=loso16, in_sample_rows_2016=[stats(x, meter(x, th)) for x in d16])
    pf = os.path.join(BLDG_CODE, 'buildingsim', 'buildingsim', 'data', 'physics.json')
    P = json.load(open(pf))
    P.update({k: (None if v is None else float(v)) for k, v in th.items()})
    P['_district'] = ('physics 1.1 fitted by tools/calibrate_district.py on the BDG2 dev buildings, 2016 '
                      f'({len(d16)} building-meters); report checks/district_fit.json')
    P.pop('_status', None)
    with open(pf, 'w') as f:
        json.dump(P, f, indent=1)
    write('district_fit.json', dict(what='physics 1.1 district plant fit on BDG2 dev (2016) with cross-checks',
                                    grid=dict(f_rh=F_RH, T_hw_off=T_HW, T_chw_off=T_CHW), **res))
    for k in ('physics_1_0_equivalent_2016', 'in_sample_2016', 'fit2016_scored_2017', 'loso_2016', 'loso_2017'):
        print('==', k)
        for m, v in res[k].items():
            print('  ', m, {kk: round(vv, 2) for kk, vv in v.items()})


if __name__ == '__main__':
    {'run': run, 'fit': do_fit}[sys.argv[1]]()
