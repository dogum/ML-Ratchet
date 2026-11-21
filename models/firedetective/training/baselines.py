"""Classical diagnostics for the three real targets (no simulation training):
  severity  Van Wagner crown-fire potential: BehavePlus-style point runs (FireSim behave) on a sample of the burned
            cells, with Nelson dead moisture from the gridMET days (hourly_from_daily) and the 15:00 LST wind, weighted
            by each day's observed growth -> fraction of cells that are passive/active crown fire.
            Also gridMET ERC and BI (max over the window) and a FWI (Canadian) max over the window.
  wind      Rothermel back-calculation: the observed head-run rate of spread per half-day step (95th-percentile
            distance of new cells from the previous perimeter / 6 h of active burning), inverted through Rothermel
            for the dominant fuel model (dev-calibrated ROS factor) for the 10 m wind; median over growth steps.
  lfmc      climatology by Globe-LFMC functional type x month from the DEV sites; the dev-site median.
Each function takes a record (real.build_record) and the gridMET daily frame of the fire.
"""
from common import *
import functools
import firesim as fs
import firedata as fd

CAL = json.load(open(f'{CHECKS}/replay_calibration.json'))   # import: was <sandbox>/fire/checks/


def group_of(num):
    return fs.simulate.fuel_group(int(num)) if hasattr(fs.simulate, 'fuel_group') else None


def _fuel_group(num):
    from firesim.simulate import fuel_group
    return fuel_group(int(num))


def day_conditions(daily, days, lat, lon):
    """Nelson dead moisture and 10 m wind at 15:00 LST for each local day (3 spin-up days from gridMET)."""
    d0 = pd.Timestamp(min(days)) - pd.Timedelta(days=4)
    d1 = pd.Timestamp(max(days)) + pd.Timedelta(days=1)
    sub = daily[(daily.index >= d0) & (daily.index <= d1)].copy()
    sub = sub.ffill().bfill()
    sub['tmmx'] = sub['tmmx'] + 273.15 if sub['tmmx'].max() < 100 else sub['tmmx']
    sub['tmmn'] = sub['tmmn'] + 273.15 if sub['tmmn'].max() < 100 else sub['tmmn']
    hw = fs.weather.hourly_from_daily(sub, lat, lon)
    md = fs.moisture.dead_moisture(hw, method='nelson')
    out = {}
    for d in days:
        t = (pd.Timestamp(d) + pd.Timedelta(hours=15 - lon / 15.0)).tz_localize('UTC')
        i = hw.index.get_indexer([t], method='nearest')[0]
        out[pd.Timestamp(d)] = dict(m1=float(md['m1h'].iloc[i]), m10=float(md['m10h'].iloc[i]),
                                    m100=float(md['m100h'].iloc[i]), ws=float(hw['wind_speed'].iloc[i]))
    return out


def crown_potential(rec, daily, n_cells=100, cal_ros=False):
    days = [pd.Timestamp(d) for d in rec['days']]
    cond = day_conditions(daily, sorted(set(days)), rec['lat'], rec['lon'])
    # growth weight per local day = observed area gained at the steps of that day
    A = np.array(rec['area_km2'])
    dA = np.diff(np.concatenate([[0.0], A]))
    w = {}
    for d, g in zip(days, dA):
        w[d] = w.get(d, 0.0) + max(g, 0.0)
    if sum(w.values()) <= 0:
        w = {d: 1.0 for d in w}
    C = rec['cells']
    n = min(n_cells, len(C['fuel']))
    if n == 0:
        return np.nan
    num = 0.0; den = 0.0
    for d, wd in w.items():
        if wd <= 0:
            continue
        c = cond[d]
        ft = []
        for i in range(n):
            b = fs.behave(int(C['fuel'][i]), m1=c['m1'], m10=c['m10'], m100=c['m100'], mherb=60, mwoody=90,
                          wind_speed=c['ws'], wind_height='10m', slope=float(C['slope'][i]), aspect=0.0,
                          cc=float(C['cc'][i]), ch=float(C['ch'][i]), cbh=float(C['cbh'][i]), cbd=float(C['cbd'][i]),
                          fmc=100)
            ft.append(b.get('fire_type', 0) >= 2)
        num += wd * float(np.mean(ft)); den += wd
    return num / den if den else np.nan


def danger_indices(rec, daily):
    days = sorted(set(pd.Timestamp(d) for d in rec['days']))
    sub = daily.reindex(days)
    out = dict(erc_max=float(sub['erc'].max()), bi_max=float(sub['bi'].max()), vpd_max=float(sub['vpd'].max()))
    # Canadian FWI from gridMET dailies (noon proxies: tmax, rmin, 10 m wind km/h, 24 h rain), started 60 days before
    try:
        d0 = days[0] - pd.Timedelta(days=60)
        s = daily[(daily.index >= d0) & (daily.index <= days[-1])].ffill()
        t = s['tmmx'] - 273.15 if s['tmmx'].max() > 100 else s['tmmx']
        dd = pd.DataFrame(dict(T=t, RH=s['rmin'], wind_kmh=s['vs'] * 3.6, rain_24h=s['pr']), index=s.index)
        F = fs.fwi.fwi_series(dd)
        out['fwi_max'] = float(F['FWI'].reindex(days).max())
    except Exception as e:  # noqa
        out['fwi_max'] = np.nan
        out['fwi_err'] = repr(e)[:120]
    return out


def ros_factor(group, first_day=False):
    f = CAL['factors'].get(group, CAL['default'])
    return f * (CAL['factor_first_day'] if first_day else 1.0) if first_day else f


def wind_backcalc(rec, daily, burn_h=6.0):
    days = [pd.Timestamp(d) for d in rec['days']]
    cond = day_conditions(daily, sorted(set(days)), rec['lat'], rec['lon'])
    C = rec['cells']
    fuels = [int(f) for f in C['fuel'] if (f < 91 or f > 99)]
    if not fuels:
        return np.nan
    vals, cnts = np.unique(fuels, return_counts=True)
    fuel = int(vals[np.argmax(cnts)])
    grp = _fuel_group(fuel)
    slope = float(np.median(C['slope'])) if len(C['slope']) else 0.0
    fac = ros_factor(grp)
    ests = []
    for k, (d, run) in enumerate(zip(days, rec['run_km'])):
        if k == 0 or run <= 0:
            continue
        ros_obs = run * 1000.0 / (burn_h * 60.0)        # m/min
        c = cond[d]
        def ros(u):
            b = fs.behave(fuel, m1=c['m1'], m10=c['m10'], m100=c['m100'], mherb=60, mwoody=90, wind_speed=u,
                          wind_height='10m', slope=slope, aspect=0.0, crown='none')
            return b['ros_mmin'] * fac
        lo, hi = 0.0, 30.0
        if ros(lo) >= ros_obs:
            ests.append(0.0); continue
        if ros(hi) <= ros_obs:
            ests.append(hi); continue
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if ros(mid) < ros_obs: lo = mid
            else: hi = mid
        ests.append(0.5 * (lo + hi))
    return float(np.median(ests)) if ests else np.nan


@functools.lru_cache(None)
def lfmc_clim():
    """Dev Globe-LFMC sites: mean LFMC by functional type x month, and the overall dev median."""
    l = pd.read_parquet(f'{fd.P}/lfmc/lfmc_us.parquet')
    ms = json.load(open(f'{fd.P}/moisture_split.json'))
    dev_sites = set(ms['dev']) if isinstance(ms['dev'], list) else set()
    sid = l['Site name']
    dl = l[sid.isin(dev_sites)] if dev_sites else l.iloc[:0]
    if not len(dl):
        raise RuntimeError('no dev LFMC sites resolved')
    v = dl['LFMC value (%)'].astype(float)
    tab = dl.assign(v=v, m=pd.DatetimeIndex(dl.date).month).groupby(['Species functional type', 'm']).v.median()
    return tab, float(v.median())


def lfmc_baselines(rec):
    tab, med = lfmc_clim()
    info = rec.get('lfmc_info')
    if not info:
        return dict(lfmc_clim=np.nan, lfmc_median=med)
    m = pd.Timestamp(rec['days'][len(rec['days']) // 2]).month
    vals = []
    for ft, key in (('Shrub', 'lfmc_shrub'), ('Tree', 'lfmc_tree')):
        if np.isfinite(info.get(key, np.nan)) and (ft, m) in tab.index:
            vals.append(tab[(ft, m)])
    return dict(lfmc_clim=float(np.mean(vals)) if vals else med, lfmc_median=med)


def all_baselines(rec, daily):
    out = dict(cfp=crown_potential(rec, daily), wind_bc=wind_backcalc(rec, daily))
    out.update(danger_indices(rec, daily))
    out.update(lfmc_baselines(rec))
    return out


if __name__ == '__main__':
    from real import load_records
    import time
    recs = [r for r in load_records(f'{REAL}/dev_records_package.pkl') if r is not None]
    res = {}
    for r in recs:
        t = time.time()
        d = fd.weather(r['fid'])['daily'].copy(); d.index = pd.DatetimeIndex(d.index).normalize()
        res[r['fid']] = all_baselines(r, d)
        print(r['fid'], round(time.time() - t, 1), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[r['fid']].items()}, flush=True)
    jdump(res, f'{ROOT}/out/baselines_dev.json')
