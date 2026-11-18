"""BuildingSim against REAL meters: the 52 BDG2 dev buildings of the frozen split (agent B), 2016, 1 h.

The simulator only knows what BDG2 metadata gives (use, floor area, year built for some, climate zone, location,
which meters exist); everything else comes from BuildingSim defaults. The system is inferred from the meters:
chilled water + steam/hot water -> district_steam_chw; steam or hot water only -> VAV with district heat and an
electric chiller; chilled water only -> district chilled water with district heat (unmetered); electricity only ->
the use-type default. This is the "generic twin from metadata" level that Phase 2 starts from, so large errors in
annual totals are expected; the hourly SHAPE (normalised by each series' mean) is the more meaningful comparison.
Only building-meters with q_ok (agent B's quality flag) are scored. Exam buildings are never touched.

    python checks/check_bdg2.py               # v1.0 generic twins -> bdg2_dev_meters.json
    python checks/check_bdg2.py w2            # Phase 2 flags on (dev only) -> bdg2_dev_meters__w2.json; see VARIANTS
    python checks/check_bdg2.py --physics 1.1 [--year 2017]   # physics version -> bdg2_dev_meters__p1.1[_y2017].json

Physics runs (`--physics`, campus.py) add the shape statistics of every meter (summer, winter and night/day levels,
daily CV(RMSE)) and, for electricity, the daily PRISM 5P r2 of real and twin with its weekday and day-to-day
residual terms, plus a two-sample KS test of the r2 distributions. At 1.2 each building's daily activity draws are
seeded by crc32 of its id.
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, BLDG_CODE)
import buildingdata as bd  # noqa: E402
from _common import write  # noqa: E402
from buildingsim import simulate  # noqa: E402

METERS = {'electricity': 'meter_electricity_kW', 'chilledwater': 'meter_chilled_water_kW',
          'steam': 'meter_steam_kW', 'hotwater': 'meter_hot_water_kW', 'gas': 'meter_gas_kW'}


def spec_from_bdg2(s):
    s = {k: v for k, v in s.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    meters = set((s.get('meters') or {}).keys()) if isinstance(s.get('meters'), dict) else set(
        str(s.get('meters', '')).split(','))
    chw = 'chilledwater' in meters
    stm = 'steam' in meters
    hw = 'hotwater' in meters
    h = {}
    if chw and (stm or hw):
        system = 'district_steam_chw'
        if hw and not stm:
            h['district_heat'] = 'hot_water'
    elif (stm or hw) and not chw:
        system = 'vav_chiller_boiler'
        h['heat_source'] = 'district'
        if hw:
            h['district_heat'] = 'hot_water'
    elif chw:
        system = 'district_steam_chw'
    else:
        system = None
    out = dict(s)
    out.pop('meters', None)
    if system:
        out['system'] = system
    out['hvac'] = h
    if 'n_floors' not in out and out.get('floor_area_m2'):
        out['n_floors'] = max(1, int(round(np.sqrt(out['floor_area_m2']) / 30)))
    return out


VARIANTS = {
    'w2': 'BuildingDetective W1 intensity + W2 campus physics (VAV min 0.45, OA/m2 x1.5 on district buildings, as '
          'calib_campus.py) at the fitted distribution factors: district_eff 0.185 (heat), district_eff_chw 0.329',
    'w2w3': 'w2 + W3 flat base loads at their means: district heat phi 0.70, chilled water phi 0.50, electricity '
            '+0.50 x its mean (no random draws)',
    'lc': "LoadCast spec_from_meters with CAMPUS_KNOBS (24x7, OA x2.5, VAV min 0.3) and options={'holidays': 'auto'}; "
          'no meter scaling',
}
W3_MEANS = dict(phi_heat=(0.70, 0.0), phi_chw=(0.50, 0.0), elec_base=(0.50, 0.0), p_elec_base=1.0)


def variant_run(variant, bid, spec, w, wm, start, end):
    """One dev twin with the Phase 2 flags of `variant` on (dev only; same spec_from_bdg2 metadata twin)."""
    from buildingsim import fill_spec, phase2 as P
    if variant == 'lc':
        s = P.spec_from_meters(bd.bdg2_spec(bid), [m for m in METERS if m in str(bd.bdg2_spec(bid).get('meters'))],
                               P.CAMPUS_KNOBS)
        return simulate(s, (w, wm), start, end, dt='1h', options={'holidays': 'auto'}), None
    f = fill_spec(spec)
    P.apply_intensity(f, P.W1_INTENSITY.get(f['primary_use'], 1.0))
    if f['system'] == 'district_steam_chw' or f['hvac'].get('heat_source') == 'district':
        f['hvac']['vav_min_frac'] = 0.45
        f['ventilation']['L_s_m2'] = float(f['ventilation'].get('L_s_m2') or 0.3) * 1.5
        f['hvac']['district_eff'] = P.W2_CAMPUS['eff_heat_median']
        f['hvac']['district_eff_chw'] = P.W2_CAMPUS['eff_chw_median']
    out = simulate(f, (w, wm), start, end, dt='1h')
    base = None
    if variant == 'w2w3':
        ts = out['ts']
        cols = {'electricity': 'meter_electricity_kW', 'chilled_water': 'meter_chilled_water_kW',
                'steam': 'meter_steam_kW', 'hot_water': 'meter_hot_water_kW'}
        x = {k: ts[c].to_numpy(float) for k, c in cols.items() if c in ts}
        y = P.flat_base_loads(x, np.random.default_rng(0), commercial=not f['primary_use'].startswith('residential'),
                              params=W3_MEANS)
        base = {cols[k]: v for k, v in y.items()}
    return out, base


def shape_extra(rv, sv, loc, mtr):
    """Summer level (Jun-Aug mean / annual mean) and weekday night/day ratio (0-5 h vs 10-16 h local), real and sim:
    the W3 twin-test statistics."""
    summer = np.isin(loc.month, [6, 7, 8])
    wk = loc.dayofweek < 5
    night = wk & ((loc.hour < 5) | (loc.hour >= 22))
    dayh = wk & (loc.hour >= 10) & (loc.hour < 16)
    f = lambda v, m: float(v[m].mean() / max(v.mean(), 1e-9)) if m.any() else None  # noqa: E731
    g = lambda v: float(v[night].mean() / max(v[dayh].mean(), 1e-9)) if night.any() and dayh.any() else None  # noqa
    return dict(summer_over_mean_real=f(rv, summer), summer_over_mean_sim=f(sv, summer),
                night_over_day_real=g(rv), night_over_day_sim=g(sv))


def prism(Td, Ed):
    """PRISM 5P change-point fit E = b + h max(Tbh - T, 0) + c max(T - Tbc, 0), h, c >= 0, on a 1.5 K grid of balance
    points (the same fit as BuildingDetective's features.prism). Returns base, h, c, Tbh, Tbc, r2."""
    m = np.isfinite(Td) & np.isfinite(Ed)
    Td, Ed = Td[m], Ed[m]
    if len(Td) < 8 or np.std(Ed) == 0:
        return np.array([np.nanmean(Ed) if len(Ed) else 0, 0, 0, 15.0, 20.0, 0.0])
    best = None
    sst = ((Ed - Ed.mean()) ** 2).sum() + 1e-12
    for tbh in np.arange(4, 23, 1.5):
        xh = np.maximum(tbh - Td, 0)
        for tbc in np.arange(max(tbh, 10), 29, 1.5):
            xc = np.maximum(Td - tbc, 0)
            X = np.c_[np.ones_like(Td), xh, xc]
            coef, *_ = np.linalg.lstsq(X, Ed, rcond=None)
            if coef[1] < 0 or coef[2] < 0:
                cols = [0] + [k for k in (1, 2) if coef[k] >= 0]
                c2, *_ = np.linalg.lstsq(X[:, cols], Ed, rcond=None)
                coef = np.zeros(3)
                coef[cols] = c2
                coef = np.maximum(coef, [-np.inf, 0, 0])
            sse = ((X @ coef - Ed) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, coef, tbh, tbc)
    sse, coef, tbh, tbc = best
    return np.array([coef[0], coef[1], coef[2], tbh if coef[1] > 0 else 15.0, tbc if coef[2] > 0 else 20.0,
                     1 - sse / sst])


def elec_stats(d, T):
    """Daily electricity (Series of daily means, local dates) vs daily mean outdoor temperature: PRISM r2, the
    weekday-minus-weekend level (x mean), the day-to-day residual after PRISM and a weekday term (sd x mean), and that
    residual's lag-1 autocorrelation."""
    ok = d.notna() & T.notna()
    d, T = d[ok], T[ok]
    pr = prism(T.values, d.values)
    fit = pr[0] + pr[1] * np.maximum(pr[3] - T.values, 0) + pr[2] * np.maximum(T.values - pr[4], 0)
    res = d.values - fit
    wd = (d.index.day_of_week < 5).astype(float)
    X = np.c_[np.ones(len(d)), wd]
    c, *_ = np.linalg.lstsq(X, res, rcond=None)
    r2 = res - X @ c
    mu = max(d.mean(), 1e-9)
    lag1 = float(np.corrcoef(r2[1:], r2[:-1])[0, 1]) if len(r2) > 3 and np.std(r2) > 0 else 0.0
    return dict(prism_r2=float(pr[5]), weekday_effect=float(c[1] / mu), resid_cv=float(r2.std() / mu), resid_lag1=lag1)


def physics_extra(rv, sv, loc, dr, ds_, dT, mtr):
    """Shape statistics of a physics run: summer / winter level, night/day, daily CV(RMSE) (daily means, x the real
    mean); electricity adds elec_stats for real and twin."""
    e = shape_extra(rv, sv, loc, mtr)
    winter = np.isin(loc.month, [12, 1, 2])
    e['winter_over_mean_real'] = float(rv[winter].mean() / max(rv.mean(), 1e-9)) if winter.any() else None
    e['winter_over_mean_sim'] = float(sv[winter].mean() / max(sv.mean(), 1e-9)) if winter.any() else None
    ok = dr.notna() & ds_.notna()
    e['daily_cvrmse'] = float(np.sqrt(np.mean((dr[ok] - ds_[ok]) ** 2)) / max(dr[ok].mean(), 1e-9))
    if mtr == 'electricity':
        for side, d in (('real', dr), ('sim', ds_)):
            for k, v in elec_stats(d, dT).items():
                e[f'{k}_{side}'] = v
    return e


def r2_compare(g):
    """Real vs twin daily PRISM r2 distributions: quantiles and the KS two-sample test, plus the other daily terms."""
    from scipy.stats import ks_2samp
    a, b = g.prism_r2_real.dropna().values, g.prism_r2_sim.dropna().values
    ks = ks_2samp(a, b)
    out = dict(prism_r2_ks_stat=float(ks.statistic), prism_r2_ks_p=float(ks.pvalue))
    for q in (25, 50, 75):
        out[f'prism_r2_real_p{q}'] = float(np.percentile(a, q))
        out[f'prism_r2_sim_p{q}'] = float(np.percentile(b, q))
    for k in ('weekday_effect', 'resid_cv', 'resid_lag1'):
        for side in ('real', 'sim'):
            out[f'{k}_{side}_median'] = float(g[f'{k}_{side}'].median())
    return out


def main(variant=None, physics=None, year=2016, sites=None, tag=None, campus=None):
    if variant is not None and variant not in VARIANTS:
        raise SystemExit(f'variant: one of {sorted(VARIANTS)}')
    if variant is not None and physics is not None:
        raise SystemExit('a Phase 2 variant or a physics version, not both')
    if physics is None and (year != 2016 or sites or campus):
        raise SystemExit('--year / --sites need --physics')
    import zlib
    meta = bd.bdg2_meta(split='dev')
    if sites:
        meta = meta[meta.site_id.isin(sites)]
    rows = []
    wcache = {}
    for _, b in meta.iterrows():
        bid = b['id']
        site = b['site_id']
        spec = spec_from_bdg2(bd.bdg2_spec(bid))
        if site not in wcache:
            w = bd.site_weather(site).copy()
            if 'LW_down' in w:
                w['IR_horiz'] = w['LW_down']
            wm = bd.site_weather_meta(site)
            wcache[site] = (w, dict(lat=wm['lat'], lon=wm['lon'], elev=wm.get('elev', 0.0), tz=wm['tz']))
        w, wm = wcache[site]
        start = pd.Timestamp(f'{year}-01-01', tz=wm['tz']).tz_convert('UTC')
        end = pd.Timestamp(f'{year + 1}-01-01', tz=wm['tz']).tz_convert('UTC')
        base = None
        try:
            if physics is not None:
                opts = {'physics': physics, 'seed': zlib.crc32(bid.encode())}
                if campus:
                    opts['campus'] = campus
                out = simulate(spec, (w, wm), start, end, dt='1h', options=opts)
            elif variant is None:
                out = simulate(spec, (w, wm), start, end, dt='1h')
            else:
                out, base = variant_run(variant, bid, spec, w, wm, start, end)
        except Exception as e:
            print('ERR', bid, e)
            continue
        ts = out['ts']
        for mtr, col in METERS.items():
            if not b.get(f'q_ok_{mtr}'):
                continue
            real = bd.bdg2(mtr, building=bid)
            real = real.iloc[:, 0].reindex(ts.index)
            sim = ts[col] if col in ts else None
            if base is not None and col in base:
                sim = pd.Series(base[col], index=ts.index)
            if sim is None or real.notna().sum() < 4000 or real.sum() <= 0:
                continue
            m = real.notna().values
            rv, sv = real.values[m], sim.values[m]
            ann_r, ann_s = rv.mean() * 8760, sv.mean() * 8760
            rn, sn = rv / max(rv.mean(), 1e-9), sv / max(sv.mean(), 1e-9)
            loc = ts.index[m].tz_convert(wm['tz'])
            wk = loc.dayofweek < 5
            prof_r = pd.Series(rn[wk]).groupby(loc.hour[wk]).mean().values
            prof_s = pd.Series(sn[wk]).groupby(loc.hour[wk]).mean().values
            dr = pd.Series(rv, index=loc).resample('D').mean()
            ds_ = pd.Series(sv, index=loc).resample('D').mean()
            dT = pd.Series(ts['T_out'].values[m], index=loc).resample('D').mean()
            ok = dr.notna() & ds_.notna()
            rows.append(dict(building=bid, use=b['primary_use'], cz=b['climate_zone'], meter=mtr, system=out['spec']['system'],
                             annual_real_kWh=float(ann_r), annual_sim_kWh=float(ann_s), ratio=float(ann_s / ann_r),
                             hourly_r_normalised=float(np.corrcoef(rn, sn)[0, 1]) if np.std(sn) > 0 else None,
                             cvrmse_hourly_normalised=float(np.sqrt(np.mean((rn - sn) ** 2))),
                             weekday_profile_r=float(np.corrcoef(prof_r, prof_s)[0, 1]) if np.std(prof_s) > 0 else None,
                             daily_r=float(np.corrcoef(dr[ok], ds_[ok])[0, 1]) if np.std(ds_[ok]) > 0 else None,
                             daily_vs_T_slope_real=float(np.polyfit(dT[ok], dr[ok] / dr[ok].mean(), 1)[0]),
                             daily_vs_T_slope_sim=float(np.polyfit(dT[ok], ds_[ok] / ds_[ok].mean(), 1)[0])))
            if physics is not None:
                rows[-1]['site'] = site
                rows[-1]['floor_area_m2'] = float(out['spec']['floor_area_m2'])
                rows[-1].update(physics_extra(rv, sv, loc, dr, ds_, dT, mtr))
            elif variant is not None:
                rows[-1].update(shape_extra(rv, sv, loc, mtr))
            print(bid, mtr, out['spec']['system'], 'ratio %.2f hourly r %.2f profile r %.2f daily r %.2f' % (
                rows[-1]['ratio'], rows[-1]['hourly_r_normalised'] or 0, rows[-1]['weekday_profile_r'] or 0,
                rows[-1]['daily_r'] or 0), flush=True)
    d = pd.DataFrame(rows)
    summ = {}
    for mtr, g in d.groupby('meter'):
        summ[mtr] = dict(n=len(g), annual_ratio_median=float(g.ratio.median()),
                         annual_ratio_p25=float(g.ratio.quantile(0.25)), annual_ratio_p75=float(g.ratio.quantile(0.75)),
                         within_25pct=float((np.abs(g.ratio - 1) <= 0.25).mean()),
                         within_50pct=float((np.abs(g.ratio - 1) <= 0.5).mean()),
                         hourly_r_median=float(g.hourly_r_normalised.median()),
                         cvrmse_normalised_median=float(g.cvrmse_hourly_normalised.median()),
                         weekday_profile_r_median=float(g.weekday_profile_r.median()),
                         daily_r_median=float(g.daily_r.median()),
                         temp_slope_sign_agreement=float((np.sign(g.daily_vs_T_slope_real) ==
                                                          np.sign(g.daily_vs_T_slope_sim)).mean()))
        if variant is not None:
            for k in ('summer_over_mean', 'night_over_day'):
                for side in ('real', 'sim'):
                    v = g[f'{k}_{side}'].dropna()
                    summ[mtr][f'{k}_{side}_median'] = float(v.median()) if len(v) else None
        if physics is not None:
            summ[mtr]['within_30pct'] = float((np.abs(g.ratio - 1) <= 0.3).mean())
            for k in ('summer_over_mean', 'winter_over_mean', 'night_over_day'):
                for side in ('real', 'sim'):
                    v = g[f'{k}_{side}'].dropna()
                    summ[mtr][f'{k}_{side}_median'] = float(v.median()) if len(v) else None
            summ[mtr]['daily_cvrmse_median'] = float(g.daily_cvrmse.median())
            if mtr == 'electricity':
                summ[mtr].update(r2_compare(g))
    if physics is not None:
        import buildingsim as bs
        name = f'bdg2_dev_meters__p{physics}' + (f'_y{year}' if year != 2016 else '') + (f'__{tag}' if tag else '')
        write(name + '.json', dict(
            what=f'BuildingSim physics {physics} dev twins (metadata only) vs REAL BDG2 meters, dev buildings, {year}',
            physics=physics, year=year, buildingsim=bs.__version__, sites=sites or 'all dev',
            campus_params=__import__('buildingsim.campus', fromlist=['params']).params(campus),
            split='bldg/data/proc/splits.json bdg2.dev (frozen by agent B); exam buildings untouched',
            per_meter=summ, buildings=rows,
            notes=['summer / winter: Jun-Aug / Dec-Feb mean over the annual mean; night_over_day: weekday 22-5 h mean '
                   'over 10-16 h, local time', 'daily_cvrmse: RMS of daily-mean differences over the real mean',
                   'electricity: daily PRISM 5P r2 (prism()) of real and twin, KS two-sample test of the two '
                   'distributions (scipy.stats.ks_2samp)']))
        return
    if variant is not None:
        import buildingsim as bs
        write(f'bdg2_dev_meters__{variant}.json', dict(
            what=f'BuildingSim dev twins with Phase 2 flags on ({variant}) vs REAL BDG2 meters, 52 dev buildings, 2016',
            variant=variant, flags=VARIANTS[variant], buildingsim=bs.__version__, physics=bs.PHYSICS_VERSION,
            split='bldg/data/proc/splits.json bdg2.dev (frozen by agent B); exam buildings untouched',
            per_meter=summ, buildings=rows,
            notes=['W1/W2 factors were fitted on these 52 dev buildings (W2 on 2017): the 2016 annual ratios are an '
                   'in-sample check, not a test', 'summer_over_mean: Jun-Aug mean / annual mean; night_over_day: '
                   'weekday 22-5 h mean / 10-16 h mean, local time']))
        return
    write('bdg2_dev_meters.json', dict(
        what='BuildingSim generic twins (metadata only) vs REAL BDG2 meters, 52 dev buildings, 2016, hourly',
        split='bldg/data/proc/splits.json bdg2.dev (frozen by agent B); exam buildings untouched',
        per_meter=summ, buildings=rows,
        notes=['envelope, schedules, setpoints, systems are BuildingSim defaults for the use type: annual errors '
               'measure the defaults + metadata gap, not the physics alone',
               'campus district meters (chilled water, steam, hot water) are compared with the district meters of the '
               'twin', 'shape metrics use each series divided by its own mean']))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', nargs='?', default=None)
    ap.add_argument('--physics', default=None)
    ap.add_argument('--year', type=int, default=2016)
    ap.add_argument('--sites', default=None, help='comma-separated dev sites (default: all)')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    main(a.variant, a.physics, a.year, a.sites.split(',') if a.sites else None, a.tag)
