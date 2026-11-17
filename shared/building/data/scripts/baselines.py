"""Classical baselines and their exam numbers (DATA.md section 9). Resumable stages:

  python3 -I baselines.py load        # BDG2 hourly load: persistence, same-hour-last-week, G14 change-point, TOWT
  python3 -I baselines.py detective   # PRISM change-point per building; floor area / use / vintage; RECS & CBECS annual
  python3 -I baselines.py faults      # APAR-style AHU rules + majority class on real/experimental fault runs
  python3 -I baselines.py thermostat  # fixed setpoint / setback / pre-cool on real tariffs, 1R1C house + heat pump
  python3 -I baselines.py summary     # proc/baselines/summary.json
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, BLDG_CODE)
import buildingdata as bd

D = f'{BLDG_ROOT}/data'; O = f'{D}/proc/baselines'; os.makedirs(O, exist_ok=True)
LOAD_METERS = ['electricity', 'chilledwater', 'steam', 'hotwater', 'gas']


# ======================================================================= change-point models (ASHRAE G14 / PRISM)
def cp_fit(T, E, kind):
    """Least-squares change-point fit on daily data. Returns (params dict, predict fn, n_params, sse)."""
    T = np.asarray(T, float); E = np.asarray(E, float)
    grid = np.arange(np.floor(np.nanpercentile(T, 5)), np.ceil(np.nanpercentile(T, 95)) + 0.01, 0.5)
    best = None

    def solve(X):
        c, *_ = np.linalg.lstsq(X, E, rcond=None); r = E - X @ c; return c, float(r @ r)
    one = np.ones_like(T)
    if kind == '2P':
        c, s = solve(np.c_[one, T]); return dict(b0=c[0], b1=c[1]), (lambda t: c[0] + c[1] * t), 2, s
    if kind in ('3PH', '3PC'):
        for tb in grid:
            x = np.maximum(tb - T, 0) if kind == '3PH' else np.maximum(T - tb, 0)
            c, s = solve(np.c_[one, x])
            if (c[1] >= 0) and (best is None or s < best[2]): best = (tb, c, s)
        if best is None: return None
        tb, c, s = best; sign = 1 if kind == '3PC' else -1
        f = (lambda t: c[0] + c[1] * np.maximum(sign * (t - tb), 0))
        return dict(base=c[0], slope=c[1], Tb=tb), f, 3, s
    if kind == '4P':
        for tb in grid:
            c, s = solve(np.c_[one, np.minimum(T - tb, 0), np.maximum(T - tb, 0)])
            if best is None or s < best[2]: best = (tb, c, s)
        tb, c, s = best
        return dict(base=c[0], slope_lo=c[1], slope_hi=c[2], Tb=tb), (lambda t: c[0] + c[1] * np.minimum(t - tb, 0) + c[2] * np.maximum(t - tb, 0)), 4, s
    if kind == '5P':
        for i, th in enumerate(grid):
            for tc in grid[i + 2:]:
                c, s = solve(np.c_[one, np.maximum(th - T, 0), np.maximum(T - tc, 0)])
                if c[1] >= 0 and c[2] >= 0 and (best is None or s < best[3]): best = (th, tc, c, s)
        if best is None: return None
        th, tc, c, s = best
        return dict(base=c[0], heat_slope=c[1], cool_slope=c[2], Tb_heat=th, Tb_cool=tc), \
            (lambda t: c[0] + c[1] * np.maximum(th - t, 0) + c[2] * np.maximum(t - tc, 0)), 5, s


def cp_best(T, E, kinds=('2P', '3PH', '3PC', '4P', '5P')):
    """Best change-point model by BIC (n log(SSE/n) + p log n)."""
    ok = np.isfinite(T) & np.isfinite(E); T, E = T[ok], E[ok]; n = len(E)
    out = None
    for k in kinds:
        r = cp_fit(T, E, k)
        if r is None: continue
        par, f, p, s = r
        bic = n * np.log(max(s, 1e-12) / n) + p * np.log(n)
        if out is None or bic < out['bic']:
            out = dict(kind=k, params=par, f=f, p=p, sse=s, bic=bic, n=n)
    if out is not None:
        out['r2'] = 1 - out['sse'] / max(float(((E - E.mean()) ** 2).sum()), 1e-12)
    return out


def g14(y, yh, p=0):
    """ASHRAE Guideline 14 CV(RMSE)% and NMBE% (n-p in the denominators)."""
    ok = np.isfinite(y) & np.isfinite(yh); y, yh = y[ok], yh[ok]; n = len(y)
    if n <= p + 1 or y.mean() <= 0: return np.nan, np.nan, n
    ym = y.mean()
    return 100 * np.sqrt(((y - yh) ** 2).sum() / (n - p)) / ym, 100 * (y - yh).sum() / ((n - p) * ym), n


# ======================================================================= LOAD
KNOTS = np.array([4.4, 12.8, 18.3, 26.7, 32.2])     # TOWT temperature knots (40, 55, 65, 80, 90 F)


def towt_X(T, tow):
    seg = [np.minimum(T, KNOTS[0])]
    for a, b in zip(KNOTS[:-1], KNOTS[1:]): seg.append(np.clip(T, a, b) - a)
    seg.append(np.maximum(T - KNOTS[-1], 0))
    X = np.zeros((len(T), 168 + len(seg)))
    X[np.arange(len(T)), tow] = 1
    X[:, 168:] = np.column_stack(seg)
    return X


def load_one(y, T):
    """y, T: hourly Series in local time (2016-2017). Returns dict of metrics per method on local 2017."""
    y = y.astype(float); T = T.reindex(y.index).astype(float).interpolate(limit=3)
    H = y.index.year == 2016; F = y.index.year == 2017
    res = {}
    yF = y[F].to_numpy()
    for name, lag in [('persistence_1h', 1), ('persistence_24h', 24), ('same_hour_last_week', 168)]:
        yh = y.shift(lag)[F].to_numpy()
        res[name] = g14(yF, yh, 0)
    # TOWT (fit on 2016)
    tow = (y.index.dayofweek * 24 + y.index.hour).to_numpy()
    okH = H & y.notna().to_numpy() & T.notna().to_numpy()
    if okH.sum() > 24 * 120:
        X = towt_X(T.to_numpy()[okH], tow[okH])
        c, *_ = np.linalg.lstsq(X, y.to_numpy()[okH], rcond=None)
        okF = F & T.notna().to_numpy()
        yh = np.full(F.sum(), np.nan)
        yh[okF[F]] = towt_X(T.to_numpy()[okF], tow[okF]) @ c
        res['towt'] = g14(yF, yh, X.shape[1])
    # G14 change-point on daily data (2016) + 2016 hour-of-week shape
    day = y.index.normalize()
    yd = y.groupby(day).mean(); cnt = y.groupby(day).count(); Td = T.groupby(day).mean()
    yd = yd[cnt >= 20]
    dH = yd[yd.index.year == 2016]
    cp = None
    if len(dH) > 120:
        cp = cp_best(Td.reindex(dH.index).to_numpy(), dH.to_numpy())
    if cp is not None:
        Tf = Td[Td.index.year == 2017]
        pd_ = pd.Series(cp['f'](Tf.to_numpy()), index=Tf.index)
        yh_day = pd_.reindex(day[F]).to_numpy()
        hw = y[H].groupby(tow[H]).mean(); hw = hw / hw.mean()
        shape = hw.reindex(tow[F]).to_numpy()
        res['g14_changepoint'] = g14(yF, yh_day * shape, cp['p'])
        dF = yd[yd.index.year == 2017]
        res['g14_changepoint_daily'] = g14(dF.to_numpy(), pd_.reindex(dF.index).to_numpy(), cp['p'])
        mF = y[F].resample('MS').mean(); mh = pd.Series(yh_day * shape, index=y.index[F]).where(y[F].notna()).resample('MS').mean()
        res['g14_changepoint_monthly'] = g14(mF.to_numpy(), mh.to_numpy(), cp['p'])
        res['cp_kind'] = cp['kind']; res['cp_r2_2016'] = cp['r2']
    return res


def stage_load():
    meta = bd.bdg2_meta()
    sp = dict(zip(meta.id, meta.split))
    rows = []
    out = f'{O}/load_bdg2.parquet'
    done = set()
    if os.path.exists(out + '.part'):
        old = pd.read_parquet(out + '.part'); rows = old.to_dict('records'); done = set(zip(old.building_id, old.meter))
    W = {}
    t0 = time.time()
    for m in LOAD_METERS:
        q = bd.bdg2_quality(m); q = q[q.q_ok]
        X = bd.bdg2(m, building=list(q.building_id))
        for b in q.building_id:
            if (b, m) in done: continue
            site = b.split('_')[0]; tz = bd.SITE_TZ[site]
            if site not in W:
                w = bd.site_weather(site).T_out; W[site] = w
            y = X[b].copy(); y.index = y.index.tz_convert(tz)
            T = W[site].copy(); T.index = T.index.tz_convert(tz)
            r = load_one(y, T)
            row = dict(building_id=b, meter=m, site_id=site, split=sp[b], cp_kind=r.pop('cp_kind', None), cp_r2_2016=r.pop('cp_r2_2016', np.nan))
            for k, (cv, nmbe, n) in r.items(): row[f'{k}_cvrmse'] = cv; row[f'{k}_nmbe'] = nmbe; row[f'{k}_n'] = n
            rows.append(row)
            if len(rows) % 100 == 0:
                pd.DataFrame(rows).to_parquet(out + '.part', index=False); print(m, len(rows), round(time.time() - t0), 's', flush=True)
    d = pd.DataFrame(rows); d.to_parquet(out, index=False)
    print(d.groupby(['meter', 'split']).size())


# ======================================================================= DETECTIVE
def stage_detective():
    meta = bd.bdg2_meta()
    q = bd.bdg2_quality()
    rows = []
    for m in ['electricity', 'chilledwater', 'steam', 'hotwater', 'gas']:
        qq = q[(q.meter == m) & q.q_ok_daily_2017]
        X = bd.bdg2(m, building=list(qq.building_id), start='2016-12-31', end='2018-01-02')
        for b in qq.building_id:
            site = b.split('_')[0]; tz = bd.SITE_TZ[site]
            y = X[b].copy(); y.index = y.index.tz_convert(tz)
            T = bd.site_weather(site).T_out; T.index = T.index.tz_convert(tz)
            y = y[y.index.year == 2017]; day = y.index.normalize()
            yd = y.groupby(day).mean(); cnt = y.groupby(day).count(); yd = yd[cnt >= 20]
            Td = T.groupby(T.index.normalize()).mean().reindex(yd.index)
            if len(yd) < 200: continue
            cp = cp_best(Td.to_numpy(), yd.to_numpy())
            if cp is None: continue
            p = cp['params']; A = meta.set_index('id').floor_area_m2.get(b)
            r = dict(building_id=b, meter=m, kind=cp['kind'], r2=cp['r2'], n_days=cp['n'], floor_area_m2=A,
                     base_kW=p.get('base', p.get('b0')),
                     Tb_heat=p.get('Tb_heat', p.get('Tb') if cp['kind'] in ('3PH', '4P') else np.nan),
                     Tb_cool=p.get('Tb_cool', p.get('Tb') if cp['kind'] in ('3PC', '4P') else np.nan),
                     heat_slope_kW_per_K=p.get('heat_slope', p.get('slope') if cp['kind'] == '3PH' else (-p['slope_lo'] if cp['kind'] == '4P' else (-p['b1'] if cp['kind'] == '2P' and p['b1'] < 0 else np.nan))),
                     cool_slope_kW_per_K=p.get('cool_slope', p.get('slope') if cp['kind'] == '3PC' else (p['slope_hi'] if cp['kind'] == '4P' else (p['b1'] if cp['kind'] == '2P' and p['b1'] > 0 else np.nan))))
            if A and A > 0:
                r['heat_slope_W_per_m2K'] = 1000 * r['heat_slope_kW_per_K'] / A if np.isfinite(r['heat_slope_kW_per_K'] or np.nan) else np.nan
                r['cool_slope_W_per_m2K'] = 1000 * r['cool_slope_kW_per_K'] / A if np.isfinite(r['cool_slope_kW_per_K'] or np.nan) else np.nan
                r['base_W_per_m2'] = 1000 * r['base_kW'] / A
            rows.append(r)
    pr = pd.DataFrame(rows).merge(meta[['id', 'primary_use', 'climate_zone', 'split', 'year_built']], left_on='building_id', right_on='id', how='left').drop(columns='id')
    pr.to_parquet(f'{O}/prism_bdg2.parquet', index=False)
    S = {}
    # PRISM parameter medians by use type (dev) -> the "median by use type" predictor, scored on exam PRISM fits
    for m in ['electricity', 'steam', 'chilledwater']:
        x = pr[pr.meter == m]
        dev = x[x.split == 'dev']; ex = x[x.split == 'exam']
        for par in ['Tb_heat', 'Tb_cool', 'heat_slope_W_per_m2K', 'cool_slope_W_per_m2K', 'base_W_per_m2']:
            med_use = dev.groupby('primary_use')[par].median(); med_all = dev[par].median()
            pred = ex.primary_use.map(med_use).fillna(med_all)
            ok = ex[par].notna() & pred.notna()
            if ok.sum() < 10: continue
            err = (ex[par][ok] - pred[ok]).abs()
            S[f'prism_{m}_{par}'] = dict(n_exam=int(ok.sum()), exam_median=float(ex[par][ok].median()),
                                          median_by_use_MAE=float(err.mean()), median_by_use_MdAE=float(err.median()))
    # floor area from annual electricity (EUI by use from dev)
    qe = q[(q.meter == 'electricity') & q.q_ok_daily_2017][['building_id', 'annual_2017']]
    a = meta.merge(qe, left_on='id', right_on='building_id')
    a = a[(a.floor_area_m2 > 0) & (a.annual_2017 > 0)]
    a['eui'] = a.annual_2017 / a.floor_area_m2
    dev, ex = a[a.split == 'dev'], a[a.split == 'exam']
    eui_use = dev.groupby('primary_use').eui.median(); eui_all = dev.eui.median()
    pred = ex.annual_2017 / ex.primary_use.map(eui_use).fillna(eui_all)
    ape = (pred - ex.floor_area_m2).abs() / ex.floor_area_m2
    dumb = (dev.floor_area_m2.median() - ex.floor_area_m2).abs() / ex.floor_area_m2
    ratio = pred / ex.floor_area_m2
    S['bdg2_floor_area'] = dict(n_exam=len(ex), eui_by_use_MdAPE=float(ape.median() * 100), eui_by_use_within_x2=float(((ratio > 0.5) & (ratio < 2)).mean()),
                                dumb_dev_median_MdAPE=float(dumb.median() * 100))
    # primary use from weekly load shape (nearest dev centroid) vs majority class
    X = bd.bdg2('electricity', building=list(a.id), start='2016-12-31', end='2018-01-02')
    prof = {}
    for b in a.id:
        y = X[b].copy(); y.index = y.index.tz_convert(bd.SITE_TZ[b.split('_')[0]]); y = y[y.index.year == 2017]
        v = y.groupby(y.index.dayofweek * 24 + y.index.hour).mean().reindex(range(168))
        if v.notna().sum() == 168 and v.mean() > 0: prof[b] = (v / v.mean()).to_numpy()
    a = a[a.id.isin(prof)]
    dev, ex = a[a.split == 'dev'], a[a.split == 'exam']
    cents = {u: np.mean([prof[b] for b in g.id], axis=0) for u, g in dev.groupby('primary_use')}
    pu = [min(cents, key=lambda u: ((prof[b] - cents[u]) ** 2).sum()) for b in ex.id]
    maj = dev.primary_use.mode().iloc[0]
    from collections import Counter
    def macro_f1(t, p):
        f = []
        for c in set(t):
            tp = sum((a_ == c) and (b_ == c) for a_, b_ in zip(t, p)); fp = sum((a_ != c) and (b_ == c) for a_, b_ in zip(t, p)); fn = sum((a_ == c) and (b_ != c) for a_, b_ in zip(t, p))
            f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        return float(np.mean(f))
    S['bdg2_primary_use'] = dict(n_exam=len(ex), classes=dict(Counter(ex.primary_use)), majority_class=maj,
                                 majority_accuracy=float((ex.primary_use == maj).mean()), majority_macroF1=macro_f1(list(ex.primary_use), [maj] * len(ex)),
                                 weekly_shape_centroid_accuracy=float(np.mean(np.array(pu) == ex.primary_use.to_numpy())),
                                 weekly_shape_centroid_macroF1=macro_f1(list(ex.primary_use), pu))
    yb = meta[meta.year_built.notna()]
    S['bdg2_year_built'] = dict(n_exam=int((yb.split == 'exam').sum()), dumb_dev_median=float(yb[yb.split == 'dev'].year_built.median()),
                                dumb_MAE_years=float((yb[yb.split == 'exam'].year_built - yb[yb.split == 'dev'].year_built.median()).abs().mean()))
    # ---------- RECS / CBECS (annual level)
    for name, fn, tcol in [('recs', bd.recs, 'housing_type'), ('cbecs', bd.cbecs, 'pba')]:
        d = fn()
        d['site_total_kWh'] = d[['elec_kWh', 'gas_kWh', 'fueloil_kWh'] + (['propane_kWh', 'wood_kWh'] if name == 'recs' else ['district_heat_kWh'])].fillna(0).sum(axis=1)
        d = d[d.site_total_kWh > 0]
        czc = 'climate_zone' if name == 'recs' else 'climate_class'
        dev, ex = d[d.split == 'dev'], d[d.split == 'exam']
        eui = (dev.site_total_kWh / dev.floor_area_m2).groupby([dev[tcol], dev[czc]]).median()
        eui_t = (dev.site_total_kWh / dev.floor_area_m2).groupby(dev[tcol]).median()
        key = list(zip(ex[tcol], ex[czc]))
        e_hat = pd.Series([eui.get(k, np.nan) for k in key], index=ex.index).fillna(ex[tcol].map(eui_t))
        A_hat = ex.site_total_kWh / e_hat
        ape = ((A_hat - ex.floor_area_m2).abs() / ex.floor_area_m2)
        dumb = (ex[tcol].map(dev.groupby(tcol).floor_area_m2.median()) - ex.floor_area_m2).abs() / ex.floor_area_m2
        # annual energy from characteristics: per-type dev regression E/A = a + b HDD + c CDD  (a "generator-level" check)
        Ehat = pd.Series(np.nan, index=ex.index)
        for t_, g in dev.groupby(tcol):
            if len(g) < 15: continue
            Xd = np.c_[np.ones(len(g)), g.hdd18C_year, g.cdd18C_year]; c, *_ = np.linalg.lstsq(Xd, g.site_total_kWh / g.floor_area_m2, rcond=None)
            sel = ex[tcol] == t_
            Ehat[sel] = (np.c_[np.ones(sel.sum()), ex.hdd18C_year[sel], ex.cdd18C_year[sel]] @ c) * ex.floor_area_m2[sel]
        ok = Ehat.notna() & (Ehat > 0)
        eape = ((Ehat[ok] - ex.site_total_kWh[ok]).abs() / ex.site_total_kWh[ok])
        # heating fuel: rule = largest non-electric fuel used, else electricity
        fuels = ['gas_kWh', 'fueloil_kWh'] + (['propane_kWh', 'wood_kWh'] if name == 'recs' else ['district_heat_kWh'])
        names = {'gas_kWh': 'natural_gas', 'fueloil_kWh': 'fuel_oil', 'propane_kWh': 'propane', 'wood_kWh': 'wood', 'district_heat_kWh': 'district'}
        f_ = ex[fuels].fillna(0)
        rule = np.where(f_.max(axis=1) > 0, f_.idxmax(axis=1).map(names), 'electricity')
        truth = ex.heating_fuel.astype(str)
        hmaj = dev.heating_fuel.astype(str).mode().iloc[0]
        S[f'{name}_floor_area'] = dict(n_exam=len(ex), eui_by_type_climate_MdAPE=float(ape.median() * 100), dumb_dev_median_by_type_MdAPE=float(dumb.median() * 100))
        S[f'{name}_annual_energy_from_characteristics'] = dict(n_exam=int(ok.sum()), per_type_HDD_CDD_regression_MdAPE=float(eape.median() * 100))
        S[f'{name}_heating_fuel'] = dict(n_exam=len(ex), majority_class=hmaj, majority_accuracy=float((truth == hmaj).mean()),
                                         largest_fuel_rule_accuracy=float((truth == rule).mean()))
        if name == 'recs':
            el = d[(d.heating_fuel == 'electricity') & d.heating_equipment.isin(['heat_pump_air', 'baseboard_resistance', 'electric_furnace'])].copy()
            el['hp'] = el.heating_equipment == 'heat_pump_air'
            el['x'] = el.elec_kWh / el.floor_area_m2 / np.maximum(el.hdd18C_year, 100)
            dv, xx = el[el.split == 'dev'], el[el.split == 'exam']
            ths = np.quantile(dv.x, np.linspace(0.05, 0.95, 91))
            bacc = lambda t, p: 0.5 * ((p[t]).mean() + (~p[~t]).mean())
            th = max(ths, key=lambda h: bacc(dv.hp.to_numpy(), (dv.x < h).to_numpy()))
            S['recs_heat_pump_vs_resistance'] = dict(n_exam=len(xx), feature='annual elec kWh per m2 per HDD18 (low -> heat pump)', threshold_dev=float(th),
                                                     rule_balanced_accuracy=float(bacc(xx.hp.to_numpy(), (xx.x < th).to_numpy())),
                                                     majority_balanced_accuracy=0.5, majority_accuracy=float(max(xx.hp.mean(), 1 - xx.hp.mean())))
    json.dump(S, open(f'{O}/detective_summary.json', 'w'), indent=1, default=float)
    print(json.dumps(S, indent=1, default=float)[:4000])


# ======================================================================= FAULTS (APAR-style)
ET, DTSF, EV = 1.5, 1.1, 0.02      # temperature tolerance, supply-fan heat (degC), valve tolerance


def apar(d):
    """APAR-style rule set (Schein et al. 2006, subset; ASHRAE/NIST). d: trend log in the contract. Returns bool matrix."""
    g = lambda c: d[c].to_numpy(float) if c in d and d[c].notna().any() else np.full(len(d), np.nan)
    Toa, Tra, Tma, Tsa = g('T_oa'), g('T_ra'), g('T_ma'), g('T_sa')
    sp = g('T_sa_sp'); sph, spc = g('T_sa_sp_heat'), g('T_sa_sp_cool')
    sph = np.where(np.isfinite(sph), sph, sp); spc = np.where(np.isfinite(spc), spc, sp)
    hc, cc, oa = np.nan_to_num(g('valve_hc')), np.nan_to_num(g('valve_cc')), g('damper_oa')
    fan = g('fan_sa_status'); spd = g('fan_sa_speed')
    on = np.where(np.isfinite(fan), fan > 0.5, np.where(np.isfinite(spd), spd > 0.05, True))
    occ = g('occ'); on &= np.where(np.isfinite(occ), occ > 0.5, True)
    oamin = np.nanpercentile(oa[on], 10) if np.isfinite(oa[on]).any() else 0.0
    M1 = on & (hc > EV) & (cc <= EV)
    M2 = on & (hc <= EV) & (cc <= EV) & (oa > oamin + 0.05)
    M3 = on & (cc > EV) & (oa >= 0.95)
    M4 = on & (cc > EV) & (oa < 0.95) & (hc <= EV)
    R = {}
    with np.errstate(invalid='ignore'):
        R['R1_heat_sat_below_mat'] = M1 & (Tsa < Tma + DTSF - ET)
        R['R3_heat_sat_low_valve_full'] = M1 & (hc >= 0.98) & (Tsa < sph - ET)
        R['R7_econ_oat_too_warm'] = M2 & (Toa > spc - DTSF + ET)
        R['R8_econ_sat_above_rat'] = M2 & (Tsa > Tra + ET)
        R['R9_econ_sat_ne_mat'] = M2 & (np.abs(Tsa - DTSF - Tma) > ET)
        R['R10_100oa_mat_ne_oat'] = M2 & (oa >= 0.95) & (np.abs(Toa - Tma) > ET)
        R['R11_mech_100oa_should_econ'] = M3 & (Toa < spc - DTSF - ET)
        R['R12_mech_sat_above_mat'] = (M3 | M4) & (Tsa > Tma + DTSF + ET)
        R['R13_cool_sat_high_valve_full'] = (M3 | M4) & (cc >= 0.98) & (Tsa > spc + ET)
        R['R14_100oa_mat_ne_oat'] = M3 & (np.abs(Toa - Tma) > ET)
        R['R23_simultaneous_heat_cool'] = on & (hc > 0.1) & (cc > 0.1)
        R['R24_mat_outside_mix'] = on & ((Tma < np.fmin(Tra, Toa) - ET) | (Tma > np.fmax(Tra, Toa) + ET))
    mode = np.select([M1, M2, M3, M4], [1, 2, 3, 4], 0)
    R = {k: np.nan_to_num(v).astype(bool) for k, v in R.items()}
    return pd.DataFrame(R), on, mode


def persist(b, n):
    if n <= 1: return b
    s = pd.Series(b.astype(int)).rolling(n, min_periods=n).sum().to_numpy()
    return np.nan_to_num(s) >= n


def run_score(d, dt):
    R, on, mode = apar(d)
    n = max(1, int(round(900 / dt)))         # rule must hold for 15 minutes
    anyr = np.zeros(len(d), bool); per = {}
    for c in R:
        p = persist(R[c].to_numpy(), n); per[c] = float(p[on].mean()) if on.any() else 0.0; anyr |= p
    mc = (np.diff(mode) != 0).sum() / max(on.sum() * dt / 3600, 1)       # mode changes per operating hour
    frac = float(anyr[on].mean()) if on.any() else 0.0
    return frac, per, float(mc), int(on.sum())


def stage_faults():
    runs = bd.faults(gradable=True)
    rows = []
    for ds, g in runs.groupby('dataset'):
        data = bd.fault_data(ds)
        for _, r in g.iterrows():
            d = data[data.run_id == r.run_id]
            if r.unit == 'RTU':
                occ = d.occ.to_numpy(float); fan = d.fan_sa_status.to_numpy(float)
                unocc = occ < 0.5
                frac = float((fan[unocc] > 0.5).mean()) if unocc.any() else 0.0
                rows.append(dict(run_id=r.run_id, dataset=ds, split=r.split, fault_type=r.fault_type, is_fault=r.fault_type != 'none',
                                 score=frac, rule='RTU: fan on while unoccupied', mode_changes_per_h=np.nan, n_on=int(len(d))))
                continue
            frac, per, mc, non = run_score(d, r.dt_s)
            rows.append(dict(run_id=r.run_id, dataset=ds, split=r.split, fault_type=r.fault_type, is_fault=r.fault_type != 'none',
                             score=frac, rule='APAR any', mode_changes_per_h=mc, n_on=non, **{f'rule_{k}': v for k, v in per.items()}))
    s = pd.DataFrame(rows)
    s.to_parquet(f'{O}/faults_apar.parquet', index=False)
    S = {}
    for grp, sel in [('ahu_all', s.rule == 'APAR any'), ('rtu', s.rule != 'APAR any')]:
        x = s[sel]; dev, ex = x[x.split == 'dev'], x[x.split == 'exam']
        cand = np.unique(np.r_[0, np.quantile(x.score, np.linspace(0, 1, 101))])
        def metr(t, p):
            tp, fp, fn, tn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum()
            return dict(accuracy=float((tp + tn) / len(t)), TPR=float(tp / max(tp + fn, 1)), FPR=float(fp / max(fp + tn, 1)),
                        F1=float(2 * tp / max(2 * tp + fp + fn, 1)), balanced_accuracy=float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))))
        th = max(cand, key=lambda h: (metr(dev.is_fault.to_numpy(), (dev.score > h).to_numpy())['balanced_accuracy'], -h))
        maj = bool(dev.is_fault.mean() >= 0.5)
        def auc(t, sc):
            t = np.asarray(t, bool); sc = np.asarray(sc, float); pos, neg = sc[t], sc[~t]
            if len(pos) == 0 or len(neg) == 0: return float('nan')
            return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
        S[grp] = dict(n_exam=len(ex), n_exam_faulty=int(ex.is_fault.sum()), n_dev=len(dev), threshold_dev=float(th),
                      exam_auc=auc(ex.is_fault, ex.score), exam_auc_by_dataset={k: auc(v.is_fault, v.score) for k, v in ex.groupby('dataset')},
                      rule=metr(ex.is_fault.to_numpy(), (ex.score > th).to_numpy()),
                      majority_class='fault' if maj else 'none', majority=metr(ex.is_fault.to_numpy(), np.full(len(ex), maj)),
                      by_dataset={k: metr(v.is_fault.to_numpy(), (v.score > th).to_numpy()) for k, v in ex.groupby('dataset')},
                      detection_rate_by_fault_type={k: float((v.score > th).mean()) for k, v in ex[ex.is_fault].groupby('fault_type')})
    json.dump(S, open(f'{O}/faults_summary.json', 'w'), indent=1)
    print(json.dumps(S, indent=1)[:3500])


# ======================================================================= THERMOSTAT
TH_PAIRS = [('pge_e_tou_c', 'Bear'), ('srp_e26', 'Fox'), ('aps_r3', 'Fox'), ('austin_res_tou', 'Bull'), ('duke_fl_rt', 'Panther'),
            ('rmp_ut_sched1_tou', 'Bobcat'), ('xcel_mn_a04_tod', 'Hog'), ('nyseg_sc12_tou', 'Cockatoo'), ('pseg_rs_tou_3p', 'Eagle'),
            ('coned_sc1_rate3', 'Eagle'), ('xcel_co_re_tou', 'Bobcat')]
HOUSE = dict(UA_W_per_K=200.0, C_J_per_K=20e6, A_win_eff_m2=4.0, Q_int_W=500.0)   # ~150 m2 single-family, tau = 28 h


QMAX = 8000.0      # W thermal: ~2.3-ton heat pump, no backup strip


def cop_heat(T): return np.clip(1.8 + 0.07 * (T + 10), 1.5, 4.5)
def cop_cool(T): return np.clip(5.5 - 0.1 * (T - 20), 2.0, 5.5)


def simulate_house(Tout, ghi, sp_heat, sp_cool, dt=3600.0, sub=6):
    UA, C = HOUSE['UA_W_per_K'], HOUSE['C_J_per_K']
    n = len(Tout); Tin = np.empty(n); E = np.empty(n); Ti = 21.0; h = dt / sub
    for k in range(n):
        Qg = HOUSE['Q_int_W'] + HOUSE['A_win_eff_m2'] * ghi[k]
        el = 0.0
        for _ in range(sub):
            Tfree = Ti + h / C * (UA * (Tout[k] - Ti) + Qg)
            if Tfree < sp_heat[k]:
                Q = min(C * (sp_heat[k] - Ti) / h - UA * (Tout[k] - Ti) - Qg, QMAX)
                Ti = Ti + h / C * (UA * (Tout[k] - Ti) + Qg + Q); el += Q / cop_heat(Tout[k]) * h
            elif Tfree > sp_cool[k]:
                Q = min(-(C * (sp_cool[k] - Ti) / h - UA * (Tout[k] - Ti) - Qg), QMAX)
                Ti = Ti + h / C * (UA * (Tout[k] - Ti) + Qg - Q); el += Q / cop_cool(Tout[k]) * h
            else:
                Ti = Tfree
        Tin[k] = Ti; E[k] = el / 3.6e6
    return Tin, E


def stage_thermostat():
    rows = []
    for key, site in TH_PAIRS:
        w = bd.site_weather(site, '2016-12-31', '2018-01-02'); tz = bd.SITE_TZ[site]
        w.index = w.index.tz_convert(tz); w = w[w.index.year == 2017]
        w = w[~w.index.duplicated()]
        To, G = w.T_out.interpolate().to_numpy(), w.GHI.fillna(0).to_numpy()
        idx = w.index; hr = np.asarray(idx.hour); wd = np.asarray(idx.dayofweek < 5)
        pr = bd.tariff_prices(key, 2017, tz=tz).reindex(idx)
        price = pr['energy_$per_kWh'].to_numpy()
        hi = price > np.nanmin(price) + 1e-6
        # peak window = hours of the day where the summer weekday price is highest
        summer = np.asarray(idx.month.isin([6, 7, 8])) & wd
        ph = pd.Series(price[summer]).groupby(hr[summer]).mean()
        peak_hours = set(ph[ph >= ph.max() - 1e-6].index) if ph.max() > ph.min() + 1e-6 else set()
        home = ~(wd & (hr >= 8) & (hr < 17)); night = (hr >= 23) | (hr < 6)
        ctl = {'fixed': (np.full(len(idx), 20.0), np.full(len(idx), 24.0)),
               'setback': (np.where(night | ~home, 16.0, 20.0), np.where(~home, 28.0, np.where(night, 26.0, 24.0)))}
        pre = np.zeros(len(idx), bool); inpk = np.isin(hr, list(peak_hours)) & wd
        if peak_hours:
            p0 = min(peak_hours)
            pre = wd & np.isin(hr, [(p0 - 2) % 24, (p0 - 1) % 24])
        ctl['precool_rule'] = (np.full(len(idx), 20.0), np.where(pre, 22.0, np.where(inpk, 26.5, 24.0)))
        for name, (sh, sc) in ctl.items():
            Tin, E = simulate_house(To, G, sh, sc)
            load = pd.Series(E, index=idx)
            b = bd.bill(key, load)
            occ_hours = home & ~night
            dh = float((np.maximum(20 - Tin, 0) + np.maximum(Tin - 24.5, 0))[occ_hours].sum())
            rows.append(dict(tariff=key, site=site, controller=name, kWh=float(E.sum()), bill_total=b['total'], bill_energy=b['energy'],
                             bill_demand=b['demand'], bill_fixed=b['fixed'], peak_kW=float(E.max()), occupied_discomfort_Kh=dh,
                             peak_hours=','.join(map(str, sorted(peak_hours)))))
        print(key, site, [(r['controller'], round(r['bill_total']), round(r['kWh'])) for r in rows[-3:]], flush=True)
    d = pd.DataFrame(rows); d.to_parquet(f'{O}/thermostat.parquet', index=False)
    S = {'model': '1R1C house (UA 200 W/K, C 20 MJ/K, tau 28 h, 4 m2 effective solar aperture, 500 W internal gains) with an '
                  'air-source heat pump limited to 8 kW thermal (no backup heat) (COP_heat = 1.8 + 0.07 (T_out + 10) in [1.5, 4.5], COP_cool = 5.5 - 0.1 (T_out - 20) in [2, 5.5]); '
                  'real 2017 site weather (BDG2 / NASA POWER) and real URDB tariffs (tier 1 energy rate + demand + fixed charges). '
                  'A rough reference, not a calibrated home: ThermoPilot should re-run the same controllers on BuildingSim.',
         'controllers': {'fixed': 'heat 20 C / cool 24 C always', 'setback': 'heat 20 C home, 16 C night (23-6) and weekday away (8-17); '
                         'cool 24 C home, 26 C night, 28 C away', 'precool_rule': 'cool 24 C; 22 C in the 2 h before the summer weekday peak window, 26.5 C during it'},
         'results': d.to_dict('records')}
    json.dump(S, open(f'{O}/thermostat_summary.json', 'w'), indent=1, default=float)


# ======================================================================= SUMMARY
def stage_summary():
    S = {'note': 'Exam numbers of classical baselines (DATA.md section 9). Medians over building-meters unless stated.'}
    l = pd.read_parquet(f'{O}/load_bdg2.parquet')
    meths = ['persistence_1h', 'persistence_24h', 'same_hour_last_week', 'towt', 'g14_changepoint', 'g14_changepoint_daily', 'g14_changepoint_monthly']
    L = {}
    for (m, sp), g in l.groupby(['meter', 'split']):
        L.setdefault(sp, {})[m] = {'n': len(g), **{k: dict(cvrmse_median=float(g[f'{k}_cvrmse'].median()), cvrmse_p25=float(g[f'{k}_cvrmse'].quantile(.25)),
                                                          cvrmse_p75=float(g[f'{k}_cvrmse'].quantile(.75)), nmbe_median=float(g[f'{k}_nmbe'].median()),
                                                          abs_nmbe_median=float(g[f'{k}_nmbe'].abs().median()))
                                                       for k in meths if f'{k}_cvrmse' in g},
                                   'g14_hourly_pass_frac_changepoint': float(((g.g14_changepoint_cvrmse <= 30) & (g.g14_changepoint_nmbe.abs() <= 10)).mean()),
                                   'g14_hourly_pass_frac_towt': float(((g.towt_cvrmse <= 30) & (g.towt_nmbe.abs() <= 10)).mean()),
                                   'g14_monthly_pass_frac_changepoint': float(((g.g14_changepoint_monthly_cvrmse <= 15) & (g.g14_changepoint_monthly_nmbe.abs() <= 5)).mean()),
                                   'cp_kind_counts': g.cp_kind.value_counts().to_dict()}
    S['load_bdg2_2016_to_2017'] = L
    for f in ['detective_summary', 'faults_summary', 'thermostat_summary']:
        if os.path.exists(f'{O}/{f}.json'): S[f.replace('_summary', '')] = json.load(open(f'{O}/{f}.json'))
    json.dump(S, open(f'{O}/summary.json', 'w'), indent=1, default=float)
    ex = L.get('exam', {})
    for m, v in ex.items():
        print(m, v['n'], {k: round(v[k]['cvrmse_median'], 1) for k in meths if k in v})


if __name__ == '__main__':
    {'load': stage_load, 'detective': stage_detective, 'faults': stage_faults, 'thermostat': stage_thermostat, 'summary': stage_summary}[sys.argv[1]]()
