"""Classical / dumb baselines on the EXAM sets (and dev, for reference). Output: proc/baselines/*.parquet + summary.json.
Sections (argv): demand | heatrate | battery | frequency | all
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, sys, json
import numpy as np, pandas as pd
PROC = f'{GRID_ROOT}/data/proc'; OUT = f'{PROC}/baselines'; os.makedirs(OUT, exist_ok=True)
SP = json.load(open(f'{PROC}/splits.json'))
FOCUS = ['ERCO', 'CISO', 'PJM', 'MISO', 'SWPP', 'ISNE', 'NYIS', 'BPAT']
SUMF = f'{OUT}/summary.json'
summary = json.load(open(SUMF)) if os.path.exists(SUMF) else {}


def yearset(name): return SP['time'][name]


def metrics(y, p):
    e = p - y
    return dict(n=int(len(y)), MAPE_pct=float(np.mean(np.abs(e) / np.abs(y)) * 100), RMSE_MW=float(np.sqrt(np.mean(e ** 2))),
                MAE_MW=float(np.mean(np.abs(e))), bias_MW=float(np.mean(e)))


def demand():
    """Day-ahead demand. Target: EIA-930 adjusted demand d where q_ok. Common mask: all predictors available.
    persistence_24h: same hour yesterday. same_hour_last_week: t-168 h. eia_day_ahead: the BA's own DF (EIA-930).
    climatology (dumb): dev-year (2019, 2021) mean by month x weekday x hour-of-day, scaled by the ratio of the
    trailing 28-day mean demand to the dev-year mean of that month (no exam labels used beyond t-24 h)."""
    out = {}; rows = []
    for ba in FOCUS:
        d = pd.read_parquet(f'{PROC}/eia930/ba/{ba}.parquet').set_index('time_utc')
        y = d.d.where(d.q_ok)
        f = pd.DataFrame({'y': y, 'persistence_24h': d.d.shift(24), 'same_hour_last_week': d.d.shift(168), 'eia_day_ahead': d.df})
        dev = d[d.index.year.isin(yearset('dev')) & d.q_ok]
        key = lambda idx: (idx.month, idx.dayofweek, idx.hour)
        clim = dev.d.groupby([dev.index.month, dev.index.dayofweek, dev.index.hour]).mean()
        cm = dev.d.groupby(dev.index.month).mean()
        base = pd.Series(clim.reindex(list(zip(*key(d.index)))).values, index=d.index)
        trail = d.d.shift(24).rolling(28 * 24, min_periods=24 * 7).mean()
        f['climatology_scaled'] = base * trail / pd.Series(cm.reindex(d.index.month).values, index=d.index)
        f['df_bad'] = (d.df <= 0) | (d.df > 3 * d.d.rolling(24 * 30, min_periods=24).max())
        for split in ('dev', 'exam'):
            s = f[f.index.year.isin(yearset(split))].dropna(subset=['y', 'persistence_24h', 'same_hour_last_week', 'eia_day_ahead', 'climatology_scaled'])
            s = s[~s.df_bad]
            res = {m: metrics(s.y.values, s[m].values) for m in ['persistence_24h', 'same_hour_last_week', 'eia_day_ahead', 'climatology_scaled']}
            res['coverage_hours'] = int(len(s)); res['hours_in_split'] = int(f.index.year.isin(yearset(split)).sum())
            out.setdefault(ba, {})[split] = res
            if split == 'exam':
                for y_ in yearset('exam'):
                    sy = s[s.index.year == y_]
                    if len(sy): out[ba].setdefault('exam_by_year', {})[y_] = {m: round(metrics(sy.y.values, sy[m].values)['MAPE_pct'], 3) for m in ['persistence_24h', 'same_hour_last_week', 'eia_day_ahead', 'climatology_scaled']}
        f['ba'] = ba
        rows.append(f[f.index.year.isin(yearset('exam'))].reset_index())
        print(ba, {k: round(v['MAPE_pct'], 2) for k, v in out[ba]['exam'].items() if isinstance(v, dict)}, flush=True)
    pd.concat(rows).to_parquet(f'{OUT}/demand_exam_predictions.parquet', index=False)
    summary['demand'] = out


def heatrate():
    """PlantDetective heat-rate baseline on EXAM plants (all years 2018-2025; the split is by plant).
    Truth: CAMPD hourly heat input on steady full operating hours (op_time = 1 with full neighbours), gross load > 0,
    heat input measured (hi_meas not substitute data). Baseline A (asked): EIA-923 annual average heat rate of the linked
    generators (fuel for electricity / net generation, same year) x gross load. Baseline B (dumb): the dev plants'
    median heat rate of the same tech x gross load. Error per unit-year on annual heat input and hourly MAPE."""
    u = pd.read_parquet(f'{PROC}/campd/units.parquet')
    hr = pd.read_parquet(f'{GRID_ROOT}/data/raw/pudl/out_eia923__yearly_generation_fuel_by_generator.parquet')
    hr['year'] = pd.to_datetime(hr.report_date).dt.year; hr['generator_id'] = hr.generator_id.astype(str)
    hr = hr.groupby(['plant_id_eia', 'generator_id', 'year'])[['fuel_consumed_for_electricity_mmbtu', 'net_generation_mwh']].sum()
    exam = set(SP['plants']['exam']); dev = set(SP['plants']['dev'])
    rows = []; hourly = []
    for _, r in u[u.crosswalk].iterrows():
        gens = str(r.generator_ids).split(';')
        for y in range(2018, 2026):
            f = f'{PROC}/campd/hourly/tx_{y}.parquet'
            d = pd.read_parquet(f, columns=['time_utc', 'op_time', 'gross_load_MW', 'heat_input_MMBtu', 'hi_meas'],
                                filters=[('plant_id_epa', '=', int(r.plant_id_epa)), ('unit_id_epa', '=', str(r.unit_id_epa))])
            if d.empty: continue
            full = (d.op_time >= 0.999) & (d.gross_load_MW > 0)
            steady = full & full.shift(1, fill_value=False) & full.shift(-1, fill_value=False)
            meas = d.hi_meas.astype(str).str.contains('Measured|Calculated', case=False, na=False) | d.hi_meas.isna()
            m = steady & meas & (d.heat_input_MMBtu > 0)
            if m.sum() < 50: continue
            k = [(int(r.plant_id_eia), g, y) for g in gens if (int(r.plant_id_eia), g, y) in hr.index]
            if not k: continue
            fs = hr.loc[k].sum()
            if fs.net_generation_mwh <= 0 or fs.fuel_consumed_for_electricity_mmbtu <= 0: continue
            hr923 = fs.fuel_consumed_for_electricity_mmbtu / fs.net_generation_mwh
            if not (5 < hr923 < 25): continue
            P = d.gross_load_MW[m].values; HI = d.heat_input_MMBtu[m].values
            split = 'dev' if r.plant_id_eia in dev else ('exam' if r.plant_id_eia in exam else 'none')
            rows.append(dict(unit_id=r.unit_id, plant_id_eia=int(r.plant_id_eia), tech=r.tech, ba=r.ba, year=y, split=split,
                             hours=int(m.sum()), hr923_net=hr923, hr_campd_gross=HI.sum() / P.sum(),
                             ape_hourly_923=float(np.median(np.abs(hr923 * P - HI) / HI)), err_annual_923=float((hr923 * P.sum() - HI.sum()) / HI.sum())))
    t = pd.DataFrame(rows)
    devmed = t[t.split == 'dev'].groupby('tech').hr_campd_gross.median()
    t['hr_devmedian'] = t.tech.map(devmed)
    t['err_annual_devmedian'] = (t.hr_devmedian - t.hr_campd_gross) / t.hr_campd_gross
    t.to_parquet(f'{OUT}/heatrate_unit_years.parquet', index=False)
    res = {}
    for split in ('dev', 'exam'):
        s = t[t.split == split]
        res[split] = {'unit_years': int(len(s)), 'units': int(s.unit_id.nunique()), 'plants': int(s.plant_id_eia.nunique()),
                      'eia923_avg': {'median_abs_err_annual_heat_input_pct': float(100 * s.err_annual_923.abs().median()),
                                     'median_bias_pct': float(100 * s.err_annual_923.median()),
                                     'median_hourly_APE_pct': float(100 * s.ape_hourly_923.median())},
                      'dev_tech_median': {'median_abs_err_annual_heat_input_pct': float(100 * s.err_annual_devmedian.abs().median())},
                      'by_tech': {k: {'n': int(len(g)), 'eia923_med_abs_err_pct': round(float(100 * g.err_annual_923.abs().median()), 2),
                                      'eia923_med_bias_pct': round(float(100 * g.err_annual_923.median()), 2),
                                      'devmedian_med_abs_err_pct': round(float(100 * g.err_annual_devmedian.abs().median()), 2)}
                                  for k, g in s.groupby('tech')}}
    summary['heatrate'] = res; print(json.dumps(res, indent=1)[:1500])


def lp_day(p, P=1.0, E=2.0, eta=0.85, soc0=0.0):
    """Perfect-foresight arbitrage for one price vector (hourly $/MWh): max sum p*(dis - ch), SOC dynamics with
    one-way efficiency sqrt(eta) on charge and discharge; SOC starts and ends at soc0. Returns revenue $."""
    from scipy.optimize import linprog
    n = len(p); se = np.sqrt(eta)
    # vars: ch[0..n), dis[0..n)
    c = np.r_[p, -p]
    # soc_t = soc0 + sum_{k<=t} (se*ch_k - dis_k/se) in [0, E]
    L = np.tril(np.ones((n, n)))
    A = np.r_[np.c_[se * L, -L / se], np.c_[-se * L, L / se]]
    b = np.r_[np.full(n, E - soc0), np.full(n, soc0)]
    Aeq = np.r_[se * np.ones(n), -np.ones(n) / se][None]
    r = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=[0.0], bounds=[(0, P)] * (2 * n), method='highs')
    return -r.fun if r.success else np.nan, (r.x[:n], r.x[n:]) if r.success else (None, None)


def battery():
    """BatteryPilot: 1 MW / 2 MWh battery, round-trip efficiency 0.85, daily cycle (empty at local midnight).
    perfect_foresight_RT: LP on the realised hourly RT price of each day (upper bound).
    perfect_foresight_DA: LP on the DAM price (also what a DA-scheduled battery earns at DA prices).
    rule_daily_spread: schedule from DAM prices only (known the day before): the LP-optimal DA schedule is not used;
       instead charge in the 2 cheapest DAM hours (if 'cheap' < 'dear' x eta) and discharge in the 2 dearest
       hours after them; settled at RT prices (a realistic rule: no RT foresight).
    rule_threshold: charge when RT price < dev-period 25th pct, discharge when > 75th pct (state-limited), RT settled.
    Revenue in $ per kW-year (= $/MW-yr / 1000)."""
    res = {}; rows = []
    def run(nodeframe, market, node, tz):
        x = nodeframe.copy(); x['day'] = x.index.tz_convert(tz).normalize()
        dev_rt = x[x.index.year.isin(yearset('dev'))].rt if market == 'ERCOT' else x[(x.index >= '2023-08-01') & (x.index < '2024-01-01')].rt
        lo, hi = (np.nanpercentile(dev_rt, 25), np.nanpercentile(dev_rt, 75)) if len(dev_rt.dropna()) else (np.nan, np.nan)
        for day, g in x.groupby('day'):
            if len(g) < 23 or g.rt.isna().any() or g.da.isna().any(): continue
            rt = g.rt.values.astype(float); da = g.da.values.astype(float)
            pf_rt, _ = lp_day(rt); pf_da, _ = lp_day(da)
            o = np.argsort(da); ch = sorted(o[:2]); later = [i for i in np.argsort(-da) if i > max(ch)][:2]
            spread = 0.0
            if len(later) == 2 and da[later].mean() * 0.85 > da[ch].mean():
                spread = float(0.85 * rt[later].sum() - rt[ch].sum())   # buy 2 MWh (1 MW x 2 h), sell 0.85 x 2 MWh
            soc = 0.0; thr = 0.0; se = np.sqrt(0.85)
            for pr in rt:
                if pr < lo and soc < 2.0:
                    q = min(1.0, (2.0 - soc) / se); soc += q * se; thr -= pr * q
                elif pr > hi and soc > 0:
                    q = min(1.0, soc * se); soc -= q / se; thr += pr * q
            rows.append(dict(market=market, node=node, day=day.tz_localize(None), year=day.year, pf_rt=pf_rt, pf_da=pf_da, rule_spread=spread, rule_threshold=thr))
    # ERCOT hubs
    da = pd.read_parquet(f'{PROC}/prices/ercot_dam.parquet'); rt = pd.read_parquet(f'{PROC}/prices/ercot_rtm_hourly.parquet')
    for node in ['HB_HOUSTON', 'HB_NORTH', 'HB_WEST', 'HB_SOUTH']:
        a = da[da.node == node].set_index('time_utc').price.rename('da'); b = rt[rt.node == node].set_index('time_utc').price.rename('rt')
        run(pd.concat([a, b], axis=1).sort_index(), 'ERCOT', node, 'America/Chicago')
    # CAISO hubs
    import glob
    cd = pd.concat([pd.read_parquet(f) for f in glob.glob(f'{GRID_ROOT}/data/raw/caiso/dam_*.parquet')])
    cr = pd.concat([pd.read_parquet(f) for f in glob.glob(f'{GRID_ROOT}/data/raw/caiso/rtpd_*.parquet')]) if glob.glob(f'{GRID_ROOT}/data/raw/caiso/rtpd_*.parquet') else None
    for node in ['TH_NP15_GEN-APND', 'TH_SP15_GEN-APND']:
        a = cd[cd.node == node].set_index('time_utc').LMP.rename('da')
        if cr is None: continue
        b = cr[cr.node == node].set_index('time_utc').LMP.resample('h').mean().rename('rt')
        run(pd.concat([a, b], axis=1).sort_index(), 'CAISO', node, 'America/Los_Angeles')
    t = pd.DataFrame(rows); t.to_parquet(f'{OUT}/battery_daily.parquet', index=False)
    for (m, n), g in t.groupby(['market', 'node']):
        for y, gy in g.groupby('year'):
            res.setdefault(m, {}).setdefault(n, {})[int(y)] = {k: round(float(gy[k].sum() / 1000 * 365 / len(gy)), 2) for k in ['pf_rt', 'pf_da', 'rule_spread', 'rule_threshold']} | {'days': int(len(gy))}
    summary['battery'] = {'units': '$ per kW-year (1 MW / 2 MWh, eta_rt 0.85, one daily cycle window), annualised over days with data', **res}
    print(json.dumps(res, indent=0)[:2000])


def frequency():
    """GridPulse baselines.
    GB (events with a loss label): RoCoF formula  dP = 2 E RoCoF / f0  (E = stored kinetic energy, GVA.s = GW.s),
      RoCoF = largest 1 s fall in the first 5 s (1 s data smooth the true initial RoCoF, so this underestimates it).
      (a) assumed E = median NESO outturn inertia over dev years (2019, 2021); (b) E = the event's own NESO outturn
      inertia (oracle input); inertia estimate E_est = dP_label f0 / (2 RoCoF) vs NESO outturn; dumb = dev median loss.
    ERCO: loss = k x delta_f with k = median(MW loss / delta f) over dev events (a frequency-response characteristic);
      dumb = dev median loss. (No RoCoF in the ERCOT reports.)"""
    res = {}
    ev = pd.read_parquet(f'{PROC}/frequency/gb_events.parquet')
    spg = SP['events'].get('GB', {})
    ev['split'] = np.where(ev.event_id.isin(spg.get('dev', [])), 'dev', np.where(ev.event_id.isin(spg.get('exam', [])), 'exam', None))
    ine = pd.read_parquet(f'{PROC}/frequency/GB/inertia.parquet')
    E_dev = float(ine[ine.time_utc.dt.year.isin(yearset('dev'))].inertia_outturn_GVAs.median())
    ev['loss_rocof_assumedE'] = 2 * E_dev * 1e3 * ev.rocof_1s_Hz_s / ev.f0_Hz
    ev['loss_rocof_outturnE'] = 2 * ev.inertia_outturn_GVAs * 1e3 * ev.rocof_1s_Hz_s / ev.f0_Hz
    ev['E_est_GVAs'] = ev.loss_MW / 1e3 * ev.f0_Hz / (2 * ev.rocof_1s_Hz_s)
    dev_med = float(ev[ev.split == 'dev'].loss_MW.median())
    ape = lambda p, y: float(np.median(np.abs(p - y) / np.abs(y)) * 100)
    for sp in ('dev', 'exam'):
        s = ev[(ev.split == sp) & ev.loss_MW.notna() & (ev.rocof_1s_Hz_s > 0)]
        s2 = s[s.inertia_outturn_GVAs.notna()]
        res.setdefault('GB', {})[sp] = {
            'n_events_with_loss': int(len(s)), 'n_with_inertia': int(len(s2)), 'E_assumed_GVAs': E_dev,
            'loss_MdAPE_pct_rocof_assumedE': ape(s.loss_rocof_assumedE, s.loss_MW) if len(s) else None,
            'loss_MdAPE_pct_rocof_outturnE': ape(s2.loss_rocof_outturnE, s2.loss_MW) if len(s2) else None,
            'loss_MdAPE_pct_dev_median': ape(dev_med, s.loss_MW) if len(s) else None,
            'inertia_MdAPE_pct_from_label_and_rocof': ape(s2.E_est_GVAs, s2.inertia_outturn_GVAs) if len(s2) else None,
            'inertia_MdAPE_pct_assumed_constant': ape(E_dev, s2.inertia_outturn_GVAs) if len(s2) else None}
    ev.to_parquet(f'{OUT}/frequency_events_gb.parquet', index=False)
    e = pd.read_parquet(f'{PROC}/frequency/ercot_events.parquet')
    spe = SP['events'].get('ERCO', {})
    e['split'] = np.where(e.event_id.isin(spe.get('dev', [])), 'dev', np.where(e.event_id.isin(spe.get('exam', [])), 'exam', None))
    e = e[(e.mw_loss > 0) & (e.delta_f_Hz > 0)]
    d = e[e.split == 'dev']
    k = float((d.mw_loss / d.delta_f_Hz).median()); med = float(d.mw_loss.median())
    e['loss_beta'] = k * e.delta_f_Hz
    # inertia-aware variant: dev fit of log(loss) = a + b log(delta_f) + c log(inertia)
    for sp in ('dev', 'exam'):
        s = e[e.split == sp]
        res.setdefault('ERCO', {})[sp] = {'n_events': int(len(s)), 'k_MW_per_Hz_dev': k,
                                          'loss_MdAPE_pct_beta_deltaf': ape(s.loss_beta, s.mw_loss) if len(s) else None,
                                          'loss_MdAPE_pct_dev_median': ape(med, s.mw_loss) if len(s) else None}
    e.to_parquet(f'{OUT}/frequency_events_ercot.parquet', index=False)
    summary['frequency'] = res; print(json.dumps(res, indent=1))


if __name__ == '__main__':
    what = sys.argv[1:] or ['demand']
    for w in what: globals()[w]()
    json.dump(summary, open(SUMF, 'w'), indent=1, default=str)
