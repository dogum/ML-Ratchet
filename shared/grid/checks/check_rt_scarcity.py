"""Check GridSim physics 1.2's real-time scarcity process (gridsim.market) against real ERCOT prices.

For each year the process is driven by that year's real EIA-930 demand-forecast misses and drawn 200 times; its
statistics (mean and 5-95 % range of the draws) are set against the real HB_HUBAVG RT - DA spread of the same year:
  - spike frequency: events (runs of RT - DA > $100), hours above $100 / $300 / $1,000, mean and longest duration
  - size: quantiles of RT - DA in event hours, and the $ sum of RT - DA over event hours
  - hour-of-day and month profiles of event hours (shares; correlation of the simulated with the real profile)
  - start rate by quintile of the demand-forecast miss
  - where events fall in the day's DA shape: share of event hours among the day's 3 dearest DA hours
Each year is drawn with the default fit ("ERCO", driven by the year's real DA prices too) and with the first fit
("ERCO_first_fit", clock-hour profile, no DA-shape term) for comparison.
First on the training-input years (2018, 2020, 2022: the fit, in-sample), then on dev (2019, and 2021 with and
without Winter Storm Uri, 8-24 Feb). Exam years are not read.
Also checks that physics 1.2 changes nothing else: with 1.1 the process returns zeros; 1.2's outage defaults and
calibration equal 1.1's; a merit-order dispatch of one ERCOT week is identical under 1.1 and 1.2.

usage: python shared/grid/checks/check_rt_scarcity.py   -> checks/rt_scarcity.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

GRID_CODE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, f"{GRID_CODE}/gridsim")
sys.path.insert(0, f"{GRID_CODE}/checks")
sys.path.insert(0, GRID_CODE)
from gridsim import market  # noqa: E402
from gridsim import fleet as gfleet  # noqa: E402
from fit_rt_scarcity import year_frame, events, U  # noqa: E402

N_DRAWS = 200
URI = ('2021-02-08', '2021-02-24')
AFTERNOON = list(range(13, 21))        # local hours 13-20


def year_stats(s, hour, month, e, err_sd, rank):
    st, L = events(s)
    ev = s > U
    out = dict(events=int(len(st)), h100=int(ev.sum()), h300=int((s > 300).sum()), h1000=int((s > 1000).sum()),
               dur_mean=float(L.mean()) if len(L) else 0.0, dur_max=int(L.max()) if len(L) else 0,
               sum_event_spread_k=float(s[ev].sum() / 1000.0))
    for qq in (0.5, 0.9, 0.99):
        out[f'event_spread_q{int(qq * 100)}'] = float(np.quantile(s[ev], qq)) if ev.any() else 0.0
    out['afternoon_share'] = float(ev[np.isin(hour, AFTERNOON)].sum() / max(ev.sum(), 1))
    out['top3_da_share'] = float(ev[rank >= 20.5 / 23].sum() / max(ev.sum(), 1))
    hp = np.bincount(hour[ev], minlength=24).astype(float)
    mp = np.bincount(month[ev] - 1, minlength=12).astype(float)
    z = e / err_sd
    qz = np.quantile(z, [0.2, 0.4, 0.6, 0.8])
    b = np.searchsorted(qz, z)
    start = np.zeros(len(s), bool)
    start[st] = True
    out['start_rate_by_miss_quintile'] = [float(start[b == k].mean() * 1000) for k in range(5)]   # per 1,000 h
    return out, hp, mp


def summarise(draws):
    keys = [k for k in draws[0] if not isinstance(draws[0][k], list)]
    out = {}
    for k in keys:
        v = np.array([d[k] for d in draws], float)
        out[k] = dict(mean=round(float(v.mean()), 3), p5=round(float(np.quantile(v, 0.05)), 3),
                      p95=round(float(np.quantile(v, 0.95)), 3))
    v = np.array([d['start_rate_by_miss_quintile'] for d in draws], float)
    out['start_rate_by_miss_quintile'] = [round(float(x), 3) for x in v.mean(0)]
    return out


def check_year(label, f, p, first=False):
    s = (f.rt - f.da).values
    hour, month, e, rank = f.hour.values, f.month.values, f.e.values, f['rank'].values
    real, hp_r, mp_r = year_stats(s, hour, month, e, p['err_sd'], rank)
    draws, hps, mps = [], [], []
    for k in range(N_DRAWS):
        a = market.rt_scarcity(f.index, 'ERCO', seed=1000 + k, demand_err=e, version='1.2', params_=p,
                               da=None if first else f.da.values)
        st, hp, mp = year_stats(a, hour, month, e, p['err_sd'], rank)
        draws.append(st); hps.append(hp); mps.append(mp)
    sim = summarise(draws)
    hp_s, mp_s = np.mean(hps, 0), np.mean(mps, 0)
    inside = {k: bool(sim[k]['p5'] <= real[k] <= sim[k]['p95']) for k in real if not isinstance(real[k], list)}
    return dict(
        label=label, hours=int(len(s)),
        real={k: (round(v, 3) if isinstance(v, float) else v) for k, v in real.items()},
        sim=sim, real_inside_sim_5_95=inside,
        hour_profile=dict(real_share=[round(float(x), 4) for x in hp_r / max(hp_r.sum(), 1)],
                          sim_share=[round(float(x), 4) for x in hp_s / max(hp_s.sum(), 1)],
                          corr=round(float(np.corrcoef(hp_r, hp_s)[0, 1]), 3)),
        month_profile=dict(real_share=[round(float(x), 4) for x in mp_r / max(mp_r.sum(), 1)],
                           sim_share=[round(float(x), 4) for x in mp_s / max(mp_s.sum(), 1)],
                           corr=round(float(np.corrcoef(mp_r, mp_s)[0, 1]), 3)),
        real_spread_quantiles={str(q): round(float(np.quantile(s, q)), 1) for q in (0.01, 0.05, 0.5, 0.95, 0.99, 0.999)},
        expected_adder_mean=round(float(market.expected(f.index, 'ERCO', version='1.2', params_=p,
                                                        da=None if first else f.da.values).mean()), 3),
        sim_adder_mean=round(float(np.mean([d['sum_event_spread_k'] for d in draws]) * 1000 / len(s)), 3),
        real_rt_minus_da_mean=round(float(s.mean()), 3))


def identity_checks():
    """Physics 1.2 changes nothing but the opt-in process."""
    out = {}
    idx = pd.date_range('2019-01-01', periods=8760, freq='h', tz='UTC')
    out['zeros_with_1_1'] = bool(not market.rt_scarcity(idx, 'ERCO', seed=1, version='1.1').any()
                                 and not market.expected(idx, 'ERCO', version='1.1').any())
    out['zeros_with_1_0'] = bool(not market.rt_scarcity(idx, 'ERCO', seed=1, version='1.0').any())
    out['zeros_for_CISO_1_2'] = bool(not market.rt_scarcity(idx, 'CISO', seed=1, version='1.2').any())
    out['calibration_1_2_equals_1_1'] = gfleet.calibration('1.2') == gfleet.calibration('1.1')
    old = os.environ.get('GRIDSIM_PHYSICS')
    try:
        vals = {}
        for v in ('1.1', '1.2'):
            os.environ['GRIDSIM_PHYSICS'] = v
            vals[v] = {(t, k): gfleet.tech_default(t, k) for t in gfleet.TECH_DEFAULTS for k in gfleet._DEF_KEYS}
        out['tech_defaults_1_2_equal_1_1'] = vals['1.1'] == vals['1.2']
        from check_ercot_dispatch import real_inputs, FLEET
        from gridsim.dispatch import dispatch
        res = {}
        for v in ('1.1', '1.2'):
            os.environ['GRIDSIM_PHYSICS'] = v
            y = real_inputs(2019, '2019-08-12', '2019-08-18 23:00')
            r = dispatch(FLEET, y.d, wind_MW=y.ng_WND, solar_MW=y.ng_SUN, imports_MW=-y.ti, hydro_MW=y.ng_WAT,
                         fixed_MW={"nuclear": y.ng_NUC}, method='merit')
            res[v] = (np.asarray(r.price, float), np.asarray(r.p, float))
        out['merit_week_2019_08_12_identical'] = bool(np.array_equal(res['1.1'][0], res['1.2'][0])
                                                      and np.array_equal(res['1.1'][1], res['1.2'][1]))
        out['merit_week_mean_price'] = round(float(res['1.2'][0].mean()), 4)
    finally:
        if old is None:
            os.environ.pop('GRIDSIM_PHYSICS', None)
        else:
            os.environ['GRIDSIM_PHYSICS'] = old
    return out


if __name__ == '__main__':
    t0 = time.time()
    p = market.params('ERCO')
    p1 = market.params('ERCO_first_fit')
    out = dict(check='rt_scarcity', physics='1.2', params=p, params_first_fit=p1, draws=N_DRAWS, threshold_usd=U,
               note='real = ERCOT HB_HUBAVG RT - DA; sim = gridsim.market.rt_scarcity alone (the part the DA cannot '
                    'see), driven by the same year\'s real EIA-930 demand-forecast misses; statistics are means and '
                    '5-95 % ranges over the draws. Fit on 2018/2020/2022 only (fit_rt_scarcity.py).',
               train={}, dev={}, first_fit=dict(train={}, dev={}))
    for y in (2018, 2020, 2022):
        f = year_frame(y)
        out['train'][str(y)] = check_year(str(y), f, p)
        out['first_fit']['train'][str(y)] = check_year(str(y), f, p1, first=True)
        print(y, json.dumps({k: out['train'][str(y)][k] for k in ('real', 'real_inside_sim_5_95')}), flush=True)
    for y in (2019, 2021):
        f = year_frame(y)
        out['dev'][str(y)] = check_year(str(y), f, p)
        out['first_fit']['dev'][str(y)] = check_year(str(y), f, p1, first=True)
        if y == 2021:
            m = ~((f.index >= pd.Timestamp(URI[0], tz='UTC')) & (f.index < pd.Timestamp(URI[1], tz='UTC')))
            out['dev']['2021_without_uri'] = check_year('2021 without 8-24 Feb', f[m], p)
            out['first_fit']['dev']['2021_without_uri'] = check_year('2021 without 8-24 Feb', f[m], p1, first=True)
    out['identity'] = identity_checks()
    out['seconds'] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(GRID_CODE, 'checks', 'rt_scarcity.json'), 'w'), indent=1)
    for part in ('train', 'dev'):
        for k, r in out[part].items():
            print(part, k, 'events', r['real']['events'], r['sim']['events'], 'h100', r['real']['h100'], r['sim']['h100'],
                  'h1000', r['real']['h1000'], r['sim']['h1000'], 'hourcorr', r['hour_profile']['corr'],
                  'monthcorr', r['month_profile']['corr'], 'top3', r['real']['top3_da_share'],
                  r['sim']['top3_da_share']['mean'], 'first fit top3',
                  out['first_fit'][part][k]['sim']['top3_da_share']['mean'],
                  'first fit hourcorr', out['first_fit'][part][k]['hour_profile']['corr'])
    print(out['identity'])
