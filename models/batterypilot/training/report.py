"""Exam tables with bootstrap intervals (weekly blocks).  python3 report.py <rung> -> data/exam/<rung>_report.json"""
import json, sys
import numpy as np, pandas as pd
import _paths  # noqa: E402  (outputs to WORK)

ROOT = _paths.WORK   # was the sandbox workspace grid/batterypilot
NB = 2000


def boot_weeks(daily, strategies, rng, nb=NB):
    """daily: rows (node, day, strategy, net) for one market-year-duration.  Resample whole weeks (same weeks for every
    node and strategy).  Returns dict strategy -> (nb,) annualised $/kW-yr (mean over nodes), and capture draws."""
    d = daily.copy()
    d['week'] = (pd.to_datetime(d.day) - pd.Timestamp('2000-01-03')).dt.days // 7
    pv = d.pivot_table(index='week', columns=['strategy', 'node'], values='net', aggfunc='sum')
    ndays = d[d.strategy == strategies[0]].groupby('week').day.nunique().reindex(pv.index).values   # distinct days
    W = len(pv)
    idx = rng.integers(0, W, (nb, W))
    out = {}
    tot_days = ndays[idx].sum(1)
    for s in strategies:
        m = pv[s].values.mean(1)                     # mean over nodes, $ per week per MW
        out[s] = m[idx].sum(1) / tot_days * 365 / 1000
    return out


def main(rung):
    rows = pd.read_parquet(f'{ROOT}/data/exam/{rung}_rows.parquet')
    daily = pd.read_parquet(f'{ROOT}/data/exam/{rung}_daily.parquet')
    monthly = pd.read_parquet(f'{ROOT}/data/exam/{rung}_monthly.parquet').reset_index(drop=True)
    rows['kWyr'] = rows.net / 1000 * 8760 / rows.hours
    rng = np.random.default_rng(20261114)
    strats = ['pf', 'rolling', 'spread', 'threshold', 'bp']
    rep = {'units': '$ per kW-year net of degradation (1 MW battery; mean over the nodes of each market), 95% intervals '
                    'from a weekly block bootstrap (2,000 draws, same weeks for every node and strategy)',
           'by_market_year_duration': [], 'by_node': [], 'overall': {}}
    for (m, y, d), g in rows.groupby(['market', 'year', 'duration']):
        dd = daily[(daily.market == m) & (daily.year == y) & (daily.duration == d)]
        bs = boot_weeks(dd, strats, rng)
        rec = dict(market=m, year=int(y), duration=int(d), nodes=int(g.node.nunique()))
        for s in strats:
            gs = g[g.strategy == s]
            rec[s] = round(float(gs.kWyr.mean()), 2)
            rec[s + '_ci'] = [round(float(np.percentile(bs[s], 2.5)), 2), round(float(np.percentile(bs[s], 97.5)), 2)]
            rec[s + '_cycles'] = round(float(gs.cycles.mean()), 0)
            rec[s + '_deg'] = round(float((gs.deg / 1000 * 8760 / gs.hours).mean()), 2)
        cap = bs['bp'] / bs['pf']
        rec['bp_capture'] = round(rec['bp'] / rec['pf'], 3)
        rec['bp_capture_ci'] = [round(float(np.percentile(cap, 2.5)), 3), round(float(np.percentile(cap, 97.5)), 3)]
        for s in ('rolling', 'spread', 'threshold'):
            rec[s + '_capture'] = round(rec[s] / rec['pf'], 3)
            diff = bs['bp'] - bs[s]
            rec['bp_minus_' + s] = round(rec['bp'] - rec[s], 2)
            rec['bp_minus_' + s + '_ci'] = [round(float(np.percentile(diff, 2.5)), 2), round(float(np.percentile(diff, 97.5)), 2)]
        # worst month of BatteryPilot: median over nodes of each node's worst month ($/kW-month), and the single worst
        mm = monthly[(monthly.market == m) & (monthly.year == y) & (monthly.duration == d) & (monthly.strategy == 'bp')]
        wm = mm.groupby('node').net.min() / 1000
        rec['bp_worst_month_median_node'] = round(float(wm.median()), 2)
        rec['bp_worst_month_any_node'] = round(float(wm.min()), 2)
        wmm = mm.loc[mm.net.idxmin()]
        rec['bp_worst_month_where'] = f"{wmm.node} month {int(wmm.month)}"
        rm = monthly[(monthly.market == m) & (monthly.year == y) & (monthly.duration == d) & (monthly.strategy == 'rolling')]
        rec['rolling_worst_month_median_node'] = round(float((rm.groupby('node').net.min() / 1000).median()), 2)
        rep['by_market_year_duration'].append(rec)
    for (m, n, y, d), g in rows.groupby(['market', 'node', 'year', 'duration']):
        r = dict(market=m, node=n, year=int(y), duration=int(d))
        for s in strats:
            r[s] = round(float(g[g.strategy == s].kWyr.iloc[0]), 2)
        r['bp_capture'] = round(r['bp'] / r['pf'], 3)
        r['bp_cycles'] = round(float(g[g.strategy == 'bp'].cycles.iloc[0]), 0)
        rep['by_node'].append(r)
    bn = pd.DataFrame(rep['by_node'])
    for m, g in bn.groupby('market'):
        rep['overall'][m] = dict(
            node_years=int(len(g) / 3),
            bp_mean_capture=round(float((g.bp / g.pf).mean()), 3),
            rolling_mean_capture=round(float((g['rolling'] / g.pf).mean()), 3),
            spread_mean_capture=round(float((g.spread / g.pf).mean()), 3),
            threshold_mean_capture=round(float((g.threshold / g.pf).mean()), 3),
            bp_beats_rolling_share=round(float((g.bp > g['rolling']).mean()), 3),
            bp_beats_spread_share=round(float((g.bp > g.spread).mean()), 3),
            bp_beats_threshold_share=round(float((g.bp > g.threshold).mean()), 3))
    json.dump(rep, open(f'{ROOT}/data/exam/{rung}_report.json', 'w'), indent=1)
    t = pd.DataFrame(rep['by_market_year_duration'])
    print(t[['market', 'year', 'duration', 'pf', 'rolling', 'spread', 'threshold', 'bp', 'bp_capture', 'bp_capture_ci',
             'bp_minus_rolling_ci', 'bp_cycles', 'bp_worst_month_median_node']].to_string())
    print(json.dumps(rep['overall'], indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
