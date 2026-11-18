"""Data for the SnowSense page (web/data.json).

Explorer stations: a fixed random draw (seed 20261118) of 4 exam stations per region stratum, among the 621 scored
exam stations. Each carries WY2016-2025 windows (P, T, gap flags) so the page re-runs the network live, plus the
observed SWE, the uncalibrated energy-balance model and B's per-station calibrated degree-day model.
Also: every scored exam station's NSE for the map/strip, the ratchet table, and summary numbers.
python3 webdata.py <run>
"""
import json
import sys
import numpy as np
import pandas as pd
from common import *
from evalset import station_set, physics

run = sys.argv[1]
sys.path.insert(0, MODEL)   # the released snowsense.py (was {ROOT}/release in the sandbox)
import snowsense
from waterdata.models import degree_day

M = snowsense.SnowSense(f'{MODEL}/snowsense_weights.npz')
meta = wd().snotel_meta().set_index('station_id')
REG = {'WA': 'Pacific', 'OR': 'Pacific', 'CA': 'Pacific', 'ID': 'Northern Rockies', 'MT': 'Northern Rockies',
       'WY': 'Central Rockies', 'CO': 'Central Rockies', 'UT': 'Central Rockies', 'NV': 'Great Basin',
       'AZ': 'Southwest', 'NM': 'Southwest', 'AK': 'Alaska', 'SD': 'Central Rockies'}
ex = station_set('exam')
b = pd.read_parquet(SNOW_SCORES)
bdd = b[(b.method == 'dd')].set_index('station_id')
rng = np.random.default_rng(SEED)
pick = []
by = {}
for sid, lat, elev, ok in ex:
    by.setdefault(REG[meta.loc[sid, 'state']], []).append((sid, lat, elev, ok))
for reg in sorted(by):
    rows = by[reg]
    for j in rng.choice(len(rows), min(4, len(rows)), replace=False):
        pick.append((reg,) + rows[j])

stations = []
for reg, sid, lat, elev, ok in pick:
    df = load_station(sid)
    eb = physics(sid, lat, elev, 'energy_balance', df=df)
    d0 = wd().snotel(sid, '2000-10-01', '2025-09-30').asfreq('D')
    th = bdd.loc[sid, ['tt', 'ddf', 'pmult', 'tmelt']].values.astype(float)
    dd = pd.Series(degree_day(th, d0.P.fillna(0).values, d0.T_mean.interpolate(limit=3).fillna(0).values), index=d0.index)
    wins = []
    for y in range(2016, 2026):
        idx = pd.date_range(f'{y - 1}-08-01', periods=L, freq='D')
        w = df.reindex(idx)
        if w['T_mean'].isna().mean() > 0.2:
            continue
        wy = idx[(idx.year + (idx.month >= 10)) == y]
        r1 = lambda s: [None if not np.isfinite(v) else round(float(v), 1) for v in s]
        ri = lambda s: [None if not np.isfinite(v) else int(round(float(v))) for v in s]
        o = df['SWE_eod'].reindex(wy)
        win = {'wy': y, 'start': f'{y - 1}-08-01', 'nwy': len(wy), 'scored': y in ok,
                     'P': r1(w['P'].fillna(0).values), 'Tm': r1(w['T_mean'].interpolate(limit_direction='both').values),
                     'Tn': r1(w['T_min'].interpolate(limit_direction='both').values),
                     'Tx': r1(w['T_max'].interpolate(limit_direction='both').values),
                     'mP': [int(i) for i in np.where(w['mP'].fillna(1).values > 0)[0]],
                     'mT': [int(i) for i in np.where(w['mT'].fillna(1).values > 0)[0]],
                     'obs': ri(o.values), 'eb': ri(eb.reindex(wy).values), 'dd': ri(dd.reindex(wy).values),
                     }
        mPv = np.zeros(L); mPv[win['mP']] = 1
        mTv = np.zeros(L); mTv[win['mT']] = 1
        Xw = snowsense.features(np.array(win['P']), np.array(win['Tm']), np.array(win['Tn']), np.array(win['Tx']),
                                mPv, mTv, idx, lat, elev)
        Qw = M.forward(Xw[None].astype(np.float64))[0][61:61 + len(wy)]
        win['py_swe'] = [round(float(v), 2) for v in 100 * np.maximum(Qw[:, 2], 0) ** 2]
        wins.append(win)
    stations.append({'id': sid, 'name': meta.loc[sid, 'name'], 'state': meta.loc[sid, 'state'], 'region': reg,
                     'lat': lat, 'lon': float(meta.loc[sid, 'lon']), 'elev': elev, 'wins': wins})
    print(sid, reg, len(wins), flush=True)

net = pd.read_parquet(f'{ROOT}/results/{run}_exam.parquet').set_index('station_id')
phy = pd.read_parquet(f'{ROOT}/results/physics_exam.parquet')
ebs = phy[phy.method == 'eb_uncal'].set_index('station_id')
bcl = b[(b.method == 'clim')].set_index('station_id')
allst = []
for sid, lat, elev, ok in ex:
    if sid not in net.index:
        continue
    allst.append([sid, round(lat, 3), round(float(meta.loc[sid, 'lon']), 3), int(elev), meta.loc[sid, 'state'],
                  round(float(net.loc[sid, 'nse']), 3), round(float(ebs.loc[sid, 'nse']), 3),
                  round(float(bdd.loc[sid, 'nse']), 3), round(float(bcl.loc[sid, 'nse']), 3)])
R = json.load(open(f'{ROOT}/ratchet.json'))
out = {'run': run, 'stations': stations, 'exam': allst,
       'exam_cols': ['id', 'lat', 'lon', 'elev', 'state', 'net', 'eb', 'dd_cal', 'clim'],
       'summary': json.load(open(f'{ROOT}/results/final_summary.json')),
       'ratchet': [{k: r[k] for k in ('rung', 'phys', 'change', 'sim', 'kept', 'reason')} | {'dev': r['dev']['nse']}
                   for r in R['rungs']]}
S = out['summary']
hr = np.random.default_rng(SEED + 3)
hs = stations[hr.integers(len(stations))]
hw = [w for w in hs['wins'] if w['scored']]
out['hero'] = {'id': hs['id'], 'wy': hw[hr.integers(len(hw))]['wy']}
ex_r = json.load(open(f'{ROOT}/results/exam_by_rung.json'))
for r in out['ratchet']:
    r['exam'] = ex_r.get(r['rung'], {}).get('nse')
out['how'] = [
    '<b>Inputs.</b> Daily precipitation and mean, minimum and maximum air temperature from the station itself, plus its latitude and elevation. Never snow depth or SWE. Missing precipitation counts as zero and missing temperature is interpolated; both are flagged to the network.',
    '<b>Physics.</b> WaterSim\'s snow routines, verified in Phase 1: a degree-day model (HBV-light, exact against a hand calculation and TUWmodel to 1e-12 mm) and a daily energy-balance model (net radiation, turbulent heat, cold content, refreeze; terms exact against hand arithmetic). Each simulated station-winter uses one of the two, 50/50.',
    '<b>Weather.</b> Real daily P and T from 30 development stations, water years up to 2010 only, shifted to other climates (P x e^N(0, 0.3), T + N(0, 1.5) °C). The snowpack in the simulator feels a slightly different weather than the gauge records: wet-day precipitation x lognormal(0.25), one-day timing slips, temperature error 0.6 °C (AR(1)) plus a 0.5 °C station bias, and random data gaps in what the network reads.',
    '<b>Snow parameters.</b> Fitted on the development stations: both routines calibrated on each dev station\'s own SWE (water years up to 2010), then drawn from a multivariate normal fitted to those 36 parameter sets. That calibration found turbulent heat transfer about 5x weaker and fresh-snow albedo decaying faster than WaterSim\'s defaults.',
    '<b>Network.</b> A 2-layer GRU (hidden size 64, 51k parameters) reads one window from 1 August to 1 October of the next year and outputs five quantiles of the square root of SWE for every day. Trained with a pinball loss on 48,000 simulated station-winters.',
    f"<b>Ratchet.</b> Frozen simulated validation sets from six held-out dev stations' weather (one per physics version), a +0.01 margin on median NSE, and a dev gate on 38 real dev stations in 2011-2025. Six rungs, five kept.",
    f"<b>Error bars.</b> One widen-only factor per output fitted on the dev stations; all three came out at 1.0, because the dev intervals were already at or above 68%.",
    '<b>Annual numbers.</b> Peak SWE, its date and the melt-out date (first day after the peak below 5 mm, as in the WaterData baselines) come from the median curve. Their intervals come from the 16% and 84% curves, as if one error were shared by the whole winter.']
out['limits'] = [
    f"The per-station calibrated degree-day model is still better (median NSE {S['dd_cal']['nse']:.3f} vs {S['net']['nse']:.3f}; it wins at {100 * (1 - S['win_frac_vs_dd_cal']):.0f}% of stations). It uses ten years of the station's own snow record; SnowSense uses none.",
    f"Peak SWE is low by {-S['net']['peak_swe_bias_pct']:.0f}% (median). The dev twin test puts about half of that in the network's averaging over unknown parameters and half in snow the physics does not make (wind drift onto the pillow, gauge undercatch beyond the prior).",
    f"Low stations do worst: median NSE {S['by_elev']['<1500 m']['net']:.2f} below 1,500 m, where rain-on-snow and thin, intermittent packs dominate, against {S['by_elev']['>2800 m']['net']:.2f} above 2,800 m. The Southwest is weakest by region ({S['by_region']['Southwest']['net']:.2f}).",
    f"Error bars are conservative: the 68% daily band holds {100 * S['net']['cov68_daily']:.0f}% of exam snow days and the 68% melt-out interval {100 * S['net']['melt_cov68']:.0f}% of melt-outs. Widen-only calibration cannot narrow them.",
    'Each water year is estimated from 1 August with no knowledge of snow already on the ground. Stations that keep snow through summer are outside its training data.',
    'Inputs are the station\'s own gauge and thermometer. Driving it with gridded weather (ERA5-Land, PRISM) is untested; SNOTEL temperature sensors changed across the network in the 2000s, which may shift results by period.']
out['foot'] = f"Release rung {S['release_rung']} ({out['run']}). JS network checked against numpy; exam scored once after the ratchet finished."
json.dump(out, open(f'{ROOT}/web/data.json', 'w'), separators=(',', ':'))
print('bytes', os.path.getsize(f'{ROOT}/web/data.json'))
