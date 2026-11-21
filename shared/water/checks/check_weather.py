"""Stochastic weather generator vs real forcing on the DEV basins: fit_climate() on 1999-10..2008-09 (non-exam), then
generate 9 years and compare statistics that matter for runoff (mean P, wet-day frequency, wet-day persistence,
99th percentile, annual-total variability, T mean/amplitude/std, dry-spell length), and the streamflow they produce
through the same dev-calibrated HBV (KGE-calibrated parameters): mean Q, Q95, BFI, flashiness."""
import json

import numpy as np
import pandas as pd

from _common import write, CAL
import waterdata as wd
import watersim as ws
from watersim import signatures as sg


def stats(df):
    P = df['P'].values
    wet = P > 0.1
    dry_runs = []
    c = 0
    for v in wet:
        if not v:
            c += 1
        elif c:
            dry_runs.append(c)
            c = 0
    ann = df['P'].groupby(df.index.year).sum()
    ann = ann[df['P'].groupby(df.index.year).count() > 360]
    return {'P_mean': P.mean(), 'wet_frac': wet.mean(), 'p11': np.mean(wet[1:][wet[:-1]]),
            'P99': np.percentile(P, 99), 'annual_cv': ann.std() / ann.mean(), 'dry_spell_mean': np.mean(dry_runs),
            'T_mean': df['T_mean'].mean(), 'T_std': df['T_mean'].std()}


cal = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dev_calibrated_params.json')))['params']
rows = []
for i, bid in enumerate(wd.basins('camels', split='dev')):
    s = wd.spec(bid)
    real = wd.forcing(bid, *CAL)
    meta = wd.forcing_meta(bid)
    clim = ws.fit_climate(real)
    gen = ws.generate_weather(s, '1998-10-01', '2008-09-30', rng=i, climate=clim)
    r, g = stats(real), stats(gen.loc[CAL[0]:])
    p = cal[bid]['hbv+degree_day']
    o_r = ws.simulate(s, (wd.forcing(bid, '1998-10-01', CAL[1]), meta), CAL[0], model='hbv', params=p, labels=False)
    o_g = ws.simulate({**s, 'srad_is_net': False}, gen, CAL[0], model='hbv', params=p, labels=False,
                      pet='penman_monteith')
    qr, qg = o_r['ts']['Q'].values, o_g['ts']['Q'].values
    rows.append({'basin': bid, 'real': r, 'gen': g,
                 'q_real': {'q_mean': qr.mean(), 'q95': np.percentile(qr, 95), 'bfi': sg.bfi(qr), 'rb': sg.richards_baker(qr)},
                 'q_gen': {'q_mean': qg.mean(), 'q95': np.percentile(qg, 95), 'bfi': sg.bfi(qg), 'rb': sg.richards_baker(qg)}})
summ = {}
for k in rows[0]['real']:
    a = np.array([x['real'][k] for x in rows])
    b = np.array([x['gen'][k] for x in rows])
    summ[k] = {'median_real': float(np.median(a)), 'median_gen': float(np.median(b)),
               'median_ratio_gen_real': float(np.median(b / a)) if k not in ('T_mean',) else None,
               'median_diff': float(np.median(b - a)), 'corr_across_basins': float(np.corrcoef(a, b)[0, 1])}
for k in rows[0]['q_real']:
    a = np.array([x['q_real'][k] for x in rows])
    b = np.array([x['q_gen'][k] for x in rows])
    summ['streamflow_' + k] = {'median_real_forcing': float(np.median(a)), 'median_gen_forcing': float(np.median(b)),
                               'median_ratio': float(np.median(b / a)), 'corr_across_basins': float(np.corrcoef(a, b)[0, 1])}
write('weather_generator.json', {'what': __doc__.strip().split('\n')[0], 'n_basins': len(rows), 'summary': summ,
                                 'note': 'PET for generated weather is FAO-56 PM from the generated variables; the real '
                                         'runs use Caravan PET', 'basins': rows})
for k, v in summ.items():
    print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
