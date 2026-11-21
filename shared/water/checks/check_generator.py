"""Generator realism: do randomly generated basins produce streamflow signatures that cover the real DEV basins?
Real side: observed Q of the 52 CAMELS-US dev basins, calibration window 1999-10..2008-09 (non-exam).
Simulated side: (a) prior='default' + stochastic weather, (b) prior='camels' (dev-basin attributes, jittered) with
real non-exam forcing of the donor basin. Each simulated basin uses a random model of the four.
Reports per-signature quantiles, the share of real basins inside the simulated 2.5-97.5 % range, the KS distance,
and a multivariate check (real-to-simulated nearest-neighbour distance vs simulated-to-simulated)."""
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from _common import write, CAL
import watersim as ws
from watersim import signatures as sg, realdata as rd
import waterdata as wd

SIGS = ['runoff_ratio', 'q_mean', 'baseflow_index', 'q5', 'q95', 'slope_fdc', 'high_q_freq', 'low_q_freq',
        'hfd_mean', 'flashiness', 'frac_snow', 'aridity']
t0 = time.time()
real = []
for bid in wd.basins('camels', split='dev'):
    q = wd.streamflow(bid, 'caravan')['Q_mm'].loc[CAL[0]:CAL[1]]
    f = wd.forcing(bid, *CAL)
    if q.notna().mean() < 0.9:
        continue
    d = sg.camels_signatures(q.values, f['P'].reindex(q.index).values, q.index)
    d['flashiness'] = sg.richards_baker(q.values)
    d['frac_snow'] = float(f['P'][f['T_mean'] < 0].sum() / f['P'].sum())   # same definition as the label
    d['aridity'] = f['PET'].sum() / f['P'].sum()
    d['id'] = bid
    real.append(d)
real = pd.DataFrame(real)
models = ['hbv', 'gr4j', 'sacsma', 'vic_lite']


def sim_sigs(spec, forcing, start, model, rng):
    o = ws.simulate(spec, forcing, start, model=model, snow='degree_day', routing='uh', rng=rng)
    lab = o['labels']
    d = {k[4:]: v for k, v in lab.items() if k.startswith('sig_')}
    d['flashiness'] = lab['flashiness_rb']
    d['frac_snow'] = lab['frac_snow_t0']
    d['aridity'] = lab['aridity']
    d['model'] = model
    d['ledger_ok'] = o['ledger']['ok']
    return d


rng = np.random.default_rng(7)
simA = []
for i, s in enumerate(ws.generate_basins(250, rng=11, prior='default')):
    f = ws.generate_weather(s, '1999-01-01', '2009-12-31', rng=i)
    simA.append(sim_sigs(s, f, '2000-01-01', models[i % 4], i))
simA = pd.DataFrame(simA)
simB = []
for i, s in enumerate(ws.generate_basins(250, rng=12, prior='camels')):
    df, meta, info = rd.random_forcing(rng, years=9, basin_id=s['donor_id'])
    simB.append(sim_sigs(s, (df, meta), info['sim_start'], models[i % 4], i))
simB = pd.DataFrame(simB)


def compare(sim):
    out = {}
    for k in SIGS:
        r = real[k].astype(float).dropna().values
        x = sim[k].astype(float).dropna().values
        lo, hi = np.percentile(x, [2.5, 97.5])
        out[k] = {'real_q05_q50_q95': np.percentile(r, [5, 50, 95]).round(4).tolist(),
                  'sim_q05_q50_q95': np.percentile(x, [5, 50, 95]).round(4).tolist(),
                  'real_inside_sim_95pct_range': float(np.mean((r >= lo) & (r <= hi))),
                  'ks_stat': float(ks_2samp(r, x).statistic)}
    keys = ['runoff_ratio', 'baseflow_index', 'slope_fdc', 'high_q_freq', 'low_q_freq', 'flashiness', 'q_mean']
    S = sim[keys].astype(float).dropna()
    R = real[keys].astype(float).dropna()
    mu, sd = S.mean(), S.std()
    Sz, Rz = ((S - mu) / sd).values, ((R - mu) / sd).values
    dRS = np.array([np.min(np.linalg.norm(Sz - r, axis=1)) for r in Rz])
    dSS = np.array([np.sort(np.linalg.norm(Sz - s, axis=1))[1] for s in Sz])
    out['multivariate_nn'] = {'keys': keys, 'median_real_to_sim_nn': float(np.median(dRS)),
                              'median_sim_to_sim_nn': float(np.median(dSS)),
                              'real_with_nn_beyond_sim_95pct': float(np.mean(dRS > np.percentile(dSS, 95)))}
    return out


res = {'what': __doc__.strip().split('\n')[0], 'n_real_dev': len(real), 'n_sim_default': len(simA),
       'n_sim_camels_prior': len(simB), 'default_prior_weather_generator': compare(simA),
       'camels_prior_real_forcing': compare(simB), 'ledgers_ok': bool(simA.ledger_ok.all() and simB.ledger_ok.all()),
       'prior_notes': ws.generator.PRIOR_NOTES, 'prior_adjust_meta': ws.generator.prior_adjustments().get('_meta'),
       'history': 'v0 (no dev correction): mean KS 0.238 / 0.251 (default / camels prior), real basins outside the '
                  'simulated cloud 10.6 % / 19.1 %; v1 (full median correction): 0.252 / 0.212, 6.4 % / 4.3 %; '
                  'v2 (half correction, SHRINK=0.5, kept): see this file. Files generator_v0_*.json, generator_v1_*.json',
       'wall_s': time.time() - t0}
write('generator.json', res)
for name in ('default_prior_weather_generator', 'camels_prior_real_forcing'):
    print(name)
    for k, v in res[name].items():
        print('  ', k, v)
