"""WaterSim demo (a few seconds): a real snowmelt DEV basin with its dev-calibrated HBV parameters, all four models
on the same basin, and a generated regulated basin with a reservoir. Writes demo_watershed.png."""
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import watersim as ws  # noqa: E402
from watersim import metrics as M  # noqa: E402

fig, ax = plt.subplots(3, 2, figsize=(13, 10))
try:
    import waterdata as wd
    bid = 'camels_09035800'                      # a DEV basin (B's splits), snowmelt-dominated, Colorado
    spec, forc = wd.spec(bid), (wd.forcing(bid, '1999-10-01', '2008-09-30'), wd.forcing_meta(bid))
    q = wd.streamflow(bid, 'caravan')['Q_mm']
    cal = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checks', 'dev_calibrated_params.json')))['params'][bid]
    outs = {}
    for m in ('hbv', 'gr4j', 'sacsma', 'vic_lite'):
        outs[m] = ws.simulate(spec, forc, '2000-10-01', '2008-09-30', model=m, params=cal[f'{m}+degree_day'])
        L = outs[m]['ledger']
        obs = q.reindex(outs[m]['ts'].index)
        print(f"{m:9s} NSE {M.nse(obs, outs[m]['ts']['Q']):.3f}  KGE {M.kge(obs, outs[m]['ts']['Q']):.3f}  "
              f"ledger ok {L['ok']} (closure {L['closure_rel']:.1e})  P {L['P_in']:.0f}  ET {L['ET_soil']:.0f}  "
              f"Q {L['Q_out']:.0f} mm")
    ts = outs['hbv']['ts'].loc['2004-10-01':'2006-09-30']
    a = ax[0, 0]
    a.plot(q.loc[ts.index[0]:ts.index[-1]], 'k', lw=1, label='observed (USGS via Caravan)')
    for m, c in zip(outs, ['C0', 'C1', 'C2', 'C3']):
        a.plot(outs[m]['ts']['Q'].loc[ts.index[0]:ts.index[-1]], c, lw=0.8, label=m)
    a.set_title(f'{bid}: streamflow, dev-calibrated models (WY2005-06)')
    a.set_ylabel('mm/day')
    a.legend(fontsize=7)
    a = ax[0, 1]
    a.plot(ts['SWE'], label='SWE')
    a.plot(ts['soil'], label='soil moisture')
    a.plot(ts['groundwater'], label='groundwater (SUZ+SLZ)')
    if 'SWE_era5' in forc[0]:
        a.plot(forc[0]['SWE_era5'].loc[ts.index[0]:ts.index[-1]], 'k:', lw=0.8, label='ERA5-Land SWE (reference)')
    a.set_title('hidden states (HBV)')
    a.set_ylabel('mm')
    a.legend(fontsize=7)
    a = ax[1, 0]
    led = outs['hbv']['ledger']
    names = ['P_in', 'ET_soil', 'sublimation', 'Q_out']
    vals = [led[k] for k in names] + [led['dS_total']]
    a.bar(names + ['dS'], vals, color=['C0', 'C2', 'C4', 'C1', 'C7'])
    a.set_title(f"ledger 2000-08 (mm): closes to {led['closure_rel']:.1e}")
except Exception as e:  # no B data available
    print('real-basin part skipped:', e)
# generated regulated basin
specs = ws.generate_basins(60, rng=3)
s = next(x for x in specs if x.get('reservoirs') and x['reservoir_storage_frac'] > 0.2 and x['p_mean'] > 3.0)
f = ws.generate_weather(s, '1999-01-01', '2004-12-31', rng=5)
o = ws.simulate(s, f, '2000-01-01', model='sacsma', snow='energy_balance', routing='muskingum', obs_noise=True, rng=1)
print('generated', s['id'], 'area %.0f km2' % s['area_km2'], 'reservoirs', len(s['reservoirs']),
      'policy', s['reservoirs'][0]['policy'], 'ledger ok', o['ledger']['ok'])
ts2 = o['ts']
a = ax[1, 1]
a.plot(ts2['Q_natural'] * s['area_km2'] / 86.4, lw=0.8, label='natural flow')
a.plot(ts2['Q_m3s'], lw=0.8, label='regulated outflow')
a.plot(ts2['Q_obs'] * s['area_km2'] / 86.4, '.', ms=1, label='gauge-like Q_obs')
a.set_title(f"generated basin {s['id']}: {len(s['reservoirs'])} reservoir(s), {s['reservoirs'][0]['policy']}")
a.set_ylabel('m3/s')
a.legend(fontsize=7)
a = ax[2, 0]
a.plot(ts2['res_storage_hm3'], label='reservoir storage (hm3)')
a2 = a.twinx()
a2.plot(ts2['res_power_MW'], 'C3', lw=0.6, label='hydropower (MW)')
a.set_title('reservoir storage and hydropower')
a.legend(fontsize=7, loc='upper left')
a2.legend(fontsize=7, loc='upper right')
a = ax[2, 1]
lab = o['labels']
keys = ['bfi_lyne_hollick', 'baseflow_frac_model', 'runoff_ratio', 'evaporative_index', 'frac_snow', 'regulation_degree']
a.barh(keys, [lab[k] for k in keys])
a.set_title('some labels of the generated basin')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'demo_watershed.png'), dpi=110)
print('wrote', os.path.join(HERE, 'demo_watershed.png'))
