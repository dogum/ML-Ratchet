"""Water-balance closure on many random runs: every model x snow routine x routing method, generated basins
(with their reservoirs, irrigation and cascades) and stochastic weather, plus real dev-basin forcing."""
import itertools
import time

import numpy as np

from _common import write
import watersim as ws

rng = np.random.default_rng(42)
specs = ws.generate_basins(40, rng=rng)
models = ['hbv', 'gr4j', 'sacsma', 'vic_lite']
snows = ['degree_day', 'energy_balance']
routs = ['none', 'uh', 'muskingum', 'kinematic']
rows = []
t0 = time.time()
for i, s in enumerate(specs):
    f = ws.generate_weather(s, '1999-01-01', '2006-12-31', rng=i)
    m = models[i % 4]
    sn = snows[(i // 4) % 2]
    r = routs[(i // 8) % 4] if i < 32 else routs[i % 4]
    o = ws.simulate(s, f, '2000-01-01', model=m, snow=sn, routing=r, rng=i)
    L = o['ledger']
    rows.append({'id': s['id'], 'model': m, 'snow': sn, 'routing': r, 'reservoirs': len(s.get('reservoirs', [])),
                 'irrigation_frac': s['irrigation_frac'], 'closure_rel': L['closure_rel'],
                 'max_daily_residual_mm': L['max_daily_residual_mm'], 'ok': L['ok'], 'P_mm': L['P_in'],
                 'Q_mm': L['Q_out'], 'ET_mm': L['ET_soil'] + L['sublimation'], 'res_evap_mm': L['reservoir_evaporation'],
                 'consumptive_mm': L['withdrawals_consumptive']})
# every model x snow x routing combination on one basin
s = specs[0]
f = ws.generate_weather(s, '1999-01-01', '2006-12-31', rng=99)
for m, sn, r in itertools.product(models, snows + ['none'], routs):
    L = ws.simulate(s, f, '2000-01-01', model=m, snow=sn, routing=r, labels=False)['ledger']
    rows.append({'id': s['id'] + '_grid', 'model': m, 'snow': sn, 'routing': r, 'closure_rel': L['closure_rel'],
                 'max_daily_residual_mm': L['max_daily_residual_mm'], 'ok': L['ok']})
# real forcing (dev basins only)
try:
    import waterdata as wd
    for bid in wd.basins('camels', split='dev')[:8]:
        sp = wd.spec(bid)
        fr = (wd.forcing(bid, '2000-10-01', '2008-09-30'), wd.forcing_meta(bid))
        for m in models:
            L = ws.simulate(sp, fr, '2001-10-01', model=m, snow='energy_balance' if m == 'hbv' else 'degree_day',
                            routing='muskingum', labels=False)['ledger']
            rows.append({'id': bid, 'model': m, 'routing': 'muskingum', 'closure_rel': L['closure_rel'],
                         'max_daily_residual_mm': L['max_daily_residual_mm'], 'ok': L['ok'], 'real_forcing': True})
except Exception as e:
    print('real forcing part skipped:', e)
# a 4-sub-basin network joined by Muskingum-Cunge reaches
sb = {f'b{i}': x for i, x in enumerate(ws.generate_basins(4, rng=8, regulation=False))}
fo = {k: ws.generate_weather(x, '1999-01-01', '2004-12-31', rng=i) for i, (k, x) in enumerate(sb.items())}
topo = {'b0': {'down': 'b2', 'length_km': 40, 'slope': 0.002, 'width_m': 20},
        'b1': {'down': 'b2', 'length_km': 25, 'slope': 0.003, 'width_m': 15},
        'b2': {'down': 'b3', 'length_km': 60, 'slope': 0.001, 'width_m': 40}, 'b3': {'down': None}}
net = ws.simulate_network(sb, fo, topo, '2000-01-01')
rows.append({'id': 'network_4_subbasins', 'model': 'hbv', 'routing': 'uh + network MC', 'closure_rel': net['ledger']['closure_rel'],
             'max_daily_residual_mm': 0.0, 'ok': net['ledger']['ok']})
# restart: two chained halves (init=final_state) equal one continuous run
restart = {}
sr = ws.generate_basins(3, rng=9, regulation=False)[1]
fr2 = ws.generate_weather(sr, '1999-01-01', '2004-12-31', rng=2)
for m, sn in itertools.product(models, snows):
    full = ws.simulate(sr, fr2, '2000-01-01', '2004-12-31', model=m, snow=sn, routing='none', labels=False)
    a = ws.simulate(sr, fr2, '2000-01-01', '2002-06-30', model=m, snow=sn, routing='none', labels=False)
    b = ws.simulate(sr, fr2, '2002-07-01', '2004-12-31', model=m, snow=sn, routing='none', labels=False,
                    init=a['final_state'])
    restart[f'{m}+{sn}'] = float(np.max(np.abs(np.r_[a['ts'].Q.values, b['ts'].Q.values] - full['ts'].Q.values)))
worst = max(r['closure_rel'] for r in rows)
worst_d = max(r['max_daily_residual_mm'] for r in rows)
write('balance.json', {
    'what': 'water-balance ledger closure: P + exchange gains = ET + sublimation + Q + exchange losses + reservoir '
            'evaporation + consumptive withdrawals + dS(snow, soil, groundwater, model UH, channel, irrigation '
            'return store, reservoirs), whole window and every day',
    'n_runs': len(rows), 'n_ok': sum(r['ok'] for r in rows), 'worst_closure_rel': worst,
    'worst_daily_residual_mm': worst_d, 'restart_chain_max_abs_diff_mm': restart, 'wall_s': time.time() - t0,
    'runs': rows})
print(len(rows), sum(r['ok'] for r in rows), worst, worst_d)
