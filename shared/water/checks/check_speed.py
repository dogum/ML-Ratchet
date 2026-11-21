"""Speed: single runs (whole simulate() call incl. pandas/labels, and the numba core alone) and batch mode
(many parameter sets, many basins). One thread on the shared 2-core machine."""
import os
import time

import numpy as np

from _common import write
import watersim as ws
from watersim import engine
from watersim.simulate import prepare_run, _run
from watersim import routing as rt

cpu = [l for l in open('/proc/cpuinfo') if l.startswith('model name')][0].strip()
spec = {'id': 'speed', 'area_km2': 800, 'lat': 45, 'elev_mean_m': 1500, 'elev_min_m': 900, 'elev_max_m': 2500}
f = ws.generate_weather(spec, '1999-01-01', '2010-12-31', rng=1)
YEAR_S = 365.25 * 86400
res = {'what': 'wall time, single thread', 'cpu': cpu, 'target': '>= 10,000x real time (a basin-decade in well under 1 s)',
       'single': {}, 'batch': {}}
for model in ['hbv', 'gr4j', 'sacsma', 'vic_lite']:
    for snow in ['degree_day', 'energy_balance']:
        ws.simulate(spec, f, '2001-01-01', '2010-12-31', model=model, snow=snow)   # compile / warm
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            o = ws.simulate(spec, f, '2001-01-01', '2010-12-31', model=model, snow=snow, routing='uh')
            ts.append(time.perf_counter() - t0)
        ts2 = []
        for _ in range(5):
            t0 = time.perf_counter()
            ws.simulate(spec, f, '2001-01-01', '2010-12-31', model=model, snow=snow, routing='uh', labels=False)
            ts2.append(time.perf_counter() - t0)
        # numba core alone (5 bands, 10 years + 1 year spin-up)
        p = prepare_run(spec, f, model, None, snow)
        mod = ws.simulate.__globals__['MODULES'][model]
        tc = []
        for _ in range(20):
            s = np.array(mod.init_state(p.pm), float)
            uh = rt.model_uh(model, p.pm)
            t0 = time.perf_counter()
            _run(p, 0, len(p.F), s, np.zeros((len(p.frac), 4)), np.zeros((3, uh.shape[1])), uh, True)
            tc.append(time.perf_counter() - t0)
        nyrs = len(p.F) / 365.25
        res['single'][f'{model}_{snow}'] = {
            'simulate_10y_s': float(np.median(ts)), 'simulate_10y_no_labels_s': float(np.median(ts2)),
            'core_s_per_basin_year': float(np.median(tc) / nyrs), 'n_bands': len(p.frac),
            'x_realtime_simulate': float(10 * YEAR_S / np.median(ts)),
            'x_realtime_core': float(nyrs * YEAR_S / np.median(tc))}
        print(model, snow, res['single'][f'{model}_{snow}'])
# batch: 1000 parameter sets for one basin, 10 years
rng = np.random.default_rng(0)
for model in ['hbv', 'gr4j', 'sacsma', 'vic_lite']:
    pars = [ws.params_from_attributes(spec, model, 'degree_day', {k: rng.normal() for k in ('storage', 'perm', 'quick', 'snow')}, rng) for _ in range(1000)]
    ws.simulate_batch(spec, f, pars[:2], model=model, start='2001-01-01', end='2010-12-31')
    t0 = time.perf_counter()
    r = ws.simulate_batch(spec, f, pars, model=model, start='2001-01-01', end='2010-12-31')
    dt = time.perf_counter() - t0
    res['batch'][f'{model}_1000_paramsets_10y'] = {'wall_s': dt, 'ms_per_basin_decade': 1000 * dt / 1000,
                                                    'x_realtime': float(1000 * 11 * YEAR_S / dt)}
    print(model, 'batch', dt)
# many basins: 200 generated basins, own forcing
specs = ws.generate_basins(200, rng=5)
forc = [ws.generate_weather(s, '2000-01-01', '2010-12-31', rng=i) for i, s in enumerate(specs)]
ws.simulate_many(specs[:2], forc[:2], model='hbv', start='2001-01-01')
t0 = time.perf_counter()
r = ws.simulate_many(specs, forc, model='hbv', start='2001-01-01')
dt = time.perf_counter() - t0
res['batch']['hbv_200_basins_10y_simulate_many'] = {'wall_s': dt, 'ms_per_basin_decade': 1000 * dt / 200}
t0 = time.perf_counter()
for s, fr in zip(specs[:50], forc[:50]):
    ws.simulate(s, fr, '2001-01-01', model='hbv', routing='muskingum')
dt = time.perf_counter() - t0
res['batch']['hbv_50_basins_full_simulate_with_reservoirs_muskingum'] = {'wall_s': dt, 'ms_per_basin_decade': 1000 * dt / 50}
print(res['batch'])
write('speed.json', res)
