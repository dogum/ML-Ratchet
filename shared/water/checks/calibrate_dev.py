"""Calibrate every model per DEV basin (B's splits) on 1999-10..2008-09 (KGE objective) and score it on two
non-exam validation windows (1980-10..1989-09 and 2008-10..2014-09). Resumable: one JSON per basin x model in
checks/out/dev_calib/. Exam basins and exam years are never read. Usage: python3 calibrate_dev.py [models...]"""
import json
import os
import sys
import time

from _common import CAL, VAL_EARLY, VAL_LATE, OUT, dev_basins, assert_not_exam_window
import numpy as np
import waterdata as wd
import watersim as ws
from watersim.calibrate import calibrate
from watersim import metrics as M

OBJ = os.environ.get('OBJ', 'kge')
MODELS = sys.argv[1:] or ['hbv', 'gr4j', 'sacsma', 'vic_lite']
D = os.path.join(OUT, 'dev_calib' if OBJ == 'kge' else f'dev_calib_{OBJ}')
os.makedirs(D, exist_ok=True)
for w in (CAL, VAL_EARLY, VAL_LATE):
    assert_not_exam_window(*w)
ids, _ = dev_basins()
for bid in ids:
    s = wd.spec(bid)
    f = wd.forcing(bid)
    meta = wd.forcing_meta(bid)
    q = wd.streamflow(bid, 'caravan')['Q_mm']
    for (a, b) in [('1989-10-01', '1999-09-30'), ('2015-10-01', '2023-09-30')]:   # blank exam years
        q = q.where((q.index < a) | (q.index > b))
    for model in MODELS:
        for snow in (['degree_day', 'energy_balance'] if model == 'hbv' and OBJ == 'kge' else ['degree_day']):
            path = os.path.join(D, f'{bid}_{model}_{snow}.json')
            if os.path.exists(path):
                continue
            t0 = time.time()
            rec = {'basin': bid, 'model': model, 'snow': snow}
            try:
                r = calibrate(s, (f, meta), q, *CAL, model=model, snow=snow, objective=OBJ, pop=40, gens=80, rng=1)
                rec['params'] = r['params']
                rec['cal_objective'] = r['value']
                rec['n_evals'] = r['n_evals']
                for name, (a, b) in (('cal', CAL), ('val_early', VAL_EARLY), ('val_late', VAL_LATE)):
                    out = ws.simulate(s, (f, meta), a, b, model=model, snow=snow, params=r['params'], labels=False)
                    sim = out['ts']['Q']
                    obs = q.reindex(sim.index)
                    if obs.notna().sum() < 365:
                        rec[name] = None
                        continue
                    rec[name] = M.all_metrics(obs.values, sim.values)
                    rec[name]['ledger_ok'] = out['ledger']['ok']
                # also the uncalibrated attribute-prior parameters (an 'ungauged' prior reference)
                out = ws.simulate(s, (f, meta), *CAL, model=model, snow=snow, labels=False)
                sim = out['ts']['Q']
                rec['prior_cal'] = M.all_metrics(q.reindex(sim.index).values, sim.values)
            except Exception as e:  # record and continue
                rec['error'] = repr(e)
            rec['wall_s'] = time.time() - t0
            json.dump(rec, open(path, 'w'), default=float)
            print(bid, model, snow, rec.get('cal', {}) and round(rec['cal']['KGE'], 3), round(rec['wall_s'], 1), flush=True)
print('DONE')
