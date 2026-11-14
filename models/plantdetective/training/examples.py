"""Demo examples for the release: three exam units from the page's fixed stratified draw (first coal, CC, CT)."""
from common import *
import json
import numpy as np, pandas as pd
from pdcore import truth, model_input, flex_class, REG
from evaluate import real_unit
from webdata import explorer_units
import realdata
units = explorer_units()
pick = []
for tech in ('coal_steam', 'gas_cc', 'gas_ct'):
    pick.append(next(u for u in units if realdata.truth_tech(realdata.units().set_index('unit_id').loc[u]) == tech))
pr = realdata.prices()
for uid in pick:
    rec, t = real_unit('exam', uid)
    m = pd.to_datetime(t, unit='s').year <= 2025
    rec = {k: v[m] for k, v in rec.items()}; t = t[m]
    tr = truth(rec['op'], rec['P'], rec['HI'], rec['meas'])
    U = realdata.units().set_index('unit_id').loc[uid]
    tt = pd.to_datetime(t, unit='s', utc=True)
    ctx = realdata.context(uid); ctx['hr923'] = {str(k): v for k, v in ctx['hr923'].items()}
    meta = dict(id=uid, name=str(U.facility_name), unit=str(U.unit_id_epa), tech=realdata.truth_tech(U),
                years=f'{tt.year.min()}-{tt.year.max()}', context=ctx,
                truth={k: (None if not np.isfinite(tr[k]) else float(tr[k])) for k in REG},
                flex_true=flex_class(tr['starts_yr'], tr['cf'], tr['med_up']))
    os.makedirs(f'{ROOT}/release/examples', exist_ok=True)
    np.savez_compressed(f'{ROOT}/release/examples/{uid.replace("*", "s")}.npz', x=model_input(rec['op'], rec['P']).astype(np.int16),
                        t=t, price=pr.price.reindex(tt).values.astype(np.float32), gas=pr.gas.reindex(tt).values.astype(np.float32),
                        meta=np.array(json.dumps(meta)))
    print(uid, meta['name'], meta['tech'])
