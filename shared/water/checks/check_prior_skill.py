"""Skill of the attribute-prior parameters (no calibration) on the dev basins, before and after the dev-fitted
prior corrections (in-sample for the medians: the corrections were fitted on these basins' calibrations).
v1 (full median correction, SHRINK=1) results are kept in prior_skill_v0_v1.json; this file has v0 and the kept
v2 (SHRINK=0.5)."""
import numpy as np

from _common import write, CAL, VAL_LATE
import waterdata as wd
import watersim as ws
from watersim import metrics as M
from watersim.generator import params_from_attributes

res = {'what': __doc__.strip().split('\n')[0], 'periods': {'cal': CAL, 'val_late': VAL_LATE}, 'models': {}}
for model in ['hbv', 'gr4j', 'sacsma', 'vic_lite']:
    rec = {'v0_unadjusted': {'cal': [], 'val_late': []}, 'v2_adjusted_shrink05': {'cal': [], 'val_late': []}}
    for bid in wd.basins('camels', split='dev'):
        s = wd.spec(bid)
        f = (wd.forcing(bid), wd.forcing_meta(bid))
        q = wd.streamflow(bid, 'caravan')['Q_mm']
        for tag, adj in (('v0_unadjusted', False), ('v2_adjusted_shrink05', True)):
            p = params_from_attributes(ws.fill_spec(s), model, 'degree_day', None, None, adjust=adj)
            for per, (a, b) in (('cal', CAL), ('val_late', VAL_LATE)):
                o = ws.simulate(s, f, a, b, model=model, params=p, labels=False)
                obs = q.reindex(o['ts'].index).values
                rec[tag][per].append((M.nse(obs, o['ts']['Q'].values), M.kge(obs, o['ts']['Q'].values)))
    res['models'][model] = {tag: {per: {'median_NSE': float(np.nanmedian([x[0] for x in v])),
                                        'median_KGE': float(np.nanmedian([x[1] for x in v]))}
                                  for per, v in d.items()} for tag, d in rec.items()}
    print(model, res['models'][model])
write('prior_skill.json', res)
