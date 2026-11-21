"""Fit the generator's dev-calibrated corrections (recipe step 5: calibrate the simulator on dev only).

For every dev basin and model, compare the KGE-calibrated parameters (checks/out/dev_calib) with the attribute
transfer-function median (latent factors 0, no scatter, no adjustment). Per parameter:
  positive parameters spanning > 3x: factor = exp(median log ratio), clipped to [0.1, 10];
                                     extra_sigma = sqrt(max(var(log ratio) - var_prior, 0))
  others (TT, X2, ...):             shift = median difference; extra_sigma likewise on differences
var_prior is the spread the generator already produces (latent + scatter), measured by sampling it.
Parameters that the calibration pins to a bound for > 40 % of basins are reported but not adjusted.
Writes watersim/watersim/data/prior_adjust.json (and a copy of the fit report in checks/prior_fit.json)."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(_CODE, 'watersim'))
sys.path.insert(0, _CODE)
import waterdata as wd  # noqa: E402
from watersim.generator import params_from_attributes  # noqa: E402
from watersim.models import MODULES  # noqa: E402
from watersim import snow as sn  # noqa: E402
from watersim.calibrate import FIXED_DEFAULT  # noqa: E402

SHRINK = float(os.environ.get('SHRINK', '1.0'))   # fraction of the median correction applied (dev-gated choice)
rows = [json.load(open(p)) for p in glob.glob(os.path.join(_DATA, 'checks', 'out', 'dev_calib', '*.json'))]
rows = [r for r in rows if r.get('params') and r['snow'] == 'degree_day']
specs = {}
rng = np.random.default_rng(0)
table, report = {}, {}
bounds = {m: {p[0]: (p[2], p[3]) for p in MODULES[m].PARAMS} for m in MODULES}
bounds['snow_dd'] = {p[0]: (p[2], p[3]) for p in sn.DD_PARAMS}
for model in MODULES:
    rr = [r for r in rows if r['model'] == model]
    groups = [(model, model, 'none')] + ([('snow_dd', 'none', 'degree_day')] if model == 'hbv' else [])
    for group, m_arg, s_arg in groups:
        diffs = {}
        for r in rr:
            b = r['basin']
            if b not in specs:
                specs[b] = wd.spec(b)
            med = params_from_attributes(specs[b], m_arg, s_arg, None, None, adjust=False)
            # generator spread for this basin (latent + scatter)
            samp = [params_from_attributes(specs[b], m_arg, s_arg, {k: rng.normal() for k in
                                           ('storage', 'perm', 'quick', 'snow')}, rng, adjust=False) for _ in range(60)]
            for k, v in r['params'].items():
                if k not in med:
                    continue
                lo, hi = bounds[group][k]
                vals = np.array([sm[k] for sm in samp])
                mult = lo > 0 and hi / lo > 3
                if mult:
                    d = np.log(v / med[k])
                    pv = np.var(np.log(vals))
                else:
                    d = v - med[k]
                    pv = np.var(vals)
                at_bound = (v <= lo + 1e-3 * (hi - lo)) or (v >= hi - 1e-3 * (hi - lo))
                diffs.setdefault(k, []).append((d, pv, mult, at_bound))
        tab = {}
        for k, lst in diffs.items():
            d = np.array([x[0] for x in lst])
            pv = float(np.median([x[1] for x in lst]))
            mult = lst[0][2]
            fb = float(np.mean([x[3] for x in lst]))
            extra = float(np.sqrt(max(np.var(d) - pv, 0.0)))
            rep = {'n': len(d), 'median_diff': float(np.median(d)), 'std_diff': float(np.std(d)),
                   'prior_std': float(np.sqrt(pv)), 'frac_at_bound': fb, 'mode': 'mult' if mult else 'add'}
            if fb <= 0.4:
                if mult:
                    tab[k] = {'mode': 'mult', 'factor': float(np.exp(SHRINK * np.clip(np.median(d), -np.log(10), np.log(10)))),
                              'extra_sigma': min(extra, 1.0)}
                else:
                    tab[k] = {'mode': 'add', 'shift': float(SHRINK * np.median(d)), 'extra_sigma': extra}
                rep['applied'] = tab[k]
            else:
                rep['applied'] = None
            report.setdefault(group, {})[k] = rep
        table[group] = tab
out = os.path.join(_CODE, 'watersim', 'watersim', 'data', 'prior_adjust.json')
meta = {'what': __doc__.strip().split('\n')[0], 'source': 'checks/out/dev_calib (52 CAMELS-US dev basins, KGE, '
        'calibration window 1999-10..2008-09)', 'n_records': len(rows), 'shrink': SHRINK}
json.dump({**table, '_meta': meta}, open(out, 'w'), indent=1)
json.dump({'what': 'generator prior fit report (dev only)', **meta, 'report': report},
          open(os.path.join(_CODE, 'checks', 'prior_fit.json'), 'w'), indent=1)
for g, t in report.items():
    print(g)
    for k, v in t.items():
        print('   %-9s med %+.2f sd %.2f prior_sd %.2f bound %.2f -> %s' % (k, v['median_diff'], v['std_diff'], v['prior_std'], v['frac_at_bound'], v['applied']))
