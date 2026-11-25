"""Check: the harness's reference rows reproduce WaterData's own baseline scores (flow_scores.parquet) on the 52 DEV
basins in the main and recent windows (dev basins only: no exam unit is read). Writes results/check_refs.json."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import numpy as np, pandas as pd
import bake, refs as R, score as S
import waterdata as wd

fs = pd.read_parquet(os.path.join(C.BASELINES, 'flow_scores.parquet'))
fs = fs[fs.split == 'dev']
name_map = {'udev': 'nn_dev', 'uk12': 'nn_k12', 'gr4j': 'gr4j', 'nwm': 'nwm', 'clim': 'clim', 'persistence': 'persist1'}
out, worst = {}, 0.0
for per, wper in (('main', 'val'), ('recent', 'recent')):
    for b in bake.dev_basins():
        obs = bake._obs(b, per)
        if obs is None or obs.notna().sum() < 365:
            continue
        full = wd.streamflow(b, C.PERIODS[per][1]).Q_mm
        for name, wn in name_map.items():
            s = R.reference(name, b, per, obs_full=full, lead=1)
            if s is None:
                continue
            r = S.score_real(obs, s.values)
            g = fs[(fs.basin == b) & (fs.method == wn) & (fs.period == wper)]
            if len(g) == 0:
                continue
            for m in ('nse', 'kge', 'fhv'):
                d = abs(r[m] - float(g[m].iloc[0]))
                if np.isfinite(d):
                    worst = max(worst, d)
                    out.setdefault(f'{per}/{name}/{m}', []).append(d)
res = {k: {'n': len(v), 'max_abs_diff': float(np.max(v))} for k, v in sorted(out.items())}
res['worst'] = worst
json.dump(res, open(os.path.join(C.RESULTS, 'check_refs.json'), 'w'), indent=1)
print(json.dumps(res, indent=0))
