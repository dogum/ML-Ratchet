"""Grid search of GB response knobs on the dev twins (median twin vs real nadir and 60 s settling, 4 dev events)."""
from common import *
import gpsim, twin_gb as T, itertools
res = []
real = {e: T.real_window(e) for e in GB_DEV}
grid = list(itertools.product([0.0], [(150, 800), (300, 1500), (600, 2500)], [0.5, 0.75, 1.25], [(0, 300), (0, 800)]))
for i, (dcp, gh, D, ffr) in enumerate(grid):
    name = gpsim.derive('p0', f'g{i}', GB=dict(dc_p=dcp, gov_head=gh, D_med=D, ffr=ffr, ffr_p=0.7))
    row = dict(i=i, dc_p=dcp, gov_head=gh, D=D, ffr=ffr)
    errs = []
    for e in GB_DEV:
        s, L = T.twins(e, name, n=40)
        rn, rs = real[e]['nadir'][0], real[e]['settle'][0]
        mn, ms = np.median(s['nadir']), np.median(s['settle'])
        row[e[3:11]] = (round(float(mn / rn), 2), round(float(ms / rs), 2))
        errs += [abs(np.log(mn / rn)), abs(np.log(ms / rs))]
    row['score'] = float(np.mean(errs)); res.append(row); print(row, flush=True)
json.dump(res, open(f'{ROOT}/logs/twin_grid1.json', 'w'), indent=1)
