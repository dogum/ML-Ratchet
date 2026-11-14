"""Grid 2: GB response knobs incl. secondary. Real metrics are taken relative to the 5 s before onset (removes the
11 Jul pre-event drift), sims are noise-free. Score = mean |log ratio| of nadir (4 events) and 60 s settling (3; 9 Aug
excluded: later trips and LFDD)."""
from common import *
import gpsim, twin_gb as T, itertools, measure as M
tag = sys.argv[1]
real = {}
for e in GB_DEV:
    r = T.real_window(e); w = r['w'][0]; ref = w[M.PRE - 5:M.PRE].mean()
    real[e] = (w[M.PRE:].min() - ref, w[M.PRE + 55:M.PRE + 61].mean() - ref)
print({k[3:11]: np.round(v, 3).tolist() for k, v in real.items()})
if tag == 'g2':
    grid = [dict(gov_head=gh, D_med=D, ffr=(0, 300), ffr_p=0.7, dc_p=0.0, sec_frac=sf, sec_T=(30, 90))
            for gh, D, sf in itertools.product([(300, 1500), (450, 2000)], [0.75, 1.0], [(0, 0), (0, 0.3), (0, 0.6)])]
else:
    grid = json.loads(sys.argv[2])
res = []
for i, kn in enumerate(grid):
    name = gpsim.derive('p0', f'{tag}_{i}', GB={k: tuple(v) if isinstance(v, list) else v for k, v in kn.items()})
    row = dict(i=i, **kn); errs = []
    for e in GB_DEV:
        s, L = T.twins(e, name, n=40)
        w = s['w']; ref = w[:, M.PRE - 5:M.PRE].mean(1)
        mn = np.median(w[:, M.PRE:].min(1) - ref); ms = np.median(w[:, M.PRE + 55:M.PRE + 61].mean(1) - ref)
        rn, rs = real[e]
        row[e[3:11]] = (round(float(mn / rn), 2), round(float(ms / rs), 2))
        errs.append(abs(np.log(mn / rn)))
        if e != 'GB_20190809T155233':
            errs.append(abs(np.log(ms / rs)))
    row['score'] = float(np.mean(errs)); res.append(row); print(row, flush=True)
json.dump(res, open(f'{ROOT}/logs/twin_{tag}.json', 'w'), indent=1)
