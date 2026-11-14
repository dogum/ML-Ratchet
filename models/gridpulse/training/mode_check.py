"""Which 1 s measurement model matches NESO's 1 s data? Statistic on the first post-onset seconds:
r1 = (first 1 s drop) / (largest 1 s drop in the first 5 s), and the share of the 2 s drop seen in the first second.
Compared between the 4 GB dev events and simulated GB events (val bank) under each mode, with real noise."""
from common import *
import bank as BK, dataset as DS, measure as M, evaluate as EV
dev, L = BK.load('val_p1'); s = (L.system == 'GB').values; dev, L = dev[s], L[s].reset_index(drop=True)
keep = DS.selected(dev, L); dev, L = dev[keep], L[keep].reset_index(drop=True)
noise = DS.noise_split('val')


def stats(w):
    d = np.diff(w[:, M.PRE:M.PRE + 6], axis=1)        # drops in seconds 1..5 after onset
    big = d.min(1)
    return d[:, 0] / big, d[:, 1] / big


real = np.array([EV.real_gb(e)['w'][0] for e in GB_DEV])
r1, r2 = stats(real)
print('real r1', np.round(r1, 2), 'r2', np.round(r2, 2))
res = {'real': dict(r1=r1.tolist(), r2=r2.tolist())}
for mode, tau in (('sample', 0), ('mean', 0), ('mean', 0.5), ('mean', 1.0)):
    rng = np.random.default_rng(5)
    meas, _, _ = DS.make(dev, L, rng, noise, mode, tau, Nfix=60)
    a, b = stats(meas['w'])
    big = meas['w'][:, M.PRE:].min(1) < -0.25        # comparable sizes to the dev events
    pct = [float((a[big] < x).mean()) for x in r1]
    print(mode, tau, 'r1 median', round(float(np.median(a[big])), 2), 'iqr', np.percentile(a[big], [25, 75]).round(2),
          'r2 median', round(float(np.median(b[big])), 2), 'real r1 percentiles', np.round(pct, 2), flush=True)
    res[f'{mode}{tau}'] = dict(r1_med=float(np.median(a[big])), r2_med=float(np.median(b[big])), real_pct=pct)
json.dump(res, open(f'{ROOT}/logs/mode_check.json', 'w'), indent=1)
