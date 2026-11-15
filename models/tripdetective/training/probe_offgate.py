"""Probe (15 Nov 2026): the 'engine stopped while moving' signal per trip window (rpm < 100 at v > 5 m/s, valid
seconds, rpm logged): its distribution for conventional and hybrid trips in a simulated bank, and on the dev cars.
usage: probe_offgate.py <bank>"""
from common import *
import simbank, realbank
import realdata as rd

def off_counts(r, f):
    """(engine-off moving seconds, moving seconds) of one window; (0, 0) without rpm"""
    if f[2] < .5: return 0, 0
    mv = (r[0] > 5) & (r[10] > .5)
    return int((mv & (r[6] < 0.1)).sum()), int(mv.sum())

b = simbank.load_bank(sys.argv[1]); o = b['offs']; meta = {m['vid']: m for m in b['meta']}
per = {}
for j in range(len(b['veh'])):
    r = b['raw'][o[j]:o[j + 1]].T.astype(np.float32); a = meta[b['veh'][j]]['arch']
    k, n = off_counts(r, b['flags'][j])
    if n >= 30: per.setdefault(a, []).append(k / n)
for a in ARCHS:
    if a in per:
        x = np.array(per[a]); print(f'sim {a:16s} trips {len(x):5d} off-share max {x.max():.3f} p99.9 {np.percentile(x, 99.9):.3f} p1 {np.percentile(x, 1):.3f} share > 0.03: {np.mean(x > .03):.3f}')
rb = realbank.load('dev'); vt = rd.ved_vehicles(); o = rb['offs']
for c in dict.fromkeys(rb['vid'].tolist()):
    js = np.where(rb['vid'] == c)[0]; ks = []; K = N = 0
    for j in js:
        k, n = off_counts(rb['raw'][o[j]:o[j + 1]].T.astype(np.float32), rb['flags'][j]); K += k; N += n
        if n >= 30: ks.append(k / n)
    if N: print(f'dev {c:8s} {vt.loc[c, "ved_type"]:4s} car off-share {K / N:.3f}  windows: max {max(ks):.3f} share > 0.03 {np.mean(np.array(ks) > .03):.2f} (n {len(ks)})')
