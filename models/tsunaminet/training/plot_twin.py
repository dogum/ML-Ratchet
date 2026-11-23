"""Plot dev DART records against their finite-fault twins (diagnostic figure for the twin test)."""
import sys, json, numpy as np, tsim, realdata, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
tag = sys.argv[1]; delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
geo = tsim.Geometry(); gf = tsim.GFBank(); E = {e['id']: e for e in json.load(open('../data/events.json'))}
fig, ax = plt.subplots(6, 2, figsize=(14, 16)); ax = ax.ravel(); k = 0
for eid in (3016, 6061, 5755, 5636, 3019):
    ev = E[eid]; p = realdata.prepare(ev, gf, geo); u, s = realdata.ffm_units(ev, geo)
    for st in sorted(p['all'], key=lambda x: x['ta']):
        if st['mask'][max(st['ta'] - 10, 0): st['ta'] + 120].mean() < 0.6 or k >= 12: continue
        tw = gf.wave(st['site'], u, s); tw = np.interp(np.arange(1441) - st['ta'] * delay, np.arange(1441), tw, left=0); t = np.arange(1441); m = st['mask']; ta = st['ta']
        a = ax[k]; a.plot(t[m], st['eta'][m] * 100, '.', ms=2, color='k'); a.plot(t, tw * 100, 'b'); a.set_xlim(max(0, ta - 60), ta + 240); a.axvline(ta, color='r')
        a.set_title(f"{ev['name'][:25]} {st['site']}  ta={ta}", fontsize=9); lim = max(np.abs(tw).max() * 100, 1.0) * 2.5; a.set_ylim(-lim, lim); k += 1
plt.tight_layout(); plt.savefig(f'../tmp/twin_{tag}.png', dpi=65)
