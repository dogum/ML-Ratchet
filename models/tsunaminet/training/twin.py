"""Twin test on dev events: each dev event's twin = USGS finite-fault slip on the unit sources, propagated with the
same Green's functions, read at every DART that recorded it. Measures (a) the arrival lag of real vs twin by
cross-correlation in [ta-10, ta+120] min, (b) the ratio of real to twin max amplitude. Writes data/twin_<tag>.json."""
import os, sys, json, numpy as np
from _paths import DATA
import tsim, realdata
tag = sys.argv[1] if len(sys.argv) > 1 else 'p0'
geo = tsim.Geometry(); gf = tsim.GFBank()
EV = [e for e in json.load(open(os.path.join(DATA, 'events.json'))) if e['split'] == 'dev']
rows = []
for ev in EV:
    prep = realdata.prepare(ev, gf, geo)
    units, slips = realdata.ffm_units(ev, geo)
    for st in prep['all']:
        ta = st['ta']; a, b = max(ta - 10, 0), min(ta + 120, 1440)
        m = st['mask'][a:b + 1]
        if m.mean() < 0.6: continue
        tw = gf.wave(st['site'], units, slips)
        obs = st['eta'][a:b + 1]; pv = st['eta'][:max(ta - 10, 0)][st['mask'][:max(ta - 10, 0)]]
        if len(pv) < 8: continue
        sig = 1.4826 * np.median(np.abs(pv - np.median(pv))) + 1e-4
        amp_o = np.abs(obs[m]).max()
        if amp_o < 5 * sig: continue
        best = (-2, 0)
        for lag in range(-6, 16):
            sim = np.interp(np.arange(a, b + 1) - lag, np.arange(len(tw)), tw, left=0.0)
            if m.sum() < 10 or np.std(sim[m]) == 0: continue
            r = np.corrcoef(obs[m], sim[m])[0, 1]
            if r > best[0]: best = (r, lag)
        amp_t = np.abs(tw[a:b + 1][m]).max()
        rows.append(dict(event=ev['id'], name=ev['name'], interface=ev['interface'], site=st['site'], ta=ta, lag=best[1], r=float(best[0]),
                         amp_obs=float(amp_o), amp_twin=float(amp_t), log10_ratio=float(np.log10(amp_o / max(amp_t, 1e-6))), input=st['site'] in prep['inputs']))
        print(rows[-1]['name'][:30], st['site'], 'ta', ta, 'lag', best[1], 'r %.2f' % best[0], 'obs %.3f twin %.3f' % (amp_o, amp_t), flush=True)
good = [r for r in rows if r['r'] > 0.6]
lagf = [r['lag'] / r['ta'] for r in good if r['ta'] > 60]
summ = dict(n=len(rows), n_good=len(good), lag_median_min=float(np.median([r['lag'] for r in good])) if good else None,
            lag_frac_median=float(np.median(lagf)) if lagf else None, lag_frac_iqr=[float(x) for x in np.percentile(lagf, [25, 75])] if lagf else None,
            log10_amp_ratio_median=float(np.median([r['log10_ratio'] for r in rows])) if rows else None,
            log10_amp_ratio_sd=float(np.std([r['log10_ratio'] for r in rows])) if rows else None,
            log10_amp_ratio_median_interface=float(np.median([r['log10_ratio'] for r in rows if r['interface']])) if any(r['interface'] for r in rows) else None)
print(json.dumps(summ, indent=1))
json.dump(dict(summary=summ, rows=rows), open(os.path.join(DATA, f'twin_{tag}.json'), 'w'), indent=1)
