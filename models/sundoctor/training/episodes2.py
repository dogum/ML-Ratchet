"""Real fault episodes in system 2107, found in the sibling answer key; how the released model saw each one
(from its single-inverter view only)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, _P.SRC)
import realexam as RX, evaluate2 as E, realexam2 as RX2

def find(fr, days):
    k = np.round(6 * (1 - fr)); clean = np.isfinite(fr) & (np.abs(6 * (1 - fr) - k) < 0.25); K = np.where(clean, k, np.nan); eps = []
    for j in range(24):
        v = pd.Series(K[:, j]).ffill(limit=2).values
        for i in range(10, len(v) - 25):
            pre, post = v[i - 10:i], v[i:i + 10]
            if np.all(pre == 0) and np.all(post[:5] == 1):
                rep = np.where(v[i + 5:i + 60] == 0)[0]
                eps.append({'kind': 'string', 'inv': j, 'day': i, 'repair': int(i + 5 + rep[0]) if len(rep) else None})
            if np.all(pre == 0) and np.all(post[:3] == 6):
                rep = np.where(v[i + 3:i + 60] == 0)[0]
                eps.append({'kind': 'offline', 'inv': j, 'day': i, 'repair': int(i + 3 + rep[0]) if len(rep) else None})
    return eps

def main(ckpt, home=False):
    if ckpt.endswith('s3_polish.pt'):                       # the v1 release, seen through v2's inputs
        import ratchet2, evaluate as E1; net = ratchet2.V1Adapter(E1.load(ckpt)).eval()
    else: net = E.load(ckpt)
    days, P, G, T, fr, geo = RX.load(); eps = find(fr, days); cache = {}
    for e in eps:
        j = e['inv']
        if j not in cache:
            cache[j] = RX2.diagnose(net, P[:, :, j].astype(float), G, T, geo, home)
        o = cache[j]; d = e['day']
        if e['kind'] == 'string':
            e['onset_found'] = bool(np.nanmax(o['step_p'][d - 1:d + 2]) > 0.5)
            after = o['frac'][d + 1:d + 5]; e['count_after'] = float(np.nanmedian(np.round(6 * (1 - np.clip(after, 0, 1)))))
            if e['repair'] is not None: e['repair_found'] = bool(np.nanmax(o['step_p'][e['repair'] - 1:e['repair'] + 2]) > 0.5)
        else:
            e['outage_found'] = bool(np.nanmean(o['outage'][d:d + 3]) > 0.5)
        e['date'] = str(days[d].date()); e['repair_date'] = str(days[e['repair']].date()) if e['repair'] is not None else None
    s = [e for e in eps if e['kind'] == 'string']; off = [e for e in eps if e['kind'] == 'offline']
    summ = {'string_failures': len(s), 'onsets_found_pm1day': sum(e['onset_found'] for e in s), 'counted_one_dead_after': sum(e['count_after'] == 1 for e in s),
            'repairs': sum(e['repair'] is not None for e in s), 'repairs_found': sum(e.get('repair_found', False) for e in s),
            'offline_events': len(off), 'offline_found': sum(e['outage_found'] for e in off)}
    print(json.dumps(summ, indent=1)); json.dump({'summary': summ, 'episodes': eps}, open(ckpt.replace('.pt', '_episodes%s.json' % ('_home' if home else '')), 'w'), indent=1, default=str)
    return summ, eps, cache

if __name__ == '__main__':
    main(sys.argv[1], '--home' in sys.argv)
