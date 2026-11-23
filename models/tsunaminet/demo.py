"""TsunamiNet demo: real exam tsunamis (a fixed seeded draw within strata, not picked by result).
For each: the source posterior from the first DART records, next to USGS, and the amplitude forecast at held-out
DARTs next to what they recorded. Runs in seconds with numpy + scipy."""
import os, json, glob, time, numpy as np
import tsunaminet as T
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
net = T.TsunamiNet(); gf = T.GreensFunctions(); gf.cal = net.cal
wf = net.cal.get('widen', {}).get('amp', 1.0)
for fn in sorted(glob.glob(os.path.join(HERE, 'examples', '*.npz'))):
    z = np.load(fn, allow_pickle=False); meta = json.loads(str(z['meta']))
    recs = {s: (z['eta'][i], z['mask'][i]) for i, s in enumerate(meta['sites'])}
    ev = gf.event(meta['lon'], meta['lat'], recs)
    print(f"\n{meta['name']}  (USGS W-phase Mw {meta['mw']:.2f}; {'interface thrust' if meta['interface'] else 'NOT an interface thrust'})")
    if not ev['inputs']:
        print('  no usable DART within 290 min'); continue
    post = net.posterior(ev)
    print(f"  input DARTs: {', '.join(s.split('_')[0] for s in ev['inputs'])} (arrivals {ev['input_ta']} min)")
    print(f"  Mw  {post['mw']:.2f}  68% [{post['mw68'][0]:.2f}, {post['mw68'][1]:.2f}]   USGS {meta['mw']:.2f}")
    if meta.get('ffm_s100') is not None:
        print(f"  centroid along trench {post['centroid_km']:+.0f} km  68% [{post['centroid_km68'][0]:+.0f}, {post['centroid_km68'][1]:+.0f}]   USGS finite fault {100 * meta['ffm_s100']:+.0f} km")
    for t in meta['targets']:
        i = meta['sites'].index(t['site']); m = z['mask'][i][t['a']:t['b'] + 1]
        f = gf.forecast(ev, post, t['site'], t['a'], t['b'], m, widen=wf)
        inside = f['lo68'] <= t['obs'] <= f['hi68']
        print(f"  forecast {t['site'].split('_')[0]}: {100 * f['median']:6.1f} cm [{100 * f['lo68']:.1f}, {100 * f['hi68']:.1f}]  observed {100 * t['obs']:6.1f} cm  {'inside' if inside else 'outside'} 68%")
print(f'\n{time.time() - t0:.1f} s')
