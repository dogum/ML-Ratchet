"""QuakeNet v2 on four real exam earthquakes (a fixed random draw: 2 shallow, 2 deep), first with their first 2 stations,
then with all their stations (10 each). None was used in training or choosing anything. Takes about 2 s."""
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from quakenet2 import QuakeNet2, gc_dist, KM
from datetime import datetime, timezone
net = QuakeNet2()
for f in sorted(glob.glob(os.path.join(HERE, 'examples', '*.npz'))):
    z = np.load(f); m = json.loads(str(z['meta'])); e = m['event']
    st = [dict(zne=x, lat=s['lat'], lon=s['lon'], t_start=s['t_start'], p_guess=s['p_guess']) for x, s in zip(z['zne'], m['stations'])]
    t = datetime.fromtimestamp(e['time'], timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{e['place']} ({t} UTC)  USGS: Mw {e['mw']:.1f} ({e['magType']}), depth {e['depth']:.0f} km, {e['lat']:.2f} {e['lon']:.2f}")
    for n in (2, len(st)):
        r = net.locate(st[:n]); miss = gc_dist(e['lat'], e['lon'], r['lat'], r['lon']) * KM; a = r['anchor']; am = gc_dist(e['lat'], e['lon'], a['lat'], a['lon']) * KM
        lo, hi = 10 ** (r['logz'] - r['logz_sig']), 10 ** (r['logz'] + r['logz_sig'])
        print(f"  {n:2d} stations  QuakeNet v2: {r['lat']:7.2f} {r['lon']:8.2f} (miss {miss:4.0f} km, 68% ellipse {r['err_ellipse']['major_km']:.0f} x {r['err_ellipse']['minor_km']:.0f} km)"
              f"  depth {r['depth']:3.0f} km ({lo:.0f}-{hi:.0f})  Mw {r['mw']:.2f} +-{r['mw_sig']:.2f}  origin {r['t0'] - e['time']:+5.1f} +-{r['t0_sig']:.1f} s")
        print(f"              v1.2 combined: miss {am:4.0f} km, depth {a['depth']:3.0f} km, Mw {a['mw']:.2f}, origin {a['t0'] - e['time']:+5.1f} s")
