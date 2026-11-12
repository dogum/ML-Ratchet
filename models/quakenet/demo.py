"""QuakeNet on four real exam records (one station each): a large shallow quake, a small shallow one, an intermediate-depth
one and a deep one. None was used in training or choosing."""
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from quakenet import QuakeNet
net = QuakeNet()
for f in sorted(glob.glob(os.path.join(HERE, 'examples', '*.npz'))):
    z = np.load(f); m = json.loads(str(z['meta'])); r = net.read(z['zne'], m['p_guess'])
    lo, hi = 10 ** (r['logz'] - r['logz_sig']), 10 ** (r['logz'] + r['logz_sig']); sb = 57.3 / np.sqrt(max(r['kappa'], 1e-3))
    print(f"\n{m['place']} ({m['time']} UTC) at {m['station']}")
    print(f"  QuakeNet  Mw {r['mw']:.2f} +-{r['mw_sig']:.2f}   distance {r['dist']:.1f} +-{r['dist_sig']:.1f} deg   depth {r['depth']:.0f} km ({lo:.0f}-{hi:.0f})   from {r['baz']:.0f} +-{sb:.0f} deg")
    print(f"  USGS      Mw {m['usgs_mw']:.1f} ({m['usgs_magType']})   distance {m['dist_deg']:.1f} deg            depth {m['usgs_depth_km']:.0f} km            from {m['baz_deg']:.0f} deg")
