"""StarQuake demo: three real Kepler red giants from the exam (a fixed random draw), read from the whole light curve
and from a 90-day piece. Runs in a few seconds with numpy and scipy."""
import os, json, time, numpy as np
from starquake import StarQuake
H = os.path.dirname(os.path.abspath(__file__)); sq = StarQuake(); truth = json.load(open(os.path.join(H, 'examples', 'truth.json')))
for f in sorted(os.listdir(os.path.join(H, 'examples'))):
    if not f.endswith('.npz'): continue
    kic = f[:-4]; z = np.load(os.path.join(H, 'examples', f)); t = truth[kic]; t0 = time.time()
    import pipeline as PL; y = PL.load_kepler(z)
    for name, seg in (('whole light curve', y), ('90 days', y[20000:20000 + int(90 * 86400 / PL.DT)])):
        r = sq.read_ppm(seg, teff=t['teff'], e_teff=t['e_teff'])
        print(f"KIC {kic} ({name}, {r['days']:.0f} d): numax {r['numax']:.2f} +- {r['numax_err']:.2f} uHz (published {t['numax']:.2f}), "
              f"dnu {r['dnu']:.3f} +- {r['dnu_err']:.3f} uHz (published {t['dnu']:.3f}), M {r['mass']:.2f} +- {r['mass_err']:.2f} (published {t['M']:.2f}), R {r['radius']:.2f} +- {r['radius_err']:.2f} (published {t['R']:.2f})")
    print(f"   {time.time() - t0:.1f} s")
