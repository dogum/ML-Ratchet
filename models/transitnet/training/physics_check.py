"""Check TransitNet's light-curve physics: transits against batman (Kreidberg 2015; exact Mandel & Agol 2002 quadratic limb
darkening), and a star eclipsed by a bigger star against brute-force pixel integration."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, batman
sys.path.insert(0, _P.SRC); import transitsim as TS
t = np.linspace(-0.15, 0.15, 3001); rows = []
for k, b, u1, u2, aR, e, w in [(0.1, 0.3, 0.4, 0.25, 12, 0, 90), (0.02, 0.0, 0.3, 0.3, 20, 0, 90), (0.15, 0.95, 0.5, 0.1, 8, 0, 90), (0.25, 1.1, 0.2, 0.4, 6, 0, 90),
                               (0.08, 0.5, 0.45, 0.2, 15, 0.3, 40), (0.05, 0.7, 0.1, 0.0, 30, 0.6, 200), (0.01, 0.4, 0.35, 0.25, 10, 0, 90)]:
    inc = np.arccos(b / aR); pm = batman.TransitParams(); pm.t0 = 0; pm.per = 3.0; pm.rp = k; pm.a = aR; pm.inc = np.degrees(inc); pm.ecc = e; pm.w = w; pm.u = [u1, u2]; pm.limb_dark = 'quadratic'
    fb = batman.TransitModel(pm, t).light_curve(pm); o = TS.planet_flux(t, 0, 3.0, k, aR, inc, u1, u2, e, np.radians(w), supersample=1)
    rows.append(dict(k=k, b=b, e=e, depth_ppm=float(1e6 * (1 - fb.min())), max_err_ppm=float(1e6 * np.abs(o - fb).max())))
big = []
n = 4001; g = np.linspace(-1, 1, n); X, Y = np.meshgrid(g, g); R2 = X ** 2 + Y ** 2; mu = np.sqrt(np.clip(1 - R2, 0, 1)); I = (1 - 0.4 * (1 - mu) - 0.2 * (1 - mu) ** 2) * (R2 < 1)
for z, p in [(0.0, 3.0), (2.5, 3.0), (3.4, 3.0), (1.2, 1.5), (0.5, 1.0)]:
    br = float((I * (((X - z) ** 2 + Y ** 2) < p * p)).sum() / I.sum()); big.append(dict(z=z, p=p, ours=float(TS.occult(z, p, 0.4, 0.2)), brute=br))
json.dump(dict(transit=rows, big_occulter=big), open(_P.WORK + '/physics_check.json', 'w'), indent=1)
for r in rows: print(r)
for r in big: print(r)
