"""Scaling-relation checks (added to physics_check.json under 'scaling'):
1. The Sun: our relations with solar reference values (3090 uHz, 135.1 uHz, 5777 K; Huber et al. 2011 / Yu et al. 2018)
   against the original Kjeldsen & Bedding 1995 constants (3050 uHz, 134.9 uHz) and the measured solar values
   (numax 3090 +- 30 uHz, Huber et al. 2009; dnu 135.1 +- 0.1 uHz, Huber et al. 2011).
2. An independent implementation check: recompute Yu et al. 2018's 'no correction' masses and radii of all ~16,000
   stars from their own numax, dnu and Teff with our formulas.
3. The simulator's draw: dnu drawn from M, Teff, numax should follow the observed dnu-numax relation
   (Yu et al. 2018: dnu = 0.267 numax^0.764 on the RGB; Mosser 2012: 0.276 numax^0.751)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import catalog, starsim as SS
NS, DS, TS = 3090.0, 135.1, 5777.0
def MR(nm, dn, t): a, b, c = nm / NS, dn / DS, t / TS; return a ** 3 * b ** -4 * c ** 1.5, a * b ** -2 * c ** 0.5
Y = catalog.read_yu(); v = [r for r in Y.values() if np.isfinite(r.get('M', np.nan)) and np.isfinite(r.get('teff', np.nan))]
M, R = MR(np.array([r['numax'] for r in v]), np.array([r['dnu'] for r in v]), np.array([r['teff'] for r in v]))
yM, yR = np.array([r['M'] for r in v]), np.array([r['R'] for r in v])
rng = np.random.default_rng(5); S = [SS.draw_star(rng) for _ in range(4000)]; nm = np.array([s['numax'] for s in S]); dn = np.array([s['dnu'] for s in S])
rg = np.array([not s['clump'] for s in S]); p = np.polyfit(np.log(nm[rg]), np.log(dn[rg]), 1)
yn = np.array([r['numax'] for r in v]); yd = np.array([r['dnu'] for r in v]); yp = np.polyfit(np.log(yn), np.log(yd), 1)
OUT = dict(sun=dict(numax_ours=NS, dnu_ours=DS, numax_KB95=3050.0, dnu_KB95=134.9, numax_diff_pct=100 * (NS / 3050 - 1), dnu_diff_pct=100 * (DS / 134.9 - 1),
                    note='numax from M=R=1, Teff=5777 K equals the measured solar 3090 +- 30 uHz and dnu 135.1 +- 0.1 uHz by construction; KB95 constants differ by 1.3% and 0.15%'),
           yu_recomputed=dict(n=len(v), M_ratio_median=float(np.median(M / yM)), M_ratio_mad=float(1.4826 * np.median(np.abs(M / yM - np.median(M / yM)))),
                              R_ratio_median=float(np.median(R / yR)), R_ratio_mad=float(1.4826 * np.median(np.abs(R / yR - np.median(R / yR))))),
           sim_dnu_numax=dict(sim_slope=float(p[0]), sim_coef=float(np.exp(p[1])), yu_all_slope=float(yp[0]), yu_all_coef=float(np.exp(yp[1])), mosser2012=[0.276, 0.751]))
J = json.load(open(_P.WORK + '/physics_check.json')); J['scaling'] = OUT; json.dump(J, open(_P.WORK + '/physics_check.json', 'w'), indent=1); print(json.dumps(OUT, indent=1))
