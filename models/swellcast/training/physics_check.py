"""Independent checks of the simulator (writes physics_check.json):
 1. deep-water group speed g/(4 pi f) against the full dispersion relation omega^2 = g k tanh(k h) (numerical d omega/dk, h = 4000 m);
 2. ridge slope and intercept from a near-point, short storm against df/dt = g/(4 pi D) and t0;
 3. energy conservation: wave energy crossing the full circle at distance D equals the energy the storm emitted (no dissipation);
 4. swell dissipation e-folding distances against the ranges reported by Ardhuin et al. (2009) and Collard et al. (2009);
 5. a worked number: a ridge slope of 0.01 Hz/day means D = g / (4 pi slope) = 6,745 km."""
import _paths as _P  # portable paths (see _paths.py)
import json, numpy as np, scipy.optimize as so
import swellsim as SS, geo
out = {}
# 1
h = 4000.0; T = np.array([8, 12, 16, 20, 25.0]); w = 2 * np.pi / T
k = np.array([so.brentq(lambda k: G_ * k * np.tanh(k * h) - wi ** 2, 1e-6, 10) for wi, G_ in zip(w, [9.81] * 5)])
dk = 1e-7; cgn = (np.sqrt(9.81 * (k + dk) * np.tanh((k + dk) * h)) - np.sqrt(9.81 * (k - dk) * np.tanh((k - dk) * h))) / (2 * dk)
out['group_speed'] = dict(T_s=T.tolist(), numerical=cgn.round(4).tolist(), formula=(9.81 * T / (4 * np.pi)).round(4).tolist(),
                          max_rel_err=float(np.max(np.abs(cgn / (9.81 * T / (4 * np.pi)) - 1))))
# 2
res = []
for Dkm in (1500, 3000, 6000, 9000):
    blat, blon = 21.0, -158.0; la, lo = geo.dest(blat, blon, 150.0, Dkm * 1e3)
    p = dict(hs=10, tp=20.0, sig1=2.0, sig2=2.0, speed=0.0, heading=0.0, R=5e3, gam=1.0, w_iso=1.0, s_beam=1, fe=0.0, r1_0=1.0)
    D = float(geo.dist(la, lo, blat, blon)); tmax = D / SS.cg(0.03) / 3600 + 400
    t = np.arange(0, tmax, 1.0); E, _, _ = SS.simulate(blat, blon, la, lo, 0.0, p, t, rng=None, noise=False)
    f_at = []; tt = []
    for i in range(len(t)):
        if E[i].max() > 1e-3 * E.max() and 0.04 <= SS.FGRID[np.argmax(E[i])] <= 0.15: f_at.append(np.sum(E[i] * SS.FGRID) / E[i].sum()); tt.append(t[i] + 0.5)
    a, b = np.polyfit(np.array(tt) * 3600, f_at, 1)
    res.append(dict(D_km=round(D / 1e3, 1), slope_sim=a, slope_theory=9.81 / (4 * np.pi * D), rel_err=a / (9.81 / (4 * np.pi * D)) - 1, t0_from_intercept_h=-b / a / 3600))
out['ridge'] = res
# 3
blat, blon = 0.0, 0.0; la, lo = geo.dest(blat, blon, 90.0, 4000e3); D = float(geo.dist(la, lo, blat, blon))
p = dict(hs=8, tp=14.0, sig1=12.0, sig2=12.0, speed=0.0, heading=0.0, R=5e3, gam=2.0, w_iso=1.0, s_beam=1, fe=0.0, r1_0=1.0)
t = np.arange(0, 600, 1.0); E, _, _ = SS.simulate(blat, blon, la, lo, 0.0, p, t, rng=None, noise=False)
Eb = np.sum(E * SS.cg(SS.FGRID)[None] * SS.DF) * 3600 * 2 * np.pi * (geo.R_E * np.sin(D / geo.R_E) + 2 * p['R'])
ts = np.arange(-3.5 * 12, 3.5 * 12 + 1e-9, 0.5); A = np.exp(-ts ** 2 / (2 * 144.0)); A = A[A > 0.02]
Es = SS.jonswap(SS.FFINE, p['hs'] * np.sqrt(A), np.maximum(p['tp'] * A ** 0.25, 4.0), p['gam'])
m = (SS.FFINE >= SS.FGRID[0] - SS.DF / 2) & (SS.FFINE <= SS.FGRID[-1] + SS.DF / 2)
Em = np.sum(Es[:, m] * SS.cg(SS.FFINE[m])[None]) * (SS.DF / SS.NSUB) * 0.5 * 3600 * 2 * p['R']
out['energy_conservation'] = dict(received_over_emitted=float(Eb / Em))
# 4
efold = []
for fe_, hs, tp in [(fe_, hs, tp) for fe_ in (0.0042, 0.019) for hs, tp in ((4.0, 15.0), (2.0, 15.0), (6.0, 13.0), (1.0, 18.0))]:
    # swell of height hs at 1000 km (equivalently, source scaled so) : e-folding distance of energy at the peak frequency
    fp = 1 / tp; L = 1e6; m0 = hs ** 2 / 16
    a1 = SS.attenuation(np.array([fp]), 3000e3, m0 / (L / (2 * np.pi * geo.R_E * np.sin(1000e3 / geo.R_E))), tp, L, 1 / (2 * np.pi), fe_)
    a2 = SS.attenuation(np.array([fp]), 1000e3, m0 / (L / (2 * np.pi * geo.R_E * np.sin(1000e3 / geo.R_E))), tp, L, 1 / (2 * np.pi), fe_)
    efold.append(dict(fe=fe_, hs_at_1000km=hs, tp=tp, efold_km_1000_to_3000=float(2000 / -np.log(a1[0] / a2[0])) if a1[0] < a2[0] else None))
out['dissipation'] = dict(prior_fe='log-uniform 0.002-0.02', note='Ardhuin et al. 2009 (GRL 36 L06607) report energy e-folding scales from ~3,300 km for steep swells to over 20,000 km for low swells; Collard et al. 2009 similar.', cases=efold)
out['munk_example'] = dict(slope_Hz_per_day=0.01, D_km=9.81 / (4 * np.pi * 0.01 / 86400) / 1e3)
json.dump(out, open(_P.WORK + '/physics_check.json', 'w'), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
