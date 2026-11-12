"""Check the simulator against the physics it claims (noise-free simulations, no gaps unless stated).
Writes physics_check.json."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import sq, starsim as SS, classical as C
rng = np.random.default_rng(5); OUT = {}; n = 71000

def star(numax, **kw):
    s = SS.draw_star(np.random.default_rng(int(numax * 10)), (numax, numax * 1.0001)); s.update(v1=SS.VIS[1], amp=float(SS.yu_amp(s['numax'], s['dnu'], s['teff'], s['clump']))); s.update(kw); return s

# 1. chi^2 with 2 degrees of freedom: granulation only, power / limit spectrum over 20-250 uHz
s = star(100.0); r = []
for i in range(5):
    x = SS.synth(s, n, rng, osc=False); f, P = sq.psd(x); M = SS.model_psd(dict(s, amp=0.0), f); m = (f > 20) & (f < 250); r.append(P[m] / M[m])
r = np.concatenate(r); OUT['chi2_2dof'] = dict(mean=float(r.mean()), var=float(r.var()), frac_gt_4=float((r > 4).mean()), expected=dict(mean=1, var=1, frac_gt_4=float(np.exp(-4))))

# 2. Parseval: time-series variance vs integral of the limit spectrum (granulation + modes)
v = []; ff = np.arange(1, 4000001) * 1e6 / (2 * n * sq.DT) / 2; ff = ff[ff < sq.NYQ]
for nm in (20.0, 60.0, 180.0):
    s = star(nm); x = np.concatenate([SS.synth(s, n, rng) for _ in range(3)]); Mi = np.trapezoid(SS.model_psd(s, ff), ff); v.append(dict(numax=nm, var_ts=float(x.var()), var_model=float(Mi), ratio=float(x.var() / Mi)))
OUT['parseval'] = v

# 3. Amplitude per radial mode recovered with Yu et al. 2018 eq. 1: A = sqrt(H_env dnu / c), c = 3.04
res = []
for nm in (15.0, 40.0, 100.0, 200.0):
    s = star(nm); H = []
    for i in range(6):
        x = SS.synth(s, n, rng); f, P = sq.psd(x); B = SS.model_psd(dict(s, amp=0.0), f); w = np.abs(f - nm) < s['dnu']; H.append(np.mean((P - B)[w]))
    eta = np.sinc(nm / (2 * sq.NYQ)) ** 2; A = np.sqrt(np.mean(H) / eta * s['dnu'] / 3.04)
    res.append(dict(numax=nm, A_in=s['amp'], A_recovered=float(A), ratio=float(A / s['amp']), expected_ratio=float(np.sqrt(sum(SS.VIS) / 3.04)),
                    H_env=float(np.mean(H)), H_mosser2012=float(2.03e7 * nm ** -2.38), B_at_numax=float(np.mean(B[w])), B_mosser2012=float(6.37e6 * nm ** -2.41)))
OUT['amplitude_and_heights'] = res

# 4. Classical pipeline on noise-free 4-year simulations: does it recover the numax and dnu we put in?
E = []
for nm in np.exp(np.linspace(np.log(8), np.log(240), 24)):
    s = SS.draw_star(rng, (nm, nm * 1.0001)); x = SS.synth(s, n, rng) + rng.normal(0, 30, n); F1, F2, sc, (f, P), T = sq.prepare(x); c = C.measure(f, P, F1, F2)
    E.append((np.log(c['numax'] / s['numax']), np.log(c['dnu'] / s['dnu'])))
E = np.array(E); OUT['classical_on_sims'] = dict(n=len(E), numax_med_abs_lnerr=float(np.median(np.abs(E[:, 0]))), numax_bias=float(np.median(E[:, 0])), dnu_med_abs_lnerr=float(np.median(np.abs(E[:, 1]))), dnu_bias=float(np.median(E[:, 1])))

# 5. Mixed modes: number of l=1 modes per order vs the asymptotic expectation N = dnu / (nu^2 DPi1)
s = star(60.0, mixed=True, clump=False, dpi=75.0, q=0.14); L = SS.mixed(s['numax'], 1.0, 0.1, s['dnu'], 75.0, 0.14)
OUT['mixed_modes'] = dict(numax=60.0, dnu=s['dnu'], dpi=75.0, n_modes=len(L), expected=float(s['dnu'] / (60.0 ** 2 * 75e-6)), min_zeta_power_frac=float(max(p for _, p, _ in L)))
json.dump(OUT, open(_P.WORK + '/physics_check.json', 'w'), indent=1); print(json.dumps(OUT, indent=1))
