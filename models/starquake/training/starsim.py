"""StarQuake simulator: a red giant's brightness fluctuations on the real Kepler long-cadence grid.

Physics (frequencies in uHz, amplitudes in ppm, power density single-sided in ppm^2/uHz):
- Granulation: two super-Lorentzians (exponent 4) of Kallinger et al. 2014 (A&A 570, A41, Table 2):
    a = 3710 numax^-0.613 M^-0.26 ppm rms each, b1 = 0.317 numax^0.970, b2 = 0.948 numax^0.992,
    P_k(nu) = xi a^2/b_k / (1 + (nu/b_k)^4), xi = 2 sqrt(2)/pi; log-normal scatter 14%, 10%, 9% as in their fits.
- Oscillation envelope: Gaussian in nu around numax with FWHM 0.66 numax^0.88 (Mosser et al. 2012, A&A 537, A30);
  rms amplitude per radial mode from Yu et al. 2018 (ApJS 236, 42) eq. 8 / Table 3 (RGB and clump coefficients).
- Mode frequencies: the universal red-giant pattern (Mosser et al. 2011, A&A 525, L9, Table 1):
    nu_{n,l} = (n + l/2 + eps - d0l + alpha/2 (n - numax/dnu)^2) dnu, eps = 0.634 + 0.546 log dnu,
    d01 = -0.056 - 0.002 log dnu, d02 = 0.131 - 0.033 log dnu, d03 = 0.280, alpha = 0.008.
  Visibilities V^2 = 1, 1.54, 0.63, 0.071 for l = 0..3 (Mosser et al. 2012).
- Optional l = 1 mixed modes from the asymptotic coupling tan(theta_p) = q tan(theta_g) (Mosser et al. 2012b, 2015):
  theta_p = pi (nu - nu_p)/dnu, theta_g = pi / (nu DPi1); inertia fraction zeta sets each mode's power (1 - zeta) and
  linewidth Gamma (1 - zeta). RGB: DPi1 50-100 s, q 0.10-0.18; clump: DPi1 240-340 s, q 0.2-0.4.
- Lorentzian profiles (radial linewidth 0.05-0.3 uHz, Vrard et al. 2018 range), integrated over each frequency bin so
  that unresolved modes are handled; random complex Gaussian amplitudes per bin give chi^2 with 2 dof statistics.
- The 29.4-min integration attenuates the stellar signal by sinc^2(pi nu / (2 nu_Nyq)).
- Synthesis: inverse FFT on a grid twice the observed span (so mode frequencies fall between analysis bins), sampled
  on the star's real cadence grid; the real light curve's gaps are kept by adding it to a real quiet-star light curve.
"""
import numpy as np
import sq
NUMAX_SUN, DNU_SUN, TEFF_SUN, A_SUN = 3090.0, 135.1, 5777.0, 3.6
VIS = (1.0, 1.54, 0.63, 0.071); XI = 2 * np.sqrt(2) / np.pi

def harvey(nu, a, b): return XI * a * a / b / (1 + (nu / b) ** 4)

def yu_amp(numax, dnu, teff, clump):
    """Yu et al. 2018 eq. 8, Table 3: rms amplitude per radial mode (ppm)"""
    al, be, ga, de = (0.36, -1.99, 1.36, -1.23) if clump else (2.02, -2.85, 2.95, -1.82)
    return A_SUN * al * (numax / NUMAX_SUN) ** be * (dnu / DNU_SUN) ** ga * (teff / TEFF_SUN) ** de

def draw_star(rng, numax_range=(5.0, 275.0), calib=None):
    """Random red giant: global parameters drawn from broad priors; dnu follows from mass, teff and numax through the
    scaling relations (plus a 2% non-scaling term)."""
    numax = float(np.exp(rng.uniform(*np.log(numax_range))))
    M = float(np.clip(np.exp(rng.normal(np.log(1.25), 0.3)), 0.7, 3.0))
    teff = float(np.clip(4750 + 350 * np.log10(numax / 40) + 140 * rng.normal() + 120 * (M - 1.25), 3900, 5400))
    pcl = 0.55 if 22 < numax < 55 else (0.25 if 55 <= numax < 110 and M > 1.6 else 0.03); clump = bool(rng.random() < pcl)
    R = np.sqrt(M / (numax / NUMAX_SUN) / np.sqrt(teff / TEFF_SUN)); dnu = DNU_SUN * np.sqrt(M) * R ** -1.5 * np.exp(rng.normal(0, 0.02))
    if clump: dnu *= np.exp(rng.normal(-0.02, 0.01))      # clump stars sit a few percent below the RGB dnu-numax relation
    s = dict(numax=numax, dnu=float(dnu), M=M, teff=teff, clump=clump)
    s['amp'] = float(yu_amp(numax, dnu, teff, clump) * np.exp(rng.normal(0, 0.10)))
    if rng.random() < 0.12: s['amp'] *= float(rng.uniform(0.25, 0.8))          # dilution by a companion / contaminating star
    s['fwhm'] = float(0.66 * numax ** 0.88 * np.exp(rng.normal(0, 0.10)))
    s['a_gran'] = float(3710 * numax ** -0.613 * M ** -0.26 * np.exp(rng.normal(0, 0.14)))
    s['b1'] = float(0.317 * numax ** 0.970 * np.exp(rng.normal(0, 0.10))); s['b2'] = float(0.948 * numax ** 0.992 * np.exp(rng.normal(0, 0.09)))
    s['gamma0'] = float(np.exp(rng.normal(np.log(0.12), 0.35)) * (teff / 4750) ** 3)
    s['eps_off'] = float(rng.normal(-0.04 if clump else 0.0, 0.03)); s['d02_off'] = float(rng.normal(0, 0.01))
    s['mixed'] = bool(rng.random() < 0.7); s['dpi'] = float(rng.uniform(240, 340) if clump else rng.uniform(50, 100)); s['q'] = float(rng.uniform(0.2, 0.4) if clump else rng.uniform(0.10, 0.18))
    s['v1'] = float(VIS[1] * (rng.uniform(0.05, 0.5) if (not clump and rng.random() < 0.08) else 1.0))   # dipole-suppressed stars
    if calib: calib(s, rng)
    return s

def mode_list(s):
    """-> list of (nu0, power ppm^2, linewidth uHz)"""
    numax, dnu = s['numax'], s['dnu']; lg = np.log10(dnu)
    eps = 0.634 + 0.546 * lg + s['eps_off']; d0 = (0.0, -0.056 - 0.002 * lg, 0.131 - 0.033 * lg + s['d02_off'], 0.280); alpha = 0.008
    sig = s['fwhm'] / (2 * np.sqrt(2 * np.log(2))); nmax = numax / dnu - eps; g0 = s['gamma0']; out = []
    env = lambda nu: s['amp'] ** 2 * np.exp(-(nu - numax) ** 2 / (2 * sig ** 2))
    for n in range(max(1, int(nmax) - 8), int(nmax) + 10):
        for l in (0, 1, 2, 3):
            nu = (n + l / 2 + eps - d0[l] + alpha / 2 * (n - nmax) ** 2) * dnu
            if nu <= 0.5 or nu > sq.NYQ * 1.02 or abs(nu - numax) > 4.5 * sig: continue
            vis = s['v1'] if l == 1 else VIS[l]; pw = env(nu) * vis
            if l == 1 and s['mixed']: out += mixed(nu, pw, g0, dnu, s['dpi'], s['q'])
            else: out.append((nu, pw, g0))
    return out

def mixed(nup, pw, g0, dnu, dpi, q):
    """l=1 mixed modes around the nominal p mode nu_p (asymptotic coupling); powers (1-zeta) normalised to the p-mode power."""
    dP = dpi * 1e-6                                   # s -> Ms, so 1/(nu[uHz] * dP) is in radians/pi units
    nu = np.linspace(nup - dnu / 2, nup + dnu / 2, 4000)
    tp = np.pi * (nu - nup) / dnu; tg = np.pi / (nu * dP)
    h = np.sin(tp) * np.cos(tg) - q * np.cos(tp) * np.sin(tg); i = np.flatnonzero(np.sign(h[:-1]) * np.sign(h[1:]) < 0)
    if len(i) == 0: return [(nup, pw, g0)]
    r = nu[i] - h[i] * (nu[i + 1] - nu[i]) / (h[i + 1] - h[i])
    tpr = np.pi * (r - nup) / dnu; tgr = np.pi / (r * dP); N = dnu / (r ** 2 * dP)
    ratio = np.cos(tgr) ** 2 / np.maximum(np.cos(tpr) ** 2, 1e-9); zeta = 1 / (1 + ratio / (q * N))
    w = (1 - zeta); keep = np.abs(np.tan(tpr) - q * np.tan(tgr)) < 0.2 * (1 + np.abs(np.tan(tpr)))   # drop spurious roots at poles
    r, w = r[keep], w[keep]
    if len(r) == 0 or w.sum() <= 0: return [(nup, pw, g0)]
    w = w / w.sum(); return [(float(a), float(pw * b), float(max(g0 * (1 - z), 1e-4))) for a, b, z in zip(r, w, zeta[keep])]

def model_psd(s, f):
    """Smooth limit spectrum on grid f (uHz): granulation and the oscillation modes' expected power density (for checks)."""
    eta = np.sinc(f / (2 * sq.NYQ)) ** 2; P = harvey(f, s['a_gran'], s['b1']) + harvey(f, s['a_gran'], s['b2'])
    for nu0, pw, g in mode_list(s): P = P + pw * (2 / (np.pi * g)) / (1 + (2 * (f - nu0) / g) ** 2)
    return P * eta

def synth(s, n, rng, osc=True):
    """Time series (ppm) of length n cadences: granulation + modes, generated on a 2n grid by inverse FFT."""
    ng = 2 * n; df = 1e6 / (ng * sq.DT); k = np.arange(ng // 2 + 1); f = k * df
    lo, hi = f - df / 2, f + df / 2; eta = np.sinc(f / (2 * sq.NYQ)) ** 2
    pw = (harvey(f, s['a_gran'], s['b1']) + harvey(f, s['a_gran'], s['b2'])) * df
    if osc:
        for nu0, p, g in mode_list(s):
            w = max(40 * g, 30 * df); j0, j1 = max(int((nu0 - w) / df), 1), min(int((nu0 + w) / df) + 2, len(f))
            pw[j0:j1] += p / np.pi * (np.arctan(2 * (hi[j0:j1] - nu0) / g) - np.arctan(2 * (lo[j0:j1] - nu0) / g))
    pw *= eta; pw[0] = 0
    X = (ng / 2) * np.sqrt(pw) * (rng.normal(size=len(f)) - 1j * rng.normal(size=len(f)))
    return np.fft.irfft(X, ng)[:n]
