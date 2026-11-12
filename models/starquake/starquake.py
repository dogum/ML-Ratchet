"""StarQuake v1.0: numax and Delta-nu, with error bars, from a red giant's light curve. numpy + scipy only.

    from starquake import StarQuake
    sq = StarQuake()                                   # loads starquake_weights.npz next to this file
    r = sq.read_kepler('kplr*_llc.fits files' or arrays, teff=4800)
    r['numax'], r['numax_err'], r['dnu'], r['dnu_err'], r['mass'], r['radius']

Output: p_osc (probability that solar-like oscillations with numax 5-275 uHz are present; 'detected' if > 0.5),
numax and dnu with error bars, and mass and radius from the scaling relations if Teff is given.
Input: a Kepler long-cadence light curve (29.4-min cadence), 27 days to 4 years. Trained only on simulated red giants
(numax 5-275 uHz) injected into real Kepler light curves of quiet dwarf stars. See MODEL_CARD.md for limits."""
import os, json, numpy as np
import pipeline as PL

def gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def conv1d(x, w, b, stride, pad):
    """x (C, L), w (O, C, K)"""
    C, L = x.shape; O, _, K = w.shape; xp = np.pad(x, ((0, 0), (pad, pad))); Lo = (L + 2 * pad - K) // stride + 1
    idx = np.arange(Lo)[:, None] * stride + np.arange(K)[None, :]; cols = xp[:, idx]            # (C, Lo, K)
    return np.einsum('clk,ock->ol', cols, w, optimize=True) + b[:, None]

def conv2d(x, w, b, stride, pad):
    """x (C, H, W), w (O, C, KH, KW); stride (sh, sw)"""
    C, H, W = x.shape; O, _, KH, KW = w.shape; sh, sw = stride; xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - KH) // sh + 1; Wo = (W + 2 * pad - KW) // sw + 1
    ih = np.arange(Ho)[:, None] * sh + np.arange(KH)[None, :]; iw = np.arange(Wo)[:, None] * sw + np.arange(KW)[None, :]
    cols = xp[:, ih[:, None, :, None], iw[None, :, None, :]]                                    # (C, Ho, Wo, KH, KW)
    return np.einsum('chwij,ocij->ohw', cols, w, optimize=True) + b[:, None, None]

class Net:
    def __init__(self, path):
        z = np.load(path); self.W = {k: z[k].astype(np.float64) for k in z.files if k != 'meta'}; self.meta = json.loads(str(z['meta']))
    def forward(self, x1, x2, s):
        W = self.W; m = self.meta
        h = gelu(conv1d(((np.asarray(x1, np.float64) - 2.5) / 2)[None], W['s1_w'], W['s1_b'], 2, 3))
        for i in range(m['n1']):
            a = gelu(conv1d(h, W[f'r1_{i}_c1_w'], W[f'r1_{i}_c1_b'], 2, 3)); a = conv1d(a, W[f'r1_{i}_c2_w'], W[f'r1_{i}_c2_b'], 1, 3)
            h = gelu(a + conv1d(h, W[f'r1_{i}_sk_w'], W[f'r1_{i}_sk_b'], 2, 0))
        h1 = h.ravel()
        h = gelu(conv2d((np.asarray(x2, np.float64) * 4)[None], W['s2_w'], W['s2_b'], (1, 2), 1))
        for i in range(m['n2']):
            a = gelu(conv2d(h, W[f'r2_{i}_c1_w'], W[f'r2_{i}_c1_b'], (2, 2), 1)); a = conv2d(a, W[f'r2_{i}_c2_w'], W[f'r2_{i}_c2_b'], (1, 1), 1)
            h = gelu(a + conv2d(h, W[f'r2_{i}_sk_w'], W[f'r2_{i}_sk_b'], (2, 2), 0))
        v = np.concatenate([h1, h.ravel(), np.asarray(s, np.float64)])
        for j in range(m['n_head']):
            v = W[f'h{j}_w'] @ v + W[f'h{j}_b']
            if j < m['n_head'] - 1: v = gelu(v)
        mu = v[:2] * np.array(m['sd_y']) + np.array(m['mu_y']); sg = np.exp(np.clip(v[2:4], -7, 3)) * np.array(m['sd_y'])
        p = 1 / (1 + np.exp(-np.clip(v[4], -30, 30))) if len(v) > 4 else 1.0
        return dict(ln_numax=mu[0], ln_dnu=mu[1], sig_numax=sg[0], sig_dnu=sg[1], p_det=float(p))

NUMAX_SUN, DNU_SUN, TEFF_SUN = 3090.0, 135.1, 5777.0      # solar references used by Yu et al. 2018

def scaling(numax, dnu, teff, e_numax=0.0, e_dnu=0.0, e_teff=0.0):
    """Asteroseismic scaling relations (no dnu correction): mass and radius in solar units, with propagated errors."""
    a, b, t = numax / NUMAX_SUN, dnu / DNU_SUN, teff / TEFF_SUN
    M = a ** 3 * b ** -4 * t ** 1.5; R = a * b ** -2 * t ** 0.5
    fn, fd, ft = e_numax / numax, e_dnu / dnu, e_teff / teff
    return dict(mass=M, mass_err=M * np.sqrt((3 * fn) ** 2 + (4 * fd) ** 2 + (1.5 * ft) ** 2), radius=R, radius_err=R * np.sqrt(fn ** 2 + (2 * fd) ** 2 + (0.5 * ft) ** 2))

class StarQuake:
    def __init__(self, weights=None):
        weights = weights or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'starquake_weights.npz')
        self.net = Net(weights); self.widen = self.net.meta.get('widen', [1.0, 1.0])
    def read_ppm(self, flux_ppm, teff=None, e_teff=100.0):
        """flux_ppm: relative flux in ppm on the regular Kepler long-cadence grid, NaN in gaps."""
        F1, F2, s, (f, P), T = PL.prepare(np.asarray(flux_ppm, np.float64)); o = self.net.forward(F1, F2, s)
        nm, dn = float(np.exp(o['ln_numax'])), float(np.exp(o['ln_dnu'])); sn, sd = o['sig_numax'] * self.widen[0], o['sig_dnu'] * self.widen[1]
        r = dict(detected=bool(o['p_det'] > 0.5), p_osc=o['p_det'], numax=nm, numax_err=nm * sn, dnu=dn, dnu_err=dn * sd, days=T, freq=f, power=P)
        if teff: r.update(scaling(nm, dn, teff, nm * sn, dn * sd, e_teff))
        return r
    def read_kepler(self, files_or_arrays, teff=None, e_teff=100.0):
        """files: Kepler *_llc.fits paths (needs astropy) or a dict with arrays cad, flux (PDCSAP), quarter, q."""
        return self.read_ppm(PL.load_kepler(files_or_arrays), teff, e_teff)
