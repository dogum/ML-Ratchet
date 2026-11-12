"""TransitNet: is that dip in a star's light a planet, and if so, how big, how tilted, how far out? Learned from physics alone.

Reads a TESS 2-minute light curve (one or more sectors) of a star with a candidate periodic dip (period, epoch, duration),
removes the star's slow variability, refines the ephemeris, folds, and returns:
  p_planet, p_eb, p_blend, p_nothing  what the dip is: a planet, an eclipsing binary on the star, a background eclipsing
                                      binary diluted by the star's light, or nothing real
  k, b, aR, T14_h                     for a planet: radius ratio Rp/R*, impact parameter, a/R*, duration (hours), each with
                                      a 68% interval (k_lo/k_hi, ...)
numpy and scipy only. See demo.py and MODEL_CARD.md."""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE); import pipeline as PL

def _gelu(x):
    a = np.abs(x) * 0.7071067811865476; t = 1 / (1 + 0.3275911 * a)
    erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return 0.5 * x * (1 + np.sign(x) * erf)

def _conv(x, w, b):
    k = w.shape[2]; p = k // 2; xp = np.pad(x, ((0, 0), (0, 0), (p, p))); win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=2)
    return np.einsum('bclk,ock->bol', win, w, optimize=True) + b[None, :, None]

class Net:
    def __init__(self, path=os.path.join(HERE, 'transitnet_weights.npz')):
        z = np.load(path); self.meta = json.loads(str(z['meta'])); self.W = {k: z[k] for k in z.files if k != 'meta'}
    def _branch(self, name, x):
        for op in self.meta['branches'][name]:
            if op[0] == 'conv': x = _gelu(_conv(x, self.W[op[1] + '_w'], self.W[op[1] + '_b']))
            else: L = x.shape[-1] // 2 * 2; x = x[..., :L].reshape(x.shape[0], x.shape[1], -1, 2).max(-1)
        return np.concatenate([x.mean(-1), x.max(-1)], 1)
    def forward(self, loc, sec, glo, oe, s, cent=None, sc=None):
        parts = [self._branch('loc', loc), self._branch('sec', sec), self._branch('glo', glo), self._branch('oe', oe), s]
        if 'cen' in self.meta['branches']: parts += [self._branch('cen', np.clip(cent, -20, 20)), np.asarray(sc, float).reshape(-1, 1)]
        z = np.concatenate(parts, 1)
        for j in range(self.meta['n_head']):
            z = z @ self.W[f'h{j}_w'].T + self.W[f'h{j}_b']
            if j < self.meta['n_head'] - 1: z = _gelu(z)
        return z

CLASSES = ['planet', 'eb', 'blend', 'nothing']; PARAMS = ['log_k', 'b', 'log_aR', 'log_T14']

def _base(s):
    s = np.asarray(s, float); z = np.zeros((len(s), 4)); z[:, 0] = 0.5 * s[:, 2]; z[:, 2] = s[:, 0] - np.log10(np.pi) - s[:, 1] + np.log10(24); z[:, 3] = s[:, 1]; return z

class TransitNet:
    """read(time, flux, period, epoch, duration_hours, ...) -> what the dip is, and the planet's parameters with 68% intervals"""
    def __init__(self, weights=os.path.join(HERE, 'transitnet_weights.npz')):
        self.net = Net(weights); m = self.net.meta; self.mu, self.sd, self.residual = np.array(m['mu_y']), np.array(m['sd_y']), m.get('residual', 0)

    def predict_views(self, V, s, sc=None):
        """V: dict of views (local, secondary, glob, oddeven, cent) with a leading batch axis; s: (B, 10) scalars; sc: (B,) log10 centroid noise"""
        cent = V.get('cent'); flip = lambda a: None if a is None else a[..., ::-1]
        o = np.stack([self.net.forward(V['local'], V['secondary'], V['glob'], V['oddeven'], s, cent, sc),
                      self.net.forward(flip(V['local']), flip(V['secondary']), flip(V['glob']), flip(V['oddeven']), s, flip(cent), sc)])
        e = np.exp(o[..., :4] - o[..., :4].max(-1, keepdims=True)); pr = (e / e.sum(-1, keepdims=True)).mean(0)
        mu = o[..., 4:8].mean(0) * self.sd + self.mu + (_base(s) if self.residual else 0)
        sg = np.sqrt((np.exp(2 * np.clip(o[..., 8:12], -6, 3)) * self.sd ** 2).mean(0) + o[..., 4:8].var(0) * self.sd ** 2)
        R = {'p_' + c: pr[:, j] for j, c in enumerate(CLASSES)}
        for j, p in enumerate(PARAMS): R[p] = mu[:, j]; R[p + '_sig'] = sg[:, j]
        return R

    def read(self, sectors, period, epoch, duration_hours, star=None, period_err=None, others=()):
        """sectors: list of dicts with 't' (days, e.g. BTJD), 'f' (flux, any units: each sector is divided by its median) and
        optionally 'c1', 'c2' (flux-weighted centroid column and row, pixels). star: dict with TESSMAG, TEFF, LOGG, RADIUS (the
        TIC header keys; any may be missing). others: (period, epoch, duration_days) of other known transiting bodies to cut out."""
        hdr = dict(star or {}); secs = []
        for s in sectors:
            t, f = np.asarray(s['t'], float), np.asarray(s['f'], float); ok = np.isfinite(t) & np.isfinite(f) & (f > 0)
            d = dict(t=t[ok], f=f[ok] / np.median(f[ok]), hdr=hdr)
            for c in ('c1', 'c2'): d[c] = np.asarray(s[c], float)[ok] if c in s else np.full(ok.sum(), np.nan)
            secs.append(d)
        V, info = PL.read(secs, period, epoch, duration_hours / 24, sigP=period_err, others=others)
        st = PL.star_scalars(hdr); s = PL.scalars(info['P'], duration_hours / 24, info, st)[None]
        R = self.predict_views({k: v[None] for k, v in V.items()}, s, np.array([np.log10(max(info['cent_noise'], 1e-4))]))
        out = {k: float(v[0]) for k, v in R.items()}
        for p, name in (('log_k', 'k'), ('log_aR', 'aR'), ('log_T14', 'T14_h')):
            out[name] = 10 ** out[p]; out[name + '_lo'] = 10 ** (out[p] - out[p + '_sig']); out[name + '_hi'] = 10 ** (out[p] + out[p + '_sig'])
        out['b_lo'], out['b_hi'] = max(0.0, out['b'] - out['b_sig']), out['b'] + out['b_sig']
        if st['have'] and np.isfinite(st['rad']): out['Rp_earth'] = out['k'] * st['rad'] * 109.1; out['Rp_earth_lo'] = out['k_lo'] * st['rad'] * 109.1; out['Rp_earth_hi'] = out['k_hi'] * st['rad'] * 109.1
        out.update(snr=info['snr'], period=info['P'], epoch=info['T0'], depth_ppm=1e6 * info['depth'], n_transits=info['ntr']); return out

def load_spoc(path):
    """a TESS SPOC light-curve FITS file -> (sector dict, star header dict); needs astropy"""
    from astropy.io import fits
    with fits.open(path) as h:
        d = h[1].data; q = d['QUALITY'] == 0; hdr = {k: h[0].header.get(k) for k in ('TESSMAG', 'TEFF', 'LOGG', 'RADIUS')}
        return dict(t=np.asarray(d['TIME'], float)[q], f=np.asarray(d['PDCSAP_FLUX'], float)[q], c1=np.asarray(d['MOM_CENTR1'], float)[q], c2=np.asarray(d['MOM_CENTR2'], float)[q]), hdr
