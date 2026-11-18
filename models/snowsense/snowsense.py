"""SnowSense v1.0: daily snow water equivalent (SWE) at a snow-pillow station from the station's own weather.

numpy-only inference. Input: daily precipitation P (mm/day, the gauge total for that local day), air temperature
T_mean, T_min, T_max (deg C), the station latitude and elevation. No SWE is ever read.
Output: daily SWE median and 68% / 90% bands (end of day, mm); per water year: peak SWE, peak date and melt-out date
(first day after the peak below 5 mm) with 68% / 90% intervals.

    import pandas as pd, snowsense
    m = snowsense.SnowSense()                      # loads snowsense_weights.npz next to this file
    out = m.predict(df, lat=40.5, elev_m=2900)      # df: daily DatetimeIndex, columns P, T_mean, T_min, T_max
    out['daily']                                    # DataFrame: swe, lo68, hi68, lo90, hi90
    out['annual']                                   # DataFrame per water year: peak, peak_date, melt_out, intervals

Trained only on WaterSim snow simulations (degree-day and energy-balance physics) driven by dev SNOTEL weather.
"""
import os
import numpy as np
import pandas as pd

L, WARM, NCH = 427, 61, 10
Z68, Z90 = 0.9945, 1.6449
HERE = os.path.dirname(os.path.abspath(__file__))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def features(P, Tm, Tn, Tx, mP, mT, dates, lat, elev_m):
    doy = pd.DatetimeIndex(dates).day_of_year.values
    ang = 2 * np.pi * (doy - 1) / 365.25
    f = np.empty((len(P), NCH), np.float32)
    f[:, 0] = np.log1p(np.maximum(P, 0)) / 2.0
    f[:, 1] = Tm / 10.0
    f[:, 2] = Tn / 10.0
    f[:, 3] = Tx / 10.0
    f[:, 4] = np.sin(ang)
    f[:, 5] = np.cos(ang)
    f[:, 6] = mP
    f[:, 7] = mT
    f[:, 8] = (elev_m - 2200.0) / 800.0
    f[:, 9] = (lat - 43.0) / 8.0
    return f


def clean(df):
    """Gap handling exactly as in training: P gaps -> 0 (flagged), T gaps interpolated (flagged)."""
    d = df.asfreq('D')
    out = pd.DataFrame(index=d.index)
    out['mP'] = d['P'].isna().astype(np.float32)
    out['P'] = d['P'].clip(lower=0).fillna(0.0)
    tm = d['T_mean'].fillna((d['T_min'] + d['T_max']) / 2)
    out['mT'] = tm.isna().astype(np.float32)
    out['T_mean'] = tm
    out['T_min'] = d['T_min'].fillna(tm - 5.0)
    out['T_max'] = d['T_max'].fillna(tm + 5.0)
    for c in ('T_mean', 'T_min', 'T_max'):
        out[c] = out[c].interpolate(limit_direction='both')
    out['T_min'] = np.minimum(out['T_min'], out['T_mean'])
    out['T_max'] = np.maximum(out['T_max'], out['T_mean'])
    return out


def melt_out_idx(s, thresh=5.0):
    i = int(np.argmax(s))
    b = np.where(s[i:] < thresh)[0]
    return i + int(b[0]) if len(b) else None


class SnowSense:
    def __init__(self, path=None):
        w = np.load(path or os.path.join(HERE, 'snowsense_weights.npz'))
        self.w = {k: w[k].astype(np.float64) for k in w.files if k != 'k'}
        self.k = dict(zip(['daily', 'peak', 'melt'], w['k'].tolist())) if 'k' in w.files else \
            {'daily': 1.0, 'peak': 1.0, 'melt': 1.0}
        self.layers = int(sum(1 for k in self.w if k.startswith('gru.weight_ih_l')))

    def forward(self, X):
        """X: (B, T, NCH) -> quantiles (B, T, 5) [q05, q16, q50, q84, q95] in z = sqrt(SWE)/10 space."""
        w = self.w
        h = np.tanh(X @ w['inp.weight'].T + w['inp.bias'])
        for l in range(self.layers):
            Wi, Wh = w[f'gru.weight_ih_l{l}'], w[f'gru.weight_hh_l{l}']
            bi, bh = w[f'gru.bias_ih_l{l}'], w[f'gru.bias_hh_l{l}']
            H = Wh.shape[1]
            gi = h @ Wi.T + bi
            hs = np.zeros((X.shape[0], H))
            out = np.empty((X.shape[0], X.shape[1], H))
            for t in range(X.shape[1]):
                gh = hs @ Wh.T + bh
                r = _sig(gi[:, t, :H] + gh[:, :H])
                z = _sig(gi[:, t, H:2 * H] + gh[:, H:2 * H])
                n = np.tanh(gi[:, t, 2 * H:] + r * gh[:, 2 * H:])
                hs = (1 - z) * n + z * hs
                out[:, t] = hs
            h = out
        o = h @ w['head.weight'].T + w['head.bias']
        if o.shape[-1] == 2:                       # Gaussian head (rungs 1-3)
            mu, sg = o[..., 0], _softplus(o[..., 1]) + 0.01
            return np.stack([mu + z * sg for z in (-Z90, -Z68, 0.0, Z68, Z90)], -1)
        d = _softplus(o[..., 1:]) + 1e-3          # quantile head: q05, q16, q50, q84, q95 (monotone)
        q50 = o[..., 0]
        return np.stack([q50 - d[..., 0] - d[..., 1], q50 - d[..., 0], q50, q50 + d[..., 2],
                         q50 + d[..., 2] + d[..., 3]], -1)

    def windows(self, df, lat, elev_m, wys=None):
        c = clean(df)
        if wys is None:
            idx = c.index
            wys = sorted(set((idx.year + (idx.month >= 10)).tolist()))
        X, D, keep = [], [], []
        for y in wys:
            idx = pd.date_range(f'{y - 1}-08-01', periods=L, freq='D')
            w = c.reindex(idx)
            inwy = (idx.year + (idx.month >= 10)) == y
            if w['T_mean'].reindex(idx[inwy]).notna().sum() < 200:
                continue
            Tm = w['T_mean'].interpolate(limit_direction='both').values
            Tn = w['T_min'].interpolate(limit_direction='both').values
            Tx = w['T_max'].interpolate(limit_direction='both').values
            mP = np.where(w['mP'].isna(), 1.0, w['mP'].values)
            mT = np.where(w['mT'].isna(), 1.0, w['mT'].values)
            X.append(features(w['P'].fillna(0).values, Tm, Tn, Tx, mP, mT, idx, lat, elev_m))
            D.append(idx)
            keep.append(y)
        return (np.stack(X) if X else np.zeros((0, L, NCH), np.float32)), D, keep

    def predict(self, df, lat, elev_m, wys=None):
        X, D, wys = self.windows(df, lat, elev_m, wys)
        if not len(wys):
            return {'daily': pd.DataFrame(), 'annual': pd.DataFrame()}
        Qa = self.forward(X.astype(np.float64))
        kd, kp, km = self.k['daily'], self.k['peak'], self.k['melt']
        band = lambda Q, j, k=1.0: 100.0 * np.maximum(Q[:, 2] + k * (Q[:, j] - Q[:, 2]), 0) ** 2
        daily, ann = [], []
        for j, y in enumerate(wys):
            wm = ((D[j].year + (D[j].month >= 10)) == y)
            idx = D[j][wm]
            Q = Qa[j][wm]
            daily.append(pd.DataFrame({'swe': band(Q, 2), 'lo68': band(Q, 1, kd), 'hi68': band(Q, 3, kd),
                                       'lo90': band(Q, 0, kd), 'hi90': band(Q, 4, kd)}, index=idx))
            med = band(Q, 2)
            r = {'water_year': y, 'peak_swe_mm': float(med.max()), 'peak_date': idx[int(np.argmax(med))]}
            mo = melt_out_idx(med)
            r['melt_out_date'] = idx[mo] if mo is not None else pd.NaT
            for tag, (jl, jh) in (('68', (1, 3)), ('90', (0, 4))):
                r[f'peak_lo{tag}'] = float(band(Q, jl, kp).max())
                r[f'peak_hi{tag}'] = float(band(Q, jh, kp).max())
                a, b = melt_out_idx(band(Q, jl, km)), melt_out_idx(band(Q, jh, km))
                r[f'melt_lo{tag}'] = idx[a] if a is not None else pd.NaT
                r[f'melt_hi{tag}'] = idx[b] if b is not None else pd.NaT
            ann.append(r)
        return {'daily': pd.concat(daily), 'annual': pd.DataFrame(ann).set_index('water_year'), 'quantiles_z': Qa}
