"""FlowCast v1.0: daily streamflow for UNGAUGED basins from forcing + static attributes (numpy only).

A small LSTM trained only on WaterSim simulations (ML Ratchet, Watershed Phase 2). It never sees streamflow from
the basin it predicts. Input: basin-average daily forcing (ERA5-Land style, Caravan columns) and a few static
attributes. Output: daily mean flow (mm/day) plus 5/25/50/75/95 % quantiles (dev-calibrated, widen-only).

    import flowcast as fc
    m = fc.FlowCast()                                   # loads flowcast_weights.npz next to this file
    out = m.predict(forcing_df, attrs, start='1995-10-01', end='1997-09-30')
    out[['Q', 'q05', 'q50', 'q95']]                     # mm/day; Q_m3s too when attrs['area_km2'] is given

forcing_df: daily DatetimeIndex with P (mm/d), PET (mm/d, FAO-56 PM), T_mean, T_min, T_max (C), SRAD (W/m2, NET
shortwave as in ERA5-Land/Caravan), vp (Pa), wind (m/s); it must start 365 days before `start` and run one day
past `end`.
attrs: area_km2, elev_mean_m, elev_min_m, elev_max_m, slope_mean (m/km; or degrees with slope_unit='degrees'),
forest_frac, soil_porosity, soil_conductivity (cm/h; or mm/h with soil_conductivity_unit='mm/h'),
carbonate_frac, reservoir_storage_frac, irrigation_frac, urban_frac. Missing values get documented fallbacks.
"""
import math
import os
import numpy as np
import pandas as pd

DYN = ['P', 'PET', 'T_mean', 'T_min', 'T_max', 'SRAD', 'vp', 'wind']
STATIC_NAMES = ['log_area', 'elev_mean_km', 'elev_range_km', 'log_slope_mkm', 'forest_frac', 'soil_porosity',
                'log_ksat_cmh', 'carbonate_frac', 'reservoir_frac', 'irrigation_frac', 'urban_frac',
                'log_p_mean', 'pet_mean', 'log_aridity', 'frac_snow', 'hp_freq', 'lp_freq', 't_mean_c']
FALLBACK = {'forest_frac': 0.5, 'soil_porosity': 0.44, 'soil_conductivity_cmh': 1.6, 'slope_mkm': 60.0}
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
WARM = 365


def _f(x, d):
    try:
        x = float(x)
        return x if np.isfinite(x) else d
    except Exception:
        return d


def climate_stats(P, PET, T):
    P = np.asarray(P, float); PET = np.asarray(PET, float); T = np.asarray(T, float)
    pm = max(np.nanmean(P), 0.05)
    em = max(np.nanmean(PET), 0.05)
    fs = float(np.nansum(P[T < 0]) / max(np.nansum(P), 1e-6))
    return {'p_mean': pm, 'pet_mean': em, 'aridity': em / pm, 'frac_snow': fs,
            'hp_freq': float(np.nanmean(P > 5 * pm)), 'lp_freq': float(np.nanmean(P < 1.0)),
            't_mean': float(np.nanmean(T))}


def spec_attrs(spec):
    if 'slope_mkm' in spec and 'ksat_cmh' in spec:      # already in FlowCast units (as in examples.json)
        a = {k: _f(spec.get(k), 0.0) for k in ('area_km2', 'elev_mean_m', 'slope_mkm', 'forest_frac', 'soil_porosity',
                                               'ksat_cmh', 'carbonate_frac', 'reservoir_storage_frac',
                                               'irrigation_frac', 'urban_frac')}
        a['elev_range_m'] = _f(spec.get('elev_range_m'), _f(spec.get('elev_max_m'), 0) - _f(spec.get('elev_min_m'), 0))
        return a
    sl = _f(spec.get('slope_mean'), None)
    if sl is None:
        sl = FALLBACK['slope_mkm']
    elif str(spec.get('slope_unit', '')).startswith('deg'):
        sl = math.tan(math.radians(sl)) * 1000.0
    k = _f(spec.get('soil_conductivity'), None)
    if k is None:
        k = FALLBACK['soil_conductivity_cmh']
    elif spec.get('soil_conductivity_unit') == 'mm/h':
        k = k / 10.0
    em = _f(spec.get('elev_mean_m'), 500.0)
    lo = _f(spec.get('elev_min_m'), em * 0.7)
    hi = _f(spec.get('elev_max_m'), em * 1.3)
    return {'area_km2': max(_f(spec.get('area_km2'), 300.0), 1.0), 'elev_mean_m': em, 'elev_range_m': max(hi - lo, 0.0),
            'slope_mkm': max(sl, 0.5), 'forest_frac': _f(spec.get('forest_frac'), FALLBACK['forest_frac']),
            'soil_porosity': _f(spec.get('soil_porosity'), FALLBACK['soil_porosity']), 'ksat_cmh': max(k, 0.01),
            'carbonate_frac': _f(spec.get('carbonate_frac'), 0.0),
            'reservoir_storage_frac': _f(spec.get('reservoir_storage_frac'), 0.0),
            'irrigation_frac': _f(spec.get('irrigation_frac'), 0.0), 'urban_frac': _f(spec.get('urban_frac'), 0.0)}


def static_vector(a, c):
    return np.array([
        math.log10(a['area_km2']), a['elev_mean_m'] / 1000, a['elev_range_m'] / 1000, math.log10(a['slope_mkm']),
        a['forest_frac'], a['soil_porosity'], math.log10(a['ksat_cmh']), a['carbonate_frac'],
        min(a['reservoir_storage_frac'], 2.0), a['irrigation_frac'], a['urban_frac'],
        math.log(c['p_mean']), c['pet_mean'], math.log(c['aridity']), c['frac_snow'], c['hp_freq'], c['lp_freq'],
        c['t_mean'] / 10], np.float32)


def dyn_features(F, doy):
    F = np.asarray(F, np.float32)
    P = np.maximum(F[:, 0], 0)
    out = np.empty((len(F), 11), np.float32)
    out[:, 0] = P / 10.0; out[:, 1] = np.log1p(P); out[:, 2] = F[:, 1] / 5.0
    out[:, 3] = F[:, 2] / 10.0; out[:, 4] = F[:, 3] / 10.0; out[:, 5] = F[:, 4] / 10.0
    out[:, 6] = F[:, 5] / 150.0; out[:, 7] = F[:, 6] / 1000.0; out[:, 8] = F[:, 7] / 3.0
    w = 2 * np.pi * np.asarray(doy, np.float32) / 365.25
    out[:, 9] = np.sin(w); out[:, 10] = np.cos(w)
    return out


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.logaddexp(0.0, x)


class FlowCast:
    def __init__(self, weights=None):
        weights = weights or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flowcast_weights.npz')
        z = np.load(weights)
        self.W = {k: z[k].astype(np.float64) for k in z.files if k not in ('k_widen',)}
        self.k = tuple(float(x) for x in z['k_widen']) if 'k_widen' in z.files else (1.0, 1.0)

    def forward(self, x, st):
        """x (B, T, 11), st (B, 18) -> mean (B, T), q (B, T, 5) in units of the basin's mean precipitation."""
        W = self.W
        s = (st - W['st_mu']) / W['st_sd']
        Bn, T, _ = x.shape
        H = W['w_hh'].shape[1]
        z = np.concatenate([x, np.repeat(s[:, None, :], T, 1)], 2) @ W['w_ih'].T + (W['b_ih'] + W['b_hh'])
        h = np.zeros((Bn, H)); c = np.zeros((Bn, H)); hs = np.empty((Bn, T, H))
        Wh = W['w_hh'].T
        for t in range(T):
            g = z[:, t] + h @ Wh
            i, f, gg, o = _sig(g[:, :H]), _sig(g[:, H:2 * H]), np.tanh(g[:, 2 * H:3 * H]), _sig(g[:, 3 * H:])
            c = f * c + i * gg
            h = o * np.tanh(c)
            hs[:, t] = h
        out = hs @ W['head_w'].T + W['head_b']
        mean = _softplus(out[..., 0])
        q = np.cumsum(np.concatenate([_softplus(out[..., 1:2]) * 0.2, _softplus(out[..., 2:])], -1), -1)
        return mean, q

    def predict_arrays(self, F, doy, st, n_target, qn=None, lead=None):
        """F (365 + n_target + 1, 8) raw forcing; blocks of 365 target days, each with a 365-day warm-up.
        qn: observed flow / p_mean over the same rows (nowcast weights only), lead in days."""
        blocks, a = [], 0
        while a < n_target:
            b = min(a + 365, n_target); blocks.append((a, b)); a = b
        L = WARM + 365
        X = []
        for a, b in blocks:
            Fw = F[a:a + WARM + (b - a)]
            if len(Fw) < L:
                Fw = np.concatenate([Fw, np.repeat(Fw[-1:], L - len(Fw), 0)])
            x = dyn_features(Fw, (doy[a] - 1 + np.arange(L)) % 365 + 1)
            if qn is not None:
                qw = qn[a:a + WARM + (b - a)]
                qw = np.concatenate([qw, np.full(L - len(qw), np.nan)])
                lag = np.full(L, np.nan); lag[lead:] = qw[:L - lead]
                ok = np.isfinite(lag)
                ch = np.zeros((L, 3), np.float32)
                ch[ok, 0] = np.log1p(np.maximum(lag[ok], 0) * 5.0); ch[:, 1] = ok; ch[:, 2] = lead / 7.0
                x = np.concatenate([x, ch], 1)
            X.append(x)
        mean, q = self.forward(np.stack(X).astype(np.float64), np.repeat(st[None].astype(np.float64), len(blocks), 0))
        mean = np.concatenate([mean[k, WARM:WARM + b - a] for k, (a, b) in enumerate(blocks)])
        q = np.concatenate([q[k, WARM:WARM + b - a] for k, (a, b) in enumerate(blocks)])
        return mean, q

    def widen(self, q):
        q = q.copy(); med = q[:, 2].copy(); k50, k90 = self.k
        q[:, 1] = med - k50 * (med - q[:, 1]); q[:, 3] = med + k50 * (q[:, 3] - med)
        q[:, 0] = med - k90 * (med - q[:, 0]); q[:, 4] = med + k90 * (q[:, 4] - med)
        return np.maximum(q, 0.0)

    def predict(self, forcing, attrs, start, end, qobs=None, lead=1):
        """Ungauged by default. With nowcast weights (flowcast_nowcast_weights.npz) pass qobs (observed mm/day,
        daily Series) and lead (1-7): each day then also uses the gauge record up to `lead` days earlier (GAUGED)."""
        idx = pd.date_range(start, end)
        s0 = pd.Timestamp(start) - pd.Timedelta(days=WARM)
        df = forcing.loc[s0:pd.Timestamp(end) + pd.Timedelta(days=1), DYN].astype(float).interpolate(limit_direction='both')
        if len(df) < WARM + len(idx):
            raise ValueError('forcing must start 365 days before `start` and cover `end`')
        F = df.values.astype(np.float32)
        clim = climate_stats(F[:, 0], F[:, 1], F[:, 2])
        st = static_vector(spec_attrs(attrs), clim)
        pm = clim['p_mean']
        qn = None
        if qobs is not None:
            qn = qobs.reindex(df.index).values.astype(float) / pm
        mean, q = self.predict_arrays(F, df.index.day_of_year.values, st, len(idx), qn, lead)
        q = self.widen(q) * pm
        out = pd.DataFrame({'Q': mean * pm, **{f'q{int(round(100 * t)):02d}': q[:, j] for j, t in enumerate(QS)}},
                           index=idx)
        if attrs.get('area_km2'):
            out['Q_m3s'] = out['Q'] * float(attrs['area_km2']) / 86.4
        return out
