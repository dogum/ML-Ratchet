"""Forcing preparation (contract columns -> engine matrix) and helpers to reuse real forcing."""
import numpy as np
import pandas as pd

from .pet import esat_kpa, extraterrestrial_radiation

COLUMNS = ['P', 'T_mean', 'T_min', 'T_max', 'SRAD', 'vp', 'wind']


def as_frame(forcing):
    """Accept a DataFrame, (DataFrame, meta) tuple or a parquet path. Returns (df, meta)."""
    meta = {}
    if isinstance(forcing, (tuple, list)):
        forcing, meta = forcing[0], dict(forcing[1] or {})
    if isinstance(forcing, str):
        forcing = pd.read_parquet(forcing)
    df = forcing.copy()
    meta = {**getattr(df, 'attrs', {}).get('meta', {}), **meta}
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ('date', 'time'):
            if c in df.columns:
                df = df.set_index(pd.to_datetime(df[c])).drop(columns=[c])
                break
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    return df.sort_index(), meta


def prepare(forcing):
    """Fill the contract columns: T_min/T_max from T_mean (+-4 C) when missing, T_mean from (T_min+T_max)/2,
    vp from RH or from T_min (FAO-56 eq. 48), wind 2 m/s, SRAD left NaN (Hargreaves radiation is used where
    needed). Missing P days are set to 0 and flagged. Returns (df, meta, notes)."""
    df, meta = as_frame(forcing)
    notes = []
    if 'P' not in df or 'T_mean' not in df and not ('T_min' in df and 'T_max' in df):
        raise ValueError('forcing needs at least P and T_mean (or T_min and T_max)')
    if 'T_mean' not in df:
        df['T_mean'] = (df['T_min'] + df['T_max']) / 2
    if 'T_min' not in df:
        df['T_min'] = df['T_mean'] - 4.0
        notes.append('T_min = T_mean - 4')
    if 'T_max' not in df:
        df['T_max'] = df['T_mean'] + 4.0
        notes.append('T_max = T_mean + 4')
    if 'vp' not in df:
        if 'RH' in df:
            df['vp'] = df['RH'].clip(0, 100) / 100.0 * esat_kpa(df['T_mean'].values) * 1000.0
            notes.append('vp from RH')
        else:
            df['vp'] = np.nan
    if 'SRAD' not in df:
        df['SRAD'] = np.nan
    if 'wind' not in df:
        df['wind'] = np.nan
    nP = int(df['P'].isna().sum())
    if nP:
        notes.append(f'{nP} missing P days set to 0')
    df['P'] = df['P'].fillna(0.0).clip(lower=0)
    note = str(meta.get('SRAD_note', '')) + str(meta.get('srad', ''))
    meta['srad_is_net'] = bool(meta.get('srad_is_net', False) or 'NET' in note.upper())
    for c in ('T_mean', 'T_min', 'T_max'):
        if df[c].isna().any():
            df[c] = df[c].interpolate(limit_direction='both')
    return df, meta, notes


def engine_matrix(df, lat, z_ref, srad_is_net=False):
    """(T, 7) float64 matrix for the engine with NaNs filled: SRAD by Hargreaves-Samani radiation
    (k_Rs = 0.16, FAO-56 eq. 50), vp by e_sat(T_min), wind by 2 m/s. Also returns clear-sky Rso (MJ/m2/day).
    The engine needs INCOMING shortwave; when SRAD is net (ERA5-Land) it is divided by (1 - albedo) with the
    albedo of the reanalysis surface approximated from its own snow (0.20 bare, up to 0.75 under SWE_era5 >> 10 mm)
    and the result is kept within [0.15, 1.0] x clear-sky."""
    doy = df.index.dayofyear.values
    ra = extraterrestrial_radiation(lat, doy)
    rso = (0.75 + 2e-5 * z_ref) * ra
    F = np.empty((len(df), 7))
    F[:, 0] = df['P'].values
    F[:, 1] = df['T_mean'].values
    F[:, 2] = df['T_min'].values
    F[:, 3] = df['T_max'].values
    srad = df['SRAD'].values.astype(float)
    est = 0.16 * np.sqrt(np.maximum(F[:, 3] - F[:, 2], 0)) * ra / 0.0864
    if srad_is_net:
        swe = df['SWE_era5'].values.astype(float) if 'SWE_era5' in df else np.zeros(len(df))
        alb = 0.20 + 0.55 * (1 - np.exp(-np.nan_to_num(swe) / 10.0))
        srad = np.clip(srad / (1 - alb), 0.15 * rso / 0.0864, rso / 0.0864)
    F[:, 4] = np.where(np.isfinite(srad), srad, est)
    vp = df['vp'].values.astype(float)
    F[:, 5] = np.where(np.isfinite(vp), vp, esat_kpa(F[:, 2]) * 1000.0)
    w = df['wind'].values.astype(float)
    F[:, 6] = np.where(np.isfinite(w), w, 2.0)
    return F, rso


def perturb(df, rng, p_scale=(0.85, 1.15), t_shift=(-1.5, 1.5), keep=True):
    """Climate-perturbed copy of a real forcing frame (for variety): P x U(p_scale), T + U(t_shift) with vp scaled
    to keep relative humidity. Returns (df, {'p_scale', 't_shift'})."""
    a = float(rng.uniform(*p_scale))
    d = float(rng.uniform(*t_shift))
    out = df.copy()
    out['P'] = out['P'] * a
    for c in ('T_mean', 'T_min', 'T_max'):
        if c in out:
            out[c] = out[c] + d
    if 'vp' in out and out['vp'].notna().any():
        out['vp'] = out['vp'] * esat_kpa(out['T_min'].values) / esat_kpa(out['T_min'].values - d)
    if 'PET' in out:
        out = out.drop(columns=['PET'])
    return out, {'p_scale': a, 't_shift': d}
