"""Real BA-days as model inputs (same featurisation as the simulator).
  python3 realbank.py dev    -> data/banks/real_dev.npz   (2019, 2021; actual weather)
  python3 realbank.py exam   -> data/banks/real_exam.npz  (2023-2025; actual weather, and forecast weather where
                                 Open-Meteo previous-run forecasts exist)
Each row also carries the classical baselines for its target day: BA day-ahead forecast (demand), persistence (t-24 h),
same hour last week, and for wind/solar the dev-year capacity-factor climatology (month x LST hour) x capacity."""
import sys, os, json, numpy as np, pandas as pd
from common import *
import real

KEYS = ('xd', 'yd', 'xw', 'yw', 'xs', 'ys', 'pw', 'ps', 'night')


def climatology(g, ba):
    """Dev-year mean capacity factor by month x LST hour for wind and solar (actual MW / fleet MW online)."""
    m = np.isin(HOURS.year, DEV_YEARS)
    mon = HOURS.month.values; hr = (HOURS.hour.values + LST[ba]) % 24
    out = {}
    for name, y, cap in (('wind', g.yw, g.wcap_tot), ('solar', g.ys, g.scap_tot)):
        tab = np.zeros((13, 24))
        cf = np.where(cap > 0, y / np.maximum(cap, 1), np.nan)
        for mm in range(1, 13):
            for h in range(24):
                s = m & (mon == mm) & (hr == h)
                tab[mm, h] = np.nanmean(cf[s]) if np.isfinite(cf[s]).any() else 0.0
        out[name] = np.nan_to_num(tab)
    return out


def build(split):
    years = DEV_YEARS if split == 'dev' else EXAM_YEARS
    acc = {k: [] for k in KEYS}
    extra = {k: [] for k in ('d_true', 'pd_total', 'df', 'pers', 'week', 'w_pers', 's_pers', 'w_clim', 's_clim', 'capw',
                             'caps', 'comp', 'T', 'hasw', 'hass')}
    meta = []
    for ba in BAS:
        real._CACHE.clear()
        g = real.real_region(ba)
        clim = climatology(g, ba)
        F = real.Forecasts(ba) if split == 'exam' else None
        for y in years:
            if split == 'dev':     # cross-year: a dev year is described by the fit of the OTHER dev year (staleness)
                g = real.real_region(ba, 'dev2021' if y == 2019 else 'dev2019')
            for day in days_of_year(ba, y):
                i0 = day_start_index(ba, day)
                modes = [None]
                if F is not None:
                    fc = F.fc(i0 - 14, i0 + 24)
                    if fc is not None: modes.append(fc)
                for fc in modes:
                    s = g.sample(i0, fc)
                    for k in KEYS:
                        acc[k].append(s.get(k, np.zeros(acc_shape(k), np.float32)))
                    sl = slice(i0, i0 + 24)
                    mon = HOURS.month.values[sl]; hr = (HOURS.hour.values[sl] + LST[ba]) % 24
                    extra['d_true'].append(s['d_true']); extra['pd_total'].append(s['pd_total']); extra['comp'].append(s['comp'])
                    extra['df'].append(g.df[sl]); extra['pers'].append(g.yd[i0 - 24:i0]); extra['week'].append(g.yd[i0 - 168:i0 - 144])
                    extra['T'].append(g.wx['T'][sl] + (0 if fc is None else fc['T'][14:]))
                    cw = s.get('capw', 0.0); cs = s.get('caps', 0.0)
                    extra['capw'].append(cw); extra['caps'].append(cs)
                    extra['hasw'].append('xw' in s); extra['hass'].append('xs' in s)
                    extra['w_pers'].append(g.yw[i0 - 24:i0] / max(cw, 1)); extra['s_pers'].append(g.ys[i0 - 24:i0] / max(cs, 1))
                    extra['w_clim'].append(clim['wind'][mon, hr]); extra['s_clim'].append(clim['solar'][mon, hr])
                    meta.append((ba, str(day.date()), 0 if fc is None else fc['bits']))
            print(ba, y, len(meta), flush=True)
    arr = {k: np.stack(v).astype(np.float32) for k, v in acc.items()}
    arr.update({k: np.stack(v).astype(np.float32) if k not in ('hasw', 'hass') else np.array(v) for k, v in extra.items()})
    arr['meta_ba'] = np.array([m[0] for m in meta]); arr['meta_day'] = np.array([m[1] for m in meta])
    arr['meta_fc'] = np.array([m[2] for m in meta])     # 0 actual weather; else bits 1 T, 2 RH, 4 GHI, 8 wind forecast
    np.savez_compressed(f'{DATA}/banks/real_{split}.npz', **arr)


_SHAPES = {'xd': 497, 'xw': 332, 'xs': 324, 'yd': 24, 'yw': 24, 'ys': 24, 'pw': 24, 'ps': 24, 'night': 24}


def acc_shape(k):
    return (_SHAPES[k],)


if __name__ == '__main__':
    os.makedirs(f'{DATA}/banks', exist_ok=True)
    build(sys.argv[1])
