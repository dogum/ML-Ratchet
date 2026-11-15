"""Real input shapes for simulated systems, from TRAIN-INPUT years only (2018, 2020, 2022; splits.json).

demand shape (EIA-930 `d`), wind and solar capacity factors (EIA-930 ng_WND / ng_SUN over the year's p99.5),
Henry Hub daily gas price.  2018 has no fuel mix before July, so 2018 windows start in July or later.
"""
from common import *
import numpy as np, pandas as pd
import griddata as gd
from gridsim.fleet import henry_hub_daily

BAS = ['ERCO', 'CISO', 'PJM', 'MISO', 'SWPP', 'ISNE', 'NYIS']
TRAIN_YEARS = [2018, 2020, 2022]
_CACHE = {}


def year_frame(ba, year):
    assert year in TRAIN_YEARS, 'training inputs only from train_inputs years'
    k = (ba, year)
    if k not in _CACHE:
        d = gd.eia930(ba, f'{year}-01-01', f'{year + 1}-01-01', columns=['d', 'ng_WND', 'ng_SUN'])
        idx = pd.date_range(f'{year}-01-01', f'{year + 1}-01-01', freq='h', tz='UTC', inclusive='left')
        d = d.reindex(idx)
        d['d'] = d.d.interpolate(limit_direction='both')
        for c, out in (('ng_WND', 'wcf'), ('ng_SUN', 'scf')):
            x = d[c].clip(lower=0)
            cap = np.nanpercentile(x, 99.5) if x.notna().any() else 1.0
            d[out] = (x / max(cap, 1.0)).clip(0, 1)
        _CACHE[k] = d
    return _CACHE[k]


def window(rng, n_days=92, ba=None, year=None):
    ba = ba or str(rng.choice(BAS))
    year = int(year or rng.choice(TRAIN_YEARS))
    d = year_frame(ba, year)
    first = 181 if year == 2018 else 0
    last = (366 if year == 2020 else 365) - n_days
    s = int(rng.integers(first, max(first + 1, last)))
    w = d.iloc[s * 24:(s + n_days) * 24].copy()
    w['wcf'] = w.wcf.interpolate(limit_direction='both').fillna(0.3)
    w['scf'] = w.scf.interpolate(limit_direction='both').fillna(0.0)
    hh = henry_hub_daily()
    days = pd.date_range(w.index[0].tz_localize(None).normalize(), w.index[-1].tz_localize(None).normalize(), freq='D')
    gas = hh.reindex(days).ffill().bfill()
    return ba, year, w, gas
