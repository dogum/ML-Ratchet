"""Weather features for FireCast: the network sees gridMET-like DAILY values for the issue day (d0) and the next
day (d1), because gridMET exists for every exam fire (HRRR does not exist before 2016-08 and is packaged only for
dev and the 48 replay fires).

Feature vector per day (all from gridMET on real fires):
    vs (m/s, 10 m daily mean), th (deg from, true north) -> spread vector toward (th + 180) in GRID coordinates,
    rmin (%), tmmx (C), vpd (kPa), pr (mm), fm100 (%)
In simulation the same quantities are computed from the hourly HRRR series the simulator ran with, then passed
through the dev-fitted HRRR -> gridMET operator (calib_wx.py -> out/calib_wx.json) with its residual noise.
"""
import numpy as np
import pandas as pd

DAY_KEYS = ['vs', 'th', 'rmin', 'tmmx', 'vpd', 'pr', 'fm100']


def esat_kpa(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def hrrr_daily(w, lon, m100=None):
    """Daily (local solar day) summaries of an hourly contract weather frame (UTC index; wind_dir FROM, in whatever
    north the frame uses). Returns DataFrame indexed by local date with raw HRRR daily stats."""
    idx = pd.DatetimeIndex(w.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    loc = idx + pd.to_timedelta(lon / 15.0, unit='h')
    day = loc.normalize()
    ws = w['wind_speed'].to_numpy(float)
    wd = np.radians(w['wind_dir'].to_numpy(float))
    d = pd.DataFrame(dict(ws=ws, ue=ws * np.sin(wd), vn=ws * np.cos(wd), T=w['T'].to_numpy(float),
                          RH=w['RH'].to_numpy(float), P=w['precip'].to_numpy(float) if 'precip' in w else 0.0,
                          day=day, hour=loc.hour))
    d['vpd'] = esat_kpa(d['T']) * (1 - d['RH'] / 100.0)
    if m100 is not None:
        d['m100'] = np.asarray(m100, float)
    g = d.groupby('day')
    out = pd.DataFrame(dict(n=g.size(), ws_mean=g.ws.mean(), ws_max=g.ws.max(), ue=g.ue.mean(), vn=g.vn.mean(),
                            rhmin=g.RH.min(), tmax=g['T'].max(), vpd_h=g.vpd.mean(), p=g.P.sum()))
    out['dir_from'] = np.degrees(np.arctan2(out.ue, out.vn)) % 360.0
    if m100 is not None:
        out['m100'] = g.m100.mean()
    return out[out.n >= 20]


def to_gridmet_like(hd, cal, rng=None):
    """Map HRRR daily stats -> gridMET-like daily features with the dev-fitted operator (+ residual noise if rng)."""
    rng = np.random.default_rng(rng) if rng is not None else None
    out = pd.DataFrame(index=hd.index)

    def lin(name, x, lo=None, hi=None, log=False):
        c = cal[name]
        xx = np.log(np.maximum(x, 1e-3)) if log else x
        y = c['a'] + c['b'] * xx
        if rng is not None:
            y = y + rng.normal(0, c['sd'], len(y))
        if log:
            y = np.exp(y)
        return np.clip(y, lo, hi)
    out['vs'] = lin('vs', hd.ws_mean.to_numpy(), 0.0, 40.0, log=True)
    noise = rng.normal(0, cal['th']['sd_deg'], len(hd)) if rng is not None else 0.0
    out['th'] = (hd.dir_from.to_numpy() + cal['th']['offset_deg'] + noise) % 360.0
    out['rmin'] = lin('rmin', hd.rhmin.to_numpy(), 1.0, 100.0)
    out['tmmx'] = lin('tmmx', hd.tmax.to_numpy(), -40, 50)
    out['vpd'] = lin('vpd', hd.vpd_h.to_numpy(), 0.0, 10.0)
    pr = hd.p.to_numpy() * cal['pr']['b']
    out['pr'] = np.maximum(pr, 0.0)
    if 'm100' in hd:
        out['fm100'] = lin('fm100', hd.m100.to_numpy(), 1.0, 40.0)
    return out


def feature_vector(d0, d1, gamma_deg):
    """Network scalar weather inputs from two gridMET-like day rows (dict-like with DAY_KEYS). gamma_deg = grid
    convergence (grid = true + gamma). Spread-toward vector in grid coords: (east, north)."""
    f = []
    for d in (d0, d1):
        vs = float(d['vs']) if np.isfinite(d['vs']) else 3.0
        th = float(d['th']) if np.isfinite(d['th']) else 0.0
        tow = np.radians(th + 180.0 + gamma_deg)
        f += [vs * np.sin(tow) / 5.0, vs * np.cos(tow) / 5.0, vs / 5.0,
              (_nz(d['rmin'], 20.0) - 20.0) / 15.0, (_nz(d['tmmx'], 28.0) - 28.0) / 8.0, (_nz(d['vpd'], 2.5) - 2.5) / 1.5,
              np.log1p(max(_nz(d['pr'], 0.0), 0.0)), (_nz(d['fm100'], 9.0) - 9.0) / 4.0]
    return np.array(f, np.float32)


def _nz(v, default):
    try:
        v = float(v)
    except Exception:
        return default
    return v if np.isfinite(v) else default


N_WX = 16
