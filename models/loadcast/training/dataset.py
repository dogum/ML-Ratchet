"""Units (one building-meter with a history year and a target year), noise injection, and feature rows.

A unit is a dict: id, kind ('sim' | 'real'), site, use, area, yb (shown year built or None), meter, meters,
hvac (shown HVAC class), sched (shown schedule class), y (UTC frame, kW / (A * IREF / 1000), NaN = invalid),
target_year, hist_year. The same code builds rows for simulated and real units (and the release module).

Noise versions for simulated units:
  'white'  3 % white noise (Phase 1 level of realism)
  'real_use' as 'real', but 75 % of residuals come from a dev building of the same use type
  'real'   + a real dev residual series on the same calendar dates (noise_bank.npz), a year-to-year level change drawn
           from dev 2017/2016 ratios (whole year or a step), and real gap patterns
"""
import os, sys, json, glob, functools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C

VARIANTS = {'full': None, 'w8': 8 * 168, 'w2': 2 * 168}
QMIN = 0.1          # floor of the warm-mode scale, in units of A * IREF (1.5 W/m2 electricity)


@functools.lru_cache(maxsize=32)
def site_feats(site):
    wx, cal = C.site_hourly_features(site)
    c = C.site_calendar(site)
    w, m = C.site_weather(site)
    T = w['T_out'].interpolate(limit_direction='both').values
    dcode = ((c['day'] - pd.Timestamp('2016-01-01')) / pd.Timedelta(days=1)).astype(int).values
    Tday = pd.Series(T).groupby(dcode).transform('mean').values
    clim = {y: C.annual_climate(T[c['year'] == y]) for y in (2016, 2017)}
    return dict(wx=wx, cal=cal, T=T, Tday=Tday, T24=wx[:, 1] * 10, dcode=dcode, tow=c['dow'] * 24 + c['hour'],
                hol=c['holiday'], hour=c['hour'], year=c['year'], lidx=c['lidx'], clim=clim, lat=c['lat'])


# ------------------------------------------------------------------ noise
@functools.lru_cache(maxsize=1)
def noise_bank():
    z = np.load(f'{C.ROOT}/data/' + os.environ.get('LC_NOISE_BANK', 'noise_bank.npz'))   # LC_NOISE_BANK: winrun.py noise y16
    use = C.bd.bdg2_meta(split='dev').set_index('id').primary_use
    return dict(E=z['E'].astype(np.float32), meter=z['meter'], bid=z['bid'], ratio=z['ratio'], gaps=z['gaps'],
                use=np.array([use.get(b, 'other') for b in z['bid']]))


def heat_group(m):
    return 'electricity' if m == 'electricity' else ('chilledwater' if m == 'chilledwater' else 'heat')


def apply_noise(y, u, kind, rng):
    """y: clean normalized series on the site frame. Returns a noisy copy."""
    y = y.astype(np.float64).copy()
    ok = np.isfinite(y)
    mu = np.nanmean(y)
    if kind == 'white':
        y[ok] += rng.normal(0, 0.03, ok.sum()) * mu
        return np.clip(y, 0, None)
    nb = noise_bank(); f = site_feats(u['site'])
    g = np.array([heat_group(m) for m in nb['meter']])
    cand = np.flatnonzero((g == heat_group(u['meter'])) & (nb['bid'] != u.get('exclude_bid', '')))
    j = cand[rng.integers(len(cand))]
    if kind == 'real_use':      # prefer a residual from a dev building of the same use (campus calendars by use)
        same = cand[nb['use'][cand] == u['use']]
        if len(same) and rng.random() < 0.75: j = same[rng.integers(len(same))]
    e = nb['E'][j][np.clip(f['lidx'], 0, nb['E'].shape[1] - 1)]
    e = np.nan_to_num(e, nan=0.0)
    y = y + e * mu
    # year-to-year level change in the target year (dev 2017/2016 ratios, either direction)
    r = float(nb['ratio'][cand[rng.integers(len(cand))]])
    r = r if rng.random() < 0.5 else 1.0 / r
    tsel = np.flatnonzero(f['year'] == u['target_year'])
    if rng.random() < 0.5:
        y[tsel] *= r
    else:
        k = rng.integers(len(tsel))
        a = tsel[k:] if u['target_year'] > u['hist_year'] else tsel[:k]
        y[a] *= r
    # occasional meter quirks: quantisation of small meters, a spike, a short flat/zero run in the history
    if rng.random() < 0.1:
        qstep = mu * rng.uniform(0.05, 0.2); y = np.round(y / qstep) * qstep
    gaps = nb['gaps'][j][np.clip(f['lidx'], 0, nb['gaps'].shape[1] - 1)]
    y[gaps] = np.nan
    y[~ok] = np.nan
    return np.clip(y, 0, None)


# ------------------------------------------------------------------ banks -> units
def load_bank(name, noise, seed=0, max_buildings=None):
    """Simulated units of a bank with noise applied (deterministic per seed)."""
    files = sorted(glob.glob(f'{C.ROOT}/data/bank_{name}/part_*.npz'))
    units = []
    for fn in files:
        z = np.load(fn, allow_pickle=False)
        recs = json.loads(str(z['meta']))
        for r in recs:
            if max_buildings and r['i'] >= max_buildings: continue
            hist, targ = (2016, 2017) if r['dir'] == 0 else (2017, 2016)
            for m in r['meters']:
                key = f"{r['i']}_{m}"
                if key not in z.files: continue
                rng = np.random.default_rng([seed, r['i'], C.METERS.index(m), 7])
                hv_true = r['hvac_class']
                shown_hv = hv_true if hv_true.startswith('district') and C.hvac_class_from_meters(r['meters']) != 'unknown' \
                    else (hv_true if rng.random() < 0.3 else 'unknown')
                shown_sc = r['sched_class'] if rng.random() < 0.3 else 'unknown'
                u = dict(id=f"{name}:{r['i']}:{m}", kind='sim', site=r['site'], use=r['use'], area=r['area'],
                         yb=r['year_built'] if r['yb_shown'] else None, meter=m, meters=r['meters'], hvac=shown_hv,
                         sched=shown_sc, target_year=targ, hist_year=hist, hvac_true=hv_true, sched_true=r['sched_class'],
                         knobs=r['knobs'])
                y = z[key].astype(np.float64)
                u['y'] = apply_noise(y, u, noise, rng).astype(np.float32)
                units.append(u)
    return units


def real_units(split, meters=C.METERS, require='q_ok'):
    """Real BDG2 building-meters (q_ok both years by default): history 2016, target 2017 (local years)."""
    meta = C.bd.bdg2_meta(split=split).set_index('id')
    q = C.bd.bdg2_quality(split=split)
    units = []
    for m in meters:
        qq = q[(q.meter == m) & q[require]]
        if not len(qq): continue
        X = C.real_meter(m, list(qq.building_id))
        for b in qq.building_id:
            r = meta.loc[b]; A = float(r['floor_area_m2']); ms = C.meters_of(r['meters'])
            yb = r['year_built'] if np.isfinite(r['year_built'] if r['year_built'] is not None else np.nan) else None
            y = X[b].values.astype(np.float64) / (A * C.IREF[m] / 1000.0)
            units.append(dict(id=f'{b}:{m}', bid=b, kind='real', site=r['site_id'], use=r['primary_use'], area=A, yb=yb,
                              meter=m, meters=ms, hvac=C.hvac_class_from_meters(ms), sched='unknown',
                              target_year=2017, hist_year=2016, cz=r['climate_zone'], y=y.astype(np.float32)))
    return units


# ------------------------------------------------------------------ per-unit preparation and rows
def hist_window(u, variant):
    f = site_feats(u['site'])
    idx = np.flatnonzero(f['year'] == u['hist_year'])
    L = VARIANTS[variant]
    if L is not None:
        idx = idx[-L:] if u['hist_year'] < u['target_year'] else idx[:L]
    return idx


def prepare(u, variants=('full', 'w8')):
    f = site_feats(u['site'])
    u['tidx'] = np.flatnonzero(f['year'] == u['target_year'])
    u['static'] = C.static_features(u['use'], u['area'], u['yb'], u['meter'], u['meters'], u['hvac'], u['sched'],
                                    f['clim'][u['target_year']], f['lat'])
    u['H'] = {}
    for v in variants:
        hi = hist_window(u, v)
        u['H'][v] = C.history_encode(u['y'][hi].astype(float), f['T'][hi], f['tow'][hi], f['dcode'][hi], f['hol'][hi],
                                     f['hour'][hi], 1.0)
    return u


def rows(u, idx, mode, variant='full', wx_override=None):
    """Feature rows for target hours idx (site-frame indices). Returns X, base, y (in the model's units), scale.
    cold: units of A*IREF; warm: units of Q = max(history mean S, QMIN). wx_override: (wx, T, Tday, T24) for what-if weather."""
    f = site_feats(u['site'])
    wx = f['wx'] if wx_override is None else wx_override['wx']
    Xs = np.broadcast_to(u['static'], (len(idx), len(u['static'])))
    X = [Xs, wx[idx], f['cal'][idx]]
    y = u['y'][idx].astype(np.float64)
    if mode == 'cold':
        return np.concatenate(X, 1).astype(np.float32), np.zeros(len(idx), np.float32), y, 1.0
    h = u['H'][variant]
    T = f['T'] if wx_override is None else wx_override['T']
    Tday = f['Tday'] if wx_override is None else wx_override['Tday']
    T24 = f['T24'] if wx_override is None else wx_override['T24']
    hh = C.history_hourly(h, T[idx], f['tow'][idx], Tday[idx], T24[idx])
    X.append(hh)
    Q = max(h['S'], QMIN)      # warm units: history mean, floored so a near-zero winter history cannot blow up
    return np.concatenate(X, 1).astype(np.float32), (hh[:, 0] * h['S'] / Q).astype(np.float32), y / Q, Q


N_COLD = len(C.STATIC_NAMES) + len(C.WX_NAMES) + len(C.CAL_NAMES)
N_WARM = N_COLD + len(C.HIST_NAMES)
