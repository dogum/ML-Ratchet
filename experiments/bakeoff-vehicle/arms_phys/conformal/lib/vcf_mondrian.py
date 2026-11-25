"""Mondrian split-conformal layer for Vehicle Bake-off arms (agent C). numpy only.

STRATA (Mondrian; fixed from the inputs before looking at any residual): powertrain x regime.
  powertrain  ice (conventional) | hev (the three hybrid systems) | phev (plug-ins) | bev       <- batch 'arch'
  regime      hd (bus / trucks, cls >= 4) | dyno (light-duty dynamometer test, V[46] = X[:, 148] = spec[:, 46]) | road
Dev units per stratum (EnergyNet's scoring units, score.real_units): ice/road 645 VED trips, hev/road 275, phev/road
318, bev/road 258, ice/dyno ~6,600 EPA test-vehicle units, hev/dyno ~230, bev/dyno ~195, ice/hd 115 (Mersin trips +
ICCT tests), bev/hd 5 ZTBus legs. A stratum with fewer than MIN_UNITS dev units falls back to the regime pooled over
powertrains, then to the powertrain pooled over regimes, then to all units (the fallback is recorded in calib.json).

SCORES in log space, per band (90 %: q05-q95, 50 %: q25-q75) and endpoint (asymmetric CQR, Romano et al. 2019):
    s_hi = log y - log q_hi,      s_lo = log q_lo - log y
SHIFT d = the ceil((n + 1)(1 - a/2)) / n empirical quantile of the stratum's scores (a = 0.10 or 0.50), every dev unit
weighted equally. APPLIED to every record of the stratum:  q_hi' = q_hi e^{d_hi},  q_lo' = q_lo e^{-d_lo}, then made
monotone (q05 <= q25 <= q50 <= q75 <= q95). pred and q50 are untouched (MdAPE / bias are the wrapped arm's).
A shift can be negative (the band narrows): conformal calibration, not widen-only.
Multiplicative shifts commute with the scorer's unit aggregation (VED valid-share scaling, EPA cold/hot FTP weighting,
Mersin route-draw means), so fitting on units and applying per record is exact.
"""
import numpy as np

PT = np.array(['ice', 'hev', 'hev', 'hev', 'phev', 'phev', 'bev'])
POWERTRAINS = ['ice', 'hev', 'phev', 'bev']
REGIMES = ['road', 'dyno', 'hd']
STRATA = [f'{p}/{r}' for p in POWERTRAINS for r in REGIMES]
BANDS = {'b90': ('q05', 'q95', 0.10), 'b50': ('q25', 'q75', 0.50)}
QK = ['q05', 'q25', 'q50', 'q75', 'q95']
MIN_UNITS = 20
# dev unit set -> stratum (the scorer's set keys; checked against the input-derived strata of every dev record)
SET_STRATUM = {'VED_ICE': 'ice/road', 'VED_HEV': 'hev/road', 'VED_EV': 'bev/road', 'VED_PHEV': 'phev/road',
               'Mersin_diesel': 'ice/hd', 'Mersin_cng': 'ice/hd', 'ICCT': 'ice/hd', 'ZTBus_legs': 'bev/hd'}
for _t in ('FTP', 'HWY', 'US06', 'FTPcold', 'SC03'):
    SET_STRATUM[f'EPA_conv_{_t}'] = 'ice/dyno'; SET_STRATUM[f'EPA_hev_{_t}'] = 'hev/dyno'
    SET_STRATUM[f'EPA_bev_{_t}'] = 'bev/dyno'


def strata_of(batch):
    """per-row stratum key from a table or trace batch."""
    a = np.asarray(batch['arch'], int); c = np.asarray(batch['cls'], int)
    if batch.get('track') == 'trace' or 'X' not in batch:
        dy = np.asarray(batch['spec'], np.float64)[:, 46] > 0.5
    else:
        dy = np.asarray(batch['X'], np.float64)[:, 148] > 0.5
    r = np.where(c >= 4, 'hd', np.where(dy, 'dyno', 'road'))
    return np.array([f'{p}/{q}' for p, q in zip(PT[a], r)])


def scores(y, Q):
    """y (n,), Q (n, 5) -> dict endpoint -> (n,) log-space conformity scores."""
    ly = np.log(np.asarray(y, float)); lQ = np.log(np.asarray(Q, float))
    out = {}
    for b, (lo, hi, a) in BANDS.items():
        out[b + '_lo'] = lQ[:, QK.index(lo)] - ly
        out[b + '_hi'] = ly - lQ[:, QK.index(hi)]
    return out


def cq(x, level):
    """finite-sample conformal quantile: the ceil((n + 1) level)-th smallest score (the max if that exceeds n)."""
    x = np.sort(np.asarray(x, float)); n = len(x)
    k = int(np.ceil((n + 1) * level))
    return float(x[min(max(k, 1), n) - 1])


def fit(units):
    """units: dict stratum -> (y (n,), Q (n, 5)). -> {stratum: {endpoint: shift, n_units, source}} for every STRATUM."""
    sc = {s: scores(*units[s]) for s in units if len(units[s][0])}
    n = {s: len(units[s][0]) for s in sc}

    def pooled(keys):
        keys = [k for k in keys if k in sc]
        if not keys:
            return None, 0
        return {e: np.concatenate([sc[k][e] for k in keys]) for e in sc[keys[0]]}, sum(n[k] for k in keys)

    out = {}
    for s in STRATA:
        p, r = s.split('/')
        cands = [([s], s), ([f'{q}/{r}' for q in POWERTRAINS], f'*/{r}'), ([f'{p}/{q}' for q in REGIMES], f'{p}/*'),
                 (list(sc), '*/*')]
        for keys, name in cands:
            S, nn = pooled(keys)
            if S is not None and nn >= MIN_UNITS:
                break
        rec = {'n_units': int(nn), 'source': name, 'n_own': int(n.get(s, 0))}
        for b, (lo, hi, a) in BANDS.items():
            for side in ('lo', 'hi'):
                rec[f'{b}_{side}'] = cq(S[f'{b}_{side}'], 1 - a / 2)
        out[s] = rec
    return out


def apply(o, strata, shifts):
    """o: arm output dict; strata: (n,) keys; shifts: fit() output. -> new output dict (pred, q50 unchanged)."""
    out = {k: (np.asarray(v, float).copy() if k != 'components' else v) for k, v in o.items()}
    Q = np.sort(np.stack([np.asarray(o[k], float) for k in QK], 1), 1)
    for j, k in enumerate(QK):
        out[k] = Q[:, j]
    for b, (lo, hi, a) in BANDS.items():
        dlo = np.array([shifts[s][f'{b}_lo'] for s in strata]); dhi = np.array([shifts[s][f'{b}_hi'] for s in strata])
        out[lo] = out[lo] * np.exp(-dlo)
        out[hi] = out[hi] * np.exp(dhi)
    q50 = out['q50']
    out['q25'] = np.minimum(out['q25'], q50); out['q75'] = np.maximum(out['q75'], q50)
    out['q05'] = np.minimum(out['q05'], out['q25']); out['q95'] = np.maximum(out['q95'], out['q75'])
    return out
