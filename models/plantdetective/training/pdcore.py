"""PlantDetective core: truth statistics (same code for simulated and real units), the model-input series, and the
fixed-length feature vector.  numpy only (vendored into the release).

Record (full information, CAMPD-like), hourly arrays:  op [0..1 h operating], P [gross MW while operating],
HI [MMBtu heat input], meas [heat input measured].  Model input: x = P * op rounded to MW (hourly-averaged output),
NaN = missing hour.  The model never sees op, P or HI separately.
"""
import numpy as np

TECHS = ['coal_steam', 'gas_cc', 'gas_ct', 'gas_steam']
PART_MIN = 0.4                 # part-load heat rate is read at max(min stable load, 40 % of pmax)
HOT_H, WARM_H = 8, 48          # start type by preceding off time: hot < 8 h, warm 8-48 h, cold >= 48 h (our convention)
REG = ['pmin_frac', 'ramp_up', 'ramp_dn', 'starts_yr', 'min_up', 'med_up', 'min_dn', 'hot', 'warm', 'cold',
       'hr_full', 'hr_min', 'cf']


def runs(on):
    on = np.asarray(on, bool)
    if len(on) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, bool)
    idx = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    s = np.r_[0, idx]; e = np.r_[idx, len(on)]
    return s, e - s, on[s]


def _quad_fit(x, y, pmax):
    """Robust fuel curve F(P) on steady hours: quadratic if the load spread allows, else linear, else constant HR."""
    spread = (np.percentile(x, 95) - np.percentile(x, 5)) / pmax
    deg = 2 if spread > 0.3 else (1 if spread > 0.08 else 0)
    for _ in range(3):
        if deg == 0:
            hr = np.median(y / x); coef = np.array([0.0, hr, 0.0])
        else:
            A = np.vander(x, deg + 1, increasing=True)
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            coef = np.r_[c, np.zeros(3 - len(c))]
        res = y - (coef[0] + coef[1] * x + coef[2] * x * x)
        mad = np.median(np.abs(res - np.median(res))) * 1.4826 + 1e-9
        k = np.abs(res) < 4 * mad
        if k.all():
            break
        x, y = x[k], y[k]
    return coef, deg, len(x)


def truth(op, P, HI=None, meas=None, hours_per_year=8760.0):
    """Observed operating character of one unit from its full-information record (dict; NaN where undefined)."""
    op = np.nan_to_num(np.asarray(op, float)); P = np.nan_to_num(np.asarray(P, float))
    n = len(op)
    r = {k: np.nan for k in REG + ['pmax', 'n_hr_fit', 'hr_deg', 'n_starts', 'hours']}
    r['hours'] = n
    on = (op > 0) & (P > 0)
    full = (op >= 0.999) & (P > 0)
    steady = full & np.r_[False, full[:-1]] & np.r_[full[1:], False]
    if full.sum() < 24:
        return r
    pmax = np.percentile(P[full], 99.5)
    r['pmax'] = pmax
    r['cf'] = float((P * op).sum() / (pmax * n))
    if steady.sum() >= 24:
        r['pmin_frac'] = float(np.percentile(P[steady], 2) / pmax)
        both = steady[1:] & steady[:-1]
        if both.sum() >= 24:
            dP = np.diff(P)[both] / pmax
            r['ramp_up'] = float(max(np.percentile(dP, 99), 0.0))
            r['ramp_dn'] = float(max(-np.percentile(dP, 1), 0.0))
    s, L, v = runs(on)
    if len(s):
        inner = np.ones(len(s), bool); inner[0] = False; inner[-1] = False
        up = L[inner & v]; dn = L[inner & ~v]
        st = np.flatnonzero(v & (s > 0))                      # starts with a known preceding off spell
        r['n_starts'] = int(len(st))
        r['starts_yr'] = float(len(st) * hours_per_year / n)
        if len(up) >= 3:
            r['min_up'] = float(np.percentile(up, 5)); r['med_up'] = float(np.median(up))
        if len(dn) >= 3:
            r['min_dn'] = float(np.percentile(dn, 5))
        prev_off = np.array([L[i - 1] for i in st if i >= 1 and (i - 1 > 0)])   # skip a censored first off spell
        if len(prev_off) >= 3:
            r['hot'] = float(np.mean(prev_off < HOT_H)); r['warm'] = float(np.mean((prev_off >= HOT_H) & (prev_off < WARM_H)))
            r['cold'] = float(np.mean(prev_off >= WARM_H))
    if HI is not None:
        HI = np.asarray(HI, float)
        m = steady & np.isfinite(HI) & (HI > 0)
        if meas is not None:
            m &= np.asarray(meas, bool)
        if m.sum() >= 48:
            coef, deg, nf = _quad_fit(P[m], HI[m], pmax)
            F = lambda p: coef[0] + coef[1] * p + coef[2] * p * p  # noqa
            pmin = (r['pmin_frac'] if np.isfinite(r['pmin_frac']) else 0.5) * pmax
            lo = max(pmin, PART_MIN * pmax, np.percentile(P[m], 1))   # part-load point: min stable load, >= 40 %
            r['f_part'] = float(lo / pmax)
            r['hr_full'] = float(F(pmax) / pmax); r['hr_min'] = float(F(lo) / lo)
            r['n_hr_fit'] = nf; r['hr_deg'] = deg; r['hr_coef'] = coef.tolist()
    return r


def flex_class(starts_yr, cf, med_up):
    """Operating-flexibility class from the observed character (rule, not learned)."""
    if not np.isfinite(starts_yr) or not np.isfinite(cf):
        return 'unknown'
    if cf < 0.15 and (not np.isfinite(med_up) or med_up < 12):
        return 'peaking'
    if starts_yr >= 50:
        return 'cycling'
    if starts_yr >= 12:
        return 'load-following'
    return 'baseload'


def model_input(op, P):
    return np.round(np.nan_to_num(np.asarray(P, float)) * np.nan_to_num(np.asarray(op, float)))


# ----------------------------------------------------------------------------- features
RUN_EDGES = np.array([1, 2, 3, 4, 7, 13, 25, 49, 97, 193, 385, 1e9])
ZBINS = np.array([-1e9, 4, 6, 7, 8, 9, 10, 11, 12, 14, 17, 25, 1e9])     # market implied heat rate, MMBtu/MWh
LOADBINS = np.linspace(0, 1.1, 23)
CTX_KEYS = ['pm_ct', 'pm_st', 'pm_ca', 'fuel_coal', 'chp', 'lnp_own', 'lnp_st', 'r_own', 'r_blk', 'minload_frac',
            'minload_has', 'cold_lh', 'cold_has', 'hr923', 'hr923_has', 'age']


def _bridge(on, miss):
    """Run detection treats a missing stretch with running hours on both sides as running. Read as zeros, one missing
    hour inside a run became a stop and a start (audit, 25 Nov 2026: hiding one interior hour on 362 days of 6179_3 took
    its predicted starts from 2.76 to 107.58 a year). Runs that contain missing hours are still left out of the
    run-length statistics below."""
    on = on.copy()
    if miss.any():
        d = np.diff(np.r_[0, miss.astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if a > 0 and b < len(on) and on[a - 1] and on[b]:
                on[a:b] = True
    return on


def _q(a, qs):
    return np.percentile(a, qs) if len(a) else np.full(len(qs), np.nan)


def features(x, hour_local=None, dow=None, price=None, gas=None, ctx=None):
    """Fixed-length feature vector from one window of hourly output x (MW, NaN = missing).

    hour_local / dow: local hour (0-23) and weekday (0-6) arrays; price ($/MWh) and gas ($/MMBtu) hourly arrays or None;
    ctx: dict of context values (CTX_KEYS, already normalised by context_vector) or None.  Returns (vector, info)."""
    x = np.asarray(x, float)
    n = len(x)
    miss = ~np.isfinite(x)
    xv = np.where(miss, 0.0, x)
    on = xv > 0.5
    known = ~miss
    f = []
    # interior on-hours (both neighbours on and known) carry the steady load level
    inter = on & np.r_[False, on[:-1]] & np.r_[on[1:], False]
    base = xv[inter] if inter.sum() >= 12 else xv[on]
    pcap = np.percentile(base, 99.5) if len(base) else 1.0
    pcap = max(pcap, 1.0)
    y = xv / pcap
    f += [np.log(n / 720.0), miss.mean(), on[known].mean() if known.any() else 0.0, np.log(pcap / 100.0),
          float(inter.sum() < 12)]
    yi = y[inter] if inter.sum() else np.zeros(1)
    h, _ = np.histogram(np.clip(yi, 0, 1.0999), LOADBINS); f += list(h / max(h.sum(), 1))
    f += list(np.nan_to_num(_q(yi, [0.5, 1, 2, 5, 10, 25, 50, 75, 90, 99]), nan=0.0))
    f += [yi.std() if len(yi) > 1 else 0.0, np.mean(np.abs(yi - 1) < 0.03) if len(yi) else 0.0]
    # first / second / last hour of each run (hourly averaging of starts and stops)
    s, L, v = runs(_bridge(on, miss))
    st = s[v & (s > 0)]
    first = y[st] if len(st) else np.zeros(0)
    second = y[st[(st + 1) < n] + 1] if len(st) else np.zeros(0)
    ends = (s + L - 1)[v & (s + L < n)]
    last = y[ends] if len(ends) else np.zeros(0)
    for a in (first, second, last):
        f += list(np.nan_to_num(_q(a, [10, 50, 90]), nan=-1.0))
    # ramps between interior hours
    both = inter[1:] & inter[:-1]
    dy = np.diff(y)[both] if both.sum() else np.zeros(1)
    f += list(np.nan_to_num(_q(dy, [0.1, 1, 5, 25, 75, 95, 99, 99.9]), nan=0.0))
    f += [np.mean(np.abs(dy) < 0.01), np.mean(np.abs(dy) > 0.1), np.log1p(np.abs(dy).mean() * 100)]
    # run and off-spell lengths (complete spells; censored by window edges or missing hours dropped)
    good = np.ones(len(s), bool)
    if len(s):
        good[0] = False; good[-1] = False
        mm = np.r_[0, np.cumsum(miss)]
        good &= (mm[s + L] - mm[s]) == 0
    up = L[good & v]; dn = L[good & ~v]
    nk = max(known.sum(), 1)
    for a in (up, dn):
        h, _ = np.histogram(a, RUN_EDGES); f += list(h * 1000.0 / nk)
        f += [np.log(np.median(a)) if len(a) else -1.0, np.log(np.percentile(a, 5)) if len(a) >= 3 else -1.0,
              float(len(a) >= 3)]
    nst = len(st)
    f += [np.log1p(nst * 8760.0 / nk), np.log1p(nst)]
    prev = np.array([L[i - 1] for i in np.flatnonzero(v & (s > 0)) if i - 1 > 0])
    if len(prev):
        f += [np.mean(prev < HOT_H), np.mean((prev >= HOT_H) & (prev < WARM_H)), np.mean(prev >= WARM_H), 1.0]
    else:
        f += [0.0, 0.0, 0.0, 0.0]
    f += [xv.sum() / (pcap * nk)]
    # diurnal and weekly pattern
    if hour_local is not None:
        hb = np.asarray(hour_local) // 3
        for b in range(8):
            m = known & (hb == b)
            f += [on[m].mean() if m.any() else 0.0]
            m2 = m & inter
            f += [y[m2].mean() if m2.any() else 0.0]
        we = np.isin(np.asarray(dow), [5, 6])
        f += [on[known & we].mean() if (known & we).any() else 0.0, on[known & ~we].mean() if (known & ~we).any() else 0.0]
    else:
        f += [0.0] * 18
    # price context: on-probability and load level against the market implied heat rate (price / gas)
    if price is not None and gas is not None:
        z = np.asarray(price, float) / np.maximum(np.asarray(gas, float), 0.5)
        ok = known & np.isfinite(z)
        zb = np.digitize(z, ZBINS[1:-1])
        for b in range(len(ZBINS) - 1):
            m = ok & (zb == b)
            frac = m.mean()
            f += [frac, on[m].mean() if m.sum() >= 5 else -1.0, y[m & inter].mean() if (m & inter).sum() >= 5 else -1.0]
        zo = z[ok & on]; zf = z[ok & ~on]
        f += [np.log(np.clip(np.median(zo), 1, 100)) if len(zo) else 0.0,
              np.log(np.clip(np.median(zf), 1, 100)) if len(zf) else 0.0,
              np.log(np.clip(np.percentile(zo, 10), 1, 100)) if len(zo) >= 10 else 0.0, 1.0]
        # correlation of load level with implied heat rate while on (load following on price)
        mm = ok & inter
        f += [np.corrcoef(z[mm], y[mm])[0, 1] if mm.sum() > 30 and y[mm].std() > 1e-6 and z[mm].std() > 1e-6 else 0.0]
    else:
        f += [0.0] * (3 * (len(ZBINS) - 1) + 5)
    # context
    if ctx is not None:
        f += [float(np.nan_to_num(ctx.get(k, 0.0))) for k in CTX_KEYS] + [1.0]
    else:
        f += [0.0] * len(CTX_KEYS) + [0.0]
    v_ = np.asarray(f, np.float32)
    return np.nan_to_num(v_, nan=0.0, posinf=0.0, neginf=0.0), dict(pcap=pcap)


def context_vector(c, pcap):
    """Normalise a raw context dict (realdata.context or the simulated emulator) for the feature vector."""
    if c is None:
        return None
    np_own = c.get('np_own') or 0.0
    np_st = c.get('np_st_share') or 0.0
    ml = c.get('minload')
    cold = c.get('cold_h')
    hr = c.get('hr923_window')
    return dict(pm_ct=c.get('pm_ct', 0.0), pm_st=c.get('pm_st', 0.0), pm_ca=c.get('pm_ca', 0.0),
                fuel_coal=c.get('fuel_coal', 0.0), chp=c.get('chp') or 0.0,
                lnp_own=np.log(max(np_own, 1.0) / 100.0), lnp_st=np.log1p(np_st / 100.0),
                r_own=pcap / np_own if np_own > 0 else 0.0, r_blk=pcap / (np_own + np_st) if np_own > 0 else 0.0,
                minload_frac=(ml / (np_own + np_st)) if (ml and np_own > 0) else 0.0, minload_has=float(bool(ml)),
                cold_lh=np.log1p(cold) if cold is not None else 0.0, cold_has=float(cold is not None),
                hr923=np.log(hr / 10.0) if hr else 0.0, hr923_has=float(bool(hr)),
                age=((c.get('ref_year') or 2020) - (c.get('online_year') or 2000)) / 30.0)
