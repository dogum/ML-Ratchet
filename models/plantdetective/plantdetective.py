"""PlantDetective v1.0 -- infer a fossil unit's operating character from its hourly output (numpy only).

    from plantdetective import Detective
    det = Detective()                                    # loads plantdetective_weights.npz next to this file
    r = det.infer(x_MW, t_utc_seconds, price=None, gas=None, context=None)
    r['tech'], r['estimates']['pmin_frac'], r['heat_rate_curve'], r['flex_class']

Inputs
  x        hourly unit output in MW (gross, hourly average; NaN = missing hour). At least 30 days.
  t        UTC hour-beginning timestamps in seconds (same length as x).
  price    optional hourly electricity price ($/MWh) and gas  optional hourly gas price ($/MMBtu) -- both or neither.
  context  optional dict of EIA-860/923 facts (see CONTEXT_HELP); improves heat rate and technology.
  utc_offset  local standard time offset (hours) for the daily pattern (default -6, Texas).
  net      True if x is net output (the model was trained mostly on gross; a flag tells it).

Outputs (each estimate with a 68 % and 95 % interval; intervals widened by dev-fitted factors):
  technology probabilities (coal steam, gas CC, gas CT, gas steam), minimum stable load (fraction of the unit's
  capacity), ramp limits (99th percentile hour-to-hour change while running, fraction of capacity per hour), starts per
  year and their hot/warm/cold shares, minimum up / median run / minimum down times, full-load and part-load heat rate
  (CAMPD gross basis, MMBtu/MWh; part load = max(min stable load, 40 %)), capacity factor, an operating-flexibility
  class, and the heat-rate curve.
"""
import os, json
import numpy as np
from pdcore import features, context_vector, TECHS, REG, PART_MIN, flex_class

HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT_HELP = dict(pm_ct='1 if a linked generator is a combustion turbine (EIA-860 prime mover CT/GT)',
                    pm_st='1 if a steam turbine (ST)', pm_ca='1 if a combined-cycle steam part (CA) is linked',
                    fuel_coal='1 if coal-fired', chp='1 if EIA-860 marks the generator as combined heat and power',
                    np_own='nameplate MW of the unit\'s own generator(s)', np_st_share='MW of a shared CC steam part / sharing units',
                    minload='EIA-860 minimum load MW', cold_h='EIA-860 time from cold shutdown to full load (h)',
                    online_year='first operating year', hr923='{year: EIA-923 fuel for electricity / net generation}')


def np_forward(W, X):
    h = (X - W['mu']) / W['sd']
    i = 0
    while f'l{i}w' in W:
        h = h @ W[f'l{i}w'].T + W[f'l{i}b']
        if f'l{i + 1}w' in W:
            h = np.maximum(h, 0) + 0.01 * np.minimum(h, 0)
        i += 1
    lg = h[:, :4]
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    n = len(REG)
    return p, h[:, 4:4 + n], np.exp(np.clip(h[:, 4 + n:4 + 2 * n], -6, 4))


def natural(v):
    sg = lambda a: 1 / (1 + np.exp(-a))  # noqa
    return dict(pmin_frac=sg(v[0]), ramp_up=np.exp(v[1]), ramp_dn=np.exp(v[2]), starts_yr=max(np.expm1(v[3]), 0.0),
                min_up=np.exp(v[4]), med_up=np.exp(v[5]), min_dn=np.exp(v[6]), hot=sg(v[7]), warm=sg(v[8]),
                cold=sg(v[9]), hr_full=np.exp(v[10]), hr_min=np.exp(v[10] + v[11]), cf=sg(v[12]))


def tech_rule(ctx):
    """Technology likelihood from EIA-860 prime mover / fuel (None if unknown)."""
    if not ctx:
        return None
    if ctx.get('pm_ca'):
        return np.array([0.01, 0.96, 0.02, 0.01])
    if ctx.get('pm_ct'):
        return np.array([0.01, 0.15, 0.83, 0.01])
    if ctx.get('pm_st'):
        return np.array([0.96, 0.01, 0.01, 0.02]) if ctx.get('fuel_coal') else np.array([0.03, 0.01, 0.01, 0.95])
    return None


def pool(p, m, s):
    lp = np.log(np.clip(p, 1e-6, 1)).sum(0); pc = np.exp(lp - lp.max()); pc /= pc.sum()
    w = 1 / s ** 2
    return pc, (w * m).sum(0) / w.sum(0), s.mean(0)


def _local(t, off):
    tl = np.asarray(t, 'int64') + off * 3600
    hour = (tl // 3600) % 24
    dow = ((tl // 86400) + 3) % 7          # 1970-01-01 was a Thursday (Mon = 0)
    year = tl.astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970
    return hour, dow, year


def window_vector(x, t, price, gas, ctx, utc_offset=-6, net=False):
    x = np.asarray(x, float)
    hour, dow, year = _local(t, 0)
    hl, dl, _ = _local(t, utc_offset)
    cv = None
    if ctx is not None:
        yrs = sorted(set(year.tolist()))
        hr = ctx.get('hr923') or {}
        hh = [hr.get(str(y), hr.get(y)) for y in yrs]; hh = [h for h in hh if h]
        c = dict(ctx); c['hr923_window'] = float(np.mean(hh)) if hh else None; c['ref_year'] = float(np.mean(yrs))
        xo = x[np.isfinite(x)]
        cv = context_vector(c, max(np.percentile(xo[xo > 0], 99.5) if (xo > 0).sum() > 10 else 1.0, 1.0))
    v, _ = features(x, hl, dl, price, gas, cv)
    return np.r_[v, float(net)].astype(np.float32)


def ens_forward(Ws, X):
    outs = [np_forward(W, X) for W in Ws]
    p = np.mean([o[0] for o in outs], 0); m = np.mean([o[1] for o in outs], 0)
    v = np.mean([o[2] ** 2 for o in outs], 0) + np.var([o[1] for o in outs], 0)
    return p, m, np.sqrt(v)


class Detective:
    def __init__(self, weights=os.path.join(HERE, 'plantdetective_weights.npz')):
        z = np.load(weights, allow_pickle=False)
        n = int(z['n_members'])
        self.Ws = [{k.split('_', 1)[1]: z[k] for k in z.files if k.startswith(f'm{i}_')} for i in range(n)]
        self.factors = json.loads(str(z['factors'])) if 'factors' in z.files else {k: 1.0 for k in REG}
        self.meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}

    @staticmethod
    def windows(t):
        """Calendar-year windows with >= 300 days; else quarters (>= 80 days); else the whole series (>= 30 days)."""
        _, _, year = _local(t, 0)
        n = len(t)
        out = []
        for y in sorted(set(year.tolist())):
            i = np.flatnonzero(year == y)
            if len(i) >= 300 * 24:
                out.append((int(i[0]), int(len(i))))
        if out:
            return out
        if n >= 80 * 24:
            k = n // (91 * 24) or 1
            L = n // k
            return [(j * L, L) for j in range(k)]
        return [(0, n)] if n >= 30 * 24 else []

    def infer(self, x, t, price=None, gas=None, context=None, utc_offset=-6, net=False, windows=None):
        x = np.asarray(x, float); t = np.asarray(t, 'int64')
        wins = windows or self.windows(t)
        if not wins:
            raise ValueError('need at least 30 days of hourly output')
        X = []
        for s, L in wins:
            sl = slice(s, s + L)
            X.append(window_vector(x[sl], t[sl], None if price is None else np.asarray(price, float)[sl],
                                   None if gas is None else np.asarray(gas, float)[sl], context, utc_offset, net))
        p, m, sg = ens_forward(self.Ws, np.array(X))
        pw = [dict(start=int(t[s]), hours=int(L), tech=dict(zip(TECHS, map(float, p[i]))),
                   estimates={k: float(v) for k, v in natural(m[i]).items()}) for i, (s, L) in enumerate(wins)]
        pc, mm, ss = pool(p, m, sg)
        rule = tech_rule(context)
        tech_net = pc.copy()
        if rule is not None:
            pc = pc * rule; pc /= pc.sum()
        f = np.array([self.factors.get(k, 1.0) for k in REG])
        ss = ss * f
        est = {}
        mid = natural(mm)
        for j, k in enumerate(REG):
            lo1, hi1, lo2, hi2 = (mm.copy() for _ in range(4))
            lo1[j] -= ss[j]; hi1[j] += ss[j]; lo2[j] -= 2 * ss[j]; hi2[j] += 2 * ss[j]
            a, b, c, d = (natural(v)[k] for v in (lo1, hi1, lo2, hi2))
            est[k] = dict(value=float(mid[k]), lo68=float(min(a, b)), hi68=float(max(a, b)), lo95=float(min(c, d)),
                          hi95=float(max(c, d)))
        fp = max(mid['pmin_frac'], PART_MIN)
        beta = (mid['hr_min'] / mid['hr_full'] - 1) / max(1 / fp - 1, 1e-3) if fp < 0.98 else 0.0
        fr = np.linspace(fp, 1.0, 7)
        curve = [[float(a), float(mid['hr_full'] * (1 + beta * (1 / a - 1)))] for a in fr]
        xo = x[np.isfinite(x)]
        pcap = float(np.percentile(xo[xo > 0], 99.5)) if (xo > 0).sum() > 10 else float('nan')
        return dict(tech=dict(zip(TECHS, map(float, pc))), tech_output_only=dict(zip(TECHS, map(float, tech_net))),
                    capacity_MW=pcap, estimates=est, heat_rate_curve=curve,
                    flex_class=flex_class(mid['starts_yr'], mid['cf'], mid['med_up']), windows=pw,
                    inputs=dict(price=price is not None, context=context is not None, net=bool(net), n_windows=len(wins)))
