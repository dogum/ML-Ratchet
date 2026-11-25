"""Common scoring for the Vehicle Bake-off (every arm and every reference row goes through these functions).

Per scored unit (EnergyNet's definitions, training/evaluate.py and exam2.py):
  ape = |pred / y - 1|, sgn = pred / y - 1, logerr = log(pred / y)
  in50 / in90  y inside [q25, q75] / [q05, q95]
  crps_log     5-quantile CRPS approximation (2 x mean pinball loss) on log energy: unit-free, comparable across sets;
               for a point forecast it is |log(pred / y)|
Summaries: MdAPE (median ape), MAPE, bias (median sgn), cov50, cov90, crps_log (mean), width90_log (median
log(q95 / q05)); medians over units. Comparisons: paired bootstrap over clusters (VED: vehicles; ZTBus legs:
missions; EPA: test vehicles; Mersin: trips) of the difference in MdAPE and in mean crps_log (10,000 resamples,
default_rng(config.SEED)); a win or loss only when the 95 % interval excludes 0.

Units of the real sets (real_units): see HARNESS.md section 7.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT_KEYS = ['pred'] + C.QKEYS
OPT_KEYS = ['components']                    # optional (n, 5) MJ: aero, rolling, friction brakes, aux/HVAC, losses
QZ = np.array([-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722])
FLOOR = {'ICE': 1 * C.LHV, 'HEV': 1 * C.LHV, 'EV': 0.01 * 3.6, 'PHEV': 0.05}   # EnergyNet's VED label floors (MJ)
GAL = 3.785411784
D16 = (['VED_ICE', 'VED_HEV', 'VED_EV', 'VED_PHEV'] +
       ['EPA_conv_FTP', 'EPA_conv_HWY', 'EPA_conv_US06', 'EPA_conv_FTPcold', 'EPA_conv_SC03', 'EPA_hev_FTP', 'EPA_hev_HWY',
        'EPA_bev_FTP', 'EPA_bev_HWY'] + ['Mersin_diesel', 'Mersin_cng', 'ICCT'])


class OutputError(ValueError):
    pass


def check_output(out, n):
    """Validate an arm's output -> pred (n,), Q (n, 5) sorted, n_cross, components (n, 5) or None."""
    if not isinstance(out, dict):
        raise OutputError('predict must return a dict')
    miss = [k for k in OUT_KEYS if k not in out]
    if miss:
        raise OutputError(f'missing keys {miss}')
    arr = {}
    for k in OUT_KEYS:
        a = np.asarray(out[k], np.float64).reshape(-1)
        if a.shape != (n,):
            raise OutputError(f'{k} has shape {a.shape}, expected ({n},)')
        if not np.isfinite(a).all():
            raise OutputError(f'{k} has {int((~np.isfinite(a)).sum())} non-finite values')
        if (a <= 0).any():
            raise OutputError(f'{k} has {int((a <= 0).sum())} values <= 0 (energy in MJ must be > 0)')
        arr[k] = a
    Q = np.stack([arr[k] for k in C.QKEYS], 1)
    n_cross = int((np.diff(Q, axis=1) < -1e-12).any(1).sum())
    if n_cross:
        Q = np.sort(Q, 1)
    comp = None
    if out.get('components') is not None:
        comp = np.asarray(out['components'], np.float64)
        if comp.shape != (n, 5) or not np.isfinite(comp).all():
            raise OutputError(f'components must be finite (n, 5), got {comp.shape}')
    return arr['pred'], Q, n_cross, comp


def lognormal_quantiles(pred, sigma_log):
    """quantiles of a log-normal band around pred (the median) with log-sd sigma_log -> (n, 5)."""
    return np.asarray(pred, float)[:, None] * np.exp(QZ[None, :] * np.asarray(sigma_log, float)[:, None])


def row_scores(y, pred, Q):
    """per-unit scores (arrays)."""
    y = np.asarray(y, float); pred = np.asarray(pred, float); Q = np.asarray(Q, float)
    ly, lQ = np.log(y), np.log(Q)
    e = ly[:, None] - lQ
    t = np.asarray(C.QLEVELS)[None, :]
    crps = 2.0 * np.maximum(t * e, (t - 1) * e).mean(1)
    return dict(ape=np.abs(pred / y - 1), sgn=pred / y - 1, logerr=np.log(pred / y),
                in50=(y >= Q[:, 1]) & (y <= Q[:, 3]), in90=(y >= Q[:, 0]) & (y <= Q[:, 4]),
                crps_log=crps, width90_log=lQ[:, 4] - lQ[:, 0])


def summarize(u):
    """u: unit table with y, pred, q05..q95 (and optional 'point' flag) -> summary dict."""
    if len(u) == 0:
        return {'n': 0}
    Q = u[C.QKEYS].values
    s = row_scores(u.y.values, u.pred.values, Q)
    point = bool(np.allclose(Q[:, 0], Q[:, 4]))
    out = dict(n=int(len(u)), MdAPE=float(np.median(s['ape'])), MAPE=float(np.mean(s['ape'])),
               bias=float(np.median(s['sgn'])), med_log_err=float(np.median(np.abs(s['logerr']))),
               crps_log=float(np.mean(s['crps_log'])), point=point)
    if 'cluster' in u:
        out['n_clusters'] = int(u.cluster.nunique())
    if not point:
        out.update(cov50=float(np.mean(s['in50'])), cov90=float(np.mean(s['in90'])),
                   width90_log=float(np.median(s['width90_log'])))
    return out


# ============================================================================ real units
def _vr_scale(F):
    F = F.copy()
    for k in OUT_KEYS:
        F[k] = F[k] * F['vr']
    return F


def real_units(F):
    """F: real frame (bake.real_frame) joined with an arm's per-record pred, q05..q95 (MJ, whole trip).
    -> {set key: unit table (unit, cluster, y, pred, q05..q95, strata ...)}; every ratio pred / y is unit-free."""
    U = {}
    v = _vr_scale(F[F.recset == 'ved'])
    if len(v):
        v = v[v.y > v.group.map(FLOOR)]
        v = v.assign(unit=v.trip_id, cluster=v.vehicle_id, v_kmh=v.v_mean * 3.6,
                     climb_per_km=v.climb / np.maximum(v.dist_km, 0.1))
        for g, d in v.groupby('group'):
            U[f'VED_{g}'] = d
            c = d.groupby('vehicle_id')[['y'] + OUT_KEYS].sum().reset_index()
            U[f'VED_{g}_cars'] = c.assign(unit=c.vehicle_id, cluster=c.vehicle_id)
        for a, d in v.groupby('arch'):
            U[f'VED_arch_{a}'] = d
    e = F[F.recset == 'epa']
    if len(e):
        e = e.assign(miles=e.dist_m / C.MI)
        rows = []
        for (vid, test), d in e.groupby(['vehicle_id', 'test'], sort=True):
            arch = d.arch.iloc[0]
            per = d[OUT_KEYS].values / d.miles.values[:, None]
            if arch != 'bev' and test in ('FTP', 'FTPcold') and len(d) == 2:
                w = np.where(d.cold.values, 0.43, 0.57)
                per = (w[:, None] * per).sum(0)
            else:
                per = per[0]
            if arch == 'bev':
                pr = per / 3.6 / 0.88; yv = float(d.y.iloc[0])                 # wall kWh / mi
            else:
                pr = per / C.LHV / 745.0 / GAL; yv = 1.0 / float(d.y.iloc[0])  # gal / mi
            r = {'unit': f'{vid}|{test}', 'cluster': vid, 'vehicle_id': vid, 'family': d.family.iloc[0], 'arch': arch,
                 'test': test, 'y': yv, 'model_year': d.model_year.iloc[0]}
            r.update(dict(zip(OUT_KEYS, pr)))
            rows.append(r)
        o = pd.DataFrame(rows)
        for (a, t), d in o.groupby(['arch', 'test']):
            U[f"EPA_{'hev' if a.startswith('hev') else a}_{t}"] = d
    h = F[F.recset == 'hd']
    if len(h):
        z = h[h.set == 'ZTBus_leg']
        if len(z):
            z = z.assign(unit=z.trip_id, cluster=z.mission)
            U['ZTBus_legs'] = z
            m = z.groupby('mission')[['y'] + OUT_KEYS].sum().reset_index()
            U['ZTBus_missions'] = m.assign(unit=m.mission, cluster=m.mission)
        ms = h[h.set == 'Mersin']
        for fuel, d in ms.groupby('fuel'):
            g = d.groupby('trip_id').agg(y=('y', 'first'), **{k: (k, 'mean') for k in OUT_KEYS}).reset_index()
            cv = C.LHV / C.CNG_LHV if fuel == 'cng' else 1.0      # predicted MJ -> grams at the fuel's LHV
            for k in OUT_KEYS:
                g[k] = g[k] * cv
            U[f'Mersin_{fuel}'] = g.assign(unit=g.trip_id, cluster=g.trip_id)
        t = h[h.set == 'ICCT']
        if len(t):
            U['ICCT'] = t.assign(unit=t.trip_id, cluster=t.trip_id)
    return U


def conditions(u):
    """VED condition strata (from inputs only): city (mean speed < 40 km/h), highway (> 55 km/h), cold (mean
    T < 0 C), hot (> 25 C), hilly (climb >= 15 m per km), short (< 2 km), long (> 15 km)."""
    if 'v_kmh' not in u:
        return {}
    return {'city': u[u.v_kmh < 40], 'highway': u[u.v_kmh > 55], 'cold': u[u['T'] < 0], 'hot': u[u['T'] > 25],
            'hilly': u[u.climb_per_km >= 15], 'short': u[u.dist_km < 2], 'long': u[u.dist_km > 15]}


def real_summary(U, with_conditions=True):
    out = {}
    for k, u in U.items():
        out[k] = summarize(u)
        if with_conditions and k in ('VED_ICE', 'VED_HEV', 'VED_EV', 'VED_PHEV'):
            out[k]['conditions'] = {c: summarize(d) for c, d in conditions(u).items()}
    return out


def d16(S):
    """the bake-off dev score: mean of the 16 dev MdAPEs of EnergyNet's D2 available on the local data (D2 minus the
    full-ZTBus legs and missions, whose traces are not here). EnergyNet v2.1: 0.1133."""
    vals = [S[k]['MdAPE'] for k in D16 if k in S and S[k].get('n')]
    return float(np.mean(vals)) if len(vals) == len(D16) else float('nan')


# ============================================================================ sim val
def simval_summary(y, pred, Q, meta, dumb=None):
    """frozen sim val: y, pred, Q over the 3,985 scored val trips; meta = bake.simval_meta(); dumb = bake.dumb_rates()."""
    s = row_scores(y, pred, Q)
    out = dict(n=int(len(y)), MdAPE=float(np.median(s['ape'])), med_log_err=float(np.median(np.abs(s['logerr']))),
               bias=float(np.median(s['sgn'])), cov50=float(np.mean(s['in50'])), cov90=float(np.mean(s['in90'])),
               crps_log=float(np.mean(s['crps_log'])))
    if dumb is not None:
        k = meta.arch.values * 10 + meta.cls.values
        pdumb = np.array([dumb.get(int(g), np.nan) for g in k]) * meta.dist.values
        pdumb = np.where(np.isfinite(pdumb) & (pdumb > 0), pdumb, np.nan)
        out['skill'] = float(1 - out['med_log_err'] / np.nanmedian(np.abs(np.log(pdumb / y))))
    import features as F
    out['by_arch'] = {F.ARCHS[a]: float(np.median(s['ape'][meta.arch.values == a])) for a in range(7) if (meta.arch.values == a).any()}
    out['by_class'] = {F.CLASSES[c]: float(np.median(s['ape'][meta.cls.values == c])) for c in range(7) if (meta.cls.values == c).any()}
    out['by_source'] = {['ved', 'synth', 'dyno', 'cyc_road'][q]: float(np.median(s['ape'][meta.src.values == q])) for q in range(4)
                        if (meta.src.values == q).any()}
    return out


# ============================================================================ paired bootstrap
def compare(ua, ub, rng=None, nboot=C.NBOOT):
    """paired bootstrap over clusters of (MdAPE_a - MdAPE_b) and (mean crps_log_a - mean crps_log_b), on the units
    both rows scored. Negative = a is better."""
    rng = rng if rng is not None else np.random.default_rng(C.SEED)
    a = ua.set_index('unit'); b = ub.set_index('unit')
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return {'n': int(len(common))}
    a, b = a.loc[common], b.loc[common]
    sa = row_scores(a.y.values, a.pred.values, a[C.QKEYS].values)
    sb = row_scores(b.y.values, b.pred.values, b[C.QKEYS].values)
    cl = a['cluster'].values if 'cluster' in a else np.arange(len(a))
    codes, inv = np.unique(cl, return_inverse=True)
    order = np.argsort(inv, kind='stable'); starts = np.searchsorted(inv[order], np.arange(len(codes) + 1))
    groups = [order[starts[i]:starts[i + 1]] for i in range(len(codes))]
    d_md = float(np.median(sa['ape']) - np.median(sb['ape']))
    d_cr = float(np.mean(sa['crps_log']) - np.mean(sb['crps_log']))
    G = len(groups)
    bm = np.empty(nboot); bc = np.empty(nboot)
    single = all(len(g) == 1 for g in groups)
    for k in range(nboot):
        pick = rng.integers(0, G, G)
        ix = pick if single else np.concatenate([groups[j] for j in pick])
        bm[k] = np.median(sa['ape'][ix]) - np.median(sb['ape'][ix])
        bc[k] = np.mean(sa['crps_log'][ix]) - np.mean(sb['crps_log'][ix])
    lo, hi = np.percentile(bm, [2.5, 97.5]); clo, chi = np.percentile(bc, [2.5, 97.5])
    verdict = lambda l, h: 'win' if h < 0 else ('loss' if l > 0 else 'tie')
    return dict(n=int(len(common)), n_clusters=int(G), d_MdAPE=d_md, ci_MdAPE=[float(lo), float(hi)],
                verdict_MdAPE=verdict(lo, hi), d_crps_log=d_cr, ci_crps_log=[float(clo), float(chi)],
                verdict_crps_log=verdict(clo, chi))
