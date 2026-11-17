"""Evaluation: per building-meter G14 CV(RMSE) / NMBE and 90 % interval coverage, for cold and warm modes.

Frozen simulated set (skill for the ratchet):
  cold skill = 1 - median CV(model) / median CV(dumb), dumb = (use, meter) median intensity x median time-of-week
               shape, both taken from the frozen set itself (a fixed reference per frozen set)
  warm skill = 1 - median CV(model) / median CV(TOWT fitted on the unit's full history year)
  objective  = mean over meter types of the mean of the two skills (see skill(); history of the definition in
               ratchet.json notes: v1 floored pooled skills, v2 unfloored pooled, v3 per meter type)
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, dataset as D, lcnet as N

Z90 = 1.6449


def unit_predict(net, u, mode, variant='full', wx_override=None):
    """-> mu, sigma (kW-normalised units: A*IREF), target y in the same units, over u['tidx']."""
    X, b, y, S = D.rows(u, u['tidx'], mode, variant, wx_override)
    mu, sd = N.predict(net, X, b)
    return np.clip(mu, 0, None) * S, sd * S, y * S


def cover(y, mu, sd, k=1.0):
    ok = np.isfinite(y)
    lo = np.maximum(mu - Z90 * k * sd, 0); hi = mu + Z90 * k * sd
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok]))) if ok.any() else np.nan


def eval_units(units, nets, modes=(('cold', 'full'), ('warm', 'full')), widen=None, keep_series=False):
    rows = []
    for u in units:
        r = dict(id=u['id'], meter=u['meter'], use=u['use'], site=u['site'], kind=u['kind'])
        for mode, v in modes:
            net = nets[mode]
            mu, sd, y = unit_predict(net, u, mode, v)
            cv, nm, n = C.g14(y, mu)
            k = (widen or {}).get(f'{mode}_{v}_{u["meter"]}', 1.0)
            key = mode if v == 'full' else f'{mode}_{v}'
            r[f'{key}_cv'] = cv; r[f'{key}_nmbe'] = nm; r[f'{key}_cov'] = cover(y, mu, sd, k); r['n'] = n
            r[f'{key}_zsd'] = float(np.nanstd((y - mu) / np.maximum(sd, 1e-6)))
            dcv, mcv = daily_monthly(u, y, mu)
            r[f'{key}_daily_cv'] = dcv[0]; r[f'{key}_daily_nmbe'] = dcv[1]; r[f'{key}_monthly_cv'] = mcv[0]; r[f'{key}_monthly_nmbe'] = mcv[1]
            if keep_series: r[f'{key}_mu'] = mu; r[f'{key}_sd'] = sd; r['y'] = y
        rows.append(r)
    return pd.DataFrame(rows)


def daily_monthly(u, y, mu):
    """G14 on daily means (days with >= 20 valid hours) and monthly means of valid hours, as BuildingData does."""
    f = D.site_feats(u['site']); t = u['tidx']; c = C.site_calendar(u['site'])
    ok = np.isfinite(y)
    d = pd.DataFrame(dict(y=np.where(ok, y, np.nan), p=np.where(ok, mu, np.nan), d=f['dcode'][t], m=c['month'][t]))
    g = d.groupby('d'); cnt = g.y.count(); dd = g[['y', 'p']].mean()[cnt >= 20]
    mm = d.groupby('m')[['y', 'p']].mean()
    return C.g14(dd.y.values, dd.p.values)[:2], C.g14(mm.y.values, mm.p.values)[:2]


def ref_cvs(units):
    """Fixed references on a unit set: dumb (use x meter median intensity x median TOW shape) and TOWT full history."""
    f_int = {}; shapes = {}
    for u in units:
        f = D.site_feats(u['site']); t = u['tidx']; y = u['y'][t]
        mu = np.nanmean(y)
        f_int.setdefault((u['use'], u['meter']), []).append(mu)
        sh = pd.Series(y / mu).groupby(f['tow'][t]).mean().reindex(range(168)).fillna(1.0).values
        shapes.setdefault((u['use'], u['meter']), []).append(sh)
    med_i = {k: np.median(v) for k, v in f_int.items()}; med_s = {k: np.median(np.array(v), 0) for k, v in shapes.items()}
    out = []
    for u in units:
        f = D.site_feats(u['site']); t = u['tidx']; y = u['y'][t]
        dumb = med_i[(u['use'], u['meter'])] * med_s[(u['use'], u['meter'])][f['tow'][t]]
        h = u['H']['full']
        towt = C.towt_predict(h['coef'], f['T'][t], f['tow'][t]) * h['S']
        out.append(dict(id=u['id'], dumb_cv=C.g14(y, dumb)[0], towt_cv=C.g14(y, towt)[0], towt_nmbe=C.g14(y, towt)[1]))
    return pd.DataFrame(out)


def skill(df, ref):
    """Per meter type (n >= 5): skill = 1 - median CV(model) / median CV(reference); objective = mean over meter types
    of (cold skill + warm skill) / 2. (Definition fixed 17 Nov after R2: pooled medians mixed meter types and could
    rank a model that is better on every meter as worse.)"""
    d = df.merge(ref, on='id')
    per = {}
    for m, g in d.groupby('meter'):
        if len(g) < 5: continue
        per[m] = dict(n=len(g), cold=float(1 - g.cold_cv.median() / g.dumb_cv.median()),
                      warm=float(1 - g.warm_cv.median() / g.towt_cv.median()))
    sc = float(np.mean([v['cold'] for v in per.values()])); sw = float(np.mean([v['warm'] for v in per.values()]))
    return dict(cold_skill=sc, warm_skill=sw, objective=float(0.5 * (sc + sw)), per_meter=per,
                cold_cv_med=float(d.cold_cv.median()), warm_cv_med=float(d.warm_cv.median()),
                dumb_cv_med=float(d.dumb_cv.median()), towt_cv_med=float(d.towt_cv.median()),
                cold_cov=float(d.cold_cov.median()), warm_cov=float(d.warm_cov.median()), n=len(d))


def summarize(df, keys=('cold', 'warm')):
    out = {}
    for m, g in df.groupby('meter'):
        out[m] = {'n': int(len(g))}
        for k in keys:
            if f'{k}_cv' not in g: continue
            out[m][k] = dict(cv_med=float(g[f'{k}_cv'].median()), cv_p25=float(g[f'{k}_cv'].quantile(.25)),
                             cv_p75=float(g[f'{k}_cv'].quantile(.75)), nmbe_med=float(g[f'{k}_nmbe'].median()),
                             abs_nmbe_med=float(g[f'{k}_nmbe'].abs().median()), cov_med=float(g[f'{k}_cov'].median()),
                             cov_mean=float(g[f'{k}_cov'].mean()))
    out['all'] = {k: dict(cv_med=float(df[f'{k}_cv'].median()), cov_mean=float(df[f'{k}_cov'].mean()))
                  for k in keys if f'{k}_cv' in df}
    return out


def dev_score(df):
    """Dev gate number: mean over cold and warm of the median CV(RMSE) over all dev building-meters."""
    return float(0.5 * (df.cold_cv.median() + df.warm_cv.median()))
