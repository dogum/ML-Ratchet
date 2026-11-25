"""Mondrian split-conformal layer for River Bake-off arms (agent C). numpy only.

Calibration (fit on the wrapped arm's DEV residuals only; evaluate.py --save-preds):
  * strata (Mondrian, fixed before looking at residuals): the first of snow / arid / humid in the batch's class list,
    else 'other' (regulated-only, large-only and unclassed basins). Dev counts 10 / 12 / 8 / 22 basins.
  * scores in log space with an offset c = C_FRAC x p_mean (the forcing's mean precipitation, static[11]):
        upper endpoint of a band:  s_hi = log(y + c) - log(q_hi + c)
        lower endpoint:            s_lo = log(q_lo + c) - log(y + c)
    for the 90 % band (q05, q95) and the 50 % band (q25, q75), separately per protocol/lead job.
  * per stratum and endpoint the shift d is the weighted (1 - a/2) quantile of the scores over the stratum's dev
    basin-days, every basin weighted equally (each day 1 / n_days of its basin). Asymmetric CQR (Romano et al. 2019)
    over basin-days (tens of thousands per stratum, so the finite-sample (1 + 1/n) term is negligible and omitted;
    days are not independent, so the leave-one-basin-out check in fit_report.json is the honest coverage estimate).
  * strata with fewer than MIN_BASINS basins use the pooled shift (all dev basins).
Applied:  q_hi' = (q_hi + c) e^{d_hi} - c,  q_lo' = (q_lo + c) e^{-d_lo} - c  (clipped at 0), then made monotone
(q05 <= q25 <= q50 <= q75 <= q95). The mean and q50 are unchanged, so NSE / KGE / FHV are the wrapped arm's.
A shift can be negative (the band narrows): this is conformal calibration, not widen-only.
"""
import numpy as np

STRATA = ['snow', 'arid', 'humid', 'other']
BANDS = {'b90': ('q05', 'q95', 0.10), 'b50': ('q25', 'q75', 0.50)}
QK = ['q05', 'q25', 'q50', 'q75', 'q95']
C_FRAC = 0.05
MIN_BASINS = 6


def stratum(classes):
    for k in ('snow', 'arid', 'humid'):
        if k in classes:
            return k
    return 'other'


def offset(static):
    return C_FRAC * float(np.exp(np.asarray(static, float)[11]))


def wquantile(x, w, q):
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w) / np.sum(w)
    i = int(np.searchsorted(cw, q, side='left'))
    return float(x[min(i, len(x) - 1)])


def _scores(P, c):
    """P (days, 7) = obs, mean, q05..q95 -> dict of endpoint scores on observed days."""
    y = P[:, 0]
    ok = np.isfinite(y)
    y = np.maximum(y[ok], 0.0)
    Q = {k: np.maximum(P[ok, 2 + j], 0.0) for j, k in enumerate(QK)}
    out = {}
    for b, (lo, hi, a) in BANDS.items():
        out[b + '_lo'] = np.log(Q[lo] + c) - np.log(y + c)
        out[b + '_hi'] = np.log(y + c) - np.log(Q[hi] + c)
    return out


def fit_job(preds, meta):
    """preds: {basin: (days, 7)}; meta: {basin: (stratum, c)}. Returns {stratum|'pooled': {endpoint: shift}}."""
    sc = {b: _scores(P, meta[b][1]) for b, P in preds.items()}
    groups = {'pooled': list(preds)}
    for s in STRATA:
        groups[s] = [b for b in preds if meta[b][0] == s]
    out = {}
    for g, ids in groups.items():
        ids = [b for b in ids if len(sc[b]['b90_lo'])]
        n = len(ids)
        if g != 'pooled' and n < MIN_BASINS:
            out[g] = {'fallback': 'pooled', 'n_basins': n}
            continue
        rec = {'n_basins': n}
        for b, (lo, hi, a) in BANDS.items():
            lev = 1 - a / 2
            for side in ('lo', 'hi'):
                x = np.concatenate([sc[i][f'{b}_{side}'] for i in ids])
                w = np.concatenate([np.full(len(sc[i][f'{b}_{side}']), 1.0 / len(sc[i][f'{b}_{side}'])) for i in ids])
                rec[f'{b}_{side}'] = wquantile(x, w, lev)
        out[g] = rec
    for g in list(out):
        if out[g].get('fallback') == 'pooled':
            out[g] = {**out['pooled'], 'fallback': 'pooled', 'n_basins': out[g]['n_basins']}
    return out


def apply(o, shifts, c):
    """o: an arm output dict; shifts: the stratum's {endpoint: d}; returns a new output dict."""
    out = {k: np.asarray(v, float).copy() for k, v in o.items()}
    for b, (lo, hi, a) in BANDS.items():
        out[lo] = np.maximum((np.maximum(out[lo], 0.0) + c) * np.exp(-shifts[f'{b}_lo']) - c, 0.0)
        out[hi] = np.maximum((np.maximum(out[hi], 0.0) + c) * np.exp(shifts[f'{b}_hi']) - c, 0.0)
    q50 = out['q50']
    out['q25'] = np.minimum(out['q25'], q50)
    out['q75'] = np.maximum(out['q75'], q50)
    out['q05'] = np.minimum(out['q05'], out['q25'])
    out['q95'] = np.maximum(out['q95'], out['q75'])
    return out


def job_key(batch):
    return batch['protocol'] if batch.get('protocol') != 'gauged' else f"gauged_L{int(batch['lead'])}"
