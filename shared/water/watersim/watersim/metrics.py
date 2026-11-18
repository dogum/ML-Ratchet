"""Streamflow skill metrics (NaN-aware). obs, sim: 1-D arrays on the same days."""
import numpy as np


def _pair(obs, sim):
    o = np.asarray(obs, float)
    s = np.asarray(sim, float)
    m = np.isfinite(o) & np.isfinite(s)
    return o[m], s[m]


def nse(obs, sim):
    """Nash & Sutcliffe (1970) efficiency."""
    o, s = _pair(obs, sim)
    if len(o) < 2:
        return np.nan
    d = np.sum((o - o.mean()) ** 2)
    return float(1 - np.sum((s - o) ** 2) / d) if d > 0 else np.nan


def log_nse(obs, sim, eps=None):
    o, s = _pair(obs, sim)
    if eps is None:
        eps = max(np.mean(o) / 100.0, 1e-6)
    return nse(np.log(o + eps), np.log(np.maximum(s, 0) + eps))


def kge(obs, sim, return_parts=False):
    """Kling-Gupta efficiency (Gupta et al. 2009): 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)."""
    o, s = _pair(obs, sim)
    if len(o) < 2 or o.std() == 0 or o.mean() == 0:
        return (np.nan, np.nan, np.nan, np.nan) if return_parts else np.nan
    r = np.corrcoef(o, s)[0, 1] if s.std() > 0 else 0.0
    a = s.std() / o.std()
    b = s.mean() / o.mean()
    k = 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)
    return (float(k), float(r), float(a), float(b)) if return_parts else float(k)


def pbias(obs, sim):
    o, s = _pair(obs, sim)
    return float(100 * (s.sum() - o.sum()) / o.sum()) if o.sum() else np.nan


def fhv(obs, sim, h=0.02):
    """Bias of the top 2 % of the flow duration curve, % (Yilmaz et al. 2008)."""
    o, s = _pair(obs, sim)
    o = np.sort(o)[::-1]
    s = np.sort(s)[::-1]
    n = max(int(round(h * len(o))), 1)
    return float(100 * (s[:n].sum() - o[:n].sum()) / o[:n].sum()) if o[:n].sum() else np.nan


def flv(obs, sim, l=0.3):
    """Bias of the bottom 30 % of the flow duration curve in log space, % (Yilmaz et al. 2008)."""
    o, s = _pair(obs, sim)
    eps = 1e-6
    o = np.log(np.maximum(np.sort(o), eps))
    s = np.log(np.maximum(np.sort(s), eps))
    n = max(int(round(l * len(o))), 1)
    oo = o[:n] - o[0]
    ss = s[:n] - s[0]
    return float(-100 * (ss.sum() - oo.sum()) / oo.sum()) if oo.sum() else np.nan


def peak_error(obs, sim, index=None, q=0.99, window=3):
    """Mean relative peak error (%) and mean absolute timing error (days) over peaks above the q quantile of obs,
    matching each observed peak to the largest simulated value within +-window days."""
    o = np.asarray(obs, float)
    s = np.asarray(sim, float)
    thr = np.nanquantile(o, q)
    idx = [i for i in range(1, len(o) - 1) if np.isfinite(o[i]) and o[i] >= thr and o[i] >= o[i - 1]
           and o[i] >= o[i + 1]]
    if not idx:
        return np.nan, np.nan
    rel, tim = [], []
    for i in idx:
        lo, hi = max(i - window, 0), min(i + window + 1, len(s))
        j = lo + int(np.nanargmax(s[lo:hi]))
        rel.append((s[j] - o[i]) / o[i])
        tim.append(abs(j - i))
    return float(100 * np.mean(rel)), float(np.mean(tim))


def all_metrics(obs, sim):
    k, r, a, b = kge(obs, sim, return_parts=True)
    pe, te = peak_error(obs, sim)
    return {'NSE': nse(obs, sim), 'KGE': k, 'r': r, 'alpha': a, 'beta': b, 'logNSE': log_nse(obs, sim),
            'PBIAS': pbias(obs, sim), 'FHV': fhv(obs, sim), 'FLV': flv(obs, sim), 'peak_err_pct': pe,
            'peak_timing_days': te, 'n_days': int(np.sum(np.isfinite(np.asarray(obs, float)) &
                                                      np.isfinite(np.asarray(sim, float))))}
