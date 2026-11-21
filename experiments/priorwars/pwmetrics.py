import numpy as np
from scipy.stats import spearmanr, rankdata


def _ok(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    return y[ok], p[ok]


def rho(y, p):
    y, p = _ok(y, p)
    if len(y) < 3 or np.std(p) == 0:
        return 0.0
    return float(spearmanr(p, y).statistic)


def mdae(y, p):
    y, p = _ok(y, p)
    return float(np.median(np.abs(p - y)))


def mae(y, p):
    y, p = _ok(y, p)
    return float(np.mean(np.abs(p - y)))


def rmse(y, p):
    y, p = _ok(y, p)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def r2(y, p):
    y, p = _ok(y, p)
    return float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))


def mape(y, p):
    y, p = _ok(y, p)
    return float(np.median(np.abs(p - y) / np.abs(y)) * 100)


def auc(y, p):
    y, p = _ok(y, p)
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(p)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def acc(y, p):
    y, p = _ok(y, p)
    return float((np.round(p) == y).mean() * 100)


def bacc(y, p):
    """balanced accuracy for integer class labels; p = predicted class"""
    y, p = _ok(y, p)
    cls = np.unique(y)
    return float(np.mean([(p[y == c] == c).mean() for c in cls]) * 100)


FUN = {'mae_med': mdae, 'rho': rho, 'mdae': mdae, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'auc': auc, 'acc': acc,
       'bacc': bacc}
HIGHER = {'mae_med': False, 'rho': True, 'mdae': False, 'mae': False, 'rmse': False, 'r2': True, 'mape': False, 'auc': True,
          'acc': True, 'bacc': True}


def primary_skill(metric, value):
    """map a primary metric to a 'higher is better' skill used in the headline composite"""
    if metric == 'auc':
        return 2 * value - 1
    return value
