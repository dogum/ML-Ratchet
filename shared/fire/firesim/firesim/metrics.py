"""Spread-forecast metrics on rasters: Sorensen/Dice, Jaccard, F1/precision/recall on newly burned pixels,
average precision for probability maps, area error."""
import numpy as np


def sorensen(pred, obs):
    p = np.asarray(pred, bool)
    o = np.asarray(obs, bool)
    s = p.sum() + o.sum()
    return float(2.0 * (p & o).sum() / s) if s > 0 else np.nan


def jaccard(pred, obs):
    p = np.asarray(pred, bool)
    o = np.asarray(obs, bool)
    u = (p | o).sum()
    return float((p & o).sum() / u) if u > 0 else np.nan


def prf(pred, obs, mask=None):
    p = np.asarray(pred, bool)
    o = np.asarray(obs, bool)
    if mask is not None:
        p = p[mask]
        o = o[mask]
    tp = float((p & o).sum())
    fp = float((p & ~o).sum())
    fn = float((~p & o).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else np.nan
    rec = tp / (tp + fn) if tp + fn > 0 else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn)


def average_precision(score, obs, mask=None):
    """Average precision (area under the precision-recall step curve) of a score/probability map."""
    s = np.asarray(score, float)
    o = np.asarray(obs, bool)
    if mask is not None:
        s = s[mask]
        o = o[mask]
    ok = np.isfinite(s)
    s, o = s[ok], o[ok]
    npos = o.sum()
    if npos == 0:
        return np.nan
    order = np.argsort(-s, kind='mergesort')
    o = o[order]
    s = s[order]
    tp = np.cumsum(o)
    fp = np.cumsum(~o)
    # evaluate at distinct thresholds
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    prec = tp[last] / (tp[last] + fp[last])
    rec = tp[last] / npos
    rprev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rprev) * prec))


def area_error(pred, obs, cell_ha=1.0):
    p = np.asarray(pred, bool).sum() * cell_ha
    o = np.asarray(obs, bool).sum() * cell_ha
    return dict(pred_ha=float(p), obs_ha=float(o), err_ha=float(p - o), rel_err=float((p - o) / o) if o > 0 else np.nan,
                log_ratio=float(np.log((p + cell_ha) / (o + cell_ha))))


def next_day_scores(pred_burned, obs_burned, prior_burned, score=None, cell_ha=1.0):
    """Scores on NEW burned area (pixels burned by day end that were not burned at the start)."""
    prior = np.asarray(prior_burned, bool)
    newp = np.asarray(pred_burned, bool) & ~prior
    newo = np.asarray(obs_burned, bool) & ~prior
    out = dict(sorensen_new=sorensen(newp, newo), sorensen_total=sorensen(np.asarray(pred_burned, bool),
                                                                          np.asarray(obs_burned, bool)))
    out.update({k + '_new': v for k, v in prf(newp, newo, ~prior).items()})
    out.update({'area_' + k: v for k, v in area_error(newp, newo, cell_ha).items()})
    if score is not None:
        out['ap_new'] = average_precision(score, newo, ~prior)
    return out
