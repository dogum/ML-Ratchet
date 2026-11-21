"""Exam metrics (Phase 2 reuses these exactly).

Next-day spread (per fire-day, on the fire's 90 m grid):
  * candidates = cells NOT burned at the issue time (inside the scoring window); target = newly burned by the
    verification time (next PM perimeter).
  * AP  = average precision of a probability (or score) map over candidate cells (sklearn definition, step-wise).
  * F1  = F1 of the thresholded new-burn prediction (threshold 0.5 unless the method states another, fixed on dev).
  * Sorensen (Dice) on the TOTAL burned area at verification: 2|P n O| / (|P| + |O|), P = burned_now | predicted new.
  * Sorensen_new = Dice on the new area only (0 when both empty is reported as NaN and skipped).
  * area error: predicted new area - observed new area (km2), and log10 ratio of (new area + 1 cell).
Aggregation: per-fire-day values -> median and mean over fire-days with observed growth >= 1 cell ("growth days"),
plus pooled ("micro") AP/F1 over all candidate cells of all growth days. Report both.
"""
import numpy as np


def average_precision(y, s):
    y = np.asarray(y, bool).ravel(); s = np.asarray(s, np.float64).ravel()
    P = y.sum()
    if P == 0: return np.nan
    o = np.argsort(-s, kind='mergesort'); y = y[o]; s = s[o]
    # group ties: precision/recall evaluated at distinct thresholds
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct]; fp = (distinct + 1) - tp
    prec = tp / (tp + fp); rec = tp / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def f1(y, yhat):
    y = np.asarray(y, bool); yhat = np.asarray(yhat, bool)
    tp = (y & yhat).sum(); fp = (~y & yhat).sum(); fn = (y & ~yhat).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan


def dice(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool); d = a.sum() + b.sum()
    return float(2 * (a & b).sum() / d) if d else np.nan


def nextday(burned_now, burned_next, prob, cell_km2, thr=0.5, window=None):
    """Score one fire-day. burned_now/next: bool grids; prob: predicted P(burned by next) for every cell (values on
    already-burned cells are ignored; they count as burned). window: optional bool mask of scored cells."""
    now = np.asarray(burned_now, bool); nxt = np.asarray(burned_next, bool) | now
    W = np.ones_like(now) if window is None else np.asarray(window, bool)
    cand = W & ~now
    y = nxt[cand]; s = np.asarray(prob, np.float64)[cand]
    pred_new = s >= thr
    total_pred = now.copy(); total_pred[cand] = pred_new
    new_obs = y.sum(); new_pred = pred_new.sum()
    return dict(ap=average_precision(y, s), f1=f1(y, pred_new), dice_total=dice(total_pred & W, nxt & W),
                dice_new=dice(pred_new, y) if (new_obs + new_pred) else np.nan,
                new_obs_km2=float(new_obs * cell_km2), new_pred_km2=float(new_pred * cell_km2),
                area_err_km2=float((new_pred - new_obs) * cell_km2),
                log10_area_ratio=float(np.log10((new_pred + 1) / (new_obs + 1))),
                n_cand=int(cand.sum()), prevalence=float(y.mean()) if len(y) else np.nan)


def score_window(burned_now, res_m=90.0, km=10.0, dist=None):
    """Standard scored region: cells within `km` of the burned area at issue time (the fire grid itself is built from
    the FINAL perimeter + 30 %, so scoring the whole grid would reward knowing where the fire ends)."""
    from scipy import ndimage
    now = np.asarray(burned_now, bool)
    if dist is None: dist = ndimage.distance_transform_edt(~now, sampling=res_m)
    return dist <= km * 1000.0


def summarize(rows, growth_only=True, min_new_km2=0.0):
    """Aggregate per-fire-day dicts (from nextday) -> summary dict of medians/means (growth days only by default)."""
    import pandas as pd
    d = pd.DataFrame(rows)
    if growth_only: d = d[d.new_obs_km2 > min_new_km2]
    out = dict(n_fire_days=int(len(d)), n_fires=int(d.fire_id.nunique()) if 'fire_id' in d else None)
    for k in ['ap', 'f1', 'dice_total', 'dice_new', 'log10_area_ratio']:
        out[f'{k}_median'] = float(np.nanmedian(d[k])) if len(d) else np.nan
        out[f'{k}_mean'] = float(np.nanmean(d[k])) if len(d) else np.nan
    out['abs_area_err_km2_median'] = float(np.nanmedian(np.abs(d.area_err_km2))) if len(d) else np.nan
    return out


class Pooled:
    """Pooled (micro) AP / F1 over many fire-days without storing every cell: histogram of scores per class."""
    def __init__(self, bins=1001):
        self.edges = np.linspace(0, 1, bins); self.pos = np.zeros(bins); self.neg = np.zeros(bins); self.tp = self.fp = self.fn = 0

    def add(self, y, s, thr=0.5):
        y = np.asarray(y, bool); s = np.clip(np.asarray(s, float), 0, 1)
        i = np.minimum((s * (len(self.edges) - 1)).round().astype(int), len(self.edges) - 1)
        self.pos += np.bincount(i[y], minlength=len(self.edges)); self.neg += np.bincount(i[~y], minlength=len(self.edges))
        p = s >= thr; self.tp += int((p & y).sum()); self.fp += int((p & ~y).sum()); self.fn += int((~p & y).sum())

    def result(self):
        tp = np.cumsum(self.pos[::-1]); fp = np.cumsum(self.neg[::-1]); P = self.pos.sum()
        keep = (tp + fp) > 0
        prec = np.where(keep, tp / np.maximum(tp + fp, 1), 0); rec = tp / max(P, 1)
        ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
        f1v = 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1)
        return dict(ap_pooled=ap, f1_pooled=float(f1v), prevalence_pooled=float(P / max(P + self.neg.sum(), 1)))


# ---------- final size ----------
def size_scores(obs_ha, pred_ha):
    """Final size: log10 errors (median |err|, bias), Spearman rho, R2 in log space."""
    from scipy.stats import spearmanr
    lo = np.log10(np.asarray(obs_ha, float)); lp = np.log10(np.maximum(np.asarray(pred_ha, float), 1e-3))
    e = lp - lo
    r2 = 1 - np.sum(e ** 2) / np.sum((lo - lo.mean()) ** 2)
    return dict(n=int(len(lo)), mae_log10=float(np.median(np.abs(e))), bias_log10=float(np.median(e)),
                rho=float(spearmanr(lo, lp)[0]), r2_log10=float(r2))


def binary_scores(y, p):
    """Escape / large-fire classification: AUC, AP, Brier, log loss."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return dict(n=int(len(y)), prevalence=float(y.mean()), auc=float(roc_auc_score(y, p)),
                ap=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p)))


# ---------- moisture ----------
def moisture_scores(obs, pred):
    o = np.asarray(obs, float); p = np.asarray(pred, float); m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    return dict(n=int(m.sum()), mae=float(np.mean(np.abs(p - o))), rmse=float(np.sqrt(np.mean((p - o) ** 2))),
                bias=float(np.mean(p - o)), r=float(np.corrcoef(o, p)[0, 1]) if len(o) > 2 else np.nan)
