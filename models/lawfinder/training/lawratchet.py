"""Generic law search + ratchet for LawFinder.

For one law (one platform):
  1. symbolic regression on the simulated TRAIN bank of a physics version -> Pareto front (complexity vs loss);
  2. every front member's constants refitted on the full train bank (never on real data);
  3. each member scored on the FROZEN simulated set (skill = R^2 of the target) and on the real DEV set (error);
  4. the ratchet walks the members in order of complexity (r0 = a constant): a rung is kept only if its frozen-set
     skill beats the incumbent's by more than MARGIN and its dev error is no more than 5 % worse;
  5. when the physics version changes, every earlier rung is re-scored on the new frozen set and every decision is
     re-taken in order (the log keeps both).
Everything is logged to results/<law>_ratchet.json.
"""
import json, time, os
import numpy as np
import lawsearch as ls

from _paths import ROOT  # the work folder (training/_paths.py)
MARGIN = 0.01
DEV_TOL = 1.05


def r2(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    return float(1 - np.sum((yh - y) ** 2) / np.sum((y - y.mean()) ** 2))


def mae(y, yh):
    return float(np.mean(np.abs(np.asarray(yh, float) - np.asarray(y, float))))


def law_eval(law, X):
    t = ls.from_json(law['tree']); c = np.array(law['consts'], float)
    return ls.evaluate(t, X, c)


def make_law(t, c, names, label=None):
    t, c = ls.renumber(t, np.asarray(c, float))
    return {'tree': ls.to_json(t), 'consts': [float(v) for v in c], 'complexity': ls.size(t),
            'expr': ls.to_str(t, c, names, 4), 'js': ls.to_js(t, c, names), 'label': label}


def search(names, Xtr, ytr, wtr=None, seconds=300, seed=0, seeds=(), log=None, **kw):
    s = ls.Search(names, seed=seed, fixed_seeds=seeds, **kw)
    s.fit(Xtr, ytr, wtr, seconds=seconds, verbose=True, log=log)
    # refit constants on a larger slice of the train bank (still simulation only)
    n = min(len(ytr), 6000)
    idx = np.random.default_rng(seed + 1).permutation(len(ytr))[:n]
    w = None if wtr is None else wtr[idx]
    front = s.front(Xtr[idx], ytr[idx], w, refit=True)
    return [make_law(t, c, names) for (_, _, t, c) in front]


def score(law, Xval, yval, Xdev, ydev, dev_err=mae, transform=None):
    yv = law_eval(law, Xval); yd = law_eval(law, Xdev)
    if transform is not None:
        yv, yd = transform(yv), transform(yd)
    ok_v = np.isfinite(yv); ok_d = np.isfinite(yd)
    sim = r2(yval[ok_v], yv[ok_v]) if ok_v.mean() > 0.99 else -np.inf
    dev = dev_err(ydev[ok_d], yd[ok_d]) if ok_d.mean() > 0.99 else np.inf
    return sim, dev


def walk(rungs, margin=MARGIN, tol=DEV_TOL):
    """rungs: list of dicts with 'sim' and 'dev' (already scored on the current frozen set). Returns decisions."""
    inc = None
    out = []
    for r in rungs:
        if inc is None:
            d, why = 'kept', 'First rung (the incumbent to beat).'
        elif not np.isfinite(r['sim']) or not np.isfinite(r['dev']):
            d, why = 'rejected', 'Not finite on the frozen set or on dev.'
        elif r['sim'] <= inc['sim'] + margin:
            d, why = 'rejected', f"Frozen-set skill {r['sim']:.3f} does not beat the incumbent's {inc['sim']:.3f} by more than {margin}."
        elif r['dev'] > tol * inc['dev']:
            d, why = 'rejected', (f"Frozen-set skill {r['sim']:.3f} beats {inc['sim']:.3f}, but the dev error "
                                  f"{r['dev']:.4g} is more than 5% worse than {inc['dev']:.4g}.")
        else:
            d, why = 'kept', (f"Frozen-set skill {r['sim']:.3f} > {inc['sim']:.3f} + {margin}, dev error "
                              f"{r['dev']:.4g} vs {inc['dev']:.4g} (within 5%).")
        out.append({'id': r['id'], 'decision': d, 'why': why})
        if d == 'kept':
            inc = r
    return out, inc


def widen_factor(resid_dev, sigma):
    """Widen-only factor so that 68% of dev residuals fall inside +-k sigma (k >= 1)."""
    z = np.abs(np.asarray(resid_dev, float)) / np.asarray(sigma, float)
    k = float(np.quantile(z, 0.6827))
    return max(1.0, k)


def knee(R, ratio=1.5):
    """The simulator's own one-line belief, chosen on simulations only (PySR's 'best' rule): on the Pareto front of
    complexity vs frozen-set loss (1 - R^2), among rungs with loss <= 1.5 x the lowest, the one with the largest drop
    in log loss per unit of added complexity."""
    rs = sorted([r for r in R['rungs'] if r.get('sim') is not None and np.isfinite(r['sim'])],
                key=lambda r: (r['complexity'], -r['sim']))
    front, best = [], np.inf
    for r in rs:
        l = max(1 - r['sim'], 1e-9)
        if l < best:
            front.append((r, l)); best = l
    lmin = min(l for _, l in front)
    cand = []
    for (a, la), (b, lb) in zip(front[:-1], front[1:]):
        if lb <= ratio * lmin:
            cand.append((-np.log(lb / la) / (b['complexity'] - a['complexity']), b))
    return max(cand, key=lambda t: t[0])[1] if cand else front[-1][0]
