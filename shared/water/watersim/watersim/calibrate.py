"""Per-basin calibration with a population optimiser that evaluates every generation in one numba batch call.

Algorithm: differential evolution DE/current-to-pbest/1/bin with archive-free JADE-style adaptation (Zhang &
Sanderson 2009) in a normalised [0, 1] parameter space (log-scaled where a parameter spans > 20x), Latin-hypercube
start, reflection at the bounds. Objective: KGE (default), NSE, or a KGE/logNSE blend, on days with observations.
Default budget: population 40 x 70 generations = 2,800 model runs per basin (a few seconds per basin-model).
"""
import math
import time

import numpy as np

from .simulate import prepare_run, simulate_batch
from .models import MODULES
from . import snow as snowmod

FIXED_DEFAULT = {   # parameters held at their defaults unless calibrate(..., free=...) says otherwise
    'hbv': [],
    'gr4j': [],
    'sacsma': ['RIVA', 'SIDE', 'RSERV'],
    'vic_lite': [],
    'snow_dd': ['CFR', 'CWH', 'TTI', 'TLAPSE', 'PGRAD'],
    'snow_eb': ['CWH', 'TTI', 'ALB_MAX', 'GHEAT', 'TLAPSE', 'PGRAD', 'CANOPY'],
}


def free_parameters(model, snow, fixed=None):
    rows = [(p[0], p[2], p[3]) for p in MODULES[model].PARAMS if p[0] not in FIXED_DEFAULT[model]] \
        if model in MODULES else []
    if snow == 'degree_day':
        rows += [(p[0], p[2], p[3]) for p in snowmod.DD_PARAMS if p[0] not in FIXED_DEFAULT['snow_dd']]
    elif snow == 'energy_balance':
        rows += [(p[0], p[2], p[3]) for p in snowmod.EB_PARAMS if p[0] not in FIXED_DEFAULT['snow_eb']]
    if fixed:
        rows = [r for r in rows if r[0] not in fixed]
    return rows


def _objective(Q, obs, kind):
    m = np.isfinite(obs)
    o = obs[m]
    S = Q[:, m]
    if kind == 'nse':
        return 1 - np.sum((S - o) ** 2, 1) / np.sum((o - o.mean()) ** 2)
    om, osd = o.mean(), o.std()
    sm, ssd = S.mean(1), S.std(1)
    cov = ((S - sm[:, None]) * (o - om)).mean(1)
    r = np.where(ssd > 0, cov / (ssd * osd + 1e-12), 0.0)
    k = 1 - np.sqrt((r - 1) ** 2 + (ssd / osd - 1) ** 2 + (sm / om - 1) ** 2)
    if kind == 'kge':
        return k
    if kind == 'kge_lognse':
        eps = om / 100
        lo = np.log(o + eps)
        ls = np.log(np.maximum(S, 0) + eps)
        ln = 1 - np.sum((ls - lo) ** 2, 1) / np.sum((lo - lo.mean()) ** 2)
        return 0.5 * k + 0.5 * ln
    raise ValueError(kind)


def calibrate(spec, forcing, obs, start, end, model='hbv', snow='degree_day', objective='kge', pop=40, gens=70,
              rng=0, spinup='1y', fixed=None, verbose=False, pet=None, timeout_s=None, target='Q', obs_shift=0):
    """Calibrate `model` (+ snow) on obs (pandas Series of daily Q in mm/day, or SWE in mm with target='SWE' and
    model='none') over [start, end]. obs_shift: compare sim[t] with obs[t + obs_shift] (e.g. 1 for SNOTEL SWE,
    which is reported at the start of the day).
    Returns dict(best params (flat), objective, history, n_evals, wall_s, free parameter list)."""
    t0 = time.time()
    rng = np.random.default_rng(rng)
    o = prepare_run(spec, forcing, model, None, snow, pet)
    free = free_parameters(model, snow, fixed)
    names = [f[0] for f in free]
    lo = np.array([f[1] for f in free], float)
    hi = np.array([f[2] for f in free], float)
    uselog = (lo > 0) & (hi / np.where(lo > 0, lo, 1) > 20)
    llo = np.where(uselog, np.log(np.where(lo > 0, lo, 1)), lo)
    lhi = np.where(uselog, np.log(hi), hi)

    def decode(U):
        X = llo + U * (lhi - llo)
        return np.where(uselog, np.exp(X), X)

    outs = (target,) if target != 'Q' else ('Q',)
    res0 = simulate_batch(None, None, [{}], model=model, snow=snow, start=start, end=end, spinup=spinup, o=o,
                          outputs=outs)
    idx = res0['index']
    ob = obs.shift(-obs_shift).reindex(idx).values.astype(float)
    if np.isfinite(ob).sum() < 365:
        raise ValueError('fewer than 365 observed days in the calibration window')
    D = len(names)
    # Latin hypercube start
    U = (rng.permuted(np.tile(np.arange(pop), (D, 1)), axis=1).T + rng.random((pop, D))) / pop

    def evaluate(U):
        X = decode(U)
        plist = [dict(zip(names, x)) for x in X]
        Q = simulate_batch(None, None, plist, model=model, snow=snow, start=start, end=end, spinup=spinup, o=o,
                           outputs=outs)[target]
        f = _objective(Q, ob, objective)
        return np.where(np.isfinite(f), f, -1e9)

    fit = evaluate(U)
    n_eval = pop
    mu_cr, mu_f = 0.5, 0.5
    hist = [float(fit.max())]
    for g in range(gens):
        if timeout_s and time.time() - t0 > timeout_s:
            break
        cr = np.clip(rng.normal(mu_cr, 0.1, pop), 0, 1)
        Fs = np.clip(mu_f + 0.1 * rng.standard_cauchy(pop), 0.05, 1.0)
        order = np.argsort(-fit)
        npb = max(2, int(0.15 * pop))
        V = np.empty_like(U)
        for i in range(pop):
            pb = order[rng.integers(npb)]
            r1, r2 = rng.choice([j for j in range(pop) if j != i], 2, replace=False)
            V[i] = U[i] + Fs[i] * (U[pb] - U[i]) + Fs[i] * (U[r1] - U[r2])
        # reflect into [0, 1]
        V = np.where(V < 0, -V, V)
        V = np.where(V > 1, 2 - V, V)
        V = np.clip(V, 0, 1)
        mask = rng.random((pop, D)) < cr[:, None]
        jr = rng.integers(D, size=pop)
        mask[np.arange(pop), jr] = True
        Ut = np.where(mask, V, U)
        ft = evaluate(Ut)
        n_eval += pop
        better = ft >= fit
        if better.any():
            w = ft[better] - fit[better] + 1e-12
            mu_cr = 0.9 * mu_cr + 0.1 * float(np.sum(w * cr[better]) / w.sum())
            mu_f = 0.9 * mu_f + 0.1 * float(np.sum(w * Fs[better] ** 2) / max(np.sum(w * Fs[better]), 1e-12))
        U[better] = Ut[better]
        fit[better] = ft[better]
        hist.append(float(fit.max()))
        if verbose and g % 10 == 0:
            print(g, hist[-1], mu_cr, mu_f)
    b = int(np.argmax(fit))
    best = dict(zip(names, decode(U[b:b + 1])[0].tolist()))
    return {'params': best, 'objective': objective, 'value': float(fit[b]), 'history': hist, 'n_evals': n_eval,
            'wall_s': time.time() - t0, 'free': names, 'model': model, 'snow': snow, 'start': str(start),
            'end': str(end)}
