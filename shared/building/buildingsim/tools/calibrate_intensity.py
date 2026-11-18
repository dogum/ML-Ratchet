"""Physics 1.4, generator intensity by use type: fit one factor per use on the CBECS 2018 / RECS 2020 PRIOR split,
check on the DEV split with a disjoint generated sample. Plan: shared/building/NOTEBOOK.md, 17 Nov 2026.

    python buildingsim/tools/calibrate_intensity.py check 1.0      # dev check of v1.0 generated buildings (the gap)
    python buildingsim/tools/calibrate_intensity.py fit            # 3 log-secant iterations on prior -> physics.json
    python buildingsim/tools/calibrate_intensity.py check 1.4      # dev check at 1.4 -> checks/out/intensity_fit.json

Fit sample: 30 buildings per use (seeds 1000 + i), check sample: 30 per use (seeds 2000 + i), one year at 1 h with
the climate-zone TMYx (as check_generator.py). Targets: weighted median site EUI of the survey respondents per use
(CBECS primary_use; RECS single-family = detached, attached, mobile; multi-family = apartments), as
check_generator.py computes them. <= 3 workers.
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, BLDG_CODE)
sys.path.insert(0, os.path.join(BLDG_CODE, 'buildingsim'))
sys.path.insert(0, os.path.join(BLDG_CODE, 'checks'))
os.environ.setdefault('OMP_NUM_THREADS', '1')

OUT = os.path.join(BLDG_ROOT, 'checks', 'out', 'intensity_fit')
N = 30
_W = {}


def wmed(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])


def targets(split):
    """Weighted median site EUI (kWh/m2/yr) of survey respondents by use type, split 'prior' or 'dev'."""
    import buildingdata as bd
    if split == 'prior':
        from buildingsim.generator import _load
        cb, rc = _load('cbecs_prior.parquet'), _load('recs_prior.parquet')
    else:
        cb, rc = bd.cbecs(split=split), bd.recs(split=split)
    ref = {}
    for u, g in cb.groupby('primary_use'):
        e = (g['site_kWh'] / g['floor_area_m2']).values
        m = np.isfinite(e) & (e > 0)
        ref[u] = dict(n=int(m.sum()), eui=wmed(e[m], g['weight'].values[m]))
    for u, ht in (('residential_single', ('single_family_detached', 'single_family_attached', 'mobile_home',
                                          'single_detached', 'single_attached')),
                  ('residential_multi', ('apartment_2_4', 'apartment_5plus'))):
        g = rc[rc['housing_type'].astype(str).isin(ht)]
        e = (g['site_kWh'] / g['floor_area_m2']).values
        m = np.isfinite(e) & (e > 0)
        ref[u] = dict(n=int(m.sum()), eui=wmed(e[m], g['weight'].values[m]))
    return ref


def _sim(job):
    spec, physics = job
    from buildingsim import simulate, read_epw, epw_path
    c = spec['weather_hint']
    if c not in _W:
        _W[c] = read_epw(epw_path(c))
    try:
        out = simulate(spec, _W[c], '2019-01-01', '2020-01-01', dt='1h', options={'physics': physics})
        return spec['primary_use'], float(out['summary']['EUI_kWh_m2_yr'])
    except Exception as e:  # noqa: BLE001
        return spec['primary_use'], None


def eui_by_use(physics, seed0, fac=None):
    from buildingsim import generate_buildings, USE_TYPES
    specs = []
    for i, u in enumerate(USE_TYPES):
        specs += generate_buildings(N, seed=seed0 + i, primary_use=u, physics=physics, intensity=fac)
    from concurrent.futures import ProcessPoolExecutor
    res = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        for u, e in ex.map(_sim, [(s, physics) for s in specs], chunksize=4):
            if e is not None and np.isfinite(e):
                res.setdefault(u, []).append(e)
    return {u: dict(n=len(v), median=float(np.median(v))) for u, v in res.items()}


def table(eui, ref):
    return {u: dict(sim_median=v['median'], n_sim=v['n'], survey_median=ref[u]['eui'], n_survey=ref[u]['n'],
                    ratio=v['median'] / ref[u]['eui'], within_15pct=bool(abs(v['median'] / ref[u]['eui'] - 1) <= 0.15))
            for u, v in eui.items() if u in ref}


def do_fit():
    os.makedirs(OUT, exist_ok=True)
    ref = targets('prior')
    hist = []
    fac = {u: 1.0 for u in ref}
    prev = None
    for it in range(3):
        t0 = time.time()
        e = eui_by_use('1.4', 1000, fac)
        r = {u: e[u]['median'] / ref[u]['eui'] for u in e}
        hist.append(dict(iteration=it, factors=dict(fac), ratio_to_prior=r))
        print(it, {u: (round(fac[u], 3), round(r[u], 3)) for u in r}, f'{time.time() - t0:.0f} s', flush=True)
        new = {}
        for u in r:
            if prev is None or abs(np.log(fac[u]) - np.log(prev[0][u])) < 1e-9:
                k = 1.0                                          # first step: assume EUI proportional to the factor
            else:
                k = (np.log(r[u]) - np.log(prev[1][u])) / (np.log(fac[u]) - np.log(prev[0][u]))
                k = float(np.clip(k, 0.2, 1.5))
            new[u] = float(np.clip(fac[u] * np.exp(-np.log(r[u]) / k), 0.2, 3.0))
        prev = (dict(fac), r)
        fac = new
    # the factors after the third evaluation's step are not evaluated on prior; keep the best evaluated set per use
    best = {}
    for u in ref:
        h = min(hist, key=lambda x: abs(np.log(x['ratio_to_prior'].get(u, np.inf))) if u in x['ratio_to_prior'] else 9)
        best[u] = h['factors'][u]
    pf = os.path.join(BLDG_CODE, 'buildingsim', 'buildingsim', 'data', 'physics.json')
    P = json.load(open(pf))
    P['INTENSITY'] = best
    P['_intensity'] = ('physics 1.4 fitted by tools/calibrate_intensity.py on the CBECS 2018 / RECS 2020 prior split '
                       '(30 generated buildings per use, seeds 1000+); dev check in checks/intensity_fit.json')
    with open(pf, 'w') as f:
        json.dump(P, f, indent=1)
    with open(os.path.join(OUT, 'fit.json'), 'w') as f:
        json.dump(dict(prior_targets=ref, history=hist, factors=best), f, indent=1)
    print('factors', best)


def do_check(physics):
    from _common import write
    os.makedirs(OUT, exist_ok=True)
    ref = targets('dev')
    e = eui_by_use(physics, 2000)
    t = table(e, ref)
    with open(os.path.join(OUT, f'check_p{physics}.json'), 'w') as f:
        json.dump(t, f, indent=1)
    for u, v in t.items():
        print(u, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
    if physics != '1.0':
        base = json.load(open(os.path.join(OUT, 'check_p1.0.json')))
        fit = json.load(open(os.path.join(OUT, 'fit.json')))
        write('intensity_fit.json', dict(
            what='physics 1.4 generator intensity: factors fitted on the prior split, checked on the dev split',
            n_per_use=N, fit_seeds='1000 + i', check_seeds='2000 + i', factors=fit['factors'],
            prior_fit_history=fit['history'], dev_check_physics_1_0=base, dev_check=t,
            all_within_15pct=bool(all(v['within_15pct'] for v in t.values()))))


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        do_fit()
    else:
        do_check(sys.argv[2])
