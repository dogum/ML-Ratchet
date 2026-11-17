"""Fit the physics 1.2 electricity parameters (campus.py: s_on, sigma_d, rho_d) on BDG2 DEV buildings only, 2016, and
test the outcome the fit does not target: real vs twin daily PRISM r2 distributions (KS + quartiles), in sample, on
2017, and leave-one-site-out. Plan: shared/building/NOTEBOOK.md, 17 Nov 2026.

    python buildingsim/tools/calibrate_electricity.py run 2016     # the grid at 1.2 (on the fitted 1.1), <= 3 workers
    python buildingsim/tools/calibrate_electricity.py fit          # fit + LOSO choices -> data/physics.json
    python buildingsim/tools/calibrate_electricity.py run 2017     # only the combos the fit and the LOSO folds chose
    python buildingsim/tools/calibrate_electricity.py report       # checks/out/electricity_fit.json

Fit: rho_d = the grid value nearest the median lag-1 autocorrelation of the real day-to-day residuals; (s_on, sigma_d)
minimise the squared relative errors of the medians of the weekday night/day ratio and of the day-to-day residual sd.
The PRISM r2 is never a fit target. Each building's draws are seeded by crc32 of its id (as check_bdg2.py).
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE
import itertools
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, BLDG_CODE)
sys.path.insert(0, os.path.join(BLDG_CODE, 'buildingsim'))
sys.path.insert(0, os.path.join(BLDG_CODE, 'checks'))
os.environ.setdefault('OMP_NUM_THREADS', '1')

OUT = os.path.join(BLDG_ROOT, 'checks', 'out', 'electricity_fit')
S_ON = (0.0, 0.15, 0.3, 0.45, 0.6)
SIGMA = (0.0, 0.05, 0.1, 0.15, 0.2)
RHO = (0.5, 0.7, 0.85)


def combos():
    out = []
    for s, g in itertools.product(S_ON, SIGMA):
        for r in (RHO if g > 0 else (0.0,)):
            out.append((s, g, r))
    return out


def dev_buildings():
    import buildingdata as bd
    meta = bd.bdg2_meta(split='dev')
    return [dict(id=b['id'], site=b['site_id']) for _, b in meta.iterrows() if b.get('q_ok_electricity')]


def stats_one(job):
    """One building, one year, a list of (s_on, sigma, rho): electricity statistics of real and twin."""
    bid, site, year, cmb = job
    path = os.path.join(OUT, f'{bid}_{year}.json')
    have = json.load(open(path)) if os.path.exists(path) else {}
    todo = [c for c in cmb if json.dumps(c) not in have]
    if not todo:
        return path
    import buildingdata as bd
    from buildingsim import simulate
    from check_bdg2 import spec_from_bdg2, elec_stats
    from calibrate_district import _weather
    w, wm = _weather(site)
    spec = spec_from_bdg2(bd.bdg2_spec(bid))
    start = pd.Timestamp(f'{year}-01-01', tz=wm['tz']).tz_convert('UTC')
    end = pd.Timestamp(f'{year + 1}-01-01', tz=wm['tz']).tz_convert('UTC')
    real = None
    for c in todo:
        out = simulate(spec, (w, wm), start, end, dt='1h', options={
            'physics': '1.2', 'seed': zlib.crc32(bid.encode()), 'campus': dict(s_on=c[0], sigma_d=c[1], rho_d=c[2])})
        ts = out['ts']
        if real is None:
            real = bd.bdg2('electricity', building=bid).iloc[:, 0].reindex(ts.index)
            m = real.notna().values
            loc = ts.index[m].tz_convert(wm['tz'])
            rv = real.values[m]
            wk = loc.day_of_week < 5
            night = wk & ((loc.hour < 5) | (loc.hour >= 22))
            dayh = wk & (loc.hour >= 10) & (loc.hour < 16)
            dT = pd.Series(ts['T_out'].values[m], index=loc).resample('D').mean()
            dr = pd.Series(rv, index=loc).resample('D').mean()
            have['real'] = dict(night_day=float(rv[night].mean() / rv[dayh].mean()), mean=float(rv.mean()),
                                **elec_stats(dr, dT))
        sv = ts['meter_electricity_kW'].values[m]
        ds = pd.Series(sv, index=loc).resample('D').mean()
        have[json.dumps(c)] = dict(night_day=float(sv[night].mean() / sv[dayh].mean()),
                                   ratio=float(sv.mean() / rv.mean()), **elec_stats(ds, dT))
    with open(path, 'w') as f:
        json.dump(have, f)
    return path


def run(year, cmb_for=None):
    os.makedirs(OUT, exist_ok=True)
    bs = dev_buildings()
    if cmb_for is None:
        jobs = [(b['id'], b['site'], year, combos()) for b in bs]
    else:
        jobs = [(b['id'], b['site'], year, cmb_for(b)) for b in bs]
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        for p in ex.map(stats_one, jobs):
            print('done', os.path.basename(p), f'{time.time() - t0:.0f} s', flush=True)


def load(year):
    rows = []
    for b in dev_buildings():
        p = os.path.join(OUT, f"{b['id']}_{year}.json")
        if os.path.exists(p):
            rows.append(dict(b=b, d=json.load(open(p))))
    return rows


def choose(rows):
    real_nd = np.median([r['d']['real']['night_day'] for r in rows])
    real_cv = np.median([r['d']['real']['resid_cv'] for r in rows])
    real_l1 = np.median([r['d']['real']['resid_lag1'] for r in rows])
    rho = min(RHO, key=lambda x: abs(x - real_l1))
    best = None
    for c in combos():
        if c[1] > 0 and c[2] != rho:
            continue
        k = json.dumps(list(c))
        nd = np.median([r['d'][k]['night_day'] for r in rows])
        cv = np.median([r['d'][k]['resid_cv'] for r in rows])
        obj = ((nd - real_nd) / real_nd) ** 2 + ((cv - real_cv) / real_cv) ** 2
        if best is None or obj < best[0]:
            best = (obj, list(c), dict(night_day_real=real_nd, night_day_sim=nd, resid_cv_real=real_cv,
                                      resid_cv_sim=cv, resid_lag1_real=real_l1))
    return best


def do_fit():
    rows = load(2016)
    obj, c, info = choose(rows)
    folds = {}
    for s in sorted({r['b']['site'] for r in rows}):
        folds[s] = choose([r for r in rows if r['b']['site'] != s])[1]
    pf = os.path.join(BLDG_CODE, 'buildingsim', 'buildingsim', 'data', 'physics.json')
    P = json.load(open(pf))
    P.update(s_on=c[0], sigma_d=c[1], rho_d=c[2])
    P['_electricity'] = ('physics 1.2 fitted by tools/calibrate_electricity.py on the BDG2 dev buildings, 2016 '
                         f'({len(rows)} electricity meters); report checks/electricity_fit.json')
    with open(pf, 'w') as f:
        json.dump(P, f, indent=1)
    with open(os.path.join(OUT, 'choice.json'), 'w') as f:
        json.dump(dict(fit=c, objective=obj, info=info, loso=folds), f, indent=1)
    print('fit', c, info, '\nloso', folds)


def r2_table(pairs):
    from scipy.stats import ks_2samp
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    ks = ks_2samp(a, b)
    out = dict(n=len(a), ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
    for q in (25, 50, 75):
        out[f'real_p{q}'] = float(np.percentile(a, q))
        out[f'sim_p{q}'] = float(np.percentile(b, q))
    out['quartiles_within_0.05'] = bool(all(abs(out[f'real_p{q}'] - out[f'sim_p{q}']) <= 0.05 for q in (25, 50, 75)))
    out['match'] = bool(out['ks_p'] > 0.05 and out['quartiles_within_0.05'])
    return out


def other(rows, pick):
    keys = ('night_day', 'resid_cv', 'weekday_effect', 'resid_lag1', 'prism_r2')
    out = {}
    for k in keys:
        out[f'{k}_real_median'] = float(np.median([r['d']['real'][k] for r in rows]))
        out[f'{k}_sim_median'] = float(np.median([r['d'][json.dumps(pick(r))][k] for r in rows]))
    out['annual_ratio_median'] = float(np.median([r['d'][json.dumps(pick(r))]['ratio'] for r in rows]))
    return out


def report():
    from _common import write
    ch = json.load(open(os.path.join(OUT, 'choice.json')))
    res = {}
    for year in (2016, 2017):
        rows = load(year)
        for name, pick in (('physics_1_1', lambda r: [0.0, 0.0, 0.0]), ('fit', lambda r: ch['fit']),
                           ('loso', lambda r: ch['loso'][r['b']['site']])):
            pairs = [(r['d']['real']['prism_r2'], r['d'][json.dumps(pick(r))]['prism_r2']) for r in rows
                     if json.dumps(pick(r)) in r['d']]
            res[f'{name}_{year}'] = dict(prism_r2=r2_table(pairs), **other(rows, pick))
    write('electricity_fit.json', dict(what='physics 1.2 electricity fit on BDG2 dev (2016) and the r2 test',
                                       grid=dict(s_on=S_ON, sigma_d=SIGMA, rho_d=RHO), choice=ch, **res))
    for k, v in res.items():
        print(k, json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v['prism_r2'].items()}),
              {kk: round(vv, 3) for kk, vv in v.items() if kk != 'prism_r2'})


if __name__ == '__main__':
    if sys.argv[1] == 'run' and sys.argv[2] == '2016':
        run(2016)
    elif sys.argv[1] == 'run':
        ch = json.load(open(os.path.join(OUT, 'choice.json')))
        need = lambda b: [[0.0, 0.0, 0.0], ch['fit'], ch['loso'][b['site']]]  # noqa: E731
        run(int(sys.argv[2]), need)
    elif sys.argv[1] == 'fit':
        do_fit()
    else:
        report()
