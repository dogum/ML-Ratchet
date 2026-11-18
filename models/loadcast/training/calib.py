"""Dev calibration (BDG2 dev buildings, local 2016 ONLY; 2017 stays for the dev gate; exam never touched).

For each of the 52 dev buildings, simulate its BDG2-metadata twin on a small grid of physical knobs
(HVAC when the meters do not say it, operating schedule, outdoor-air multiplier, VAV minimum flow + fixed supply air),
fit y_real = a * y_sim + b * A per q_ok-2016 meter (a, b >= 0, hourly least squares), and keep the grid point with
the lowest mean hourly CV(RMSE). Output: data/dev_knobs.json (one knob vector per dev building, plus fit quality
before/after). The bank (bank.py) then draws its calibration knobs from these 52 vectors: an empirical prior.

    nice -n 5 python3 calib.py
"""
import sys, os, json, time, itertools
import numpy as np, pandas as pd
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, lcsim as L

OUT = f"{C.ROOT}/data/{os.environ.get('LC_KNOBS', 'dev_knobs.json')}"   # $LC_KNOBS, e.g. dev_knobs_p11.json


def grid_for(meters):
    ms = set(meters)
    vav = bool(ms & {'chilledwater', 'steam', 'hotwater'})
    if not (ms & {'chilledwater', 'steam', 'hotwater', 'gas'}): sysopts = ['auto', 'district', 'electric']
    elif 'gas' in ms and not (ms & {'chilledwater', 'steam', 'hotwater'}): sysopts = ['gas_rtu', 'gas_vav']; vav = True
    else: sysopts = ['auto']
    G = []
    for sy, sc, oa in itertools.product(sysopts, ['default', 'extended', '24x7'], [1.0, 2.5]):
        vs = [(0.3, True), (0.8, False)] if (vav or sy in ('district', 'gas_vav')) else [(0.3, True)]
        for vm, sr in vs:
            G.append(dict(sys=sy, sched=sc, oa=oa, vav_min=vm, sat_reset=sr))
    return G


def fit_ab(y, s, A):
    ok = np.isfinite(y) & np.isfinite(s)
    X = np.column_stack([s[ok], np.full(ok.sum(), A / 1000.0)])
    c, _ = nnls(X, y[ok])
    yh = X @ c
    cv = 100 * np.sqrt(np.mean((y[ok] - yh) ** 2)) / max(y[ok].mean(), 1e-9)
    return c, cv


def main():
    meta = C.bd.bdg2_meta(split='dev')
    q = C.bd.bdg2_quality(split='dev')
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    real = {m: C.real_meter(m, list(q[q.meter == m].building_id)) for m in C.METERS if (q.meter == m).any()}
    t0 = time.time()
    for _, b in meta.iterrows():
        bid = b['id']
        if bid in res: continue
        site = b['site_id']; cal = C.site_calendar(site)
        h16 = cal['year'] == 2016
        meters = C.meters_of(b['meters'])
        qq = q[(q.building_id == bid)].set_index('meter')
        fitm = [m for m in meters if m in qq.index and bool(qq.loc[m, 'q_ok_2016'])]
        base = C.bd.bdg2_spec(bid); A = float(b['floor_area_m2'])
        w, wm = C.site_weather(site)
        end = w.index[np.flatnonzero(h16)[-1]] + pd.Timedelta(hours=25)
        best = None; rows = []; generic = None
        for kn in grid_for(meters):
            spec = L.make_spec(base, meters, kn)
            try:
                sim, out = L.simulate_site(spec, site, end=end)
            except Exception as e:
                print('ERR', bid, kn, e, flush=True); continue
            per = {}
            for m in fitm:
                y = real[m][bid].values.copy(); y[~h16] = np.nan
                s = sim[m].copy(); s[~h16] = np.nan
                if np.nanmean(s) <= 0: s = np.zeros_like(s)
                c, cv = fit_ab(y, s, A)
                g = C.g14(y, s)
                per[m] = dict(a=float(c[0]), b=float(c[1]), cv_fit=float(cv), cv_raw=float(g[0]), nmbe_raw=float(g[1]),
                              sim_mean=float(np.nanmean(s)), real_mean=float(np.nanmean(y)))
            score = float(np.mean([v['cv_fit'] for v in per.values()])) if per else 0.0
            r = dict(knobs=kn, per=per, score=score, system=out['spec']['system'])
            rows.append(r)
            if generic is None and kn['sched'] == 'default' and kn['oa'] == 1.0 and kn['vav_min'] == 0.3 and kn['sys'] in ('auto', 'gas_rtu'):
                generic = r
            if best is None or score < best['score']: best = r
        res[bid] = dict(site=site, use=b['primary_use'], cz=b['climate_zone'], area=A, meters=meters, fit_meters=fitm,
                        best=best, generic=generic, n_grid=len(rows),
                        grid=[dict(knobs=r['knobs'], score=r['score']) for r in rows])
        print(f"{bid:32s} {len(rows):2d} sims  best {best['knobs']}  score {best['score']:.1f} (generic+ab "
              f"{generic['score'] if generic else float('nan'):.1f})  {round(time.time() - t0)} s", flush=True)
        C.jdump(res, OUT)


if __name__ == '__main__':
    main()
