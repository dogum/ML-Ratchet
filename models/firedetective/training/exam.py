"""FINAL EXAM (run once, at the end). Exam fires are loaded only here, with for_exam_scoring=True.

Exam set (fixed rule, seed 20261119): exam fires 2017-2023 (main 2017-21 + recent 2022-23; 2024 has no LANDFIRE
disturbance layer yet) with >= 1 FEDS growth day; all whose dominant fuel group is timber (TU/TL), plus a seeded
draw of 250 of the others.
Targets: MTBS high-severity fraction (where MTBS-coded), gridMET wind (network in wind-hidden mode), Globe-LFMC.
Baselines: dev median (dumb), Van Wagner crown-fire potential, ERC/BI/FWI max, Rothermel wind back-calculation,
LFMC dev-site climatology. Uses the incumbent rung of ratchet.json and its dev calibration (models/calib_<rung>.json).

python3 exam.py build   # resumable per fire (data/real/exam/<fid>.pkl)
python3 exam.py score   # -> out/exam.json
"""
from common import *
import pickle, time, traceback
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import spearmanr
import firedata as fd
import real as RL
import baselines as BL

EXD = f'{REAL}/exam'
os.makedirs(EXD, exist_ok=True)


def exam_list():
    f = fd.fires('exam')
    f = f[(f.year >= 2017) & (f.year <= 2023) & (f.feds_growth_days >= 1)].sort_values('Event_ID')
    forest = f[f.fuel_group.isin(['TU', 'TL'])]
    other = f[~f.fuel_group.isin(['TU', 'TL'])]
    rs = np.random.default_rng(20261119)
    pick = other.iloc[np.sort(rs.choice(len(other), min(250, len(other)), replace=False))]
    return list(forest.Event_ID) + list(pick.Event_ID)


def prefetch(fid):
    """Fetch the LANDFIRE layers for one exam fire with 4 threads (network-bound)."""
    r = fd._row(fid)
    if os.path.exists(f'{fd.P}/landscapes/{fid}.npz') or os.path.exists(f'{REAL}/lf/{fid}.npz'):
        return
    import importlib
    G = importlib.import_module('firedata.grid')
    g180 = G.coarsen(fd.grid(fid), 2)
    lays = (('FBFM40', 'fuel', 1), ('CC', 'cc', 1), ('CH', 'ch', 10), ('CBH', 'cbh', 10), ('CBD', 'cbd', 100),
            ('SlpD', 'slope', 1), ('Elev', 'elev', 1))
    with ThreadPoolExecutor(4) as ex:
        arrs = list(ex.map(lambda l: RL.lfps_export(RL.lf_service(l[0], int(r.year)), g180).astype(np.float32), lays))
    out = {}
    for (lay, key, scale), a in zip(lays, arrs):
        out[key] = np.where(a <= 0, 99, a).astype(np.int64) if key == 'fuel' else np.where(a < 0, 0, a / scale).astype(np.float32)
    os.makedirs(f'{REAL}/lf', exist_ok=True)
    np.savez_compressed(f'{REAL}/lf/{fid}.npz', **out)


def build():
    ids = exam_list()
    print('exam fires', len(ids), flush=True)
    for i, fid in enumerate(ids):
        fn = f'{EXD}/{fid}.pkl'
        if os.path.exists(fn):
            continue
        t = time.time()
        try:
            prefetch(fid)
            rec = RL.build_record(fid, for_exam=True, use_package=True, n_cells=100)
            if rec is not None:
                d = fd.weather(fid, for_exam_scoring=True)['daily'].copy()
                d.index = pd.DatetimeIndex(d.index).normalize()
                rec['base'] = BL.all_baselines(rec, d)
            with open(fn + '.tmp', 'wb') as f:
                pickle.dump(rec, f)
            os.replace(fn + '.tmp', fn)
        except Exception as e:
            print(fid, 'FAIL', repr(e)[:300], flush=True)
            with open(fn, 'wb') as f:
                pickle.dump(dict(fid=fid, error=repr(e)[:500]), f)
        if i % 10 == 0:
            print(i, fid, round(time.time() - t, 1), flush=True)


def boot_ci(f, n, B=1000, seed=0):
    rs = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rs.integers(0, n, n)
        v = f(i)
        if np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [np.nan, np.nan]


def rho_ci(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return dict(rho=np.nan, ci=[np.nan, np.nan], n=int(len(x)))
    return dict(rho=float(spearmanr(x, y)[0]), ci=boot_ci(lambda i: spearmanr(x[i], y[i])[0], len(x)), n=int(len(x)))


def mae_ci(p, y):
    ok = np.isfinite(p) & np.isfinite(y)
    e = np.abs(p[ok] - y[ok])
    if not len(e):
        return dict(mae=np.nan, ci=[np.nan, np.nan], n=0)
    return dict(mae=float(e.mean()), ci=boot_ci(lambda i: e[i].mean(), len(e)), n=int(len(e)))


def score():
    import torch  # noqa
    from ratchet import load_net
    from train import real_predict
    R = json.load(open(latest('ratchet.json')))
    rung = R['incumbent']
    cal = json.load(open(latest(f'models/calib_{rung}.json')))
    net, ns = load_net(rung)
    recs = []
    for fid in exam_list():
        fn = f'{EXD}/{fid}.pkl'
        if not os.path.exists(fn):
            continue
        r = pickle.load(open(fn, 'rb'))
        if r is None or 'error' in r:
            continue
        recs.append(r)
    p = real_predict(net, ns, recs)
    hs = np.array([r['hs'] for r in recs]); wind = np.array([r['wind'] for r in recs]); lf = np.array([r['lfmc'] for r in recs])
    ff = np.array([r['forest_frac'] for r in recs]); per = np.array([r['period'] for r in recs])
    B = {k: np.array([r['base'].get(k, np.nan) for r in recs], float) for k in
         ('cfp', 'erc_max', 'bi_max', 'fwi_max', 'wind_bc', 'lfmc_clim', 'lfmc_median')}
    # calibrated predictions
    hs_pred = cal['hs_map']['a'] + cal['hs_map']['b'] * p['crown']
    dcdy = 1.04 * np.exp(p['crown_y']) / (1 + np.exp(p['crown_y'])) ** 2
    hs_sd = abs(cal['hs_map']['b']) * dcdy * p['crown_sd'] * cal['hs_widen']
    wind_sd = p['wind_sd'] * cal['wind_widen']
    lf_sd = p['lfmc_sd'] * cal['lfmc_widen']
    out = dict(rung=rung, n_records=len(recs), n_listed=len(exam_list()), calib=cal)
    Z = 1.2816
    for sub_name, sub in (('all', np.ones(len(recs), bool)), ('main', per == 'main'), ('recent', per == 'recent')):
        S = {}
        for fs_name, fsel in (('forest', ff >= 0.5), ('all_fuels', np.ones(len(recs), bool))):
            m = sub & fsel & np.isfinite(hs)
            e = dict(n=int(m.sum()))
            e['rho'] = dict(network=rho_ci(p['crown'][m], hs[m]),
                            **{k: rho_ci(B[k][m], hs[m]) for k in ('cfp', 'erc_max', 'bi_max', 'fwi_max')})
            e['mae'] = dict(network=mae_ci(hs_pred[m], hs[m]), dumb=mae_ci(np.full(m.sum(), cal['dev_median_hs']), hs[m]),
                            **{k: mae_ci(cal[f'map_{k}'][0] + cal[f'map_{k}'][1] * B[k][m], hs[m])
                               for k in ('cfp', 'erc_max')})
            e['cov80_network'] = float(np.mean(np.abs(hs_pred[m] - hs[m]) < Z * hs_sd[m])) if m.any() else np.nan
            e['truth_median'] = float(np.median(hs[m])) if m.any() else np.nan
            S[fs_name] = e
        m = sub & np.isfinite(wind)
        W = dict(n=int(m.sum()), rho=dict(network=rho_ci(p['wind'][m], wind[m]), rothermel_bc=rho_ci(B['wind_bc'][m], wind[m])),
                 mae=dict(network=mae_ci(p['wind'][m], wind[m]), dumb=mae_ci(np.full(m.sum(), cal['dev_median_wind']), wind[m]),
                          rothermel_bc=mae_ci(cal['map_wind_bc'][0] + cal['map_wind_bc'][1] * B['wind_bc'][m], wind[m])),
                 cov80_network=float(np.mean(np.abs(p['wind_y'][m] - np.log(np.maximum(wind[m], 0.3))) < Z * wind_sd[m])) if m.any() else np.nan)
        S['wind'] = W
        m = sub & np.isfinite(lf)
        S['lfmc'] = dict(n=int(m.sum()), rho=dict(network=rho_ci(p['lfmc'][m], lf[m]), clim=rho_ci(B['lfmc_clim'][m], lf[m])),
                         mae=dict(network=mae_ci(p['lfmc'][m], lf[m]), dumb=mae_ci(B['lfmc_median'][m], lf[m]),
                                  clim=mae_ci(B['lfmc_clim'][m], lf[m])),
                         cov80_network=float(np.mean(np.abs(p['lfmc'][m] - lf[m]) < Z * lf_sd[m])) if m.any() else np.nan)
        out[sub_name] = S
    # per-fire table for the page (no selection: every scored fire)
    rows = []
    for i, r in enumerate(recs):
        rows.append(dict(fid=r['fid'], name=r['name'], year=r['year'], period=r['period'], region=r['region'],
                         fuel=r['fuel_group'], forest=round(float(ff[i]), 2), n_steps=r['n_steps'],
                         hs=None if not np.isfinite(hs[i]) else round(float(hs[i]), 3),
                         hs_pred=round(float(hs_pred[i]), 3), hs_sd=round(float(hs_sd[i]), 3),
                         crown=round(float(p['crown'][i]), 3), wind=round(float(wind[i]), 2),
                         wind_pred=round(float(p['wind'][i]), 2), lfmc=None if not np.isfinite(lf[i]) else round(float(lf[i]), 1),
                         lfmc_pred=round(float(p['lfmc'][i]), 1), ros=round(float(p['ros'][i]), 3),
                         m1=round(float(p['m1'][i]), 2), spot=round(float(p['spot'][i]), 2),
                         cfp=None if not np.isfinite(B['cfp'][i]) else round(float(B['cfp'][i]), 3),
                         erc=None if not np.isfinite(B['erc_max'][i]) else float(B['erc_max'][i]),
                         wind_bc=None if not np.isfinite(B['wind_bc'][i]) else round(float(B['wind_bc'][i]), 2)))
    out['fires'] = rows
    jdump(out, f'{ROOT}/out/exam.json')
    print(json.dumps({k: v for k, v in out.items() if k not in ('fires', 'calib')}, indent=1)[:6000])


if __name__ == '__main__':
    {'build': build, 'score': score}[sys.argv[1]]()
