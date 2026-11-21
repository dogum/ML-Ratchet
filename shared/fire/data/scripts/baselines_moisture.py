"""Live fuel moisture (LFMC) classical baselines: Globe-LFMC 2.0 CONUS samples 2012-2022 vs gridMET drought indices.
Moisture split (frozen here, seed 20261118 + 3, by SITE): 10 % of sites dev, the rest exam -> proc/moisture_split.json.
Features at the sampling date (gridMET at the site, local day): ERC, FM100, FM1000, VPD and Tmax (7-day means),
precip 30-day and 90-day sums, day of year, functional type.  Fitted on dev sites only.
Methods: median (dumb), clim (functional type x month), erc_lin (LFMC ~ ERC per functional type, the classic
NFDRS relation), gbm (HistGradientBoosting on all features; real labels fitted on dev).
-> proc/baselines/moisture_exam.json
Dead fuel (10-h stick) observations are NOT here: NFMD moved from wfas.net to FEMS (see DATA.md)."""
import os, sys, json, glob
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
sys.path.insert(0, _CODE)
from firedata import metrics as M
P = f'{_DATA}/data/proc'
SPLIT = f'{P}/moisture_split.json'


def site_split(sites):
    if os.path.exists(SPLIT): return json.load(open(SPLIT))
    s = sorted(sites); r = np.random.default_rng(20261118 + 3).random(len(s))
    d = dict(rule='Globe-LFMC CONUS sites; seeded 10% dev (seed 20261118+3), rest exam; all years 2012-2022',
             dev=[x for x, v in zip(s, r) if v < 0.10], n_sites=len(s))
    json.dump(d, open(SPLIT, 'w'), indent=1); return d


def load():
    lf = pd.read_parquet(f'{P}/lfmc/lfmc_us.parquet')
    lf = lf.rename(columns={'Site name': 'site', 'LFMC value (%)': 'lfmc', 'Species functional type': 'ftype'})
    lf['date'] = pd.to_datetime(lf['date']).dt.normalize()
    lf = lf[(lf.date.dt.year >= 2012) & (lf.date.dt.year <= 2022) & lf.lfmc.between(20, 400)]
    rows = []
    for fn in sorted(glob.glob(f'{P}/weather/gridmet/lfmc_*.npz')):
        z = np.load(fn); days = z['days'].astype('datetime64[D]'); sites = list(z['sites'])
        V = {k: z[k].astype(np.float32) for k in z.files if k not in ('days', 'sites')}
        y = lf[lf.date.dt.year == int(str(days[0])[:4])]
        si = {s: i for i, s in enumerate(sites)}
        for r in y.itertuples():
            j = si.get(r.site)
            if j is None: continue
            t = int((np.datetime64(r.date.date()) - days[0]).astype(int))
            if t < 0 or t >= len(days): continue
            a = max(t - 6, 0)
            f = dict(idx=r.Index)
            for k in ['erc', 'fm100', 'fm1000', 'vpd', 'tmmx', 'rmin', 'vs']:
                if k in V: f[k + '7'] = float(np.nanmean(V[k][a:t + 1, j]))
            f['pr30'] = float(np.nansum(V['pr'][max(t - 29, 0):t + 1, j])) if t >= 29 else np.nan
            f['pr90'] = float(np.nansum(V['pr'][max(t - 89, 0):t + 1, j])) if t >= 89 else np.nan
            rows.append(f)
    F = pd.DataFrame(rows).set_index('idx')
    d = lf.join(F, how='inner')
    d['doy'] = d.date.dt.dayofyear; d['month'] = d.date.dt.month
    return d


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor as HGR
    d = load()
    sp = site_split(d.site.unique())
    d['split'] = np.where(d.site.isin(sp['dev']), 'dev', 'exam')
    dev, ex = d[d.split == 'dev'], d[d.split == 'exam']
    print('samples dev', len(dev), 'exam', len(ex), 'sites', d.site.nunique(), flush=True)
    pr = {'median': np.full(len(ex), dev.lfmc.median())}
    c = dev.groupby(['ftype', 'month']).lfmc.median()
    c1 = dev.groupby('month').lfmc.median()
    pr['clim'] = np.array([c.get((a, b), c1.get(b, dev.lfmc.median())) for a, b in zip(ex.ftype, ex.month)])
    p = np.full(len(ex), dev.lfmc.median())
    for ft, g in dev.groupby('ftype'):
        g = g[g.erc7.notna()]
        if len(g) < 30: continue
        A = np.c_[np.ones(len(g)), g.erc7]; co = np.linalg.lstsq(A, g.lfmc, rcond=None)[0]
        m = (ex.ftype == ft).values & ex.erc7.notna().values
        p[m] = co[0] + co[1] * ex.erc7.values[m]
    pr['erc_lin'] = p
    cols = ['erc7', 'fm1007', 'fm10007', 'vpd7', 'tmmx7', 'rmin7', 'vs7', 'pr30', 'pr90', 'doy',
            'Latitude (WGS84, EPSG:4326)', 'Longitude (WGS84, EPSG:4326)']
    X = lambda q: q[cols].assign(ftype=pd.Categorical(q.ftype.astype(str), categories=sorted(d.ftype.astype(str).unique())))
    g = HGR(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, categorical_features='from_dtype',
            random_state=0).fit(X(dev), dev.lfmc)
    pr['gbm'] = g.predict(X(ex))
    res = {}
    for k, v in pr.items():
        res[k] = M.moisture_scores(ex.lfmc.values, v)
        print(k, {a: round(b, 2) for a, b in res[k].items()}, flush=True)
    json.dump(dict(n_dev=int(len(dev)), n_exam=int(len(ex)), n_dev_sites=len(sp['dev']), n_sites=int(d.site.nunique()),
                   units='LFMC % dry mass', results=res), open(f'{P}/baselines/moisture_exam.json', 'w'), indent=1)
    print('DONE')


if __name__ == '__main__':
    main()
