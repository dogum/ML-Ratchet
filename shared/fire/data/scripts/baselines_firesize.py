"""FireSize classical baselines on FPA-FOD 2012-2020 CONUS (Short 2022), split from proc/firesize_split.parquet.
Targets: final size (ha, log10), escape = final size >= 100 acres (class D+), large = >= 1000 acres (class F+).
Features at discovery (all fitted on DEV fires only): region, NWCG cause class, discovery day-of-year and hour,
lat/lon, owner, gridMET at the discovery day and the day before (erc, bi, fm100, fm1000, vs, rmin, rmax, tmmx, pr, vpd).
Methods: median (dumb), clim (region x cause), erc (logistic / linear on ERC + BI + wind), gbm (HistGradientBoosting,
"real labels fitted on dev").  -> proc/baselines/firesize_exam.json, firesize_dev_cv.json"""
import os, sys, json, glob
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
sys.path.insert(0, _CODE)
from firedata import metrics as M
P = f'{_DATA}/data/proc'
OUT = f'{P}/baselines'
REGION = json.load(open(f'{P}/splits.json'))['meta']['region_map']
WX = ['erc', 'bi', 'fm100', 'fm1000', 'vs', 'rmin', 'rmax', 'tmmx', 'pr', 'vpd']


def load():
    fod = pd.read_parquet(f'{P}/fpafod/fod.parquet')
    fod = fod[fod.FIRE_YEAR >= 2012]
    sp = pd.read_parquet(f'{P}/firesize_split.parquet')[['FOD_ID', 'split', 'exam_period']]
    d = fod.merge(sp, on='FOD_ID')
    wx = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{P}/weather/gridmet/fod_*.parquet'))])
    d = d.merge(wx, on='FOD_ID', how='left')
    d['region'] = d.STATE.astype(str).map(REGION).fillna('NE_MW')
    d['cause'] = d.NWCG_CAUSE_CLASSIFICATION.astype(str)
    d['hour'] = pd.to_numeric(d.DISCOVERY_TIME.astype(str).str[:2], errors='coerce')
    d['size_ha'] = d.FIRE_SIZE.astype(float) * 0.404686
    d['escape'] = (d.FIRE_SIZE >= 100).astype(int)
    d['large'] = (d.FIRE_SIZE >= 1000).astype(int)
    d['lsize'] = np.log10(np.maximum(d.size_ha, 1e-3))
    return d


def feats(d):
    X = pd.DataFrame({'lat': d.LATITUDE, 'lon': d.LONGITUDE, 'doy': d.DISCOVERY_DOY, 'hour': d.hour})
    for v in WX:
        if v in d: X[v] = d[v]; X[v + '_m1'] = d.get(v + '_m1')
    for c in ['region', 'cause', 'OWNER_DESCR']:
        X[c] = pd.Categorical(d[c].astype(str))
    return X


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
    from sklearn.linear_model import LogisticRegression, LinearRegression
    d = load()
    dev, ex = d[d.split == 'dev'].copy(), d[d.split == 'exam'].copy()
    print('dev', len(dev), 'exam', len(ex), 'escape prev dev', dev.escape.mean(), flush=True)
    preds = {}
    # 1 median
    preds['median'] = (np.full(len(ex), dev.lsize.median()), np.full(len(ex), dev.escape.mean()), np.full(len(ex), dev.large.mean()))
    # 2 climatology region x cause (shrunk to region with k=20 pseudo-counts)
    g0 = dev.groupby('region').agg(m=('lsize', 'median'), e=('escape', 'mean'), l=('large', 'mean'))
    g = dev.groupby(['region', 'cause']).agg(m=('lsize', 'median'), e=('escape', 'mean'), l=('large', 'mean'), n=('lsize', 'size')).reset_index()
    g = g.merge(g0, left_on='region', right_index=True, suffixes=('', '_r'))
    w = g.n / (g.n + 20)
    for k in ['m', 'e', 'l']: g[k] = w * g[k] + (1 - w) * g[k + '_r']
    e2 = ex[['region', 'cause']].merge(g[['region', 'cause', 'm', 'e', 'l']], how='left', on=['region', 'cause'])
    for k, glob_ in [('m', dev.lsize.median()), ('e', dev.escape.mean()), ('l', dev.large.mean())]:
        e2[k] = e2[k].fillna(e2.region.map(g0[k])).fillna(glob_)
    preds['clim'] = (e2.m.values, e2.e.values, e2.l.values)
    # 3 ERC/BI/wind linear + logistic
    cols = ['erc', 'bi', 'vs']
    ok = dev[cols].notna().all(1)
    mu, sd = dev.loc[ok, cols].mean(), dev.loc[ok, cols].std()
    Z = lambda x: ((x[cols] - mu) / sd).fillna(0).values
    lr = LinearRegression().fit(Z(dev[ok]), dev.lsize[ok])
    le = LogisticRegression(C=1.0, max_iter=1000).fit(Z(dev[ok]), dev.escape[ok])
    ll = LogisticRegression(C=1.0, max_iter=1000).fit(Z(dev[ok]), dev.large[ok])
    preds['erc'] = (lr.predict(Z(ex)), le.predict_proba(Z(ex))[:, 1], ll.predict_proba(Z(ex))[:, 1])
    # 4 GBM on all features (real labels fitted on dev)
    Xd, Xe = feats(dev), feats(ex)
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
              categorical_features='from_dtype', random_state=0)
    gr = HGR(**kw).fit(Xd, dev.lsize); gc = HGC(**kw).fit(Xd, dev.escape); gl = HGC(**kw).fit(Xd, dev.large)
    preds['gbm'] = (gr.predict(Xe), gc.predict_proba(Xe)[:, 1], gl.predict_proba(Xe)[:, 1])
    res = {}
    for per, m in [('all', np.ones(len(ex), bool)), ('main_2012_2019', (ex.exam_period == 'main').values),
                   ('recent_2020', (ex.exam_period == 'recent').values),
                   ('mtbs_matched', ex.MTBS_ID.notna().values)]:
        res[per] = {}
        for k, (ls, pe, pl) in preds.items():
            r = dict(size=M.size_scores(ex.size_ha.values[m], 10 ** ls[m]),
                     escape=M.binary_scores(ex.escape.values[m], pe[m]) if ex.escape.values[m].std() > 0 else None,
                     large=M.binary_scores(ex.large.values[m], pl[m]) if ex.large.values[m].std() > 0 else None)
            res[per][k] = r
            print(per, k, json.dumps({a: {b: round(c, 3) for b, c in v.items()} for a, v in r.items() if v}), flush=True)
    json.dump(dict(n_dev=int(len(dev)), n_exam=int(len(ex)), labels=dict(escape='>=100 acres', large='>=1000 acres'),
                   wx_missing_frac_exam=float(ex.erc.isna().mean()), results=res), open(f'{OUT}/firesize_exam.json', 'w'), indent=1)
    print('DONE')


if __name__ == '__main__':
    main()
