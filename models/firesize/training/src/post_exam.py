"""Post-exam analysis (after the single exam run; chooses nothing): re-runs B's FireSize baselines exactly as in
shared/fire/data/scripts/baselines_firesize.py (imported, not edited) to get per-region scores of climatology, ERC
and the dev-fitted GBM, checks they reproduce B's stored all-exam numbers, and bootstraps the escape-AUC differences.
-> results/post_exam.json"""
import sys, json
from common import *
sys.path.insert(0, f'{FIRE}/data/scripts')   # was the sandbox's fire/data/scripts
import baselines_firesize as BF
from firedata import metrics as M
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier as HGC
from sklearn.linear_model import LogisticRegression


def main():
    d = BF.load()
    dev, ex = d[d.split == 'dev'].copy(), d[d.split == 'exam'].reset_index(drop=True)
    pr = pd.read_parquet(f'{ROOT}/results/exam_preds.parquet', columns=['FOD_ID', 'p_escape', 'p_large'])
    ex = ex.merge(pr, on='FOD_ID', how='left')
    assert ex.p_escape.notna().all()
    # climatology escape (B's shrunk region x cause)
    g0 = dev.groupby('region').agg(e=('escape', 'mean'))
    g = dev.groupby(['region', 'cause']).agg(e=('escape', 'mean'), n=('lsize', 'size')).reset_index()
    g = g.merge(g0, left_on='region', right_index=True, suffixes=('', '_r'))
    w = g.n / (g.n + 20); g['e'] = w * g.e + (1 - w) * g.e_r
    e2 = ex[['region', 'cause']].merge(g[['region', 'cause', 'e']], how='left', on=['region', 'cause'])
    ex['clim'] = e2.e.fillna(e2.region.map(g0.e)).fillna(dev.escape.mean()).values
    cols = ['erc', 'bi', 'vs']; ok = dev[cols].notna().all(1)
    mu, sd = dev.loc[ok, cols].mean(), dev.loc[ok, cols].std()
    Z = lambda x: ((x[cols] - mu) / sd).fillna(0).values
    ex['erc_p'] = LogisticRegression(C=1.0, max_iter=1000).fit(Z(dev[ok]), dev.escape[ok]).predict_proba(Z(ex))[:, 1]
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
              categorical_features='from_dtype', random_state=0)
    ex['gbm'] = HGC(**kw).fit(BF.feats(dev), dev.escape).predict_proba(BF.feats(ex))[:, 1]
    stored = json.load(open(f'{FP}/baselines/firesize_exam.json'))['results']['all']
    res = dict(check={k: dict(ours=float(roc_auc_score(ex.escape, ex[c])), stored=stored[k]['escape']['auc'])
                      for k, c in [('clim', 'clim'), ('erc', 'erc_p'), ('gbm', 'gbm')]}, region={})
    print('reproduce', res['check'], flush=True)
    for rg in REGIONS:
        m = (ex.region == rg).values
        res['region'][rg] = {k: dict(auc=float(roc_auc_score(ex.escape[m], ex[c][m])),
                                     ap=M.binary_scores(ex.escape.values[m], ex[c].values[m])['ap'])
                             for k, c in [('firesize', 'p_escape'), ('clim', 'clim'), ('erc', 'erc_p'), ('gbm', 'gbm')]}
        res['region'][rg]['n'] = int(m.sum()); res['region'][rg]['prev'] = float(ex.escape[m].mean())
        print(rg, {k: round(v['auc'], 3) for k, v in res['region'][rg].items() if isinstance(v, dict)}, flush=True)
    # bootstrap (ignitions resampled, 100 draws, seeded) of AUC differences
    rs = np.random.default_rng(SEED)
    y = ex.escape.values; n = len(y)
    diffs = {'firesize-gbm': [], 'firesize-clim': [], 'firesize-erc': []}
    for b in range(100):
        i = rs.integers(0, n, n)
        a = roc_auc_score(y[i], ex.p_escape.values[i])
        for k, c in [('firesize-gbm', 'gbm'), ('firesize-clim', 'clim'), ('firesize-erc', 'erc_p')]:
            diffs[k].append(a - roc_auc_score(y[i], ex[c].values[i]))
    res['bootstrap_auc_diff'] = {k: dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
                                 for k, v in diffs.items()}
    print(res['bootstrap_auc_diff'], flush=True)
    json.dump(res, open(f'{ROOT}/results/post_exam.json', 'w'), indent=1, default=float)
    print('DONE')


if __name__ == '__main__':
    main()
