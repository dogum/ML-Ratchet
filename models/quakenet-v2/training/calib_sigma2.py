"""Widen-only error-bar factors, one per output, fitted on the real dev samples only (40 events: first 2 stations and all
stations), for QuakeNet v2 (the released rung) and for the classical combination (so both are compared fairly).
Factor = max(1, the 68th percentile of |error| / sigma); for the epicentre, of the Mahalanobis radius against chi2(2)."""
import sys, json, numpy as np
from _paths import H; sys.path.insert(0, H + 'src')
import score2 as SC, train2 as T2, ratchet2 as RT

if __name__ == '__main__':
    name = sys.argv[1]; D = T2.load(H + 'data/dev_samples.npz'); rows = np.arange(len(D['K']))
    P = RT.preds(name, D, rows, real=True); e = SC.errors(P, D, rows); f2 = SC.widen_factors(e)
    V = RT.preds('anchor', D, rows, real=True); ev = SC.errors(V, D, rows); fv = SC.widen_factors(ev)
    raw = {k: float(np.mean(e[k + '_ratio'] < 1)) for k in ('z', 'mw', 't0')}; raw['epi'] = float(np.mean(e['epi_m2'] < SC.CHI2_68))
    out = dict(rung=name, v2=f2, v12_combined=fv, dev_raw_coverage_v2=raw, n=len(rows)); json.dump(out, open(H + 'data/sigma_factors.json', 'w'), indent=1); print(json.dumps(out, indent=1))
