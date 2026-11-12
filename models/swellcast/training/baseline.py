"""Classical ridge fit (Munk) and a dumb baseline on the frozen sim set, dev and exam (confirmed / visible / all pairs).
Dumb baseline: always answer the dev set's median log distance, median t0 and circular-mean direction.
Classical error bars: one constant sigma per output, set to the dev MAE x 1.25 (sigma of a Laplace-ish error ~ MAE)."""
import os, sys, json, numpy as np
import classical as CL, evaluate as E
ROOT = E.ROOT

def run(X):
    R = [CL.read(x) for x in X]; return dict(D_km=np.array([r['D_km'] for r in R]), t0=np.array([r['t0'] for r in R]), dir=np.array([r['dir'] for r in R]), hs=np.array([r['hs'] for r in R]))

def truth(ev):
    return np.array([e['D'] / 1e3 for e in ev]), np.array([e['t0_rel'] for e in ev]), np.array([e['dir'] for e in ev]), np.array([e['dir_coverage'] > 0.5 for e in ev]), np.array([e['wind'] for e in ev])

if __name__ == '__main__':
    from scipy.stats import spearmanr
    out = {}; evd, Xd = E.real_windows('dev', 'confirmed'); Dd, td, dd, _, _ = truth(evd)
    dumb = dict(logD=float(np.median(np.log10(Dd))), t0=float(np.median(td)), dir=float(np.degrees(np.arctan2(np.sin(np.radians(dd)).mean(), np.cos(np.radians(dd)).mean())) % 360))
    Pd = run(Xd); m = E.metrics(Pd, Dd, td, dd)
    sig = dict(logD=1.25 * m['logD_mae'], t0=1.25 * m['t0_mae_h'], dir=1.25 * m['dir_mean_deg']); out['dumb_values'] = dumb; out['classical_sigma'] = sig
    for split, vis in (('dev', 'confirmed'), ('exam', 'confirmed'), ('exam', True), ('exam', None)):
        ev, X = E.real_windows(split, vis); D, t, d, hd, w = truth(ev); P = run(X)
        P.update(logD_sig=np.full(len(ev), sig['logD']), t0_sig=np.full(len(ev), sig['t0']), kappa=np.full(len(ev), 1 / np.radians(sig['dir']) ** 2))
        mc = E.metrics(P, D, t, d, None, hd); mc['hs_vs_wind_spearman'] = float(spearmanr(np.nan_to_num(P['hs']), w).correlation)
        Q = dict(D_km=np.full(len(ev), 10 ** dumb['logD']), t0=np.full(len(ev), dumb['t0']), dir=np.full(len(ev), dumb['dir']))
        md = E.metrics(Q, D, t, d, None, hd); key = f'{split}_{vis}'; out[key] = dict(classical=mc, dumb=md); out[key + '_classical_pred'] = {k: np.asarray(v).tolist() for k, v in P.items()}
        print(key, 'n', len(ev), '\n  classical', {k: round(v, 3) for k, v in mc.items()}, '\n  dumb', {k: round(v, 3) for k, v in md.items()}, flush=True)
    Xv, Yv = E.val_set(); vis = Yv[:, 4] >= 0.3; idx = np.where(vis)[0][:600]; P = run(np.asarray(Xv[idx]))
    out['val_classical'] = E.metrics(P, Yv[idx, 0], Yv[idx, 1], Yv[idx, 2]); print('sim val classical', {k: round(v, 3) for k, v in out['val_classical'].items()})
    json.dump(out, open(os.path.join(ROOT, 'baseline.json'), 'w'), indent=1)
