"""Twin test scores: the same rung on real dev samples and on their twins (twin_dev_samples.npz). Usage: twin_compare.py <ckpt|anchor> ..."""
import sys, json, numpy as np
from _paths import H; sys.path.insert(0, H + 'src')
import score2 as SC, train2 as T2, ratchet2 as RT, tt as TT
out = {}
for name in sys.argv[1:]:
    for tag, f in (('real', 'dev_samples.npz'), ('twin', 'twin_dev_samples.npz')):
        D = T2.load(H + 'data/' + f); rows = np.arange(len(D['K'])); SC._FEAT.clear()
        P = RT.preds(name, D, rows, real=True); e = SC.errors(P, D, rows); z = 10 ** D['truth'][:, 2]; sh = z <= 70
        dz = np.array([p['depth'] for p in P]); tp = []
        for s in rows:
            K = int(D['K'][s]); idx = D['idx'][s, :K]; t = D['truth'][s]; tp += list(D['n_tpick'][idx] - TT.tt('P', TT.dist(t[0], t[1], D['n_slat'][idx], D['n_slon'][idx]), 10 ** t[2]))
        tp = np.array(tp)
        out[f'{name}/{tag}'] = dict(epi_med_km=float(np.median(e['epi_km'])), epi_mean_km=float(np.mean(e['epi_km'])), mw_mae=float(np.mean(e['mw'])), mw_bias=float(np.mean([p['mw'] for p in P] - D['truth'][:, 3])),
                                    t0_med_s=float(np.median(e['t0'])), depth_ratio_shallow=float(np.median(dz[sh] / z[sh])), depth_ratio_deep=float(np.median(dz[~sh] / z[~sh])),
                                    pick_robust_sd=float(1.4826 * np.median(np.abs(tp - np.median(tp)))), pick_within5=float(np.mean(np.abs(tp) < 5)))
        print(name, tag, json.dumps({k: round(v, 3) for k, v in out[f'{name}/{tag}'].items()}), flush=True)
json.dump(out, open(H + 'data/twin_compare.json', 'w'), indent=1)
