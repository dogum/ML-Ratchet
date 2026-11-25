"""Does the rebuilt val bank reproduce EnergyNet's frozen set val2[h]? EnergyNet v2.1 (released weights) scored with
EnergyNet's own code (dataset.draw_tiers seed 777, dataset.assemble, target version 'e' on the h banks) against the
numbers in training/ratchet2.json (rung r17). -> results/check_bank_val.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import numpy as np
import dataset as DS, features as F
from model_np import Ensemble


def main():
    B = dict(np.load(os.path.join(C.BANK, 'val', 'rows.npz')))
    T = DS.draw_tiers(B, np.random.default_rng(C.VAL_TIER_SEED))
    X, E = DS.assemble(B, T)
    y = DS.target(B, T, 'e')
    ok = y > 0.02
    net = Ensemble(sorted(os.path.join(C.ENET, 'weights', f) for f in os.listdir(os.path.join(C.ENET, 'weights'))))
    p, s, c = net.predict(X[ok], E[ok]); yy = y[ok]; a = B['arch'][ok]; cl = B['cls'][ok]
    out = dict(n=int(ok.sum()), MdAPE=float(np.median(np.abs(p / yy - 1))), phys_MdAPE=float(np.median(np.abs(E[ok] / yy - 1))),
               z_cover68=float(np.mean(np.abs(np.log(p / yy)) < s)), med_log_err=float(np.median(np.abs(np.log(p / yy)))))
    for i in range(7):
        m = a == i
        if m.any(): out[f'MdAPE_{F.ARCHS[i]}'] = float(np.median(np.abs(p[m] / yy[m] - 1)))
    for i in range(7):
        m = cl == i
        if m.any(): out[f'MdAPE_{F.CLASSES[i]}'] = float(np.median(np.abs(p[m] / yy[m] - 1)))
    if os.path.exists(os.path.join(C.BANK, 'train', 'rows.npz')):
        Bt = dict(np.load(os.path.join(C.BANK, 'train', 'rows.npz')))
        Tt = DS.draw_tiers(Bt, np.random.default_rng(1)); yt = DS.target(Bt, Tt, 'e'); kt = Bt['arch'] * 10 + Bt['cls']
        g = (yt > 0.02) & (Bt['dist'] > 100)
        rates = {int(q): float(np.median(yt[g][kt[g] == q] / Bt['dist'][g][kt[g] == q])) for q in np.unique(kt[g])}
        k = a * 10 + cl
        pdumb = np.array([rates.get(int(q), np.nan) for q in k]) * B['dist'][ok]
        pdumb = np.where(np.isfinite(pdumb) & (pdumb > 0), pdumb, np.nan)
        out['skill'] = float(1 - np.median(np.abs(np.log(p / yy))) / np.nanmedian(np.abs(np.log(pdumb / yy))))
        out['dumb_MdAPE'] = float(np.nanmedian(np.abs(pdumb / yy - 1)))
    pub = [r for r in json.load(open(C.ENET_RATCHET2))['rungs'] if r['name'] == 'r17_noidle_inputs'][0]['val']['h']
    res = {'local': out, 'published_val2_h': pub,
           'diff': {k: out[k] - pub[k] for k in out if k in pub and isinstance(pub[k], (int, float))}}
    for k in out:
        print(f'{k:24s} local {out[k]:.4f}  published {pub.get(k, float("nan")):.4f}')
    json.dump(res, open(os.path.join(C.RESULTS, 'check_bank_val.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
