"""Reproduce EnergyNet v2.1's published dev terms (training/ratchet2.json, rung r17_noidle_inputs) from its released
weights on the local data, with EnergyNet's own assembly (training/dataset.py) and metrics (training/evaluate.py).
-> results/check_enet_dev.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import numpy as np, pandas as pd
import dataset as DS, evaluate as EV
from model_np import Ensemble


def load(split, name):
    p = os.path.join(C.REAL, split, name)
    B = dict(np.load(p + '.npz')); n = len(B['arch'])
    T = {k: B['T_' + k] if ('T_' + k) in B else np.zeros(n, bool) for k in DS.TIER_KEYS}
    return B, T, pd.read_parquet(p + '_meta.parquet')


def frame(net, B, T, M):
    X, E = DS.assemble(B, T)
    p, s, c = net.predict(X, E)
    M = M.copy(); M['pred'] = p * B['vr']; M['sig'] = s; M['y'] = B['y']; M['E_phys'] = E * B['vr']
    return M


def main():
    net = Ensemble(sorted(os.path.join(C.ENET, 'weights', f) for f in os.listdir(os.path.join(C.ENET, 'weights'))))
    pub = [r for r in json.load(open(C.ENET_RATCHET2))['rungs'] if r['name'] == 'r17_noidle_inputs'][0]['dev']
    out = {'published_D2': pub['D'], 'terms': {}}
    ved = frame(net, *load('dev', 'ved'))
    rv = EV.ved_metrics(ved)
    B, T, M = load('dev', 'epa')
    epa = frame(net, B, T, M); epa['miles'] = B['dist'] / C.MI
    re_, _ = EV.epa_metrics(epa)
    B, T, M = load('dev', 'hd')
    hd_ = frame(net, B, T, M)                                    # v2.1's inputs (ZTBus told the 19 t curb mass)
    T0 = dict(T); T0['m_level'] = np.where(M.set.values == 'ZTBus_leg', M.m_level0.values, T['m_level'])
    hd0 = frame(net, B, T0, M)                                   # EnergyNet's own 'hd' records (no curb mass)
    rh = EV.hd_metrics(hd0, 'dev'); rh1 = EV.hd_metrics(hd_, 'dev')
    terms = []
    for g in ('ICE', 'HEV', 'EV', 'PHEV'):
        terms.append((f'VED_{g}', rv[g]['trip_MdAPE'], pub['VED'][g]['trip_MdAPE'], rv[g]['n'], pub['VED'][g]['n']))
    for k in EV.D2_EPA + ('hev_powersplit_US06', 'hev_powersplit_SC03', 'hev_powersplit_FTPcold'):
        terms.append((f'EPA_{k}', re_[k]['MdAPE'], pub['EPA'][k]['MdAPE'], re_[k]['n'], pub['EPA'][k]['n']))
    for k in ('Mersin_diesel', 'Mersin_cng', 'ICCT', 'ZTBus_legs', 'ZTBus_traced_missions'):
        terms.append((f'HD_{k}', rh[k]['MdAPE'], pub['HD'][k]['MdAPE'], rh[k]['n'], pub['HD'][k]['n']))
    for name, a, b, n, nb in terms:
        out['terms'][name] = dict(local=a, published=b, diff=a - b, n_local=n, n_published=nb)
        print(f'{name:28s} local {a:.4f} (n {n:5d})  published {b:.4f} (n {nb:5d})  diff {a - b:+.5f}')
    d16 = [t[1] for t in terms if t[0] in [f'VED_{g}' for g in ('ICE', 'HEV', 'EV', 'PHEV')] + [f'EPA_{k}' for k in EV.D2_EPA]
           + ['HD_Mersin_diesel', 'HD_Mersin_cng', 'HD_ICCT']]
    p16 = [t[2] for t in terms if t[0] in [f'VED_{g}' for g in ('ICE', 'HEV', 'EV', 'PHEV')] + [f'EPA_{k}' for k in EV.D2_EPA]
           + ['HD_Mersin_diesel', 'HD_Mersin_cng', 'HD_ICCT']]
    out['D16_local'] = float(np.mean(d16)); out['D16_published'] = float(np.mean(p16)); out['n_terms_D16'] = len(d16)
    out['ZTBus_full_terms_published_not_local'] = dict(legs=pub['ZTB']['legs'], missions=pub['ZTB']['missions'])
    out['ZTBus_v1_legs_with_curb_mass'] = rh1.get('ZTBus_legs')
    out['VED_full'] = rv
    print('D16 local', round(out['D16_local'], 5), 'published', round(out['D16_published'], 5))
    print('ZTBus v1 dev legs with v2.1 curb mass input:', rh1.get('ZTBus_legs'))
    json.dump(out, open(os.path.join(C.RESULTS, 'check_enet_dev.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
