"""Twin test (dev only): median real BDG2 dev feature vs median of its simulated twins, per channel, for a few
interpretable features. usage: python3 twintest.py devtwin_v1 devtwin_v2"""
import json
import sys

import numpy as np

import common as C
import features as F

P = 3 + 2 * F.NB + 168 + 48
S = P + 6 + 12 + 10
NAMES = {'log10 mean kW': 2, 'night/day': S + 3, 'weekend/weekday': S + 4, 'heating slope': P + 1,
         'cooling slope': P + 2, 'PRISM r2': P + 5, 'p99/3': S + 2}


def table(tag):
    r = np.load('../feats/real/dev_bdg2.npz')
    L = json.load(open('../feats/real/dev_bdg2.json'))
    t = np.load(f'../feats/real/{tag}_hourly.npz')
    Xr, Xt, bid = r['hourly'], t['X'], t['bid']
    out = {}
    for c in C.CHANNELS:
        o = F.N_W_H + C.CHANNELS.index(c) * F.N_C_H
        for n, k in NAMES.items():
            rr, tt = [], []
            for i, l in enumerate(L):
                m = bid == l['id']
                if m.any() and Xr[i, o] > 0 and (Xt[m, o] > 0).any():
                    rr.append(Xr[i, o + k]); tt.append(np.median(Xt[m, o + k][Xt[m, o] > 0]))
            if rr:
                out[f'{c}|{n}'] = dict(n=len(rr), real=float(np.median(rr)), twin=float(np.median(tt)))
    return out


if __name__ == '__main__':
    res = {tag: table(tag) for tag in sys.argv[1:]}
    json.dump(res, open('../models/twintest.json', 'w'), indent=1)
    keys = list(res[sys.argv[1]])
    print('%-28s %4s %8s ' % ('feature', 'n', 'real') + ' '.join('%10s' % t for t in sys.argv[1:]))
    for k in keys:
        print('%-28s %4d %8.3f ' % (k, res[sys.argv[1]][k]['n'], res[sys.argv[1]][k]['real']) +
              ' '.join('%10.3f' % res[t][k]['twin'] if k in res[t] else '%10s' % '-' for t in sys.argv[1:]))
