"""print per-reading dev predictions of a checkpoint next to the classical scores"""
import _paths as _P  # portable paths (see _paths.py)
import sys, numpy as np
sys.path.insert(0, _P.SRC); import bnet as BN, exam as EX
net = BN.load_net(_P.WORK + '/ckpt/' + sys.argv[1]); R = BN.readings(net, 'dev')
tau, ba = EX.fit_tau([r for r in R if r['kind'] == 'main']); print('classical tau', tau, 'dev bal acc', round(ba, 3))
import collections; agg = collections.defaultdict(list)
for r in R:
    key = (r['src'], r['rec'].split('/')[0] if r['src'] == 'PB' else r['rec'], r.get('label_txt', ''), r['chan'], r['loc'], r['kind'], r['cls'])
    agg[key].append(r)
for k, rs in agg.items():
    p = np.mean([x['p'] for x in rs], 0); preds = collections.Counter(x['pred'] for x in rs); cl = np.mean([x['classical'] for x in rs], 0)
    print(k, 'n', len(rs), 'p', np.round(p, 2), dict(preds), 'classical', np.round(cl, 2), 'size %.2f->%.2f' % (rs[0]['size_mm'], np.mean([10 ** x['lsize_pred'] for x in rs])))
