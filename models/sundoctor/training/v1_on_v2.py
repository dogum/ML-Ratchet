"""Score the released v1 (rung 3) on the v2 frozen set: it gets its 10 familiar channels; it has no snow or stuck-tracker
outputs (counted as predicting 0) and was never trained without an irradiance sensor."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC)
import evaluate as E1, evaluate2 as E2
OLD = [0, 1, 3, 4, 6, 7, 8, 9, 10, 11]
def pred(net, X):
    P = E1.predict(net, X[:, OLD])
    P['snow'] = np.zeros_like(P['pf']); P['stuck'] = np.zeros_like(P['pf']); P['snow_sig'] = np.full_like(P['pf'], 0.1); P['stuck_sig'] = np.full_like(P['pf'], 0.1)
    return P
r = E2.evaluate(E1.load(_P.WORK + '/ckpt/s3_polish.pt'), pred)
json.dump(r, open(_P.WORK + '/ckpt/s3_polish_v2val.json', 'w'), indent=1)
for m in ('poa', 'home', 'trackers'):
    print(m, 'objective %.3f' % r[m]['objective'], {k: round(r[m][k]['skill'], 2) for k in E2.SKILLS}, 'step %.3f count %.3f' % (r[m]['step_auc'], r[m]['count_acc']))
print('overall %.3f' % r['objective'])
