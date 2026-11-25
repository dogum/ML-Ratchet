"""LightGBM booster -> flat numpy arrays, and a numpy-only predictor (no LightGBM needed at inference).

Arrays: feat (int32, -1 at leaves), thr (float64), left, right (int32; leaves point to themselves), value (float64,
leaf outputs incl. shrinkage), roots (int32, one per tree), depth (int, max node depth). A row goes left when
x <= thr (LightGBM's numerical '<=' split; the arm's features never contain NaN).
"""
import numpy as np


def export(bst):
    dump = bst.dump_model(num_iteration=bst.best_iteration if bst.best_iteration > 0 else None)
    feat, thr, left, right, value, roots = [], [], [], [], [], []
    maxdepth = 0

    def add(node, d):
        nonlocal maxdepth
        i = len(feat)
        feat.append(-1); thr.append(0.0); left.append(i); right.append(i); value.append(0.0)
        if 'leaf_value' in node and 'split_feature' not in node:
            value[i] = float(node['leaf_value'])
            maxdepth = max(maxdepth, d)
            return i
        assert node['decision_type'] == '<=', node['decision_type']
        feat[i] = int(node['split_feature']); thr[i] = float(node['threshold'])
        left[i] = add(node['left_child'], d + 1)
        right[i] = add(node['right_child'], d + 1)
        return i

    for t in dump['tree_info']:
        roots.append(add(t['tree_structure'], 0))
    return {'feat': np.array(feat, np.int32), 'thr': np.array(thr, np.float64), 'left': np.array(left, np.int32),
            'right': np.array(right, np.int32), 'value': np.array(value, np.float64),
            'roots': np.array(roots, np.int32), 'depth': np.array(maxdepth, np.int32)}


def predict(a, X, chunk=4000):
    X = np.asarray(X, np.float64)
    feat, thr, left, right, value, roots = a['feat'], a['thr'], a['left'], a['right'], a['value'], a['roots']
    depth = int(a['depth'])
    out = np.empty(len(X))
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        n = len(x)
        idx = np.broadcast_to(roots[None, :], (n, len(roots))).copy()
        rows = np.arange(n)[:, None]
        for _ in range(depth):
            f = feat[idx]
            leaf = f < 0
            if leaf.all():
                break
            xv = x[rows, np.maximum(f, 0)]
            nxt = np.where(xv <= thr[idx], left[idx], right[idx])
            idx = np.where(leaf, idx, nxt)
        out[s:s + chunk] = value[idx].sum(1)
    return out
