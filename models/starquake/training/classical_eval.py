"""Recompute the classical pipeline (v1 comb-start or v2 multi-start) on real readings of a split. Used on dev to choose
the classical baseline's version; then run once on the exam."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, time, numpy as np
sys.path.insert(0, _P.SRC); import sq, classical as C, realbank as RB
D = _P.WORK + '/data/'
def run(split, multistart, lens=(0, 365, 90, 54, 27)):
    sp = json.load(open(D + 'split.json')); out = {}; t = time.time()
    for k in sp[split]:
        y = sq.read_npz(f'{D}lc/{k}.npz')
        for L, seg in RB.pieces(y, k):
            if seg is None or L not in lens: continue
            F1, F2, s, (f, P), T = sq.prepare(seg); c = C.measure(f, P, F1, F2, multistart=multistart); out[f'{k}_{L}'] = [c['numax'], c['dnu']]
    print(split, multistart, len(out), '%.0fs' % (time.time() - t), flush=True); return out
if __name__ == '__main__':
    split, ms = sys.argv[1], bool(int(sys.argv[2])); json.dump(run(split, ms, tuple(int(v) for v in sys.argv[3].split(',')) if len(sys.argv) > 3 else (0, 365, 90, 54, 27)), open(f'{_P.WORK}/data/classical_{split}_v{2 if ms else 1}.json', 'w'))
