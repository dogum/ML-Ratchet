"""Run the least-squares transit fit on every exam planet with signal-to-noise of 7 or more (and the dev planets)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import baseline as BL
D = _P.WORK + '/data/'; S = json.load(open(D + 'split.json')); Mt = json.load(open(D + 'real_meta.json'))
idx = [i for i in S['exam'] + S['dev'] if Mt[i]['kind'] == 'planet' and Mt[i]['info']['snr'] >= 7]
res, dt = BL.run(idx); json.dump(dict(idx=idx, fits=res, sec_per_fit=dt), open(D + 'baseline_fits.json', 'w')); print(len(idx), 'fits, %.2f s each' % dt)
