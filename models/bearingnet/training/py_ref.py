"""reference outputs for web/test_js.js: Python features + numpy network on the same int16-quantised page examples"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, base64, numpy as np
sys.path.insert(0, _P.WORK + '/release'); import bearingnet as RB, pipeline as PL
D = json.load(open(_P.LAB + '/data_auto.json')); net = RB.BearingNet(); out = []
for e in D['examples'][:6]:
    x = np.frombuffer(base64.b64decode(e['x']), np.int16).astype(float) / 32767 * e['q']
    F, sc, _ = PL.window_features(x, e['geom'], e['f_nom']); o = net.forward(F[None], sc[None])[0]
    out.append(dict(F=F.ravel().tolist(), o=o.tolist()))
json.dump(out, open(_P.LAB + '/py_ref.json', 'w'))
