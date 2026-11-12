"""Assemble the single-file page: web/page.html + data_auto.json + weights.json + bnet.js + lab.js -> web/bearingnet.html"""
import _paths as _P  # portable paths (see _paths.py)
B = _P.LAB + '/'
s = open(B + 'page.html').read()
for k, f in (('/*DATA*/', 'data_auto.json'), ('/*WEIGHTS*/', 'weights.json'), ('/*BNET*/', 'bnet.js'), ('/*LAB*/', 'lab.js')):
    t = open(B + f).read()
    if f.endswith('.json'): t = t.replace('</', '<\\/')
    s = s.replace(k, t, 1)
open(B + 'bearingnet.html', 'w').write(s); print('bearingnet.html', len(s))
