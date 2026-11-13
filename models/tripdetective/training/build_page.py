"""Inline weights, data, lab, tdet.js and page.js into web/tripdetective.html."""
import _paths as _P  # portable paths (see _paths.py)
import os
W = _P.WORK + '/web'
p = open(f'{W}/page.html').read()
for tag, f in [('/*WEIGHTS*/', 'weights.json'), ('/*DATA*/', 'data.json'), ('/*LAB*/', 'lab.json'), ('/*TDETJS*/', 'tdet.js'), ('/*PAGEJS*/', 'page.js')]:
    s = open(f'{W}/{f}').read()
    if f.endswith('.json'): s = s.replace('</', '<\\/')
    p = p.replace(tag, s)
open(f'{W}/tripdetective.html', 'w').write(p)
print('page', os.path.getsize(f'{W}/tripdetective.html') // 1024, 'kB')
