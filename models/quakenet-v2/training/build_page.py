"""Inline qnet2.js, data.json and land.json into page.html -> web/quakenet2.html"""
import os
from _paths import H as _W, MODEL; H = _W + 'web/'
SRC = lambda n: H + n if os.path.exists(H + n) else os.path.join(MODEL, 'web', n)   # page.html, qnet2.js as shipped in web/
p = open(SRC('page.html')).read().replace('/*QNET2*/', open(SRC('qnet2.js')).read()).replace('/*DATA*/null', open(H + 'data.json').read()).replace('/*LAND*/null', open(H + 'land.json').read())
open(H + 'quakenet2.html', 'w').write(p); print(len(p), 'bytes')
