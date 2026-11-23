"""Inline qnet2.js, data.json and land.json into page.html -> web/quakenet2.html"""
from _paths import H as _W; H = _W + 'web/'
p = open(H + 'page.html').read().replace('/*QNET2*/', open(H + 'qnet2.js').read()).replace('/*DATA*/null', open(H + 'data.json').read()).replace('/*LAND*/null', open(H + 'land.json').read())
open(H + 'quakenet2.html', 'w').write(p); print(len(p), 'bytes')
