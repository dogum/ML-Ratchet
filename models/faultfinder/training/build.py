"""Assemble the single-file page: page.html + ffnet.js + exam.js + data.json + weights.json -> faultfinder.html"""
import os
H = os.path.dirname(os.path.abspath(__file__))
r = lambda f: open(os.path.join(H, f)).read()
p = r('page.html').replace('/*FFNET*/', r('ffnet.js')).replace('/*EXAMJS*/', r('exam.js'))
p = p.replace('/*DATA*/null', r('data.json').replace('</', '<\\/')).replace('/*WEIGHTS*/null', r('weights.json'))
open(os.path.join(H, 'faultfinder.html'), 'w').write(p)
print('faultfinder.html MB', len(p) / 1e6)
