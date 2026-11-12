"""Inline the JS, heart, data into web/page.html -> web/beatnet.html"""
import _paths as _P  # portable paths (see _paths.py)
import json
W = _P.LAB + '/'
s = open(W + 'page.html').read()
s = s.replace('/*BNET*/', open(W + 'bnet.js').read()).replace('/*BEATSIM*/', open(W + 'beatsim.js').read())
s = s.replace('/*DATA*/null', open(W + 'data.json').read()).replace('/*HEART*/null', open(W + 'heart.json').read())
open(W + 'beatnet.html', 'w').write(s); print('beatnet.html', len(s) // 1024, 'KB')
