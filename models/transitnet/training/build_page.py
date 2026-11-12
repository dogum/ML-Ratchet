"""Assemble the single-file TransitNet page: page.html with the physics, network, folding and app scripts, the data and the weights inlined."""
import _paths as _P  # portable paths (see _paths.py)
W = _P.LAB + '/'
strip = lambda s, name: s.replace(f"if (typeof module !== 'undefined') module.exports = {name};", '')
page = open(W + 'page.html').read()
out = (page.replace('/*__TRANSITJS__*/', strip(open(W + 'transit.js').read(), 'Transit')).replace('/*__TNETJS__*/', strip(open(W + 'tnetjs.js').read(), 'TNet'))
           .replace('/*__FOLDJS__*/', strip(open(W + 'fold.js').read(), 'Fold')).replace('/*__DATA__*/', open(W + 'data.json').read())
           .replace('/*__WEIGHTS__*/', open(W + 'weights.json').read()).replace('/*__APP__*/', open(W + 'app.js').read()))
open(W + 'transitnet.html', 'w').write(out)
open(W + 'test.html', 'w').write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>[hidden]{display:none!important}</style></head><body>' + out + '</body></html>')
print('page MB %.2f' % (len(out) / 1e6))
