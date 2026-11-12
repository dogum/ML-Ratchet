import _paths as _P  # portable paths (see _paths.py)
import json
W = _P.LAB + '/'
strip = lambda s, name: s.replace(f"if (typeof module !== 'undefined') module.exports = {name};", '')
page = open(W + 'page.html').read()
out = page.replace('/*__SIMJS__*/', strip(open(W + 'cellsim.js').read(), 'CellSim')).replace('/*__NETJS__*/', strip(open(W + 'cellnet.js').read(), 'CellNet')) \
          .replace('/*__DATA__*/', open(W + 'data.json').read()).replace('/*__WEIGHTS__*/', open(W + 'weights.json').read()).replace('/*__WEIGHTSP__*/', open(W + 'weights_partial.json').read()).replace('/*__APP__*/', open(W + 'app.js').read())
open(W + 'cellsense.html', 'w').write(out)
open(W + 'test.html', 'w').write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>[hidden]{display:none!important}</style></head><body>' + out + '</body></html>')
print('page MB %.2f' % (len(out) / 1e6))
