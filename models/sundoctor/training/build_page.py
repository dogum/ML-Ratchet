import _paths as _P  # portable paths (see _paths.py)
import json
W = _P.LAB + '/'
page = open(W + 'page.html').read(); js = open(W + 'sdnet.js').read().replace("if (typeof module !== 'undefined') module.exports = SunDoctorJS;", '')
out = page.replace('/*__SDJS__*/', js).replace('/*__DATA__*/', open(W + 'data.json').read()).replace('/*__WEIGHTS__*/', open(W + 'weights.json').read()).replace('/*__APP__*/', open(W + 'app.js').read())
open(W + 'sundoctor.html', 'w').write(out)
open(W + 'test.html', 'w').write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>[hidden]{display:none!important}</style></head><body>' + out + '</body></html>')
print('page MB %.2f' % (len(out) / 1e6))
