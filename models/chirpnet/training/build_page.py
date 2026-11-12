import _paths as _P  # portable paths (see _paths.py)
import json, sys
sys.path.insert(0, _P.SRC)
from export_js import pack
page = open(_P.LAB + '/page.html').read()
cj = open(_P.LAB + '/chirpnet.js').read().replace("if (typeof module !== 'undefined') module.exports = ChirpJS;", '')
app = open(_P.LAB + '/app.js').read()
import os
D = json.load(open(_P.LAB + '/data.json'))
for extra in ('pe', 'bns', 'o3'):
    fp = f'{_P.LAB}/{extra}.json'
    if os.path.exists(fp): D[extra] = json.load(open(fp))
data = json.dumps(D, separators=(',', ':'))
man, b64 = pack(_P.WORK + '/release/chirpnet_weights.npz')
W = {'man': man, 'b64': b64}
if man.get('veto') is not None:
    cm, cb = pack(_P.WORK + '/release/chirpnet_coinc_weights.npz')
    meta = json.loads(str(__import__('numpy').load(_P.WORK + '/release/chirpnet_weights.npz')['meta']))
    W['checker'] = {'man': cm, 'b64': cb, 'cfg': {'veto': meta['veto'], 'drop': meta.get('drop', 30.0)}}
out = page.replace('/*__CHIRPJS__*/', cj).replace('/*__DATA__*/', data).replace('/*__WEIGHTS__*/', json.dumps(W)).replace('/*__APP__*/', app)
open(_P.LAB + '/chirpnet.html', 'w').write(out)
open(_P.LAB + '/test.html', 'w').write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>[hidden]{display:none!important}</style></head><body>' + out + '</body></html>')
print('page MB %.2f' % (len(out) / 1e6))
