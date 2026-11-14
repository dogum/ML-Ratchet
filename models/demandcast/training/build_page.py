"""Assemble web/demandcast.html from template.html + data.json + weights.json + exam results + ratchet + text.
python3 build_page.py <ckpt>"""
import sys, json
from common import *

ck = sys.argv[1]
W = f'{ROOT}/web'
t = open(f'{W}/template.html').read()
info = json.load(open(f'{W}/info.json'))
ex = json.load(open(f'{DATA}/exam/{ck}.json'))
ev = json.load(open(f'{DATA}/exam/{ck}_events.json'))
rat = json.load(open(f'{ROOT}/ratchet.json'))
rat_small = [dict(ckpt=r['ckpt'], note=r['note'], why=r['why'], accepted=r['accepted'], objective=r['objective'],
                  dev={k: dict(err=r['dev'][k]['err']) for k in 'dws'}) for r in rat]
ex_small = {k: ex[k] for k in ('overall', 'table', 'temp_bands', 'sigma_scale_dev')}
rep = {'__DATA__': open(f'{W}/data.json').read(), '__WEIGHTS__': open(f'{W}/weights.json').read(),
       '__EXAM__': json.dumps(ex_small, separators=(',', ':')), '__EVENTS__': json.dumps(ev, separators=(',', ':')),
       '__RATCHET__': json.dumps(rat_small, separators=(',', ':')), '__INFO__': json.dumps(info, separators=(',', ':')),
       '__DCNET__': open(f'{W}/dcnet.js').read()}
for k, v in rep.items():
    t = t.replace(k, v)
open(f'{W}/demandcast.html', 'w').write(t)
print('size MB', len(t) / 1e6)
