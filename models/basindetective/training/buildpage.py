"""Assemble web/basindetective.html from page_template.html + detective.js + prose.js + data.json."""
import common  # noqa: F401
import json

from common import D
from _paths import SRC

W = f'{D}/web'
t = open(f'{SRC}/page_template.html', encoding='utf-8').read()   # template and scripts are kept in training/
js = open(f'{SRC}/detective.js', encoding='utf-8').read()
prose = open(f'{SRC}/prose.js', encoding='utf-8').read()
data = json.load(open(f'{W}/data.json'))
# the page does not need the full ratchet records' nested per-target dicts
for r in data['ratchet']['rungs']:
    r.pop('sim', None)
    r.pop('path', None)
s = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
t = t.replace('/*DETECTIVE_JS*/', js).replace('/*PROSE*/', prose).replace('/*DATA*/', s)
open(f'{W}/basindetective.html', 'w').write(t)
print(len(t) / 1e6, 'MB')
