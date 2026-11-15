"""print the current HybridPilot configuration the next rung builds on: the incumbent of the v2 ladder
(WORK/ratchet_v2.json) if it is a HybridPilot rung, else the R3 configuration (v1's release recipe: the round-1 DAgger
net, cost-based on/off, Kp 6), which every new rung then has to carry past the incumbent.
usage: incumbent.py ckpt | params ['json to merge'] | name | banks"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as _P
from hp_env import PTAG

R = json.load(open(os.path.join(_P.WORK, f'ratchet_v2{PTAG}.json')))
inc = [r for r in R if r.get('kept')][-1]
if not inc['ckpt'] or inc['ckpt'] == '-':
    inc = dict(rung='R3 recipe', ckpt='ckpt/r2_dagger.npz', params={'Kp': 6, 'onoff': 'ecms'})
if sys.argv[1] == 'ckpt':
    print(inc['ckpt'])
elif sys.argv[1] == 'name':
    print(inc['rung'])
elif sys.argv[1] == 'banks':
    print(' '.join(json.load(open(os.path.join(_P.WORK, inc['ckpt'][:-4] + '.json')))['banks']))
else:
    p = dict(inc['params'])
    if len(sys.argv) > 2:
        p.update(json.loads(sys.argv[2]))
    print(json.dumps(p))
