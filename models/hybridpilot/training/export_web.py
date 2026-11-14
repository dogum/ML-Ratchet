"""Export the network + real base states for the page's decision explorer (JS port check data)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hp_env import *
from hp_policy import HybridPilot, MLP, make_features, FEAT_NAMES
from roadsim import simulate
import lab

CK = f'{ROOT}/ckpt/r2_dagger.npz'
net = MLP.load(CK)
z = np.load(CK)


class Rec(HybridPilot):
    def select(self, C, ctrl, st, ctx):
        x = make_features(self.vf, self.tc, self.veh, st, ctx, self)
        j = super().select(C, ctrl, st, ctx)
        self.X.append((ctx['k'], x, float(st['soc'])))
        return j


out = dict(weights={k: np.round(z[k], 7).tolist() for k in z.files}, feat_names=FEAT_NAMES, combos=[])
for cb in lab.combos():
    if cb['key'] not in ('prius_aa', 'p2_aa', 'bus_zh', 'volt_aa'): continue
    st = Rec(net=net, onoff='ecms', Kp=6, preview_s=30, nav=False); st.X = []
    o = simulate(cb['spec'], cb['trip'], strategy=st, init_soc=cb['soc0'], init_temps=cb['temps'], record=True, options='roadsim-physics-v1')
    veh = get_vehicle(cb['spec']); b = veh.battery
    X = st.X
    # base states: decision steps at evenly spaced moments while moving
    idx = np.linspace(len(X) * 0.1, len(X) * 0.9, 8).astype(int)
    states = []
    for i in idx:
        k, x, soc = X[i]
        y = net(x)
        states.append(dict(k=int(k), x=np.round(x, 6).tolist(), y=[float(y[0]), float(y[1])], v=float(cb['trip']['v'][k]), soc=soc))
    out['combos'].append(dict(key=cb['key'], label=cb['label'], soc_lo=b.soc_lo, soc_hi=b.soc_hi, ref=st.soc_ref_cs,
                              Pe_kW=veh.engine.P_max / 1e3, mass=veh.mass, phev=veh.arch.startswith('phev'), states=states))
    print(cb['key'], len(X))
json.dump(out, open(f'{ROOT}/web/net.json', 'w'))
print(os.path.getsize(f'{ROOT}/web/net.json'))
