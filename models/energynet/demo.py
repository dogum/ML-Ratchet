"""EnergyNet demo on real EXAM trips from the Vehicle Energy Dataset (first trip of each powertrain in a seeded draw).
python demo.py   (numpy only; runs in a few seconds)"""
import os, json, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from energynet import EnergyNet

net = EnergyNet()
ex = json.load(open(os.path.join(HERE, 'examples', 'examples.json')))
print(f"{'trip':16s} {'powertrain':16s} {'km':>5s} {'T C':>5s}  {'measured':>9s} {'EnergyNet':>9s} {'68% range':>15s} {'label x d':>9s} {'MOVES-like':>10s}   (MJ)")
for e in ex:
    z = np.load(os.path.join(HERE, 'examples', e['trip_id'] + '.npz'))
    trip = {k: z[k] for k in z.files if k != 'valid'}
    r = net.predict(trip, e['vehicle'])
    E = r['energy_MJ']
    # the label covers only seconds the logger recorded: scale by the share of road-load energy in valid seconds
    import features as F
    P = F.prep(trip); m = e['vehicle']['mass_kg']; cls = e['vehicle']['class']
    mc, Cc, Rc, cvc, rot = F.CLS[cls]
    pa, pr, pg, pi = F.wheel_power(P, m, Cc * (m / mc) ** 0.5, Rc, cvc, rot)
    w = np.clip(pa + pr + pg + pi, 0, None) / 0.25 + 5000.0 * (m / 1500.0) ** 0.7
    vr = w[z['valid'] == 1].sum() / w.sum()
    E *= vr; lo, hi = r['lo68'] / r['value'] * E, r['hi68'] / r['value'] * E
    print(f"{e['trip_id']:16s} {e['vehicle']['architecture']:16s} {r['distance_km']:5.1f} {np.mean(trip['T_amb']):5.1f}  {e['measured']:9.2f} {E:9.2f} {lo:7.2f}-{hi:<7.2f} {e['baselines_MJ']['label']:9.2f} {e['baselines_MJ']['moves_like']:10.2f}")
    print(f"{'':16s} components (MJ): " + ', '.join(f"{k} {v:.2f}" for k, v in r['components_MJ'].items()) + f"; exam run gave {e['exam_prediction_MJ']:.2f}")
