"""Baselines on the DEV airfoils (allowed: dev is for choosing). -> work/dev_baselines.json"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..'); sys.path.insert(0, HERE)
import realeval as rev, baselines as bl
units = json.load(open(os.path.join(ROOT, 'data', 'units.json'))); splits = json.load(open(os.path.join(ROOT, 'splits.json')))
z = np.load(os.path.join(ROOT, 'work', 'shapes.npz')); pmap = {k: p for k, p in zip(z['tested_keys'], z['tested_p'])}
out = {}
for name, f in (('NeuralFoil', bl.NeuralFoilMethod()), ('thin_airfoil', bl.thin_airfoil), ('mean_polar', bl.MeanPolar(units, splits['dev']))):
    s, per = rev.evaluate(f, units, splits['dev'], pmap); out[name] = s
    print(name, {k: round(v, 4) for k, v in s.items() if isinstance(v, float)})
json.dump(out, open(os.path.join(ROOT, 'work', 'dev_baselines.json'), 'w'), indent=1, default=float)
