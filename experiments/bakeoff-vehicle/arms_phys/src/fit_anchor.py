"""Simulation bands for the physics anchor: quantiles of log(y / E_phys) on the TRAIN bank (tier draws 11, 12) per
architecture x duty (light: car/SUV/pickup/van, heavy: bus/trucks). -> phys_anchor/bands.json"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, '/home/claude/bakeoff/vehicle/harness')
import bake
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
c0 = time.process_time()
B = bake.load_bank('train')
L = [[[] for _ in range(2)] for _ in range(7)]
for s in (11, 12):
    T = bake.draw_tiers(B, np.random.default_rng(s)); D = bake.sim_table(B, T)
    ok = D['y'] > 0.02
    lr = np.log(D['y'][ok] / D['E_phys'][ok]); a = D['arch'][ok]; hd = (D['cls'][ok] >= 4).astype(int)
    for i in range(7):
        for h in range(2):
            L[i][h].append(lr[(a == i) & (hd == h)])
q = np.zeros((7, 2, 5)); n = np.zeros((7, 2), int)
for i in range(7):
    for h in range(2):
        x = np.concatenate(L[i][h]); n[i, h] = len(x)
        if len(x) < 50:          # no heavy-duty plug-in power-split in the bank: use the light-duty quantiles
            x = np.concatenate(L[i][0])
        q[i, h] = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
out = {'what': 'quantiles 5/25/50/75/95 % of log(y / E_phys) on the sim train bank (tier draws 11, 12)',
       'archs': ['conv', 'hev_parallel_p2', 'hev_series', 'hev_powersplit', 'phev_powersplit', 'phev_series', 'bev'],
       'duty': ['light', 'heavy'], 'n': n.tolist(), 'logq': q.round(6).tolist(), 'cpu_s': round(time.process_time() - c0, 1)}
json.dump(out, open(os.path.join(ROOT, 'phys_anchor', 'bands.json'), 'w'), indent=1)
print(json.dumps(out)[:1500])
