"""AirfoilNet demo: four real exam airfoils (UIUC wind tunnel), read from their coordinates alone.
A fixed random draw within two Reynolds-number strata, not picked by result. Runs in a few seconds.
"""
import os, json, glob
import numpy as np
from airfoilnet import AirfoilNet

HERE = os.path.dirname(os.path.abspath(__file__))
net = AirfoilNet()
print('AirfoilNet demo: polars from shape + Reynolds number, against UIUC tunnel data (exam airfoils)\n')
for f in sorted(glob.glob(os.path.join(HERE, 'examples', '*.json'))):
    ex = json.load(open(f))
    L = np.array([[np.nan if v is None else v for v in r] for r in ex['lift']['alpha_cl_cm']], float)
    k = int(np.argmax(L[:, 0])); L = L[:k + 1]                     # up-sweep
    pol = net.polar(np.array(ex['coords']), Re=ex['Re'])
    r = net.predict(pol['p'], ex['Re'], L[:, 0])
    km = int(np.argmax(L[:, 1]))
    print(f"{ex['airfoil']}  ({ex['family']}, Re = {ex['Re']:,.0f})")
    print(f"  Cl max   tunnel {L[km, 1]:.2f} at {L[km, 0]:.1f} deg | AirfoilNet {pol['clmax']:.2f} at {pol['alpha_stall']:.1f} deg")
    print('  alpha  Cl tunnel  Cl AirfoilNet (68% bar)')
    for i in range(0, len(L), max(1, len(L) // 6)):
        print(f"  {L[i, 0]:5.1f}  {L[i, 1]:8.3f}  {r['cl'][i]:8.3f} +- {r['cl_sd'][i]:.3f}")
    D = np.array(ex['drag']['alpha_cl_cd'], float)
    rd = net.predict(pol['p'], ex['Re'], D[:, 0])
    j = np.argsort(D[:, 2])[0]
    print(f"  min Cd   tunnel {D[j, 2]:.4f} at {D[j, 0]:.1f} deg | AirfoilNet {rd['cd'][j]:.4f} "
          f"(x/ {np.exp(rd['lncd_sd'][j]):.2f})")
    print(f"  mean |Cl error| {np.nanmean(np.abs(r['cl'] - L[:, 1])):.3f},  "
          f"median |ln Cd ratio| {np.median(np.abs(np.log(rd['cd'] / D[:, 2]))):.3f}\n")
