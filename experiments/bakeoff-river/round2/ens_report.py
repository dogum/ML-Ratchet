"""PREREG section 5: the ensembles' dev gate and paired comparisons on dev (no exam data).

usage: python ens_report.py <ens_id> <member eval dir> x5      (dirs under work/eval/, each with dev_dev_ungauged.parquet)
-> results/ensembles_dev.json (one entry per ensemble; earlier entries kept)
Gate: registered if dev median NSE >= 0.95 x the best member's. "Ensembling helps on dev" only if > best + 0.01.
"""
import json
import os
import sys

import r2
import pandas as pd
import bake
import score as S
import evaluate as EV


def dev_table(d):
    p = os.path.join(r2.WORK, 'eval', d, 'dev_dev_ungauged.parquet')
    if not os.path.exists(p):
        p = os.path.join(r2.WORK, 'eval', d, 'own_dev_ungauged.parquet')
    return pd.read_parquet(p)


def main(ens, members):
    import numpy as np
    classes = {b: bake.basin_classes(b) for b in bake.dev_basins()}
    E = dev_table(ens)
    rng = np.random.default_rng(20261118)
    rec = {'members': members, 'dev': S.summarize(E), 'by_class': S.by_class(E, classes), 'vs': {}}
    mem = {}
    for m in members:
        g = dev_table(m)
        mem[m] = float(g['nse'].median())
        rec['vs'][m] = S.compare(E, g, rng=rng, classes=classes)
    refs = EV.dev_refs()
    for name in ('udev', 'nwm', 'gr4j'):
        g = refs[(refs.method == name) & (refs.lead == 0)].set_index('basin')
        rec['vs'][name] = S.compare(E, g, rng=rng, classes=classes)
    best = max(mem, key=mem.get)
    rec.update({'member_dev_nse': mem, 'best_member': best, 'best_member_dev_nse': mem[best],
                'gate_passes': bool(rec['dev']['nse'] >= 0.95 * mem[best]),
                'helps_on_dev': bool(rec['dev']['nse'] > mem[best] + 0.01)})
    fn = os.path.join(r2.RESULTS, 'ensembles_dev.json')
    allr = json.load(open(fn)) if os.path.exists(fn) else {}
    allr[ens] = rec
    json.dump(allr, open(fn, 'w'), indent=1, default=float)
    print(ens, 'dev', round(rec['dev']['nse'], 4), 'best member', best, round(mem[best], 4), 'gate', rec['gate_passes'],
          'helps', rec['helps_on_dev'])
    for k, v in rec['vs'].items():
        n = v['nse'] if 'nse' in v else v
        print(' vs', k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in n.items()
                         if kk in ('diff', 'lo', 'hi', 'verdict', 'ci95')})


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
