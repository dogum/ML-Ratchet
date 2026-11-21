"""Write the demo examples for the release: 4 exam basins, one per climate stratum, a fixed seeded draw (never chosen
by score). Each: <id>_cal.csv (date, P, T_mean, PET, Q_mm for 1999-10..2008-09), <id>_tgt.csv (date, P, T_mean, PET
for 1989-10..1999-09) and <id>_truth.json (observed change, GR4J, regulation truth), plus expected.json (outputs of
the release model, for the demo's self-check)."""
import common  # noqa: F401
from _paths import MODEL
import json
import sys

import numpy as np

import waterdata as wd
from common import CAL, VAL, D, DELTA_T
import evaluate as E

OUT = f'{D}/release/examples'


def main(model_path):
    sys.path.insert(0, MODEL)   # the released package (the sandbox staged it in water/detective/release/)
    import basindetective as bd
    det = bd.load(model_path)
    X, meta, names = E.real_set('exam')
    fs = X[:, names.index('c_fsnow')]
    ar = X[:, names.index('c_arid')]
    rng = np.random.default_rng(7)
    ok = meta['true_d_logq'].notna().values
    picks = []
    for sel in ((fs >= 0.15) & (ar <= 1), (fs < 0.15) & (ar <= 1), (fs < 0.15) & (ar > 1), meta.dor.values > 0.1):
        picks.append(int(rng.choice(np.where(sel & ok)[0])))
    expected = {}
    for i in picks:
        b = meta.basin.iloc[i]
        f = wd.forcing(b, VAL[0], CAL[1])[['P', 'T_mean', 'PET']]
        q = wd.streamflow(b, 'caravan').Q_mm.reindex(f.index)
        cal = f.loc[CAL[0]:CAL[1]].copy()
        cal['Q_mm'] = q.loc[CAL[0]:CAL[1]]
        tgt = f.loc[VAL[0]:VAL[1]]
        cal.round(6).to_csv(f'{OUT}/{b}_cal.csv', index_label='date')
        tgt.round(6).to_csv(f'{OUT}/{b}_tgt.csv', index_label='date')
        truth = {'basin': b, 'name': str(wd.attributes('camels').loc[b, 'gauge_name']),
                 'observed_change': {t: float(meta[f'true_{t}'].iloc[i]) for t in DELTA_T},
                 'gr4j_change': {t: float(meta[f'gr4j_{t}'].iloc[i]) for t in DELTA_T},
                 'reservoir_storage_over_annual_flow': float(meta.dor.iloc[i])}
        json.dump(truth, open(f'{OUT}/{b}_truth.json', 'w'), indent=1)
        import pandas as pd
        c2 = pd.read_csv(f'{OUT}/{b}_cal.csv', index_col=0, parse_dates=True)
        t2 = pd.read_csv(f'{OUT}/{b}_tgt.csv', index_col=0, parse_dates=True)
        r = bd.predict(det, c2[['P', 'T_mean', 'PET']], c2.Q_mm.values, t2)
        expected[b] = {t: r['changes'][t]['mean'] for t in DELTA_T} | {'p_regulated': r['regulation']['p_regulated']}
        print(b)
    json.dump(expected, open(f'{OUT}/expected.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
