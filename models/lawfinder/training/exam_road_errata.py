"""Errata for the road law's exam run 1 (22 Nov 2026). Reads results/exam_road.json and results/exam_road_points.parquet
as run 1 wrote them (left untouched) and writes results/exam_road_reported.json, which the page, the release and the
card read. Chooses nothing; the law's exam is not re-run.

The bug: exam_road.py multiplied EnergyNet's VED frame (exam2_ved_v2.parquet) by its `conv` column, but that frame
already stores y and pred in MJ (EnergyNet's realfeat.py multiplies by conv before saving). EnergyNet's VED rows were
therefore scaled by 0.042 (petrol, hybrids) and 3.6 (electric). Nothing else depends on it: the law, every rung,
road-load, RoadSim's law, the dumb rows and RoadData's baselines are correct (the baselines were not multiplied, and
they match EnergyNet's published RoadData rows: petrol 13.0 / 10.3 / 22.8 %). EnergyNet's EPA rows were computed as
ratios to its own truth and are correct.
The fix (the owner agreed to a re-score, 22 Nov 2026): EnergyNet's VED rows only are re-computed from its stored
per-trip predictions (pred / trip km, in MJ/km, no unit factor), against run 1's stored truth for the same trips (the
script checks EnergyNet's own truth agrees trip by trip). Its published values on the same trips are kept beside them
as a cross-check (models/energynet/model.yaml).
"""
import json
import numpy as np, pandas as pd
from _paths import ROOT, other_work
import law_road as L
from exam_road import mets

PUBLISHED = {'VED_ICE': (8.6, 14469), 'VED_HEV': (13.4, 8241), 'VED_EV': (23.2, 167)}   # EnergyNet's model.yaml: MdAPE %, n
NAME = 'EnergyNet v2.1 (network; its exam predictions, re-computed)'


def main():
    E = json.load(open(f'{ROOT}/results/exam_road.json'))
    P = pd.read_parquet(f'{ROOT}/results/exam_road_points.parquet')
    v = P[P.src == 'VED'].set_index('id')
    en = pd.read_parquet(f"{other_work('energynet')}/data/exam2_ved_v2.parquet").set_index('trip_id').loc[v.index]
    pred = en.pred.values / v.dist_km.values                    # MJ/km (the frame is in MJ)
    truth_en = en.y.values / v.dist_km.values
    rel = np.abs(truth_en / v.E.values - 1)
    chk = {'ved_trips': int(len(v)), 'truth_max_rel_diff': float(np.nanmax(rel)), 'truth_median_rel_diff': float(np.nanmedian(rel)),
           'run1_energynet_ratio_median': float(np.nanmedian(v.pred_energynet.values / pred))}
    assert chk['truth_max_rel_diff'] < 1e-6, chk
    before = {k: {kk: E['strata'][k]['energynet'].get(kk) for kk in ('n', 'mdape', 'bias_median_pct')} for k in ('VED_ICE', 'VED_HEV', 'VED_EV', 'VED_all')}
    y = v.E.values
    for g in L.GROUPS + ('all',):
        m = (v.group == g).values if g != 'all' else np.ones(len(v), bool)
        k = f'VED_{g}'
        row = {'name': NAME, **mets(y[m], pred[m])}
        if k in PUBLISHED:
            row['published_mdape'] = PUBLISHED[k][0]
            assert row['n'] == PUBLISHED[k][1], (k, row['n'])
        E['strata'][k]['energynet'] = row
    E['rows'] = {n: r for n, r in E['strata']['VED_all'].items() if r.get('n', 0) >= 3}
    after = {k: {kk: E['strata'][k]['energynet'].get(kk) for kk in ('n', 'mdape', 'bias_median_pct')} for k in before}
    d = __doc__.split('\n\n')[1].replace('\n', ' ')
    E['errata'] = {'date': '2026-11-22', 'run': 1, 'what': d[:d.index('The fix')].strip(),
                   'fix': d[d.index('The fix'):].strip(), 'checks': chk,
                   'energynet_ved_before': before, 'energynet_ved_after': after}
    json.dump(E, open(f'{ROOT}/results/exam_road_reported.json', 'w'), indent=1, default=str)
    print(json.dumps(chk))
    for k in before:
        print(k, 'before', before[k], 'after', after[k], 'published', PUBLISHED.get(k, ('-',))[0])


if __name__ == '__main__':
    main()
