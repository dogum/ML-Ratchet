"""Errata for the road law's exam run 1 (22 Nov 2026). Reads results/exam_road.json as written (left untouched) and
writes results/exam_road_reported.json, which the page, the release and the card read. Chooses nothing; runs nothing.

The bug: exam_road.py multiplied EnergyNet's VED frame (exam2_ved_v2.parquet) by its `conv` column, but that frame
already stores y and pred in MJ (EnergyNet's realfeat.py multiplies by conv before saving). EnergyNet's three VED rows
are therefore scaled by 0.042 (petrol, hybrids) and 3.6 (electric). Nothing else depends on it: the law, every rung,
road-load, RoadSim's law, the dumb rows and RoadData's baselines are correct (the baselines were not multiplied, and
they match EnergyNet's published RoadData rows: petrol 13.0 / 10.3 / 22.8 %).
The fix here: on each VED stratum, EnergyNet's row is replaced by its own published exam value on the same trips
(models/energynet/model.yaml; the trip counts match: 14,469 / 8,241 / 167); the pooled VED row, which EnergyNet never
published, is marked not scored. The EPA rows were computed as ratios to EnergyNet's own truth and are correct.
Re-scoring EnergyNet's VED rows from its frame is left to the owner (the exam runs once).
"""
import json
from _paths import ROOT

PUBLISHED = {'VED_ICE': (8.6, 14469, 0.3), 'VED_HEV': (13.4, 8241, 3.8), 'VED_EV': (23.2, 167, -20.3)}   # MdAPE %, n, bias %
NOTE = ('EnergyNet v2.1, published exam value on the same trips (models/energynet/model.yaml); run 1 of this exam '
        'mis-scaled its VED rows (units), see training/exam_road_errata.py')


def main():
    E = json.load(open(f'{ROOT}/results/exam_road.json'))
    for k, (md, n, b) in PUBLISHED.items():
        assert E['strata'][k]['law']['n'] == n, (k, E['strata'][k]['law']['n'], n)
        E['strata'][k]['energynet'] = {'name': 'EnergyNet v2.1 (published, same trips)', 'n': n, 'mdape': md,
                                       'bias_median_pct': b, 'note': NOTE}
    E['strata']['VED_all']['energynet'] = {'name': 'EnergyNet v2.1 (not scored on the pooled trips)', 'n': 0, 'note': NOTE}
    E['rows'] = {n: v for n, v in E['strata']['VED_all'].items() if v.get('n', 0) >= 3}
    E['errata'] = {'date': '2026-11-22', 'run': 1, 'what': __doc__.split('\n\n')[1].replace('\n', ' '),
                   'checks_run1': E.get('checks')}
    json.dump(E, open(f'{ROOT}/results/exam_road_reported.json', 'w'), indent=1, default=str)
    for k in ('VED_ICE', 'VED_HEV', 'VED_EV', 'VED_all', 'EPA_ICE', 'EPA_HEV', 'EPA_EV', 'EPA_all'):
        row = E['strata'][k]
        print(k, {n: (v.get('mdape') and round(v['mdape'], 1)) for n, v in row.items()})


if __name__ == '__main__':
    main()
