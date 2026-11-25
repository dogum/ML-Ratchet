"""The road law's REAL exam, LawFinder side. It runs nothing new: the law was graded ONCE in the Vehicle Bake-off exam
(harness/exam.py, every arm together, after registration) as the arm `formula`. This script reads that exam's per-unit
scores and writes LawFinder's exam files. Run it once, after the bake-off exam is done. It chooses nothing.

usage: python3 exam_road.py [--results /home/claude/bakeoff/vehicle/harness/results] [--rehearsal] [--out FILE]
  --rehearsal: use the harness's latest dev rehearsal instead (a plumbing check; the formula arm may be absent there)
Writes results/exam_road.json and results/exam_road_points.parquet (LawFinder's layout; ROOT = LawFinder's folder).

Rows (LawFinder names): law = the formula arm (released rung; its conventional law on conventional units),
textbook = road-load E_phys (EnergyNet's simulation constants; the harness row 'ephys'), network = EnergyNet v2.1,
classical = MOVES-like (VED) / dev regression (EPA) / dev median (heavy duty), dumb = climatology. Headline set: VED ICE
trips (MdAPE); every conventional set is in by_set.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # LawFinder's folder once merged into src/
ROWS = {'law': 'formula', 'law_conformal': 'formula_conformal', 'textbook': 'ephys', 'network': 'enet_v21',
        'moves_like': 'moves_like', 'label': 'label', 'regression': 'regression', 'dev_median': 'dev_median',
        'dumb': 'climatology'}
NAMES = {'law': 'LawFinder road law (bake-off formula arm)', 'law_conformal': 'Road law + Mondrian conformal bands',
         'textbook': 'Road-load estimate E_phys (EnergyNet constants)', 'network': 'EnergyNet v2.1',
         'moves_like': 'MOVES-like VSP model (dev-fitted)', 'label': 'Window-sticker label x distance',
         'regression': 'EPA dev regression', 'dev_median': 'Dev median', 'dumb': 'Climatology (sim median per km)'}
CONV_SETS = ['VED_ICE', 'EPA_conv_FTP', 'EPA_conv_HWY', 'EPA_conv_US06', 'EPA_conv_FTPcold', 'EPA_conv_SC03',
             'Mersin_diesel', 'Mersin_cng', 'ICCT']


def mets(d):
    if len(d) < 3:
        return {'n': int(len(d))}
    y, p = d.y.values.astype(float), d.pred.values.astype(float)
    out = {'n': int(len(d)), 'mdape': float(np.median(np.abs(p / y - 1)) * 100),
           'bias_median_pct': float(np.median(p / y - 1) * 100),
           'rho': float(pd.Series(y).corr(pd.Series(p), method='spearman')),
           'crps_log': float(d.crps_log.mean())}
    if not np.allclose(d.q05.values, d.q95.values):
        out['cov50'] = float(np.mean((y >= d.q25) & (y <= d.q75)))
        out['cov90'] = float(np.mean((y >= d.q05) & (y <= d.q95)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='/home/claude/bakeoff/vehicle/harness/results')
    ap.add_argument('--rehearsal', action='store_true')
    ap.add_argument('--out', default=None, help='output JSON (default: LawFinder results/exam_road.json)')
    a = ap.parse_args()
    if a.rehearsal:
        rid = sorted(glob.glob(os.path.join(a.results, 'rehearsal_*', 'scores.parquet')))[-1]
        sc = pd.read_parquet(rid); out = a.out or os.path.join(ROOT, 'results', 'dryrun_road.json')
    else:
        done = json.load(open(os.path.join(a.results, 'EXAM_DONE.json')))
        rid = done.get('run_id') or done.get('id')
        sc = pd.read_parquet(os.path.join(a.results, f'exam_{rid}', 'scores.parquet'))
        out = a.out or os.path.join(ROOT, 'results', 'exam_road.json')
        if os.path.exists(out):
            raise SystemExit(f'{out} exists: the exam is converted once.')
    sc = sc[sc.track == 'table']                       # the law reads the same variables on both tracks
    ent = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'road_law_entry.json')))['road']
    res = {'source': rid, 'released': ent['rung'], 'expr': ent['expr'], 'sigma': ent['sigma'], 'sigma_kind': 'log',
           'headline_set': 'VED_ICE', 'rows': {}, 'by_set': {}}
    for s in CONV_SETS:
        d = sc[sc.set == s]
        res['by_set'][s] = {k: {'name': NAMES[k], **mets(d[d.row == r])} for k, r in ROWS.items() if (d.row == r).any()}
    res['rows'] = res['by_set']['VED_ICE']
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1, default=float)
    v = sc[(sc.set == 'VED_ICE')]
    if not a.rehearsal:
        pts = v[v.row == 'formula'][['unit', 'y', 'pred', 'v_kmh', 'T', 'dist_km', 'climb_per_km']].copy()
        pts['t'] = 60.0 / np.maximum(pts.v_kmh, 1.0); pts['d'] = pts.dist_km
        pts['e'] = pts.y / pts.dist_km                    # valid-seconds energy / trip distance (VED valid share ~1)
        for r in ('ephys', 'enet_v21', 'moves_like'):
            pts[f'pred_{r}'] = v[v.row == r].set_index('unit').pred.reindex(pts.unit).values
        pts.to_parquet(os.path.join(os.path.dirname(out), 'exam_road_points.parquet'))
    for k, m in res['rows'].items():
        if m.get('n', 0) >= 3:
            print(f"{m['name']:55s} n={m['n']} MdAPE={m['mdape']:.1f}% bias={m['bias_median_pct']:+.1f}% rho={m['rho']:.3f}")


if __name__ == '__main__':
    main()
