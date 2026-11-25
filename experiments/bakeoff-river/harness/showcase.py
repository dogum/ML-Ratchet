"""Seeded showcase draw for the page: 2 exam basins per class and 3 storms each.

usage: python3 showcase.py      -> showcase.json (run once in Phase 1; frozen, hashed into the harness hash)

Chosen by seed and class only, never by results and without reading any exam streamflow:
  - classes from WaterData's static attributes (bake.basin_classes: snow / arid / humid / regulated / large);
  - eligible = NWIS daily coverage >= 90 % in both exam windows (proc/nwis/coverage.parquet: counts, not values);
  - per class in order, 2 basins drawn uniformly (numpy default_rng(20261118)) from eligible basins not yet drawn;
  - storms from the FORCING only: per exam window, the 10 largest 3-day precipitation totals at least 30 days
    apart; 2 drawn from the main window and 1 from the recent window; each storm window = peak day -10 .. +30.
exam.py saves every arm's (and every reference row's) daily series for these basins over both exam windows.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import bake  # noqa: E402

OUT = os.path.join(C.HARNESS, 'showcase.json')


def storms(bid, period, rng, k):
    import waterdata as wd
    (s0, s1), _ = C.PERIODS[period]
    P = wd.forcing(bid, s0, s1)['P']
    p3 = P.rolling(3, center=True, min_periods=1).sum()
    top = []
    for t, v in p3.sort_values(ascending=False).items():
        if all(abs((t - u).days) >= 30 for u, _ in top) and t - pd.Timedelta(days=10) >= pd.Timestamp(s0) \
                and t + pd.Timedelta(days=30) <= pd.Timestamp(s1):
            top.append((t, v))
        if len(top) == 10:
            break
    pick = sorted(rng.choice(len(top), k, replace=False))
    return [{'period': period, 'peak_p_day': str(top[j][0].date()), 'p3_mm': round(float(top[j][1]), 1),
             'rank_among_top10': int(j + 1), 'start': str((top[j][0] - pd.Timedelta(days=10)).date()),
             'end': str((top[j][0] + pd.Timedelta(days=30)).date())} for j in pick]


def main():
    if os.path.exists(OUT):
        print('showcase.json exists (frozen); delete it by hand only if the harness is rebuilt from scratch')
        return
    rng = np.random.default_rng(C.SEED)
    cov = pd.read_parquet(os.path.join(C.WATER_DATA_ROOT, 'data', 'proc', 'nwis', 'coverage.parquet')).set_index('gauge_id')
    ids = bake._exam_ids()
    ok = [b for b in ids if b in cov.index and cov.loc[b, 'cov_val'] >= 0.9 and cov.loc[b, 'cov_recent'] >= 0.9]
    cls = {b: bake.basin_classes(b) for b in ok}
    chosen = []
    for c in C.CLASSES:
        cand = sorted(b for b in ok if c in cls[b] and b not in [x['basin'] for x in chosen])
        for j in sorted(rng.choice(len(cand), 2, replace=False)):
            chosen.append({'basin': cand[j], 'drawn_for': c, 'classes': cls[cand[j]], 'n_candidates': len(cand)})
    for x in chosen:
        x['storms'] = storms(x['basin'], 'main', rng, 2) + storms(x['basin'], 'recent', rng, 1)
    rec = {'what': 'River Bake-off showcase: seeded draw (2 exam basins per class, 3 storms each from the forcing '
                   'only); see showcase.py', 'seed': C.SEED, 'eligible': len(ok), 'basins': chosen}
    json.dump(rec, open(OUT, 'w'), indent=1)
    for x in chosen:
        print(x['basin'], x['drawn_for'], x['classes'], [s['peak_p_day'] for s in x['storms']])


if __name__ == '__main__':
    main()
