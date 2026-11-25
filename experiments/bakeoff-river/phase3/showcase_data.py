"""Cut the exam's saved showcase series (results/exam_<id>/showcase.npz) to the 30 storm windows of showcase.json, for
the page. Adds each storm's basin-mean precipitation (the forcing, an input, as showcase.py read it) for display.
Reads nothing from the exam beyond what exam.py saved; run only after results/EXAM_DONE.json exists.

usage: python3 showcase_data.py            -> phase3/results/showcase_page.json
"""
import json
import os
import sys

HARNESS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'harness')  # cloud: /home/claude/bakeoff/river/harness
sys.path.insert(0, HARNESS)
BASE = HARNESS if os.path.isdir(os.path.join(HARNESS, 'results')) else os.path.dirname(HARNESS)  # repo: results/ sits beside harness/
sys.path.insert(0, os.path.join(HARNESS, 'vendor'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import config as C  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work', 'phase3', 'showcase_page.json')  # cloud: phase3/results/; shipped copy in results/
UNG = ['flowcast_v1', 'seq_lstm', 'seq_ssm', 'seq_tcn', 'seq_transformer', 'tab_gbm', 'tab_gp', 'tab_dpl', 'tab_formula',
       'climatology']
REFS = ['udev', 'uk12', 'gr4j', 'nwm']
GAUGED = ['enkf_watersim', 'flowcast_v1']


def sig(a, n=3):
    out = []
    for x in np.asarray(a, float):
        if not np.isfinite(x):
            out.append(None)
        elif x == 0:
            out.append(0)
        else:
            out.append(float(f'{x:.{n}g}'))
    return out


def main():
    done = json.load(open(os.path.join(BASE, 'results', 'EXAM_DONE.json')))
    rid = done['run_id']
    Z = np.load(os.path.join(BASE, 'results', f'exam_{rid}', 'showcase.npz'))
    keys = set(Z.files)
    show = json.load(open(os.path.join(HARNESS, 'showcase.json')))
    import waterdata as wd
    attrs = wd.attributes('camels')
    basins = []
    missing = []
    for B in show['basins']:
        bid = B['basin']
        a = attrs.loc[bid]
        rec = {'id': bid.replace('camels_', ''), 'name': str(a['gauge_name']).title().replace(', ', ', ').strip(),
               'drawn_for': B['drawn_for'], 'classes': B['classes'], 'area_km2': round(float(a['area'])),
               'frac_snow': round(float(a['frac_snow']), 2), 'lat': round(float(a['gauge_lat']), 3),
               'lon': round(float(a['gauge_lon']), 3), 'storms': []}
        for s in B['storms']:
            per = s['period']
            p0, p1 = C.PERIODS[per][0]
            dates = pd.date_range(p0, p1)
            i0 = dates.get_loc(pd.Timestamp(s['start']))
            i1 = dates.get_loc(pd.Timestamp(s['end'])) + 1
            P = wd.forcing(bid, s['start'], s['end'])['P']
            st = {'period': per, 'start': s['start'], 'end': s['end'], 'peak_p_day': s['peak_p_day'], 'p3_mm': s['p3_mm'],
                  'P': sig(P.values, 3), 'ung': {}, 'cf': {}, 'ref': {}, 'g': {}}

            def get(k):
                if k not in keys:
                    missing.append(k)
                    return None
                return Z[k][i0:i1]
            o = get(f'obs|{per}_ref|{bid}')
            st['obs'] = sig(o) if o is not None else None
            for arm in UNG:
                x = get(f'{arm}|{per}_ungauged|{bid}')
                if x is not None:     # columns mean, q05, q25, q50, q75, q95
                    st['ung'][arm] = [sig(x[:, j]) for j in (0, 1, 2, 4, 5)]
                if arm != 'climatology':
                    y = get(f'{arm}_conformal|{per}_ungauged|{bid}')
                    if y is not None:
                        st['cf'][arm] = [sig(y[:, j]) for j in (1, 2, 4, 5)]
            for r in REFS:
                x = get(f'{r}|{per}_ref|{bid}')
                if x is not None:
                    st['ref'][r] = sig(x)
            for L in C.LEADS:
                g = {}
                for arm in GAUGED:
                    x = get(f'{arm}|{per}_gauged_L{L}|{bid}')
                    y = get(f'{arm}_conformal|{per}_gauged_L{L}|{bid}')
                    if x is not None:
                        g[arm] = [sig(x[:, j]) for j in (0, 1, 2, 4, 5)]
                    if y is not None:
                        g[arm + '_cf'] = [sig(y[:, j]) for j in (1, 2, 4, 5)]
                x = get(f'persistence|{per}_persistence_L{L}|{bid}')
                if x is not None:
                    g['persistence'] = sig(x)
                st['g'][f'L{L}'] = g
            rec['storms'].append(st)
        basins.append(rec)
    out = {'run_id': rid, 'basins': basins, 'missing_keys': sorted(set(missing))}
    json.dump(out, open(OUT, 'w'), separators=(',', ':'))
    print('wrote', OUT, os.path.getsize(OUT), 'bytes; missing keys:', len(set(missing)))


if __name__ == '__main__':
    main()
